"""
Fast, revision-agnostic view over the local Hugging Face Hub cache.

This module provides `HfFastCache`, a tiny, read-only index on top of the on-disk
hub layout (e.g. `~/.cache/huggingface/hub`). It is optimized for *local* cache
queries and intentionally avoids:

  * Importing `huggingface_hub` (no extra import-time side effects).
  * Walking the entire cache tree just to answer per-repo questions.
  * Performing any HTTP calls.

High-level behavior
-------------------
  * Cache resolution is per repo. For a given `repo_id`, we:
      - Locate the repo directory (`models--org--name`, `datasets--...`, etc.).
      - Identify the "active" snapshot:
          - Prefer the commit pointed to by `refs/main` when present.
          - Otherwise, fall back to the newest snapshot directory by mtime.
      - Track only that snapshot directory for lookups.
  * Change detection is driven by mtimes:
      - We cache, per repo, the mtime of `refs/main` (or the newest ref file).
      - We also cache the mtime of the active snapshot directory.
      - On each access, if either mtime changes, we refresh the state and clear
        any cached file index for that repo.
  * Lookups are O(1) path joins:
      - For `resolve(repo_id, relpath)`, we simply join `snapshot_dir / relpath`
        and check existence (no directory walks).
      - `list_files(repo_id)` walks only the active snapshot once and then reuses
        an in-memory index keyed by relative path for subsequent calls.

Thread-safety / concurrency
---------------------------
  * `HfFastCache` is safe for concurrent use across threads in a single process.
    All mutation of internal state (`_repos`) is guarded by an `RLock`.
  * No background threads are spawned; refresh happens on demand in the caller’s
    thread when mtimes indicate something changed.

Intended usage
--------------
  * Fast existence checks before we attempt a download:

        cache = HfFastCache()
        if cache.exists(\"org/repo\", \"config.json\", repo_type=\"model\"):
            ...

  * Resolving a local path for a known repo + filename:

        path = cache.resolve(\"org/repo\", \"models/model.safetensors\")
        if path is not None:
            # Use the file at `path` without any network access.

  * Admin operations on a specific repo (e.g., deletion, listing):

        root = cache.repo_root(\"org/repo\", repo_type=\"model\")
        snapshot_files = cache.list_files(\"org/repo\", repo_type=\"model\")

The implementation is deliberately minimal and self-contained so it can be used
early in process startup and from environments where importing the full
`huggingface_hub` package is undesirable.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_default_hf_cache_dir() -> Path:
    """Return the default Hugging Face Hub cache directory.

    The directory is resolved without importing ``huggingface_hub`` so this
    function is safe to call in lightweight environments and early in process
    startup.

    Resolution order:
      1. ``$HF_HUB_CACHE`` if set.
      2. ``$HF_HOME/hub`` if set.
      3. ``~/.cache/huggingface/hub`` as a final fallback.

    Returns:
        Path: Absolute path to the cache directory.
    """
    env_cache = os.getenv("HF_HUB_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


class HfFastCache:
    """
    Fast, read-only view over the local HF file cache.

    Key properties:
      - No full cache walk during normal operation.
      - For each repo, only the "current" snapshot is tracked, preferring refs/main.
      - Change detection uses mtime of refs/main and the active snapshot directory.
      - Lookups are direct path joins inside the active snapshot.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize a fast view over a local Hugging Face cache.

        Args:
            cache_dir: Optional cache root. When omitted, uses
                :func:`get_default_hf_cache_dir`.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_hf_cache_dir()
        self._lock = threading.RLock()
        self._repos: Dict[str, _RepoState] = {}

    def resolve(
        self,
        repo_id: str,
        relpath: str,
        repo_type: Optional[str] = None,
        follow_symlinks: bool = False,
    ) -> Optional[str]:
        """Resolve a repo-relative path into the local cache.

        This performs a constant-time join inside the active snapshot for the
        repo (no directory walks) and checks whether the target exists.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"`` or
                ``\"name\"``. Also accepts prefixed forms like
                ``\"model/owner/name\"`` or ``\"dataset/owner/name\"``.
            relpath: POSIX-style relative path inside the repo snapshot.
            repo_type: Optional repository type (``\"model\"``, ``\"dataset\"``,
                or ``\"space\"``). If omitted, tries models, datasets, then
                spaces in that order.
            follow_symlinks: Whether to return the real path of the underlying
                blob target instead of the snapshot path. Symlinks are not
                resolved if this is ``False``.

        Returns:
            Optional[str]: Absolute path to the file if present in the active
            snapshot, otherwise ``None``.
        """
        state = self._ensure_repo_state(repo_id, repo_type)
        if state is None or state.snapshot_dir is None:
            return None

        rel = _normalize_relpath(relpath)
        candidate = state.snapshot_dir / rel

        if candidate.exists() or candidate.is_symlink():
            return str(candidate.resolve(strict=False) if follow_symlinks else candidate)
        return None

    def exists(self, repo_id: str, relpath: str, repo_type: Optional[str] = None) -> bool:
        """Return whether a repo-relative path exists in the cache.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"``.
            relpath: POSIX-style relative path inside the repo snapshot.
            repo_type: Optional repository type (``\"model\"``, ``\"dataset\"``,
                or ``\"space\"``). If omitted, tries models, datasets, then
                spaces.

        Returns:
            bool: ``True`` if the file exists in the active snapshot, otherwise
            ``False``.
        """
        return self.resolve(repo_id, relpath, repo_type=repo_type) is not None

    def list_files(self, repo_id: str, repo_type: Optional[str] = None) -> List[str]:
        """List files in the active snapshot for a repo.

        The first call for a given repo walks only that repo's active snapshot
        directory and builds a small in-memory index. Subsequent calls reuse
        the index until a change in refs or snapshot mtime is detected.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"``.
            repo_type: Optional repository type (``\"model\"``, ``\"dataset\"``,
                or ``\"space\"``).

        Returns:
            list[str]: POSIX-style relative paths present in the active snapshot.
                Returns an empty list if the repo is not cached locally.
        """
        state = self._ensure_repo_state(repo_id, repo_type)
        if state is None or state.snapshot_dir is None or not state.snapshot_dir.exists():
            return []

        if state.file_index is not None:
            return list(state.file_index.keys())

        files: List[str] = []
        for p in state.snapshot_dir.rglob("*"):
            if p.is_file() or p.is_symlink():
                files.append(p.relative_to(state.snapshot_dir).as_posix())

        state.file_index = {rp: state.snapshot_dir / rp for rp in files}
        return files

    def invalidate(self, repo_id: Optional[str] = None, repo_type: Optional[str] = None) -> None:
        """Forget cached state for one repo or all repos.

        Args:
            repo_id: Repository identifier to invalidate. When ``None``, clears
                all cached state.
            repo_type: Optional repository type hint used to narrow the keys
                cleared when ``repo_id`` is provided.
        """
        with self._lock:
            if repo_id is None:
                self._repos.clear()
                return
            key_candidates = _candidate_repo_keys(repo_id, repo_type)
            for key in key_candidates:
                self._repos.pop(key, None)

    def repo_root(self, repo_id: str, repo_type: Optional[str] = None) -> Optional[str]:
        """Return the cache directory for a given repo.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"``.
            repo_type: Optional repository type (``\"model\"``, ``\"dataset\"``,
                or ``\"space\"``).

        Returns:
            Optional[str]: Absolute path to the repo directory under the cache
            root, or ``None`` if the repo is not present.
        """
        state = self._ensure_repo_state(repo_id, repo_type, create_if_missing=False)
        return str(state.repo_dir) if state is not None else None

    def active_snapshot_dir(self, repo_id: str, repo_type: Optional[str] = None) -> Optional[str]:
        """Return the active snapshot directory for a given repo.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"``.
            repo_type: Optional repository type (``\"model\"``, ``\"dataset\"``,
                or ``\"space\"``).

        Returns:
            Optional[str]: Absolute path to the active snapshot directory, or
            ``None`` if no snapshot is available locally.
        """
        state = self._ensure_repo_state(repo_id, repo_type)
        return str(state.snapshot_dir) if state and state.snapshot_dir else None

    def _ensure_repo_state(
        self,
        repo_id: str,
        repo_type: Optional[str],
        create_if_missing: bool = True,
    ) -> Optional["_RepoState"]:
        """Load or refresh cached state for a repo.

        This method is responsible for creating and maintaining the in-memory
        :class:`_RepoState` for a repo. It:

        * Resolves the on-disk repo directory.
        * Reads refs to determine the active commit.
        * Picks an appropriate snapshot directory.
        * Detects when refs or snapshot mtimes change and refreshes state.

        Args:
            repo_id: Repository identifier such as ``\"owner/name\"``.
            repo_type: Optional repository type hint.
            create_if_missing: When ``False``, returns ``None`` if the repo is
                not already tracked or present on disk.

        Returns:
            Optional[_RepoState]: The current state object for the repo, or
            ``None`` if the repo is not present and ``create_if_missing`` is
            ``False``.
        """
        key_candidates = _candidate_repo_keys(repo_id, repo_type)

        with self._lock:
            for key in key_candidates:
                state = self._repos.get(key)
                if state is not None:
                    if not state.repo_dir.exists():
                        self._repos.pop(key, None)
                        continue
                    self._maybe_refresh_state(state)
                    return state

            for key in key_candidates:
                repo_type_normalized, norm_repo_id = key.split(":", 1)
                repo_dir = _find_repo_dir(self.cache_dir, norm_repo_id, repo_type_normalized)
                if repo_dir is None:
                    continue
                state = _RepoState(repo_id=norm_repo_id, repo_type=repo_type_normalized, repo_dir=repo_dir)
                self._populate_initial_state(state)
                self._repos[key] = state
                return state

            return None if not create_if_missing else None

    def _maybe_refresh_state(self, state: "_RepoState") -> None:
        """Refresh a repo state if refs or snapshot mtimes changed.

        This method is cheap to call and is invoked on each access to ensure
        the in-memory view stays in sync with the on-disk cache.

        Args:
            state: Repo state to refresh.
        """
        refs_mtime_now, commit_now = _read_current_ref(state.repo_dir)
        snapshot_dir_now = _snapshot_dir_for_commit(state.repo_dir, commit_now)

        if _changed(refs_mtime_now, state.refs_mtime) or commit_now != state.commit:
            state.commit = commit_now
            state.refs_mtime = refs_mtime_now
            state.snapshot_dir = snapshot_dir_now
            state.snapshot_mtime = _mtime_or_none(snapshot_dir_now)
            state.file_index = None
            return

        snap_mtime_now = _mtime_or_none(snapshot_dir_now)
        if _changed(snap_mtime_now, state.snapshot_mtime):
            state.snapshot_mtime = snap_mtime_now
            state.file_index = None

        if state.commit is None and (state.snapshot_dir is None or not state.snapshot_dir.exists()):
            state.snapshot_dir = _pick_latest_snapshot(state.repo_dir)
            state.snapshot_mtime = _mtime_or_none(state.snapshot_dir)
            state.file_index = None

    def _populate_initial_state(self, state: "_RepoState") -> None:
        """Populate repo state from refs and snapshots on first discovery.

        Args:
            state: Repo state to initialize. ``repo_dir`` must be set.
        """
        refs_mtime, commit = _read_current_ref(state.repo_dir)
        state.commit = commit
        state.refs_mtime = refs_mtime

        if commit:
            state.snapshot_dir = _snapshot_dir_for_commit(state.repo_dir, commit)
        else:
            state.snapshot_dir = _pick_latest_snapshot(state.repo_dir)

        state.snapshot_mtime = _mtime_or_none(state.snapshot_dir)


@dataclass
class _RepoState:
    """In-memory view of a single Hugging Face repo in the cache."""

    repo_id: str
    repo_type: str
    repo_dir: Path
    commit: Optional[str] = None
    refs_mtime: Optional[float] = None
    snapshot_dir: Optional[Path] = None
    snapshot_mtime: Optional[float] = None
    file_index: Optional[Dict[str, Path]] = field(default=None)


def _candidate_repo_keys(repo_id: str, repo_type: Optional[str]) -> List[str]:
    """Return ordered cache keys to try for locating a repo.

    The function normalizes the repo identifier and type and produces keys of
    the form ``\"models:owner/name\"``. It also handles prefixed repo IDs such
    as ``\"model/owner/name\"``.

    Args:
        repo_id: Raw repository identifier (with or without type prefix).
        repo_type: Optional repository type hint.

    Returns:
        list[str]: Candidate keys in priority order.
    """
    norm_type, norm_repo = _normalize_repo_id_and_type(repo_id, repo_type)
    types_to_try = [norm_type] if norm_type else ["models", "datasets", "spaces"]

    keys: List[str] = []
    for repo_type_candidate in types_to_try:
        keys.append(f"{repo_type_candidate}:{norm_repo}")
    return keys


def _normalize_repo_id_and_type(repo_id: str, repo_type: Optional[str]) -> Tuple[Optional[str], str]:
    """Normalize repo ID and type into a canonical pair.

    This helper understands repo IDs that may be prefixed with their type
    (e.g. ``\"model/owner/name\"``) and uses that information when present.

    Args:
        repo_id: Raw repository identifier, possibly including a type prefix.
        repo_type: Optional repository type hint.

    Returns:
        Tuple[Optional[str], str]: A tuple ``(normalized_type, normalized_repo)``
        where ``normalized_type`` is one of ``\"models\"``, ``\"datasets\"``,
        ``\"spaces\"``, or ``None`` when no type is known.
    """
    repo_id = repo_id.strip().strip("/")
    parts = repo_id.split("/")
    inferred_type: Optional[str] = None
    if parts and parts[0] in {"model", "dataset", "space", "models", "datasets", "spaces"}:
        prefix = parts[0]
        inferred_type = _normalize_repo_type(prefix)
        repo_id = "/".join(parts[1:]) if len(parts) > 1 else ""

    norm_type = _normalize_repo_type(repo_type) if repo_type else inferred_type
    return norm_type, repo_id


def _normalize_repo_type(repo_type: Optional[str]) -> Optional[str]:
    """Normalize a repo type string to the internal form.

    Args:
        repo_type: Raw repo type (e.g. ``\"model\"``, ``\"datasets\"``).

    Returns:
        Optional[str]: Normalized type (``\"models\"``, ``\"datasets\"``,
        ``\"spaces\"``), or ``None`` if ``repo_type`` is ``None``.

    Raises:
        ValueError: If ``repo_type`` is not recognized.
    """
    if repo_type is None:
        return None
    t = repo_type.lower().strip()
    if t in {"model", "models"}:
        return "models"
    if t in {"dataset", "datasets"}:
        return "datasets"
    if t in {"space", "spaces"}:
        return "spaces"
    raise ValueError(f"Unknown repo_type: {repo_type}")


def _find_repo_dir(cache_dir: Path, repo_id: str, repo_type: str) -> Optional[Path]:
    """Translate a repo ID into an on-disk directory under the cache.

    On disk the naming convention is one of:

      * ``models--owner--name``
      * ``models--name``
      * ``datasets--owner--name``
      * ``spaces--owner--name``

    Args:
        cache_dir: Root of the Hugging Face Hub cache.
        repo_id: Normalized repo identifier (no type prefix).
        repo_type: Normalized repo type (``\"models\"``, ``\"datasets\"``,
            or ``\"spaces\"``).

    Returns:
        Optional[Path]: Path to the repo directory, or ``None`` if it does
        not exist.
    """
    assert repo_type in {"models", "datasets", "spaces"}
    repo_bits = [bit for bit in repo_id.split("/") if bit]
    if not repo_bits:
        return None

    candidates: List[str] = []
    if len(repo_bits) == 1:
        candidates.append(f"{repo_type}--{repo_bits[0]}")
    else:
        candidates.append(f"{repo_type}--{'--'.join(repo_bits)}")

    for name in candidates:
        path = cache_dir / name
        if path.is_dir():
            return path
    return None


def _read_current_ref(repo_dir: Path) -> Tuple[Optional[float], Optional[str]]:
    """Return the mtime and commit hash for the preferred ref.

    Preference order:

      1. ``refs/main`` if present.
      2. Newest regular file inside ``refs/`` by mtime.
      3. ``None`` if no refs exist.

    Args:
        repo_dir: Path to the repo directory.

    Returns:
        Tuple[Optional[float], Optional[str]]: A pair of
        ``(refs_mtime, commit_hash)``, where either element may be ``None``.
    """
    refs_dir = repo_dir / "refs"
    if not refs_dir.exists():
        return None, None

    main = refs_dir / "main"
    if main.exists():
        return _mtime_or_none(main), _read_first_line(main)

    newest_mtime: Optional[float] = None
    newest_commit: Optional[str] = None
    for file in refs_dir.iterdir():
        if file.is_file():
            mt = _mtime_or_none(file)
            if newest_mtime is None or (mt is not None and mt > newest_mtime):
                newest_mtime = mt
                newest_commit = _read_first_line(file)

    return (newest_mtime if newest_mtime is not None else _mtime_or_none(refs_dir), newest_commit)


def _snapshot_dir_for_commit(repo_dir: Path, commit: Optional[str]) -> Optional[Path]:
    """Return the snapshot directory for a given commit hash.

    Args:
        repo_dir: Path to the repo directory.
        commit: Commit hash string or ``None``.

    Returns:
        Optional[Path]: Snapshot directory path if it exists, otherwise
        ``None``.
    """
    if not commit:
        return None
    snapshot = repo_dir / "snapshots" / commit.strip()
    return snapshot if snapshot.exists() else None


def _pick_latest_snapshot(repo_dir: Path) -> Optional[Path]:
    """Return the newest snapshot directory for a repo by mtime.

    Args:
        repo_dir: Path to the repo directory.

    Returns:
        Optional[Path]: Path to the newest snapshot directory, or ``None`` if
        no snapshots exist.
    """
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    newest_path: Optional[Path] = None
    newest_mtime: Optional[float] = None
    for path in snapshots_dir.iterdir():
        if path.is_dir():
            mt = _mtime_or_none(path)
            if newest_mtime is None or (mt is not None and mt > newest_mtime):
                newest_mtime = mt
                newest_path = path
    return newest_path


def _read_first_line(path: Path) -> Optional[str]:
    """Read and return the first line of a text file.

    Args:
        path: File path to read.

    Returns:
        Optional[str]: First line with surrounding whitespace stripped, or
        ``None`` if the file cannot be read or is empty.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            line = handle.readline().strip()
        return line or None
    except OSError:
        return None


def _mtime_or_none(path: Optional[Path]) -> Optional[float]:
    """Return the mtime for a path, or ``None`` on error.

    Args:
        path: Path to stat, or ``None``.

    Returns:
        Optional[float]: POSIX mtime timestamp, or ``None`` if the path is
        ``None`` or cannot be stat'ed.
    """
    try:
        if path is None:
            return None
        stat_result = path.stat()
        return stat_result.st_mtime
    except OSError:
        return None


def _normalize_relpath(path: str) -> Path:
    """Normalize a repo-relative path into a :class:`Path`.

    This converts Windows separators to POSIX-style ``/`` and strips any
    leading slashes so that the result is always relative.

    Args:
        path: Raw path string (POSIX or Windows style).

    Returns:
        Path: Normalized relative path.
    """
    path = path.replace("\\", "/").lstrip("/")
    return Path(path)


def _changed(now: Optional[float], old: Optional[float]) -> bool:
    """Return whether two timestamps represent a change.

    ``None`` is treated as "unknown"; transitioning between ``None`` and a
    real value always counts as a change.

    Args:
        now: Current timestamp or ``None``.
        old: Previous timestamp or ``None``.

    Returns:
        bool: ``True`` if the timestamps differ, otherwise ``False``.
    """
    if now is None and old is None:
        return False
    if now is None or old is None:
        return True
    return now != old
