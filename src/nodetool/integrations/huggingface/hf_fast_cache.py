"""
Fast, revision-agnostic view over the local Hugging Face Hub cache (async).

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
  * `HfFastCache` is safe for concurrent use across async tasks in a single process.
    All mutation of internal state (`_repos`) is guarded by an `asyncio.Lock`.
  * No background tasks are spawned; refresh happens on demand when mtimes indicate
    something changed.
"""

from __future__ import annotations

import asyncio
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import aiofiles
import aiofiles.os

from nodetool.ml.models.model_cache import ModelCache

DEFAULT_MODEL_INFO_CACHE_TTL = 30 * 24 * 3600


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
    Fast, read-only view over the local HF file cache (async).

    Key properties:
      - No full cache walk during normal operation.
      - For each repo, only the "current" snapshot is tracked, preferring refs/main.
      - Change detection uses mtime of refs/main and the active snapshot directory.
      - Lookups are direct path joins inside the active snapshot.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        model_info_cache: ModelCache | None = None,
    ) -> None:
        """Initialize a fast view over a local Hugging Face cache.

        Args:
            cache_dir: Optional cache root. When omitted, uses
                :func:`get_default_hf_cache_dir`.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_hf_cache_dir()
        # Lazy-initialized lock to handle multiple event loops
        self._lock: asyncio.Lock | None = None
        self._lock_loop_id: int | None = None
        self._repos: dict[str, _RepoState] = {}
        # Share the model info cache between callers so metadata lookups can leverage the disk cache.
        self.model_info_cache = model_info_cache or ModelCache("model_info")

    def _get_lock(self) -> asyncio.Lock:
        """Get the asyncio lock, creating a new one if the event loop has changed.

        This handles the case where the cache is used from different event loops
        (e.g., when running in threaded execution mode with separate loops per thread).
        """
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            # No running loop - create a new lock that will bind when first used
            current_loop_id = None

        if self._lock is None or self._lock_loop_id != current_loop_id:
            self._lock = asyncio.Lock()
            self._lock_loop_id = current_loop_id

        return self._lock

    async def resolve(
        self,
        repo_id: str,
        relpath: str,
        repo_type: str | None = None,
        follow_symlinks: bool = False,
    ) -> str | None:
        """Resolve a repo-relative path into the local cache.

        This performs a constant-time join inside the active snapshot for the
        repo (no directory walks) and checks whether the target exists.

        Args:
            repo_id: Repository identifier such as ``"owner/name"`` or
                ``"name"``. Also accepts prefixed forms like
                ``"model/owner/name"`` or ``"dataset/owner/name"``.
            relpath: POSIX-style relative path inside the repo snapshot.
            repo_type: Optional repository type (``"model"``, ``"dataset"``,
                or ``"space"``). If omitted, tries models, datasets, then
                spaces in that order.
            follow_symlinks: Whether to return the real path of the underlying
                blob target instead of the snapshot path. Symlinks are not
                resolved if this is ``False``.

        Returns:
            Optional[str]: Absolute path to the file if present in the active
            snapshot, otherwise ``None``.
        """
        state = await self._ensure_repo_state(repo_id, repo_type)
        if state is None or state.snapshot_dir is None:
            return None

        rel = _normalize_relpath(relpath)
        candidate = state.snapshot_dir / rel

        try:
            exists = await aiofiles.os.path.exists(str(candidate))
            is_link = await aiofiles.os.path.islink(str(candidate))
        except OSError:
            return None

        if not (exists or is_link):
            return None

        if follow_symlinks:
            # Use a thread for realpath to avoid blocking the event loop on resolve
            resolved = await asyncio.to_thread(candidate.resolve, False)
            return str(resolved)
        return str(candidate)

    async def exists(self, repo_id: str, relpath: str, repo_type: str | None = None) -> bool:
        """Return whether a repo-relative path exists in the cache.

        Args:
            repo_id: Repository identifier such as ``"owner/name"``.
            relpath: POSIX-style relative path inside the repo snapshot.
            repo_type: Optional repository type (``"model"``, ``"dataset"``,
                or ``"space"``). If omitted, tries models, datasets, then
                spaces.

        Returns:
            bool: ``True`` if the file exists in the active snapshot, otherwise
            ``False``.
        """
        return (await self.resolve(repo_id, relpath, repo_type=repo_type)) is not None

    async def list_files(self, repo_id: str, repo_type: str | None = None) -> list[str]:
        """List files in the active snapshot for a repo.

        The first call for a given repo walks only that repo's active snapshot
        directory and builds a small in-memory index. Subsequent calls reuse
        the index until a change in refs or snapshot mtime is detected.

        Args:
            repo_id: Repository identifier such as ``"owner/name"``.
            repo_type: Optional repository type (``"model"``, ``"dataset"``,
                or ``"space"``).

        Returns:
            list[str]: POSIX-style relative paths present in the active snapshot.
                Returns an empty list if the repo is not cached locally.
        """
        state = await self._ensure_repo_state(repo_id, repo_type)
        if state is None or state.snapshot_dir is None or not await _exists(state.snapshot_dir):
            return []

        if state.file_index is not None:
            state.snapshot_file_count = len(state.file_index)
            return list(state.file_index.keys())

        files: list[str] = []
        for path in await _rglob_files_async(state.snapshot_dir):
            files.append(path.relative_to(state.snapshot_dir).as_posix())

        state.file_index = {rp: state.snapshot_dir / rp for rp in files}
        state.snapshot_file_count = len(files)
        return files

    async def invalidate(self, repo_id: str | None = None, repo_type: str | None = None) -> None:
        """Forget cached state for one repo or all repos.

        Args:
            repo_id: Repository identifier to invalidate. When ``None``, clears
                all cached state.
            repo_type: Optional repository type hint used to narrow the keys
                cleared when ``repo_id`` is provided.
        """
        async with self._get_lock():
            if repo_id is None:
                self._repos.clear()
                return
            key_candidates = _candidate_repo_keys(repo_id, repo_type)
            for key in key_candidates:
                self._repos.pop(key, None)

    async def repo_root(self, repo_id: str, repo_type: str | None = None) -> str | None:
        """Return the cache directory for a given repo.

        Args:
            repo_id: Repository identifier such as ``"owner/name"``.
            repo_type: Optional repository type (``"model"``, ``"dataset"``,
                or ``"space"``).

        Returns:
            Optional[str]: Absolute path to the repo directory under the cache
            root, or ``None`` if the repo is not present.
        """
        state = await self._ensure_repo_state(repo_id, repo_type, create_if_missing=False)
        return str(state.repo_dir) if state is not None else None

    async def active_snapshot_dir(self, repo_id: str, repo_type: str | None = None) -> str | None:
        """Return the active snapshot directory for a given repo.

        Args:

            repo_id: Repository identifier such as ``"owner/name"``.
            repo_type: Optional repository type (``"model"``, ``"dataset"``,
                or ``"space"``).

        Returns:
            Optional[str]: Absolute path to the active snapshot directory, or
            ``None`` if no snapshot is available locally.
        """
        state = await self._ensure_repo_state(repo_id, repo_type)
        return str(state.snapshot_dir) if state and state.snapshot_dir else None

    async def discover_repos(self, repo_type: str = "model") -> list[tuple[str, Path]]:
        """Discover cached Hugging Face repos by listing the cache directory.

        This is lightweight compared to scan_cache_dir as it only lists directories
        without walking the entire tree.

        Args:
            repo_type: Repository type ("model", "dataset", or "space")

        Returns:
            List of tuples (repo_id, repo_dir) for discovered repos
        """
        if not await _exists(self.cache_dir):
            return []

        type_prefix = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        repos: list[tuple[str, Path]] = []

        try:
            for name in await aiofiles.os.listdir(str(self.cache_dir)):
                item = self.cache_dir / name
                if not await _is_dir(item):
                    continue
                if not item.name.startswith(f"{type_prefix}--"):
                    continue

                parts = item.name[len(f"{type_prefix}--") :].split("--")
                repo_id = "/".join(parts) if len(parts) > 1 else parts[0]
                repos.append((repo_id, item))
        except OSError:
            return []

        return repos

    async def _ensure_repo_state(
        self,
        repo_id: str,
        repo_type: str | None,
        create_if_missing: bool = True,
    ) -> _RepoState | None:
        """Load or refresh cached state for a repo.

        This method is responsible for creating and maintaining the in-memory
        :class:`_RepoState` for a repo. It:

        * Resolves the on-disk repo directory.
        * Reads refs to determine the active commit.
        * Picks an appropriate snapshot directory.
        * Detects when refs or snapshot mtimes change and refreshes state.

        Args:
            repo_id: Repository identifier such as ``"owner/name"``.
            repo_type: Optional repository type hint.
            create_if_missing: When ``False``, returns ``None`` if the repo is
                not already tracked or present on disk.

        Returns:
            Optional[_RepoState]: The current state object for the repo, or
            ``None`` if the repo is not present and ``create_if_missing`` is
            ``False``.
        """
        key_candidates = _candidate_repo_keys(repo_id, repo_type)

        async with self._get_lock():
            for key in key_candidates:
                state = self._repos.get(key)
                if state is not None:
                    if not await _exists(state.repo_dir):
                        self._repos.pop(key, None)
                        continue
                    await self._maybe_refresh_state(state)
                    return state

            for key in key_candidates:
                repo_type_normalized, norm_repo_id = key.split(":", 1)
                repo_dir = await _find_repo_dir_async(self.cache_dir, norm_repo_id, repo_type_normalized)
                if repo_dir is None:
                    continue
                state = _RepoState(
                    repo_id=norm_repo_id,
                    repo_type=repo_type_normalized,
                    repo_dir=repo_dir,
                )
                await self._populate_initial_state(state)
                self._repos[key] = state
                return state

            return None

    async def _maybe_refresh_state(self, state: _RepoState) -> None:
        """Refresh a repo state if refs or snapshot mtimes changed."""
        refs_mtime_now, commit_now = await _read_current_ref_async(state.repo_dir)
        snapshot_dir_now = await _snapshot_dir_for_commit_async(state.repo_dir, commit_now)

        if _changed(refs_mtime_now, state.refs_mtime) or commit_now != state.commit:
            state.commit = commit_now
            state.refs_mtime = refs_mtime_now
            state.snapshot_dir = snapshot_dir_now
            state.snapshot_mtime = await _mtime_or_none_async(snapshot_dir_now)
            state.file_index = None
            state.snapshot_file_count = None
            return

        snap_mtime_now = await _mtime_or_none_async(snapshot_dir_now)
        snapshot_changed = _changed(snap_mtime_now, state.snapshot_mtime)
        if snapshot_changed:
            state.snapshot_mtime = snap_mtime_now
            state.file_index = None
            state.snapshot_file_count = None
        elif state.file_index is not None and snapshot_dir_now is not None:
            current_count = await _count_files_async(snapshot_dir_now)
            if state.snapshot_file_count is None or current_count != state.snapshot_file_count:
                state.file_index = None
            state.snapshot_file_count = current_count

        if state.commit is None and (state.snapshot_dir is None or not await _exists(state.snapshot_dir)):
            state.snapshot_dir = await _pick_latest_snapshot_async(state.repo_dir)
            state.snapshot_mtime = await _mtime_or_none_async(state.snapshot_dir)
            state.file_index = None
            state.snapshot_file_count = None

    async def _populate_initial_state(self, state: _RepoState) -> None:
        """Populate repo state from refs and snapshots on first discovery."""
        refs_mtime, commit = await _read_current_ref_async(state.repo_dir)
        state.commit = commit
        state.refs_mtime = refs_mtime

        if commit:
            state.snapshot_dir = await _snapshot_dir_for_commit_async(state.repo_dir, commit)
        else:
            state.snapshot_dir = await _pick_latest_snapshot_async(state.repo_dir)

        state.snapshot_mtime = await _mtime_or_none_async(state.snapshot_dir)


@dataclass
class _RepoState:
    """In-memory view of a single Hugging Face repo in the cache."""

    repo_id: str
    repo_type: str
    repo_dir: Path
    commit: str | None = None
    refs_mtime: float | None = None
    snapshot_dir: Path | None = None
    snapshot_mtime: float | None = None
    snapshot_file_count: int | None = None
    file_index: dict[str, Path] | None = field(default=None)


def _candidate_repo_keys(repo_id: str, repo_type: str | None) -> list[str]:
    """Return ordered cache keys to try for locating a repo.

    The function normalizes the repo identifier and type and produces keys of
    the form ``"models:owner/name"``. It also handles prefixed repo IDs such
    as ``"model/owner/name"``.
    """
    norm_type, norm_repo = _normalize_repo_id_and_type(repo_id, repo_type)
    types_to_try = [norm_type] if norm_type else ["models", "datasets", "spaces"]

    keys: list[str] = []
    for repo_type_candidate in types_to_try:
        keys.append(f"{repo_type_candidate}:{norm_repo}")
    return keys


def _normalize_repo_id_and_type(repo_id: str, repo_type: str | None) -> tuple[str | None, str]:
    """Normalize repo ID and type into a canonical pair."""
    repo_id = repo_id.strip().strip("/")
    parts = repo_id.split("/")
    inferred_type: str | None = None
    if parts and parts[0] in {
        "model",
        "dataset",
        "space",
        "models",
        "datasets",
        "spaces",
    }:
        prefix = parts[0]
        inferred_type = _normalize_repo_type(prefix)
        repo_id = "/".join(parts[1:]) if len(parts) > 1 else ""

    norm_type = _normalize_repo_type(repo_type) if repo_type else inferred_type
    return norm_type, repo_id


def _normalize_repo_type(repo_type: str | None) -> str | None:
    """Normalize a repo type string to the internal form."""
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


async def _find_repo_dir_async(cache_dir: Path, repo_id: str, repo_type: str) -> Path | None:
    """Translate a repo ID into an on-disk directory under the cache."""
    assert repo_type in {"models", "datasets", "spaces"}
    repo_bits = [bit for bit in repo_id.split("/") if bit]
    if not repo_bits:
        return None

    candidates: list[str] = []
    if len(repo_bits) == 1:
        candidates.append(f"{repo_type}--{repo_bits[0]}")
    else:
        candidates.append(f"{repo_type}--{'--'.join(repo_bits)}")

    for name in candidates:
        path = cache_dir / name
        if await _is_dir(path):
            return path
    return None


async def _read_current_ref_async(
    repo_dir: Path,
) -> tuple[float | None, str | None]:
    """Return the mtime and commit hash for the preferred ref."""
    refs_dir = repo_dir / "refs"
    if not await _exists(refs_dir):
        return None, None

    main = refs_dir / "main"
    if await _exists(main):
        return await _mtime_or_none_async(main), await _read_first_line_async(main)

    newest_mtime: float | None = None
    newest_commit: str | None = None
    try:
        for name in await aiofiles.os.listdir(str(refs_dir)):
            file = refs_dir / name
            if not await _is_file(file):
                continue
            mt = await _mtime_or_none_async(file)
            if newest_mtime is None or (mt is not None and mt > newest_mtime):
                newest_mtime = mt
                newest_commit = await _read_first_line_async(file)
    except OSError:
        return None, None

    return (
        (newest_mtime if newest_mtime is not None else await _mtime_or_none_async(refs_dir)),
        newest_commit,
    )


async def _snapshot_dir_for_commit_async(repo_dir: Path, commit: str | None) -> Path | None:
    """Return the snapshot directory for a given commit hash."""
    if not commit:
        return None
    snapshot = repo_dir / "snapshots" / commit.strip()
    return snapshot if await _exists(snapshot) else None


async def _pick_latest_snapshot_async(repo_dir: Path) -> Path | None:
    """Return the newest snapshot directory for a repo by mtime."""
    snapshots_dir = repo_dir / "snapshots"
    if not await _exists(snapshots_dir):
        return None
    newest_path: Path | None = None
    newest_mtime: float | None = None
    try:
        for name in await aiofiles.os.listdir(str(snapshots_dir)):
            path = snapshots_dir / name
            if not await _is_dir(path):
                continue
            mt = await _mtime_or_none_async(path)
            if newest_mtime is None or (mt is not None and mt > newest_mtime):
                newest_mtime = mt
                newest_path = path
    except OSError:
        return None
    return newest_path


async def _read_first_line_async(path: Path) -> str | None:
    """Read and return the first line of a text file."""
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            line = await handle.readline()
        stripped = line.strip()
        return stripped or None
    except OSError:
        return None


async def _mtime_or_none_async(path: Path | None) -> float | None:
    """Return the mtime for a path, or ``None`` on error."""
    if path is None:
        return None
    try:
        stat_result = await aiofiles.os.stat(str(path))
        return stat_result.st_mtime
    except OSError:
        return None


async def _exists(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        return await aiofiles.os.path.exists(str(path))
    except OSError:
        return False


async def _is_dir(path: Path) -> bool:
    try:
        st = await aiofiles.os.stat(str(path))
        return stat.S_ISDIR(st.st_mode)
    except OSError:
        return False


async def _is_file(path: Path) -> bool:
    try:
        st = await aiofiles.os.stat(str(path))
        return stat.S_ISREG(st.st_mode)
    except OSError:
        return False


async def _rglob_files_async(root: Path) -> list[Path]:
    """Recursively collect files and symlinks under root without blocking the loop."""
    results: list[Path] = []

    async def _walk(dir_path: Path) -> None:
        try:
            entries = await aiofiles.os.listdir(str(dir_path))
        except OSError:
            return

        for name in entries:
            full = dir_path / name
            try:
                if await _is_dir(full):
                    await _walk(full)
                    continue
                is_file = await _is_file(full)
                is_link = await aiofiles.os.path.islink(str(full))
            except OSError:
                continue
            if is_file or is_link:
                results.append(full)

    await _walk(root)
    return results


async def _count_files_async(root: Path) -> int:
    """Recursively count files and symlinks under root without storing paths."""
    count = 0

    async def _walk(dir_path: Path) -> None:
        nonlocal count
        try:
            entries = await aiofiles.os.listdir(str(dir_path))
        except OSError:
            return

        for name in entries:
            full = dir_path / name
            try:
                if await _is_dir(full):
                    await _walk(full)
                    continue
                is_file = await _is_file(full)
                is_link = await aiofiles.os.path.islink(str(full))
            except OSError:
                continue
            if is_file or is_link:
                count += 1

    await _walk(root)
    return count


def _normalize_relpath(path: str) -> Path:
    """Normalize a repo-relative path into a :class:`Path`."""
    path = path.replace("\\", "/").lstrip("/")
    return Path(path)


def _changed(now: float | None, old: float | None) -> bool:
    """Return whether two timestamps represent a change."""
    if now is None and old is None:
        return False
    if now is None or old is None:
        return True
    return now != old


if __name__ == "__main__":

    async def main():
        cache = HfFastCache()
        exists = await cache.exists(
            "nunchaku-tech/nunchaku-flux.1-schnell",
            "svdq-int4_r32-flux.1-schnell.safetensors",
        )
        print(exists)

    asyncio.run(main())
