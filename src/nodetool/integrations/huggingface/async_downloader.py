import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


HF_ENDPOINT = "https://huggingface.co"
HF_HEADER_X_REPO_COMMIT = "X-Repo-Commit"
HF_HEADER_X_LINKED_ETAG = "X-Linked-Etag"
HF_HEADER_X_LINKED_SIZE = "X-Linked-Size"

# Simple in-process cache for the resolved HF token
_CACHED_HF_TOKEN: Optional[str] = None

# Same-origin redirect hops hf_head_metadata will follow while it still has no
# ETag. One is enough for a renamed repo; the cap only has to stop a loop.
_MAX_METADATA_REDIRECTS = 5

# Exponential backoff between resumable-download retries, in seconds.
_RETRY_BACKOFF_BASE = 1.0
_RETRY_BACKOFF_MAX = 30.0


@dataclass
class HfFileMeta:
    # Metadata returned by HEAD
    url: str  # final download URL (possibly CDN)
    etag: str  # normalized without quotes
    size: Optional[int]  # bytes, None if unknown
    commit_hash: Optional[str]
    accept_ranges: bool
    original_url: str  # original /resolve url


def _validate_path_component(component: str, *, allow_subdirs: bool = False) -> str:
    """Validate a server/listing-controlled value used to build a cache path.

    Rejects absolute paths, parent-directory traversal (``..``), NUL bytes and
    (unless ``allow_subdirs``) embedded path separators. Returns the normalized
    (forward-slash) value. Mirrors the ``_validate_relpath`` guard used by the
    worker's comfy handler.

    Args:
        component: Raw value (e.g. an ETag, commit hash or filename).
        allow_subdirs: When True, forward-slash subpaths such as
            ``"onnx/model.onnx"`` are permitted (but ``..`` segments are not).
    """
    if not component:
        raise ValueError("Empty path component")
    if "\x00" in component:
        raise ValueError(f"Invalid NUL byte in path component: {component!r}")
    candidate = component.replace("\\", "/")
    p = Path(candidate)
    if p.is_absolute():
        raise ValueError(f"Absolute path not allowed: {component!r}")
    if any(seg in ("..", "") for seg in p.parts):
        raise ValueError(f"Unsafe path component: {component!r}")
    if not allow_subdirs and len(p.parts) > 1:
        raise ValueError(f"Path separators not allowed in component: {component!r}")
    return candidate


def _same_origin(a: str, b: str) -> bool:
    """True when two URLs share scheme and host:port."""
    pa, pb = urlparse(a), urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _env_bool(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    return v.strip().upper() in {"1", "TRUE", "YES", "ON"}


def _hf_home_dir() -> Path:
    """
    HF_HOME rules:
      - HF_HOME
      - else XDG_CACHE_HOME/huggingface
      - else ~/.cache/huggingface
    """
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def _get_hf_token_from_env() -> Optional[str]:
    """
    Resolve token from env, preferring HF_TOKEN, but also
    accepting some common alternatives.
    """
    for name in ("HF_TOKEN", "HF_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def _get_hf_token_from_file() -> Optional[str]:
    """
    Resolve token from token file:
      - HF_TOKEN_PATH, or
      - HF_HOME/token (HF_HOME default is ~/.cache/huggingface)
    """
    token_path = os.getenv("HF_TOKEN_PATH")
    path = Path(token_path).expanduser() if token_path else _hf_home_dir() / "token"

    try:
        txt = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError):
        return None

    txt = txt.strip()
    return txt or None


def _get_cached_hf_token() -> Optional[str]:
    global _CACHED_HF_TOKEN
    if _CACHED_HF_TOKEN is not None:
        return _CACHED_HF_TOKEN

    token = _get_hf_token_from_env() or _get_hf_token_from_file()
    _CACHED_HF_TOKEN = token
    return token


def _resolve_hf_token(token: str | bool | None) -> Optional[str]:
    """
    Hugging Face style semantics:

      - token is str: use it
      - token is False: explicitly disable auth
      - token is True: require a locally saved token or error
      - token is None:
            if HF_HUB_DISABLE_IMPLICIT_TOKEN is set -> no token
            else -> use cached token if present, otherwise None
    """
    if token is False:
        return None

    if isinstance(token, str):
        return token

    cached = _get_cached_hf_token()
    disable_implicit = _env_bool("HF_HUB_DISABLE_IMPLICIT_TOKEN")

    if token is True:
        if cached is None:
            raise OSError(
                "Token is required (token=True), but no Hugging Face token "
                "was found in env or token file. Run `hf auth login` or set HF_TOKEN."
            )
        return cached

    # token is None
    if disable_implicit:
        return None
    return cached


def _hf_cache_root() -> Path:
    """
    Cache root resolution, mirroring huggingface_hub:

      - HF_HUB_CACHE (preferred)
      - HUGGINGFACE_HUB_CACHE (deprecated)
      - HF_HOME/hub
      - XDG_CACHE_HOME/huggingface/hub
      - ~/.cache/huggingface/hub
    """
    cache = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if cache:
        return Path(cache).expanduser()

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    xdg = os.getenv("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


_COMMIT_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


def _write_revision_ref(repo_cache: Path, revision: str, commit_hash: str) -> None:
    """Record `refs/<revision> -> commit_hash`, the way huggingface_hub does.

    Our cache layout matches huggingface_hub's `blobs/` + `snapshots/<commit>/`
    exactly, but nothing wrote the third directory it keeps. `from_pretrained`
    resolves a branch name (the default "main") to a commit by reading
    `refs/<revision>`; with the file absent, an offline load cannot find the
    snapshot sitting right next to it and raises `LocalEntryNotFoundError`.

    Written as soon as the commit is known, which is what huggingface_hub does
    too: a repo is fetched one file at a time, so there is no later moment at
    which the snapshot becomes "complete" enough to point at.
    """
    if not commit_hash or _COMMIT_HASH_RE.match(revision):
        # A revision that IS a commit hash needs no indirection.
        return
    try:
        safe_revision = _validate_path_component(revision, allow_subdirs=True)
    except ValueError:
        log.warning("Refusing to write a ref for unsafe revision %r", revision)
        return
    ref_path = repo_cache / "refs" / safe_revision
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ref_path.with_name(ref_path.name + ".tmp")
    tmp_path.write_text(commit_hash)
    os.replace(tmp_path, ref_path)


def _hf_repo_cache_dir(
    repo_id: str,
    repo_type: str = "model",
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Layout: <cache_dir>/<repo_type>s--namespace--name
    e.g. models--gpt2, datasets--wmt14--en-de
    """
    if cache_dir is None:
        cache_dir = _hf_cache_root()
    parts = [f"{repo_type}s", *repo_id.split("/")]
    folder = "--".join(parts)
    return cache_dir / folder


def hf_hub_file_url(
    repo_id: str,
    filename: str,
    revision: str = "main",
    repo_type: str = "model",
    endpoint: str = HF_ENDPOINT,
) -> str:
    """
    Recreate huggingface_hub.hf_hub_url:

      - model:   https://huggingface.co/{repo_id}/resolve/{rev}/{filename}
      - dataset: https://huggingface.co/datasets/{repo_id}/resolve/{rev}/{filename}
      - space:   https://huggingface.co/spaces/{repo_id}/resolve/{rev}/{filename}
    """
    if repo_type in (None, "model"):
        prefix = ""
    elif repo_type == "dataset":
        prefix = "datasets/"
    elif repo_type == "space":
        prefix = "spaces/"
    else:
        raise ValueError(f"Unsupported repo_type {repo_type!r}")
    filename = filename.lstrip("/")
    return f"{endpoint.rstrip('/')}/{prefix}{repo_id}/resolve/{revision}/{filename}"


async def hf_head_metadata(
    client: httpx.AsyncClient,
    url: str,
    token: Optional[str] = None,
    timeout: float = 10.0,
    user_agent: str = "custom-hf-downloader",
) -> HfFileMeta:
    """
    HEAD /resolve to get ETag, size, commit hash and final location.

    Hugging Face uses X-Linked-Etag and X-Linked-Size on redirects to LFS/CDN
    plus X-Repo-Commit for the resolved commit.

    The first hop usually carries them. A repo that has been RENAMED answers
    with a rename redirect first, and that hop carries no ETag and no commit —
    both appear one hop later. So when a redirect has no ETag, follow it, but
    only while the target stays on the same origin: the hop that leaves
    huggingface.co is the CDN hand-off, and its own ETag is a different hash
    than the one the cache layout is keyed on.

    Every header of the returned metadata comes from the same response, so the
    commit can never be paired with an ETag resolved on a different hop.
    """
    headers = {
        "Accept-Encoding": "identity",  # avoid compression; want real Content-Length
        "User-Agent": user_agent,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    current_url = url
    try:
        for _ in range(_MAX_METADATA_REDIRECTS + 1):
            resp = await client.head(
                current_url,
                headers=headers,
                follow_redirects=False,  # never chase the CDN hand-off here
                timeout=timeout,
            )
            if not (300 <= resp.status_code < 400):
                resp.raise_for_status()
                break

            # A redirect is expected for LFS/CDN, and it normally carries the
            # metadata. Take it and stop.
            if resp.headers.get(HF_HEADER_X_LINKED_ETAG) or resp.headers.get("ETag"):
                break

            location = resp.headers.get("Location")
            if not location:
                break
            target = urljoin(current_url, location)
            if not _same_origin(current_url, target):
                break
            current_url = target
        else:
            raise RuntimeError(
                f"Too many redirects (>{_MAX_METADATA_REDIRECTS}) resolving metadata for url={url!r}"
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            raise PermissionError(
                f"Unauthorized to access {url!r}. "
                f"Status: {e.response.status_code}. "
                "Check your Hugging Face token and permissions. "
                f"Token present: {bool(token)}"
            ) from e
        log.error(f"HTTP error fetching metadata for {url}: {e}")
        raise

    etag = resp.headers.get(HF_HEADER_X_LINKED_ETAG) or resp.headers.get("ETag")
    if etag is None:
        raise RuntimeError(f"No ETag received from Hugging Face for url={url!r}")
    if etag.startswith('"') and etag.endswith('"'):
        etag = etag[1:-1]

    size_header = resp.headers.get(HF_HEADER_X_LINKED_SIZE)
    if size_header is None and 200 <= resp.status_code < 300:
        size_header = resp.headers.get("Content-Length")
    size = int(size_header) if size_header is not None else None

    location = resp.headers.get("Location") or str(resp.url)
    if location and not location.startswith(("http://", "https://")):
        location = urljoin(str(resp.url), location)

    commit = resp.headers.get(HF_HEADER_X_REPO_COMMIT)
    accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"

    return HfFileMeta(
        url=location,
        etag=etag,
        size=size,
        commit_hash=commit,
        accept_ranges=accept_ranges,
        original_url=str(resp.url),
    )


async def _download_with_resume(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    *,
    token: Optional[str],
    expected_size: Optional[int],
    accept_ranges: bool,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
    timeout: float = 60.0,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Stream a file to disk with resume support using HTTP Range.

    - Writes to dest.with_suffix(dest.suffix + ".incomplete")
    - Uses Range: bytes=<offset>- when possible
    - On transient network errors, retries and resumes from the last byte
    """
    dest = dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".incomplete")

    # Final file already complete
    if dest.exists() and expected_size is not None:
        if dest.stat().st_size == expected_size:
            if progress_callback:
                progress_callback(expected_size, expected_size)
            return

    # Incomplete file already fully downloaded (crash after download)
    if tmp.exists() and expected_size is not None and tmp.stat().st_size == expected_size:
        tmp.replace(dest)
        if progress_callback:
            progress_callback(expected_size, expected_size)
        return

    attempt = 0
    while True:
        attempt += 1
        resume_from = tmp.stat().st_size if tmp.exists() else 0

        if expected_size is not None:
            if resume_from == expected_size:
                tmp.replace(dest)
                return
            if resume_from > expected_size:
                log.warning(f"Local file larger than expected ({resume_from} > {expected_size}). Restarting download.")
                tmp.unlink()
                resume_from = 0

        headers = {"Accept-Encoding": "identity"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if accept_ranges and resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            async with client.stream(
                "GET",
                url,
                headers=headers,
                follow_redirects=True,
                timeout=timeout,
            ) as resp:
                # Handle 416 Range Not Satisfiable
                if resp.status_code == 416:
                    log.warning(f"Range not satisfiable (resume_from={resume_from}). Restarting download.")
                    if tmp.exists():
                        tmp.unlink()
                    if attempt >= max_retries:
                        raise RuntimeError(f"Download failed after {attempt} attempts: server kept returning 416")
                    continue

                # Only keep the partial file when the server actually served a
                # partial response. Any other status carries the WHOLE body, so
                # appending it to the partial file would corrupt the result —
                # silently so when expected_size is unknown and the size check
                # below cannot run. This covers the ``accept_ranges is False``
                # case too, where no Range header was ever sent.
                if resume_from > 0 and resp.status_code != 206:
                    log.warning(
                        "Server did not honour Range for %s (status %s); restarting download.",
                        url,
                        resp.status_code,
                    )
                    if tmp.exists():
                        tmp.unlink()
                    resume_from = 0

                resp.raise_for_status()

                mode = "ab" if resume_from > 0 else "wb"
                with tmp.open(mode) as f:
                    async for chunk in resp.aiter_bytes(chunk_size):
                        if cancel_event and cancel_event.is_set():
                            log.debug(f"Download cancelled for {url}")
                            raise asyncio.CancelledError("Download cancelled")
                        if not chunk:
                            continue
                        f.write(chunk)
                        if progress_callback:
                            progress_callback(len(chunk), expected_size)

            if expected_size is not None:
                actual_size = tmp.stat().st_size
                if actual_size != expected_size:
                    raise RuntimeError(f"Size mismatch for {url}: expected {expected_size}, got {actual_size}")

            tmp.replace(dest)
            return

        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Download failed after {attempt} attempts") from exc
            # Back off before resuming from whatever is on disk. Without this,
            # a fast-failing error (connection refused, DNS failure) burns every
            # attempt in a tight loop, so the retries buy nothing.
            delay = min(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), _RETRY_BACKOFF_MAX)
            log.warning(
                "Download of %s failed (%s); retrying in %.1fs (attempt %d/%d)",
                url,
                exc,
                delay,
                attempt,
                max_retries,
            )
            await asyncio.sleep(delay)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 416:
                # Should be handled above, but just in case raise_for_status catches it first
                log.warning("Range not satisfiable (caught via exception). Restarting download.")
                if tmp.exists():
                    tmp.unlink()
                if attempt >= max_retries:
                    raise RuntimeError(f"Download failed after {attempt} attempts: server kept returning 416") from exc
                continue
            raise


async def async_hf_download(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    repo_type: str = "model",  # "model", "dataset", or "space"
    token: str | bool | None = None,
    cache_dir: Optional[Path] = None,
    client: Optional[httpx.AsyncClient] = None,
    chunk_size: int = 1024 * 1024,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> Path:
    """
    Async, minimal clone of huggingface_hub.hf_hub_download using bare httpx.

    Authentication behavior:

      - token="hf_xxx": use that value as Bearer token
      - token=True: require a local token (env or token file) and fail if none
      - token=False: never send a token, even if one is configured
      - token=None (default):
            use local token if available, unless HF_HUB_DISABLE_IMPLICIT_TOKEN=1

    Token is only sent to huggingface.co host, not to redirected S3/CDN URLs.
    """
    token_str = _resolve_hf_token(token)

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(verify=True)

    try:
        # 1) Build /resolve URL on the Hub
        resolve_url = hf_hub_file_url(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,
        )

        # 2) HEAD to get metadata (ETag, size, commit, redirect)
        log.info(f"async_hf_download: Fetching metadata for {repo_id}/{filename}")
        meta = await hf_head_metadata(client, resolve_url, token=token_str)
        log.info(f"async_hf_download: Metadata received. Size: {meta.size}, ETag: {meta.etag}")

        # 3) Compute cache paths (HF-style layout)
        if cache_dir is None:
            cache_dir = _hf_cache_root()
        cache_dir = cache_dir.expanduser()

        repo_cache = _hf_repo_cache_dir(repo_id, repo_type=repo_type, cache_dir=cache_dir)
        blobs_dir = repo_cache / "blobs"
        commit_or_rev = meta.commit_hash or revision

        # Sanitize server/listing-controlled values before joining them into
        # cache paths (etag + commit from response headers, filename from hub
        # listing) so they cannot traverse outside the intended directories.
        safe_etag = _validate_path_component(meta.etag)
        safe_commit = _validate_path_component(commit_or_rev)
        safe_filename = _validate_path_component(filename, allow_subdirs=True)

        snapshot_root = repo_cache / "snapshots" / safe_commit
        snapshot_path = snapshot_root / safe_filename
        # filename may legitimately contain subdirs; assert the joined path
        # stays within the snapshot commit directory.
        #
        # Resolve the ROOT only, and normalize the joined name textually.
        # Once a file has been downloaded, `snapshot_path` IS a symlink into
        # ../../blobs, and `Path.resolve()` follows it — so resolving the full
        # path reported every already-cached file as an escape and made a
        # second download of any repo fail. Normalizing the text answers the
        # question actually being asked (does this *name* escape?) without
        # consulting the filesystem.
        snapshot_root_resolved = snapshot_root.resolve()
        if not Path(
            os.path.normpath(snapshot_root_resolved / safe_filename)
        ).is_relative_to(snapshot_root_resolved):
            raise ValueError(f"Filename escapes snapshot directory: {filename!r}")
        blob_path = blobs_dir / safe_etag

        # Point the branch name at this commit so an offline load can resolve
        # the snapshot we are about to populate (see _write_revision_ref).
        _write_revision_ref(repo_cache, revision, meta.commit_hash or "")

        # Blob already cached and looks complete
        if blob_path.exists() and (meta.size is None or blob_path.stat().st_size == meta.size):
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            rel = os.path.relpath(blob_path, start=snapshot_path.parent)
            try:
                if snapshot_path.exists():
                    snapshot_path.unlink()
                snapshot_path.symlink_to(rel)
            except OSError:
                shutil.copy2(blob_path, snapshot_path)

            # Report full progress if cached
            if progress_callback and meta.size:
                progress_callback(meta.size, meta.size)

            return snapshot_path

        blobs_dir.mkdir(parents=True, exist_ok=True)

        # 4) For the actual data, only send Authorization if host matches
        orig_host = urlparse(meta.original_url).netloc
        target_host = urlparse(meta.url).netloc
        token_for_data = token_str if orig_host == target_host else None

        # 5) Download to blob path (with resume)
        await _download_with_resume(
            client,
            meta.url,
            blob_path,
            token=token_for_data,
            expected_size=meta.size,
            accept_ranges=meta.accept_ranges,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

        # 6) Expose snapshot path as symlink or copy
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(blob_path, start=snapshot_path.parent)
        try:
            if snapshot_path.exists():
                snapshot_path.unlink()
            snapshot_path.symlink_to(rel)
        except OSError:
            shutil.copy2(blob_path, snapshot_path)

        return snapshot_path

    finally:
        if owns_client:
            await client.aclose()
