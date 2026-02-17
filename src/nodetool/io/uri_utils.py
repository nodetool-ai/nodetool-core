from pathlib import Path

from nodetool.io.media_fetch import (
    fetch_uri_bytes_and_mime_async as _fetch_async,
)


def create_file_uri(path: str) -> str:
    """
    Create a cross-platform file:// URI from a file path.
    Handles Windows paths correctly by using pathlib.Path.as_uri().
    Accepts already-formed file URIs.
    """
    if isinstance(path, str) and path.startswith("file://"):
        return path

    try:
        resolved_path = Path(path).expanduser().resolve(strict=False)
        return resolved_path.as_uri()
    except (OSError, RuntimeError):
        # Fallback: best-effort POSIX-style URI for paths that can't be resolved
        # (e.g., non-existent paths on Windows, permission issues)
        posix_path = Path(path).as_posix()
        prefix = "file:///" if not posix_path.startswith("/") else "file://"
        return f"{prefix}{posix_path}"


async def fetch_uri_bytes_and_mime(uri: str) -> tuple[str, bytes]:
    """Delegate to shared async media fetcher to ensure DRY behavior."""
    return await _fetch_async(uri)
