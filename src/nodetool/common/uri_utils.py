from pathlib import Path


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
    except Exception:
        # Fallback: best-effort POSIX-style URI
        posix_path = Path(path).as_posix()
        prefix = "file:///" if not posix_path.startswith("/") else "file://"
        return f"{prefix}{posix_path}"
