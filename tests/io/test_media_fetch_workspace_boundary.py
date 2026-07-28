"""Regression tests: file:// URIs must stay inside the workspace directory."""

import pytest

from nodetool.io.media_fetch import _fetch_file_uri, fetch_uri_bytes_and_mime_async

pytestmark = pytest.mark.asyncio


async def test_file_uri_without_workspace_is_rejected() -> None:
    """Reading a file:// URI without a workspace must not disclose local files."""
    with pytest.raises(PermissionError):
        await fetch_uri_bytes_and_mime_async("file:///etc/passwd")


async def test_file_uri_inside_workspace_is_read(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("hello")

    mime, data = await fetch_uri_bytes_and_mime_async("file:///note.txt", workspace_dir=str(tmp_path))

    assert data == b"hello"
    assert mime == "text/plain"


async def test_file_uri_escaping_workspace_is_blocked(tmp_path) -> None:
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside the workspace"):
        await fetch_uri_bytes_and_mime_async(
            f"file:///../{outside.name}",
            workspace_dir=str(workspace),
        )


async def test_fetch_file_uri_unquotes_and_enforces_boundary(tmp_path) -> None:
    (tmp_path / "my file.txt").write_text("data")

    mime, data = _fetch_file_uri("file:///my%20file.txt", str(tmp_path))

    assert data == b"data"
    assert mime == "text/plain"
