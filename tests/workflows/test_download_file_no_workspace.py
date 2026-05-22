import tempfile
from pathlib import Path

import pytest

from nodetool.workflows.processing_context import create_file_uri
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_download_file_reads_file_uri_when_no_workspace():
    """With no workspace configured (desktop mode), a user-selected file://
    URI must be readable directly instead of failing with 'No workspace assigned'."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "audio.wav"
        target.write_bytes(b"RIFFdata")

        ctx = ProcessingContext(user_id="u", auth_token="t", workspace_dir=None)

        f = await ctx.download_file(create_file_uri(str(target)))
        assert f.read() == b"RIFFdata"


@pytest.mark.asyncio
async def test_download_file_reads_absolute_path_when_no_workspace():
    """Empty-scheme absolute paths must also be readable without a workspace."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "audio.wav"
        target.write_bytes(b"RIFFdata")

        ctx = ProcessingContext(user_id="u", auth_token="t", workspace_dir=None)

        f = await ctx.download_file(str(target))
        assert f.read() == b"RIFFdata"


@pytest.mark.asyncio
async def test_download_file_still_enforces_workspace_when_set():
    """When a workspace IS configured, reads outside it stay blocked (LFI guard)."""
    with tempfile.TemporaryDirectory() as ws, tempfile.TemporaryDirectory() as outside:
        secret = Path(outside) / "secret.txt"
        secret.write_bytes(b"top-secret")

        ctx = ProcessingContext(user_id="u", auth_token="t", workspace_dir=ws)

        with pytest.raises(FileNotFoundError):
            await ctx.download_file(create_file_uri(str(secret)))
