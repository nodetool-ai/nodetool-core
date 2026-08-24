"""Regression tests for downloading a repo that is already cached.

The snapshot-escape guard called ``Path.resolve()`` on the snapshot path. Once a
file has been downloaded, that path IS a symlink into ``../../blobs``, and
``resolve()`` follows it — so the guard reported every already-cached file as
escaping its own snapshot directory and raised
``ValueError: Filename escapes snapshot directory``. The first download of a
repo succeeded and every later one failed, on the worker and locally alike.
"""

import os
from pathlib import Path

import httpx
import pytest

from nodetool.integrations.huggingface.async_downloader import async_hf_download

BODY = b"weights"
ETAG = "deadbeef"
COMMIT = "a" * 40


def _handler(request: httpx.Request) -> httpx.Response:
    headers = {
        "ETag": f'"{ETAG}"',
        "Content-Length": str(len(BODY)),
        "X-Repo-Commit": COMMIT,
        "Accept-Ranges": "bytes",
    }
    if request.method == "HEAD":
        return httpx.Response(200, headers=headers)
    return httpx.Response(200, content=BODY, headers=headers)


async def _download(cache_dir: Path, filename: str = ".gitattributes") -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_handler)
    ) as client:
        await async_hf_download(
            repo_id="org/repo",
            filename=filename,
            cache_dir=cache_dir,
            client=client,
        )


@pytest.mark.asyncio
async def test_second_download_of_a_cached_file_succeeds(tmp_path):
    """The cached path is a symlink; the guard must not follow it."""
    await _download(tmp_path)
    # Before the fix this raised "Filename escapes snapshot directory".
    await _download(tmp_path)


@pytest.mark.asyncio
async def test_cached_file_is_a_symlink_into_blobs(tmp_path):
    """Pin the premise: without a symlink here the test above proves nothing."""
    await _download(tmp_path)
    snapshot = next(tmp_path.rglob("snapshots/*/.gitattributes"))
    assert snapshot.is_symlink()
    assert "blobs" in os.path.realpath(snapshot)


@pytest.mark.asyncio
async def test_traversal_in_a_filename_is_still_rejected(tmp_path):
    """The guard still has to refuse what it was written to refuse."""
    with pytest.raises(ValueError):
        await _download(tmp_path, filename="../escaped.bin")
