"""Regression tests for the `refs/<revision>` file in the HF cache.

`async_hf_download` built huggingface_hub's cache layout — `blobs/` plus
`snapshots/<commit>/` — but never wrote the third directory it keeps.
`from_pretrained` resolves a branch name (the default "main") to a commit by
reading `refs/<revision>`, so with the file absent an offline load could not
find the snapshot sitting right next to it:

    LocalEntryNotFoundError: Cannot find an appropriate cached snapshot folder
    for the specified revision on the local disk ...

Every node that loads with `local_files_only=True` hit it — the diffusers
pipelines in text_to_audio, text_to_image and text_to_video. Found by running
ACE-Step on a worker: the 11 GB download succeeded and the load failed.
"""

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


async def _download(cache_dir: Path, revision: str = "main") -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        await async_hf_download(
            repo_id="org/repo",
            filename="config.json",
            revision=revision,
            cache_dir=cache_dir,
            client=client,
        )


def _repo_dir(cache_dir: Path) -> Path:
    return next(cache_dir.glob("models--*"))


@pytest.mark.asyncio
async def test_branch_revision_gets_a_ref(tmp_path):
    """The whole bug: this file did not exist, so offline loads failed."""
    await _download(tmp_path)

    ref = _repo_dir(tmp_path) / "refs" / "main"
    assert ref.is_file()
    assert ref.read_text() == COMMIT


@pytest.mark.asyncio
async def test_the_ref_points_at_a_snapshot_that_exists(tmp_path):
    """A ref aimed at nothing would satisfy the test above and still not load."""
    await _download(tmp_path)

    repo = _repo_dir(tmp_path)
    commit = (repo / "refs" / "main").read_text()
    assert (repo / "snapshots" / commit).is_dir()
    assert (repo / "snapshots" / commit / "config.json").exists()


@pytest.mark.asyncio
async def test_a_commit_revision_needs_no_ref(tmp_path):
    """huggingface_hub resolves a commit hash directly; a ref named after one
    would be a file called `aaaa...` in refs/, which it never looks for."""
    await _download(tmp_path, revision=COMMIT)

    assert not (_repo_dir(tmp_path) / "refs").exists()


@pytest.mark.asyncio
async def test_a_traversing_revision_writes_no_ref(tmp_path):
    """The revision reaches a path join, so it gets the same guard as the rest."""
    await _download(tmp_path, revision="../../escaped")

    for path in tmp_path.rglob("escaped"):
        raise AssertionError(f"wrote a ref outside the repo cache: {path}")
