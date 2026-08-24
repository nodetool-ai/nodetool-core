"""Repo files download concurrently, and the progress totals still add up.

The loop used to transfer one file at a time, so a repo of many shards moved at
a single connection's throughput — SmolVLM-Instruct took 1798 s for 29.5 GB
(16 MB/s) on an A40 with far more bandwidth available.

Concurrency breaks the old progress arithmetic. It kept a running
`file_base = done_bytes` and added each file's progress to it, which is only
correct while exactly one file is in flight; with two, each callback publishes a
total that omits the other's bytes and the next file's base is wrong. Each file
now accumulates into its own slot and the reported total is their sum.
"""

import asyncio
import os

import pytest

from nodetool.worker import model_handler
from nodetool.worker.model_handler import _handle_download

FILES = [(f"shard-{i}.safetensors", 1000) for i in range(8)]


class _Recorder:
    def __init__(self) -> None:
        self.progress: list[dict] = []
        self.results: list[dict] = []

    async def send_progress(self, _request_id, frame):
        self.progress.append(frame)

    async def send_result(self, _request_id, result):
        self.results.append(result)


@pytest.fixture
def patched(monkeypatch):
    """Stub the network: list a fixed file set, and 'download' with a sleep."""
    monkeypatch.setattr(
        model_handler, "_list_repo_files", lambda *a, **k: _async(FILES)
    )
    monkeypatch.setattr(model_handler, "get_hf_token", lambda *a, **k: _async(None))

    state = {"in_flight": 0, "peak": 0}

    async def fake_download(repo_id, filename, token=None, progress_callback=None,
                            cancel_event=None):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            for _ in range(4):
                await asyncio.sleep(0.01)
                if progress_callback:
                    progress_callback(250)
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(model_handler, "async_hf_download", fake_download)
    return state


def _async(value):
    async def run():
        return value
    return run()


async def _run(recorder) -> None:
    await _handle_download(
        {"repo_id": "org/repo"},
        "req-1",
        {},
        recorder.send_progress,
        recorder.send_result,
    )


@pytest.mark.asyncio
async def test_files_download_concurrently(patched):
    """The whole point: peak in-flight above one."""
    rec = _Recorder()
    await _run(rec)

    assert patched["peak"] > 1, "files still transferred one at a time"


@pytest.mark.asyncio
async def test_concurrency_is_bounded(patched, monkeypatch):
    monkeypatch.setenv("NODETOOL_HF_DOWNLOAD_CONCURRENCY", "3")
    rec = _Recorder()
    await _run(rec)

    assert patched["peak"] <= 3


@pytest.mark.asyncio
async def test_progress_totals_reach_the_full_size(patched):
    rec = _Recorder()
    await _run(rec)

    total = sum(size for _, size in FILES)
    final = rec.progress[-1]
    assert final["total_bytes"] == total
    assert final["downloaded_bytes"] == total
    assert final["downloaded_files"] == len(FILES)


@pytest.mark.asyncio
async def test_progress_never_exceeds_the_total(patched):
    """The old base arithmetic overshot once two files overlapped."""
    rec = _Recorder()
    await _run(rec)

    total = sum(size for _, size in FILES)
    assert all(f["downloaded_bytes"] <= total for f in rec.progress)


@pytest.mark.asyncio
async def test_progress_is_monotonic(patched):
    """A total that goes backwards is what a shared running base produced."""
    rec = _Recorder()
    await _run(rec)

    seen = [f["downloaded_bytes"] for f in rec.progress]
    assert seen == sorted(seen), "downloaded_bytes went backwards"


def test_the_concurrency_knob_clamps():
    # Imported here, not at module scope: the behavioural tests above must stay
    # collectable against a build without this helper, so they can be shown to
    # fail on the serial loop rather than erroring out.
    from nodetool.worker.model_handler import _download_concurrency

    assert _download_concurrency() == 8

    for value, expected in [("1", 1), ("16", 16), ("0", 1), ("999", 32), ("nonsense", 8)]:
        os.environ["NODETOOL_HF_DOWNLOAD_CONCURRENCY"] = value
        try:
            assert _download_concurrency() == expected, value
        finally:
            del os.environ["NODETOOL_HF_DOWNLOAD_CONCURRENCY"]
