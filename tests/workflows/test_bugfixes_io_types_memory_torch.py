"""Regression tests for fixes in io.py, types.py, memory_utils.py and torch_support.py."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel

from nodetool.workflows.base_node import OutputNode
from nodetool.workflows.io import NodeOutputs
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import sanitize_memory_uris_for_client


class _FakeRunner:
    def __init__(self) -> None:
        self.outputs: dict[str, list[Any]] = {}
        self.sent: list[Any] = []

    def send_messages(self, node, values, context, metadata=None) -> None:
        self.sent.append((node, values, metadata))


class _Out(OutputNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "tests.bugfix.Out"


def _make_outputs() -> tuple[_FakeRunner, NodeOutputs]:
    runner = _FakeRunner()
    node = _Out(id="out1", name="result")
    context = ProcessingContext(user_id="u", workflow_id="w")
    return runner, NodeOutputs(runner, node, context, capture_only=True)


# --- Bug 1: NodeOutputs.emit dropped duplicates / crashed on numpy ---------


def test_emit_records_consecutive_duplicates():
    runner, outputs = _make_outputs()
    for value in (1, 1, 2):
        asyncio.run(outputs.emit("output", value))
    assert runner.outputs["result"] == [1, 1, 2]


def test_emit_accepts_numpy_arrays():
    runner, outputs = _make_outputs()
    arr = np.array([1, 2, 3])
    asyncio.run(outputs.emit("output", arr))
    asyncio.run(outputs.emit("output", arr))
    assert len(runner.outputs["result"]) == 2
    assert np.array_equal(runner.outputs["result"][0], arr)


# --- Bug 2: sanitize_memory_uris_for_client missed nested models ----------


class _Ref(BaseModel):
    type: str = "image"
    uri: str = ""
    asset_id: str | None = None
    data: Any = None


class _Result(BaseModel):
    image: _Ref
    label: str = "x"


class _Nested(BaseModel):
    inner: _Result
    items: list[_Ref] = []


def test_sanitize_recurses_into_nested_models():
    value = _Result(image=_Ref(uri="memory://abc"))
    out = sanitize_memory_uris_for_client(value)
    assert out.image.uri == ""
    # non-mutating
    assert value.image.uri == "memory://abc"


def test_sanitize_recurses_deeply_and_into_model_lists():
    value = _Nested(
        inner=_Result(image=_Ref(uri="memory://a", asset_id="aid")),
        items=[_Ref(uri="memory://b", data=b"x"), _Ref(uri="asset://keep")],
    )
    out = sanitize_memory_uris_for_client(value)
    assert out.inner.image.uri == "asset://aid"
    assert out.items[0].uri == ""
    assert out.items[1].uri == "asset://keep"
    assert value.inner.image.uri == "memory://a"


def test_sanitize_handles_cycles():
    class Node(BaseModel):
        uri: str = ""
        peer: Any = None

    a = Node(uri="memory://a")
    b = Node(uri="memory://b")
    a.peer = b
    b.peer = a
    out = sanitize_memory_uris_for_client(a)
    assert out.uri == ""


# --- Bug 3: memory cache stats used the wrong attribute -------------------


def test_memory_uri_cache_stats_and_clear():
    from nodetool.runtime.resources import ResourceScope
    from nodetool.workflows.memory_utils import (
        clear_memory_uri_cache,
        get_memory_uri_cache_stats,
    )

    async def run() -> None:
        async with ResourceScope() as scope:
            cache = scope.get_memory_uri_cache()
            cache.set("a", 1)
            cache.set("b", 2)
            assert get_memory_uri_cache_stats() == {"count": 2}
            assert clear_memory_uri_cache(log_stats=False) == 2
            assert get_memory_uri_cache_stats() == {"count": 0}

    asyncio.run(run())


# --- Bug 4: OOM retry counter double increment ---------------------------


@pytest.mark.asyncio
async def test_oom_retry_counter_increments_once_per_attempt(monkeypatch):
    import nodetool.workflows.torch_support as ts

    class FakeOOM(Exception):
        pass

    attempts: list[int] = []
    delays: list[float] = []

    support = ts.TorchWorkflowSupport.__new__(ts.TorchWorkflowSupport)
    support.max_retries = 3
    support.base_delay = 1
    support.max_delay = 1000

    monkeypatch.setattr(support, "is_cuda_oom_exception", lambda exc: True)
    monkeypatch.setattr(ts, "is_cuda_available", lambda: False)
    monkeypatch.setattr(ts.random, "uniform", lambda a, b: 0.0)

    async def _fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(ts.asyncio, "sleep", _fake_sleep)

    class _Node:
        _requires_grad = True
        _id = "n1"

        def get_title(self) -> str:
            return "n1"

        async def process(self, context):
            attempts.append(1)
            raise FakeOOM("oom")

    with pytest.raises(FakeOOM):
        await support.process_with_gpu(None, None, _Node())

    # max_retries=3 -> 3 attempts total (initial + 2 retries), one increment each
    assert len(attempts) == 3
    # Backoff doubles without skipping a step
    assert delays == [1.0, 2.0]


# --- OOM cleanup must not run with the failed frames still pinned ----------


@pytest.mark.asyncio
async def test_oom_cleanup_runs_after_the_traceback_is_released(monkeypatch):
    """The activations that caused the OOM live on ``exc.__traceback__``.

    Reclaiming while the traceback is still bound frees little or nothing, so
    the retry runs against the same occupied VRAM. The traceback must be gone
    by the time ``free_vram_if_needed`` is called.
    """
    import nodetool.workflows.torch_support as ts

    class FakeOOM(Exception):
        pass

    observed: list[object] = []

    support = ts.TorchWorkflowSupport.__new__(ts.TorchWorkflowSupport)
    support.max_retries = 1
    support.base_delay = 1
    support.max_delay = 1

    monkeypatch.setattr(support, "is_cuda_oom_exception", lambda exc: True)
    monkeypatch.setattr(ts, "is_cuda_available", lambda: True)
    monkeypatch.setattr(support, "get_available_vram", lambda: 0)
    monkeypatch.setattr(support, "empty_cuda_cache", lambda: None)
    monkeypatch.setattr(
        ts,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None),
            no_grad=nullcontext,
        ),
    )
    monkeypatch.setattr(ts.ModelManager, "get_vram_snapshot", classmethod(lambda cls: None))

    raised: dict[str, BaseException] = {}

    def _record_reclaim(cls, **kwargs):
        observed.append(raised["exc"].__traceback__)

    monkeypatch.setattr(ts.ModelManager, "free_vram_if_needed", classmethod(_record_reclaim))

    class _Node:
        _requires_grad = True
        _id = "n1"

        def get_title(self) -> str:
            return "n1"

        async def process(self, context):
            exc = FakeOOM("out of memory")
            raised["exc"] = exc
            raise exc

    with pytest.raises(FakeOOM):
        await support.process_with_gpu(None, None, _Node())

    assert observed, "reclaim was never attempted"
    assert observed[0] is None, "VRAM was reclaimed while the failed frames were still alive"
    assert raised["exc"].__cause__ is None
    assert raised["exc"].__context__ is None
