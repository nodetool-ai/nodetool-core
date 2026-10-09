"""The GPU OOM retry must free memory before it retries, on CUDA and on MPS."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import nodetool.workflows.torch_support as ts
from nodetool.config.environment import Environment
from nodetool.ml.core.model_manager import ModelManager


class FakeOOM(RuntimeError):
    pass


class _Model:
    def __init__(self) -> None:
        self.moved_to: list[str] = []

    def to(self, device: str) -> "_Model":
        self.moved_to.append(device)
        return self


class _Node:
    """Fails with an OOM ``failures`` times, then returns "ok"."""

    _requires_grad = True
    _id = "n1"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts = 0
        self.cache_at_attempt: list[set[str]] = []

    def get_title(self) -> str:
        return "n1"

    async def process(self, context):
        # Using its model pins it to this execution, as a real node does.
        assert ModelManager.get_model("in_use") is not None
        self.attempts += 1
        self.cache_at_attempt.append(set(ModelManager._models))
        if self.attempts <= self.failures:
            raise FakeOOM("MPS backend out of memory")
        return "ok"


@pytest.fixture(autouse=True)
def clean_model_manager(monkeypatch):
    Environment.set_env("development")
    ModelManager.clear()
    monkeypatch.setattr(ts.random, "uniform", lambda a, b: 0.0)

    async def _no_sleep(delay):
        return None

    monkeypatch.setattr(ts.asyncio, "sleep", _no_sleep)
    yield
    ModelManager.clear()


def _support(max_retries: int) -> ts.TorchWorkflowSupport:
    support = ts.TorchWorkflowSupport(base_delay=0, max_delay=0, max_retries=max_retries)
    support.is_cuda_oom_exception = lambda exc: isinstance(exc, FakeOOM)  # type: ignore[method-assign]
    return support


def _cache_models() -> tuple[_Model, _Model]:
    in_use, idle = _Model(), _Model()
    ModelManager.set_model("node-1", "in_use", in_use)
    ModelManager.set_model("node-2", "idle", idle)
    return in_use, idle


@pytest.mark.asyncio
async def test_mps_oom_evicts_idle_models_and_empties_the_cache_before_retrying(monkeypatch):
    emptied: list[set[str]] = []
    monkeypatch.setattr(ts, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(ts, "is_cuda_available", lambda: False)
    monkeypatch.setattr(
        ts,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            mps=SimpleNamespace(empty_cache=lambda: emptied.append(set(ModelManager._models))),
            no_grad=nullcontext,
        ),
    )
    in_use, _idle = _cache_models()
    node = _Node(failures=1)

    with ModelManager.execution_scope():
        result = await _support(max_retries=2).process_with_gpu(None, None, node)

    assert result == "ok"
    assert node.attempts == 2
    assert node.cache_at_attempt[1] == {"in_use"}, "the retry ran with the idle model still cached"
    assert emptied == [{"in_use"}], "torch.mps.empty_cache did not run after the eviction"
    assert in_use.moved_to == [], "evicted the model the retried node is using"


@pytest.mark.asyncio
async def test_cuda_aggressive_cleanup_is_followed_by_a_retry(monkeypatch):
    """With two attempts, the one cleanup is the aggressive one, and it is retried."""
    monkeypatch.setattr(ts, "is_cuda_available", lambda: True)
    monkeypatch.setattr(ts, "cuda_device_index", lambda: 0)
    monkeypatch.setattr(
        ts,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            cuda=SimpleNamespace(synchronize=lambda device=None: None, ipc_collect=lambda: None),
            no_grad=nullcontext,
        ),
    )
    support = _support(max_retries=2)
    monkeypatch.setattr(support, "get_available_vram", lambda: 0)
    monkeypatch.setattr(support, "empty_cuda_cache", lambda: None)
    monkeypatch.setattr(ModelManager, "get_vram_snapshot", classmethod(lambda cls: None))
    in_use, idle = _cache_models()
    node = _Node(failures=1)

    with ModelManager.execution_scope():
        result = await support.process_with_gpu(None, None, node)

    assert result == "ok"
    assert node.cache_at_attempt[1] == {"in_use"}
    assert idle.moved_to == ["cpu"]
    assert in_use.moved_to == []


@pytest.mark.asyncio
async def test_first_of_several_cleanups_keeps_idle_models_cached(monkeypatch):
    """Eviction escalates: only the cleanup before the last retry drops models."""
    monkeypatch.setattr(ts, "is_cuda_available", lambda: True)
    monkeypatch.setattr(ts, "cuda_device_index", lambda: 0)
    monkeypatch.setattr(
        ts,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            cuda=SimpleNamespace(synchronize=lambda device=None: None, ipc_collect=lambda: None),
            no_grad=nullcontext,
        ),
    )
    support = _support(max_retries=3)
    monkeypatch.setattr(support, "get_available_vram", lambda: 0)
    monkeypatch.setattr(support, "empty_cuda_cache", lambda: None)
    monkeypatch.setattr(ModelManager, "get_vram_snapshot", classmethod(lambda cls: None))
    _cache_models()
    node = _Node(failures=2)

    with ModelManager.execution_scope():
        assert await support.process_with_gpu(None, None, node) == "ok"

    assert node.cache_at_attempt == [{"in_use", "idle"}, {"in_use", "idle"}, {"in_use"}]


class _StreamingNode:
    """Yields "a" then "b", raising an OOM at ``fail_at`` on the first ``failures`` runs."""

    _id = "s1"

    def __init__(self, *, fail_at: int, failures: int) -> None:
        self.fail_at = fail_at
        self.failures = failures
        self.runs = 0

    def get_title(self) -> str:
        return "s1"

    async def gen_process(self, context):
        self.runs += 1
        for position, item in enumerate(["a", "b"]):
            if position == self.fail_at and self.runs <= self.failures:
                raise FakeOOM("CUDA out of memory")
            yield item


def _cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(ts, "is_cuda_available", lambda: False)
    monkeypatch.setattr(ts, "_is_mps_available", lambda: False)


@pytest.mark.asyncio
async def test_streaming_oom_before_the_first_item_is_retried(monkeypatch):
    _cpu_only(monkeypatch)
    node = _StreamingNode(fail_at=0, failures=1)

    items = [item async for item in _support(max_retries=2).stream_with_gpu(None, node)]

    assert items == ["a", "b"]
    assert node.runs == 2


@pytest.mark.asyncio
async def test_streaming_oom_after_an_item_was_emitted_is_not_retried(monkeypatch):
    _cpu_only(monkeypatch)
    node = _StreamingNode(fail_at=1, failures=1)
    items: list[str] = []

    with pytest.raises(FakeOOM):
        async for item in _support(max_retries=2).stream_with_gpu(None, node):
            items.append(item)

    assert items == ["a"], "an emitted item would be sent twice by a re-run"
    assert node.runs == 1


@pytest.mark.asyncio
async def test_streaming_oom_raises_after_max_retries(monkeypatch):
    _cpu_only(monkeypatch)
    node = _StreamingNode(fail_at=0, failures=5)

    with pytest.raises(FakeOOM):
        async for _ in _support(max_retries=2).stream_with_gpu(None, node):
            pass

    assert node.runs == 2
