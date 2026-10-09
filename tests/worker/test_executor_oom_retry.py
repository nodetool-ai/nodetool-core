"""execute_node retries a buffered node after a GPU out-of-memory error."""

import pytest

import nodetool.workflows.torch_support as ts
from nodetool.worker import executor
from nodetool.worker.executor import execute_node
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class FakeOOM(RuntimeError):
    pass


calls: list[int] = []


class OOMOnceNode(BaseNode):
    """Fails with an OOM on the first call, then succeeds."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.OOMOnceNode"

    async def process(self, context: ProcessingContext) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise FakeOOM("MPS backend out of memory")
        return "ok"


class AlwaysFailsNode(BaseNode):
    @classmethod
    def get_node_type(cls) -> str:
        return "test.AlwaysFailsNode"

    async def process(self, context: ProcessingContext) -> str:
        calls.append(1)
        raise ValueError("not an OOM")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    calls.clear()
    NODE_BY_TYPE["test.OOMOnceNode"] = OOMOnceNode
    NODE_BY_TYPE["test.AlwaysFailsNode"] = AlwaysFailsNode

    support = ts.TorchWorkflowSupport(base_delay=0, max_delay=0, max_retries=2)
    monkeypatch.setattr(support, "is_cuda_oom_exception", lambda exc: isinstance(exc, FakeOOM))
    monkeypatch.setattr(ts, "is_cuda_available", lambda: False)
    monkeypatch.setattr(ts, "_is_mps_available", lambda: False)

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(ts.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(executor, "_TORCH_SUPPORT", support)
    yield
    NODE_BY_TYPE.pop("test.OOMOnceNode", None)
    NODE_BY_TYPE.pop("test.AlwaysFailsNode", None)


@pytest.mark.asyncio
async def test_execute_node_retries_after_oom():
    result = await execute_node(node_type="test.OOMOnceNode", fields={}, secrets={}, input_blobs={})
    assert result["outputs"]["output"] == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_execute_node_does_not_retry_other_errors():
    with pytest.raises(ValueError, match="not an OOM"):
        await execute_node(node_type="test.AlwaysFailsNode", fields={}, secrets={}, input_blobs={})
    assert len(calls) == 1
