import asyncio
import types
import pytest

from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Error
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.job import JobUpdate
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.graph import Graph


class DummyRunner(WorkflowRunner):
    async def run(
        self,
        request: RunJobRequest,
        context: ProcessingContext,
        send_job_updates: bool = True,
        initialize_graph: bool = True,
        validate_graph: bool = True,
    ):
        self.status = "completed"
        await asyncio.sleep(0)


def make_context():
    return ProcessingContext(user_id="u", auth_token="t", graph=Graph())


@pytest.mark.asyncio
async def test_run_workflow_loads_graph(monkeypatch):
    req_graph = Graph()
    workflow_obj = types.SimpleNamespace(graph=req_graph)
    context = make_context()

    async def fake_get_workflow(wf_id: str):
        assert wf_id == "wf1"
        return workflow_obj

    monkeypatch.setattr(context, "get_workflow", fake_get_workflow)

    messages = [
        {"type": "node_update", "node_id": "1"},
        {"type": "job_update", "job_id": "job1", "status": "completed"},
    ]

    async def fake_process_messages(ctx, runner):
        await asyncio.sleep(0)
        for msg in messages:
            yield msg
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "nodetool.workflows.run_workflow.process_workflow_messages",
        fake_process_messages,
    )

    runner = DummyRunner(job_id="job1")
    req = pytest.importorskip("nodetool.workflows.run_job_request").RunJobRequest(
        workflow_id="wf1", user_id="u", auth_token="t"
    )

    results = []
    async for m in run_workflow(req, runner=runner, context=context, use_thread=False):
        results.append(m)

    assert req.graph is req_graph
    assert results == messages


@pytest.mark.asyncio
async def test_run_workflow_error(monkeypatch):
    context = make_context()

    async def fake_process_messages(ctx, runner):
        await asyncio.sleep(0)
        yield {"type": "node_update", "node_id": "1"}
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "nodetool.workflows.run_workflow.process_workflow_messages",
        fake_process_messages,
    )

    runner = DummyRunner(job_id="job2")
    workflow_obj = types.SimpleNamespace(graph=APIGraph(nodes=[], edges=[]))

    async def fake_get_workflow(wf_id):
        return workflow_obj

    monkeypatch.setattr(context, "get_workflow", fake_get_workflow)
    req = pytest.importorskip("nodetool.workflows.run_job_request").RunJobRequest(
        workflow_id="wf2", user_id="u", auth_token="t"
    )

    results = []
    with pytest.raises(RuntimeError, match="boom"):
        async for m in run_workflow(
            req, runner=runner, context=context, use_thread=False
        ):
            results.append(m)

    assert any(isinstance(m, JobUpdate) and m.status == "failed" for m in results)


class FailingRunner(WorkflowRunner):
    async def run(
        self,
        request: RunJobRequest,
        context: ProcessingContext,
        send_job_updates: bool = True,
        initialize_graph: bool = True,
        validate_graph: bool = True,
    ):
        raise ModuleNotFoundError("test.missing.module")


@pytest.mark.asyncio
async def test_run_workflow_propagates_initialization_error():
    context = make_context()
    req = RunJobRequest(
        workflow_id="wf3",
        user_id="u",
        auth_token="t",
        graph=APIGraph(nodes=[], edges=[]),
    )
    runner = FailingRunner(job_id="job3")

    agen = run_workflow(req, runner=runner, context=context, use_thread=False)

    messages: list[JobUpdate | Error] = []
    try:
        messages.append(await agen.__anext__())
        messages.append(await agen.__anext__())
        with pytest.raises(ModuleNotFoundError):
            await agen.__anext__()
    finally:
        await agen.aclose()

    assert isinstance(messages[0], Error)
    assert messages[0].error == "test.missing.module"

    assert isinstance(messages[1], JobUpdate)
    assert messages[1].status == "failed"
