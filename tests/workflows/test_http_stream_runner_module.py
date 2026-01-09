import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from nodetool.workflows.http_stream_runner import HTTPStreamRunner
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import NodeUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner


@pytest.mark.asyncio
async def test_run_job(monkeypatch):
    runner = HTTPStreamRunner()

    async def fake_run_workflow(req, runner_obj, context_obj):
        yield NodeUpdate(node_id="1", node_name="n", status="ok", node_type="node_type")

    monkeypatch.setattr("nodetool.workflows.http_stream_runner.run_workflow", fake_run_workflow)

    outputs = []
    async for msg in runner.run_job(RunJobRequest()):
        outputs.append(json.loads(msg))

    assert outputs[0]["node_id"] == "1"
    assert outputs[-1]["type"] == "job_completed"


@pytest.mark.asyncio
async def test_cancel_job(monkeypatch):
    runner = HTTPStreamRunner()
    runner.runner = WorkflowRunner(job_id="1")
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    result = await runner.cancel_job()
    assert result["message"] == "Job cancelled"
    assert runner.runner is None


def test_get_status():
    runner = HTTPStreamRunner()
    assert runner.get_status() == {"status": "idle", "job_id": None}
    runner.runner = WorkflowRunner(job_id="1")
    runner.job_id = "1"
    assert runner.get_status()["status"] == "running"
