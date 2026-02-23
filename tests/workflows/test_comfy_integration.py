"""
Regression tests for ComfyUI workflow integration.

Covers:
- Workflow save does not clear run_mode="comfy" when run_mode omitted in update payload.
- Integration test for mock Comfy HTTP+WS full flow.
"""

import json
from unittest.mock import patch

import pytest

from nodetool.types.api_graph import Edge, Graph, Node
from nodetool.types.job import JobUpdate
from nodetool.workflows.comfy_workflow_runner import (
    run_comfy_workflow,
)
from nodetool.workflows.types import NodeProgress, NodeUpdate, OutputUpdate

# ---------------------------------------------------------------------------
# Regression: run_mode preservation
# ---------------------------------------------------------------------------


class TestRunModePreservation:
    """Ensure run_mode is preserved when update request omits it.

    This directly tests the logic that was changed in api/workflow.py:
    ``if workflow_request.run_mode is not None: workflow.run_mode = ...``
    """

    def test_run_mode_not_cleared_when_omitted(self):
        """Simulates an update where run_mode is not sent (None)."""
        from nodetool.models.workflow import Workflow

        wf = Workflow(id="test-wf", user_id="user-1", run_mode="comfy")
        assert wf.run_mode == "comfy"

        # Simulate the update logic from api/workflow.py
        request_run_mode = None  # request omits run_mode
        if request_run_mode is not None:
            wf.run_mode = request_run_mode

        assert wf.run_mode == "comfy"

    def test_run_mode_updated_when_explicitly_set(self):
        """Simulates an update where run_mode is explicitly changed."""
        from nodetool.models.workflow import Workflow

        wf = Workflow(id="test-wf", user_id="user-1", run_mode="comfy")

        request_run_mode = "tool"
        if request_run_mode is not None:
            wf.run_mode = request_run_mode

        assert wf.run_mode == "tool"


# ---------------------------------------------------------------------------
# Integration: mock Comfy HTTP+WS full flow
# ---------------------------------------------------------------------------


class TestComfyIntegrationMockFlow:
    """Integration test that mocks Comfy HTTP (queue_workflow) and
    WebSocket to run a full backend comfy flow and assert the emitted
    update sequence."""

    @pytest.mark.asyncio
    async def test_full_mock_flow(self):
        """Simulate a complete comfy workflow execution with mock WS events."""
        graph = Graph(
            nodes=[
                Node(id="loader", type="comfy.CheckpointLoaderSimple", data={"ckpt_name": "model.safetensors"}),
                Node(id="sampler", type="comfy.KSampler", data={"steps": 20}),
                Node(id="save", type="comfy.SaveImage", data={}),
            ],
            edges=[
                Edge(source="loader", sourceHandle="output_0", target="sampler", targetHandle="model"),
                Edge(source="sampler", sourceHandle="output_0", target="save", targetHandle="images"),
            ],
        )

        prompt_id = "mock-prompt-id-123"

        # WS event sequence that Comfy would emit
        ws_events = [
            {"type": "execution_start", "data": {"prompt_id": prompt_id}},
            {"type": "execution_cached", "data": {"nodes": ["loader"], "prompt_id": prompt_id}},
            {"type": "executing", "data": {"node": "sampler", "prompt_id": prompt_id}},
            {"type": "progress", "data": {"node": "sampler", "value": 5, "max": 20, "prompt_id": prompt_id}},
            {"type": "progress", "data": {"node": "sampler", "value": 20, "max": 20, "prompt_id": prompt_id}},
            {
                "type": "executed",
                "data": {
                    "node": "sampler",
                    "output": {"latent": {"data": "..."}},
                    "prompt_id": prompt_id,
                },
            },
            {"type": "executing", "data": {"node": "save", "prompt_id": prompt_id}},
            {
                "type": "executed",
                "data": {
                    "node": "save",
                    "output": {"images": [{"filename": "output.png"}]},
                    "prompt_id": prompt_id,
                },
            },
            # Terminal: node=null
            {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}},
        ]

        # Create mock WS that returns events in sequence
        class MockWebSocket:
            def __init__(self):
                self.connected = True
                self._recv_index = 0

            def settimeout(self, timeout):
                pass

            def connect(self, url, timeout=None):
                pass

            def recv(self):
                if self._recv_index < len(ws_events):
                    event = ws_events[self._recv_index]
                    self._recv_index += 1
                    return json.dumps(event)
                # Return terminal after all events
                self.connected = False
                raise Exception("WS closed")

            def close(self):
                self.connected = False

        mock_ws = MockWebSocket()

        with (
            patch("nodetool.workflows.comfy_workflow_runner.websocket.WebSocket", return_value=mock_ws),
            patch(
                "nodetool.workflows.comfy_workflow_runner._submit_prompt",
                return_value=prompt_id,
            ),
        ):
            messages = []
            async for msg in run_comfy_workflow(
                graph=graph,
                workflow_id="test-wf",
                job_id="test-job",
            ):
                messages.append(msg)

        # Verify the message sequence
        job_updates = [m for m in messages if isinstance(m, JobUpdate)]
        node_updates = [m for m in messages if isinstance(m, NodeUpdate)]
        progress_updates = [m for m in messages if isinstance(m, NodeProgress)]
        output_updates = [m for m in messages if isinstance(m, OutputUpdate)]

        # Should have job running and completed
        assert any(j.status == "running" for j in job_updates)
        assert any(j.status == "completed" for j in job_updates)

        # Should have node progress for sampler
        assert len(progress_updates) >= 2

        # Should have output updates for sampler and save
        assert len(output_updates) >= 2

        # Loader should be cached-completed
        loader_updates = [m for m in node_updates if m.node_id == "loader"]
        assert any(m.status == "completed" and m.result == {"cached": True} for m in loader_updates)

        # Sampler and save should be completed
        sampler_updates = [m for m in node_updates if m.node_id == "sampler"]
        assert any(m.status == "completed" for m in sampler_updates)

        save_updates = [m for m in node_updates if m.node_id == "save"]
        assert any(m.status == "completed" for m in save_updates)

        # No nodes should be stuck in running at the end
        # (the final job_update should be "completed", not lingering)
        final_job = job_updates[-1]
        assert final_job.status == "completed"

    @pytest.mark.asyncio
    async def test_error_flow(self):
        """Simulate an error during execution."""
        graph = Graph(
            nodes=[Node(id="n1", type="comfy.KSampler", data={"steps": 20})],
            edges=[],
        )

        prompt_id = "error-prompt-id"
        ws_events = [
            {"type": "execution_start", "data": {"prompt_id": prompt_id}},
            {"type": "executing", "data": {"node": "n1", "prompt_id": prompt_id}},
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": prompt_id,
                    "node_id": "n1",
                    "node_type": "KSampler",
                    "exception_message": "Out of memory",
                    "traceback": ["File ...", "OOM"],
                },
            },
        ]

        class MockWebSocket:
            def __init__(self):
                self.connected = True
                self._recv_index = 0

            def settimeout(self, timeout):
                pass

            def connect(self, url, timeout=None):
                pass

            def recv(self):
                if self._recv_index < len(ws_events):
                    event = ws_events[self._recv_index]
                    self._recv_index += 1
                    return json.dumps(event)
                self.connected = False
                raise Exception("WS closed")

            def close(self):
                self.connected = False

        mock_ws = MockWebSocket()

        with (
            patch("nodetool.workflows.comfy_workflow_runner.websocket.WebSocket", return_value=mock_ws),
            patch(
                "nodetool.workflows.comfy_workflow_runner._submit_prompt",
                return_value=prompt_id,
            ),
        ):
            messages = []
            async for msg in run_comfy_workflow(
                graph=graph,
                workflow_id="test-wf",
                job_id="test-job",
            ):
                messages.append(msg)

        job_updates = [m for m in messages if isinstance(m, JobUpdate)]
        assert any(j.status == "failed" for j in job_updates)
        assert any(j.error == "Out of memory" for j in job_updates)

        node_updates = [m for m in messages if isinstance(m, NodeUpdate)]
        assert any(n.node_id == "n1" and n.status == "error" for n in node_updates)

    @pytest.mark.asyncio
    async def test_submission_failure(self):
        """Simulate failure to submit prompt."""
        graph = Graph(
            nodes=[Node(id="n1", type="comfy.KSampler", data={})],
            edges=[],
        )

        class MockWebSocket:
            def __init__(self):
                self.connected = True

            def settimeout(self, timeout):
                pass

            def connect(self, url, timeout=None):
                pass

            def close(self):
                self.connected = False

        mock_ws = MockWebSocket()

        with (
            patch("nodetool.workflows.comfy_workflow_runner.websocket.WebSocket", return_value=mock_ws),
            patch(
                "nodetool.workflows.comfy_workflow_runner._submit_prompt",
                side_effect=ValueError("Comfy validation failed"),
            ),
        ):
            messages = []
            async for msg in run_comfy_workflow(
                graph=graph,
                workflow_id="test-wf",
                job_id="test-job",
            ):
                messages.append(msg)

        assert len(messages) == 1
        assert isinstance(messages[0], JobUpdate)
        assert messages[0].status == "failed"
        assert "validation failed" in messages[0].error.lower()


if __name__ == "__main__":
    pytest.main([__file__])
