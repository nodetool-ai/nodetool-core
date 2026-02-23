"""
Tests for the ComfyUI event translator and workflow runner helpers
(``comfy_workflow_runner.py``).

Covers:
- Each Comfy event type maps to expected NodeTool updates
- prompt_id filtering works
- Lingering running nodes are finalized on success
- should_use_comfy_runner routing logic
"""

import pytest

from nodetool.types.api_graph import Graph, Node
from nodetool.types.job import JobUpdate
from nodetool.workflows.comfy_workflow_runner import (
    ComfyEventTranslator,
    should_use_comfy_runner,
)
from nodetool.workflows.types import (
    NodeProgress,
    NodeUpdate,
    OutputUpdate,
)

PROMPT_ID = "test-prompt-123"
WORKFLOW_ID = "wf-456"


def _make_translator(**kwargs) -> ComfyEventTranslator:
    return ComfyEventTranslator(
        prompt_id=kwargs.get("prompt_id", PROMPT_ID),
        workflow_id=kwargs.get("workflow_id", WORKFLOW_ID),
    )


# ---------------------------------------------------------------------------
# execution_start
# ---------------------------------------------------------------------------


class TestExecutionStart:
    def test_emits_job_update_running(self):
        t = _make_translator()
        msgs = t.translate({"type": "execution_start", "data": {"prompt_id": PROMPT_ID}})
        assert len(msgs) == 1
        assert isinstance(msgs[0], JobUpdate)
        assert msgs[0].status == "running"
        assert msgs[0].job_id == PROMPT_ID


# ---------------------------------------------------------------------------
# executing
# ---------------------------------------------------------------------------


class TestExecuting:
    def test_node_not_null_emits_node_running(self):
        t = _make_translator()
        msgs = t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        assert len(msgs) == 1
        assert isinstance(msgs[0], NodeUpdate)
        assert msgs[0].node_id == "node_1"
        assert msgs[0].status == "running"

    def test_node_null_triggers_finalization(self):
        t = _make_translator()
        # Start a node
        t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        # Terminal pattern
        msgs = t.translate({"type": "executing", "data": {"node": None, "prompt_id": PROMPT_ID}})
        # Should finalize node_1 + emit job completed
        assert any(isinstance(m, NodeUpdate) and m.node_id == "node_1" and m.status == "completed" for m in msgs)
        assert any(isinstance(m, JobUpdate) and m.status == "completed" for m in msgs)
        assert t.is_completed

    def test_node_null_no_duplicate_terminal(self):
        """Calling node=null twice should not emit duplicate terminal messages."""
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": None, "prompt_id": PROMPT_ID}})
        msgs2 = t.translate({"type": "executing", "data": {"node": None, "prompt_id": PROMPT_ID}})
        assert msgs2 == []


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------


class TestProgress:
    def test_emits_node_progress(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        msgs = t.translate(
            {"type": "progress", "data": {"node": "node_1", "value": 5, "max": 10, "prompt_id": PROMPT_ID}}
        )
        assert len(msgs) == 1
        assert isinstance(msgs[0], NodeProgress)
        assert msgs[0].node_id == "node_1"
        assert msgs[0].progress == 5
        assert msgs[0].total == 10

    def test_progress_without_node_uses_current(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        msgs = t.translate({"type": "progress", "data": {"value": 3, "max": 10, "prompt_id": PROMPT_ID}})
        assert msgs[0].node_id == "node_1"


# ---------------------------------------------------------------------------
# executed
# ---------------------------------------------------------------------------


class TestExecuted:
    def test_emits_output_and_node_completed(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        msgs = t.translate(
            {
                "type": "executed",
                "data": {
                    "node": "node_1",
                    "output": {"images": [{"filename": "test.png"}]},
                    "prompt_id": PROMPT_ID,
                },
            }
        )
        # Should have OutputUpdate + NodeUpdate(completed)
        output_msgs = [m for m in msgs if isinstance(m, OutputUpdate)]
        node_msgs = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(output_msgs) == 1
        assert output_msgs[0].output_name == "images"
        assert len(node_msgs) == 1
        assert node_msgs[0].status == "completed"

    def test_multiple_output_keys(self):
        t = _make_translator()
        msgs = t.translate(
            {
                "type": "executed",
                "data": {
                    "node": "node_1",
                    "output": {"images": [], "metadata": {"width": 512}},
                    "prompt_id": PROMPT_ID,
                },
            }
        )
        output_msgs = [m for m in msgs if isinstance(m, OutputUpdate)]
        assert len(output_msgs) == 2
        output_names = {m.output_name for m in output_msgs}
        assert output_names == {"images", "metadata"}

    def test_executed_removes_from_running(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "node_1", "prompt_id": PROMPT_ID}})
        t.translate(
            {"type": "executed", "data": {"node": "node_1", "output": {}, "prompt_id": PROMPT_ID}}
        )
        # On finalize, node_1 should not be finalized again
        msgs = t.translate({"type": "execution_success", "data": {"prompt_id": PROMPT_ID}})
        node_updates = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(node_updates) == 0  # no lingering nodes


# ---------------------------------------------------------------------------
# execution_cached
# ---------------------------------------------------------------------------


class TestExecutionCached:
    def test_emits_completed_with_cached_result(self):
        t = _make_translator()
        msgs = t.translate(
            {"type": "execution_cached", "data": {"nodes": ["n1", "n2"], "prompt_id": PROMPT_ID}}
        )
        assert len(msgs) == 2
        for m in msgs:
            assert isinstance(m, NodeUpdate)
            assert m.status == "completed"
            assert m.result == {"cached": True}

    def test_cached_removes_from_running_set(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "n1", "prompt_id": PROMPT_ID}})
        t.translate({"type": "execution_cached", "data": {"nodes": ["n1"], "prompt_id": PROMPT_ID}})
        # Finalize should not re-complete n1
        msgs = t.translate({"type": "execution_success", "data": {"prompt_id": PROMPT_ID}})
        node_updates = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(node_updates) == 0


# ---------------------------------------------------------------------------
# execution_success
# ---------------------------------------------------------------------------


class TestExecutionSuccess:
    def test_finalizes_lingering_nodes(self):
        t = _make_translator()
        t.translate({"type": "executing", "data": {"node": "n1", "prompt_id": PROMPT_ID}})
        t.translate({"type": "executing", "data": {"node": "n2", "prompt_id": PROMPT_ID}})
        # Only n1 completed
        t.translate({"type": "executed", "data": {"node": "n1", "output": {}, "prompt_id": PROMPT_ID}})
        # Success should force-complete n2
        msgs = t.translate({"type": "execution_success", "data": {"prompt_id": PROMPT_ID}})
        node_updates = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(node_updates) == 1
        assert node_updates[0].node_id == "n2"
        assert node_updates[0].status == "completed"
        # Plus job completion
        job_updates = [m for m in msgs if isinstance(m, JobUpdate)]
        assert len(job_updates) == 1
        assert job_updates[0].status == "completed"

    def test_no_lingering_nodes(self):
        t = _make_translator()
        msgs = t.translate({"type": "execution_success", "data": {"prompt_id": PROMPT_ID}})
        job_updates = [m for m in msgs if isinstance(m, JobUpdate)]
        assert len(job_updates) == 1
        assert job_updates[0].status == "completed"
        node_updates = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(node_updates) == 0


# ---------------------------------------------------------------------------
# execution_error
# ---------------------------------------------------------------------------


class TestExecutionError:
    def test_emits_node_error_and_job_failed(self):
        t = _make_translator()
        msgs = t.translate(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": PROMPT_ID,
                    "node_id": "n1",
                    "node_type": "KSampler",
                    "exception_message": "OOM",
                    "traceback": ["line 1", "line 2"],
                },
            }
        )
        node_errors = [m for m in msgs if isinstance(m, NodeUpdate) and m.status == "error"]
        assert len(node_errors) == 1
        assert node_errors[0].node_id == "n1"
        assert node_errors[0].error == "OOM"

        job_errors = [m for m in msgs if isinstance(m, JobUpdate) and m.status == "failed"]
        assert len(job_errors) == 1
        assert job_errors[0].error == "OOM"
        assert "line 1" in job_errors[0].traceback
        assert t.is_completed

    def test_error_without_node_id(self):
        t = _make_translator()
        msgs = t.translate(
            {
                "type": "execution_error",
                "data": {
                    "prompt_id": PROMPT_ID,
                    "exception_message": "Server crash",
                },
            }
        )
        node_errors = [m for m in msgs if isinstance(m, NodeUpdate)]
        assert len(node_errors) == 0  # no node_id provided
        job_errors = [m for m in msgs if isinstance(m, JobUpdate)]
        assert len(job_errors) == 1
        assert job_errors[0].status == "failed"


# ---------------------------------------------------------------------------
# execution_interrupted
# ---------------------------------------------------------------------------


class TestExecutionInterrupted:
    def test_emits_job_cancelled(self):
        t = _make_translator()
        msgs = t.translate({"type": "execution_interrupted", "data": {"prompt_id": PROMPT_ID}})
        assert len(msgs) == 1
        assert isinstance(msgs[0], JobUpdate)
        assert msgs[0].status == "cancelled"
        assert t.is_completed


# ---------------------------------------------------------------------------
# prompt_id filtering
# ---------------------------------------------------------------------------


class TestPromptIdFiltering:
    def test_ignores_messages_for_other_prompt(self):
        t = _make_translator()
        msgs = t.translate({"type": "executing", "data": {"node": "n1", "prompt_id": "other-prompt"}})
        assert msgs == []

    def test_accepts_messages_without_prompt_id(self):
        """Messages without prompt_id (e.g., status updates) are accepted."""
        t = _make_translator()
        msgs = t.translate({"type": "execution_start", "data": {}})
        assert len(msgs) == 1


# ---------------------------------------------------------------------------
# should_use_comfy_runner
# ---------------------------------------------------------------------------


class TestShouldUseComfyRunner:
    def test_run_mode_comfy(self):
        assert should_use_comfy_runner("comfy", None) is True

    def test_run_mode_other(self):
        assert should_use_comfy_runner("tool", None) is False

    def test_run_mode_none_with_comfy_nodes(self):
        graph = Graph(
            nodes=[Node(id="n1", type="comfy.KSampler", data={})],
            edges=[],
        )
        assert should_use_comfy_runner(None, graph) is True

    def test_run_mode_none_without_comfy_nodes(self):
        graph = Graph(
            nodes=[Node(id="n1", type="nodetool.math.Add", data={})],
            edges=[],
        )
        assert should_use_comfy_runner(None, graph) is False

    def test_run_mode_none_no_graph(self):
        assert should_use_comfy_runner(None, None) is False


if __name__ == "__main__":
    pytest.main([__file__])
