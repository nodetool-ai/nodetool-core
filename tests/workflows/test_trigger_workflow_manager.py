"""
Tests for the TriggerWorkflowManager.
"""

from __future__ import annotations

import pytest

from nodetool.workflows.trigger_workflow_manager import (
    TriggerWorkflowManager,
    workflow_has_trigger_nodes,
)


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self, id: str, graph: dict):
        self.id = id
        self.graph = graph
        self.name = "Test Workflow"
        self.user_id = "1"


def test_workflow_has_trigger_nodes_with_trigger():
    """Test that workflow with trigger node is detected."""
    workflow = MockWorkflow(
        id="test-1",
        graph={
            "nodes": [
                {"type": "triggers.interval.IntervalTrigger"},
                {"type": "nodetool.output.TextOutput"},
            ],
            "edges": [],
        },
    )
    assert workflow_has_trigger_nodes(workflow) is True


def test_workflow_has_trigger_nodes_without_trigger():
    """Test that workflow without trigger node is not detected."""
    workflow = MockWorkflow(
        id="test-2",
        graph={
            "nodes": [
                {"type": "nodetool.input.TextInput"},
                {"type": "nodetool.output.TextOutput"},
            ],
            "edges": [],
        },
    )
    assert workflow_has_trigger_nodes(workflow) is False


def test_workflow_has_trigger_nodes_empty_graph():
    """Test that empty graph returns False."""
    workflow = MockWorkflow(
        id="test-3",
        graph={},
    )
    assert workflow_has_trigger_nodes(workflow) is False


def test_workflow_has_trigger_nodes_no_nodes():
    """Test that graph with no nodes returns False."""
    workflow = MockWorkflow(
        id="test-4",
        graph={"nodes": [], "edges": []},
    )
    assert workflow_has_trigger_nodes(workflow) is False


def test_trigger_workflow_manager_singleton():
    """Test that TriggerWorkflowManager is a singleton."""
    manager1 = TriggerWorkflowManager.get_instance()
    manager2 = TriggerWorkflowManager.get_instance()
    assert manager1 is manager2


def test_trigger_workflow_manager_is_workflow_running_not_running():
    """Test is_workflow_running for non-running workflow."""
    manager = TriggerWorkflowManager.get_instance()
    assert manager.is_workflow_running("nonexistent-workflow") is False


def test_trigger_workflow_manager_list_running_workflows():
    """Test listing running workflows."""
    manager = TriggerWorkflowManager.get_instance()
    running = manager.list_running_workflows()
    assert isinstance(running, dict)
