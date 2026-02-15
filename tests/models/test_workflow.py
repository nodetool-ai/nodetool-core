"""
Tests for the Workflow model (models/workflow.py).

The Workflow model represents workflows in the nodetool system,
including graph structure, metadata, and user association.
"""

from datetime import datetime

import pytest

from nodetool.models.workflow import Workflow


class TestWorkflowModel:
    """Test suite for Workflow model core functionality."""

    def test_workflow_creation_with_defaults(self):
        """Test creating a workflow with default values."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
        )
        assert workflow.id == "test-id"
        assert workflow.user_id == "user-123"
        assert workflow.access == "private"
        assert workflow.name == ""
        assert workflow.tags == []
        assert workflow.description == ""
        assert workflow.graph == {}
        assert workflow.settings == {}
        assert isinstance(workflow.created_at, datetime)
        assert isinstance(workflow.updated_at, datetime)

    def test_workflow_creation_with_values(self):
        """Test creating a workflow with specific values."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            access="public",
            name="Test Workflow",
            tags=["tag1", "tag2"],
            description="A test workflow",
            package_name="test_package",
            thumbnail="thumb.jpg",
            graph={"nodes": [], "edges": []},
            settings={"setting1": "value1"},
            receive_clipboard=True,
            run_mode="tool",
            workspace_id="workspace-123",
        )
        assert workflow.id == "test-id"
        assert workflow.user_id == "user-123"
        assert workflow.access == "public"
        assert workflow.name == "Test Workflow"
        assert workflow.tags == ["tag1", "tag2"]
        assert workflow.description == "A test workflow"
        assert workflow.package_name == "test_package"
        assert workflow.thumbnail == "thumb.jpg"
        assert workflow.graph == {"nodes": [], "edges": []}
        assert workflow.settings == {"setting1": "value1"}
        assert workflow.receive_clipboard is True
        assert workflow.run_mode == "tool"
        assert workflow.workspace_id == "workspace-123"

    def test_before_save_updates_timestamp(self):
        """Test that before_save updates the updated_at timestamp."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
        )
        old_updated_at = workflow.updated_at
        # Simulate a small delay
        import time
        time.sleep(0.01)
        workflow.before_save()
        assert workflow.updated_at > old_updated_at

    def test_from_dict_creates_workflow(self):
        """Test creating a workflow from a dictionary."""
        data = {
            "id": "test-id",
            "user_id": "user-123",
            "access": "public",
            "name": "Test Workflow",
            "tags": ["tag1", "tag2"],
            "description": "A test workflow",
            "package_name": "test_package",
            "thumbnail": "thumb.jpg",
            "settings": {"setting1": "value1"},
            "graph": {"nodes": [], "edges": []},
            "run_mode": "tool",
            "workspace_id": "workspace-123",
        }
        workflow = Workflow.from_dict(data)
        assert workflow.id == "test-id"
        assert workflow.user_id == "user-123"
        assert workflow.access == "public"
        assert workflow.name == "Test Workflow"
        assert workflow.tags == ["tag1", "tag2"]
        assert workflow.description == "A test workflow"
        assert workflow.package_name == "test_package"
        assert workflow.thumbnail == "thumb.jpg"
        assert workflow.settings == {"setting1": "value1"}
        assert workflow.graph == {"nodes": [], "edges": []}
        assert workflow.run_mode == "tool"
        assert workflow.workspace_id == "workspace-123"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing optional fields."""
        data = {
            "id": "test-id",
            "user_id": "user-123",
            "name": "Test Workflow",
        }
        workflow = Workflow.from_dict(data)
        assert workflow.id == "test-id"
        assert workflow.user_id == "user-123"
        assert workflow.name == "Test Workflow"
        assert workflow.access == ""  # Empty string default
        assert workflow.tags == []  # Empty list default
        assert workflow.description == ""  # Empty string default
        assert workflow.graph == {"nodes": [], "edges": []}  # Default graph
        assert workflow.settings == {}  # Empty dict default

    def test_has_trigger_nodes_with_trigger(self):
        """Test has_trigger_nodes returns True when trigger nodes present."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph={
                "nodes": [
                    {"type": "nodetool.triggers.cron"},
                    {"type": "nodetool.text.generate"},
                ]
            },
        )
        assert workflow.has_trigger_nodes() is True

    def test_has_trigger_nodes_without_trigger(self):
        """Test has_trigger_nodes returns False when no trigger nodes."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph={
                "nodes": [
                    {"type": "nodetool.text.generate"},
                    {"type": "nodetool.image.create"},
                ]
            },
        )
        assert workflow.has_trigger_nodes() is False

    def test_has_trigger_nodes_empty_graph(self):
        """Test has_trigger_nodes with empty graph."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph={},
        )
        assert workflow.has_trigger_nodes() is False

    def test_has_trigger_nodes_no_nodes_key(self):
        """Test has_trigger_nodes when graph has no nodes key."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph={"edges": []},
        )
        assert workflow.has_trigger_nodes() is False

    def test_has_trigger_nodes_empty_nodes(self):
        """Test has_trigger_nodes when nodes list is empty."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph={"nodes": []},
        )
        assert workflow.has_trigger_nodes() is False

    def test_get_api_graph(self):
        """Test get_api_graph returns the API graph representation."""
        graph_data = {
            "nodes": [
                {"id": "node1", "type": "test"},
            ],
            "edges": [
                {
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "node2",
                    "targetHandle": "input",
                    "id": "edge1",
                },
            ],
        }
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            graph=graph_data,
        )
        api_graph = workflow.get_api_graph()
        assert len(api_graph.nodes) == len(graph_data["nodes"])
        assert api_graph.nodes[0].id == "node1"
        assert api_graph.nodes[0].type == "test"
        assert len(api_graph.edges) == len(graph_data["edges"])
        assert api_graph.edges[0].source == "node1"
        assert api_graph.edges[0].target == "node2"

    def test_get_table_schema(self):
        """Test get_table_schema returns correct table name."""
        schema = Workflow.get_table_schema()
        assert schema["table_name"] == "nodetool_workflows"


class TestWorkflowRunMode:
    """Test suite for workflow run modes."""

    def test_run_mode_tool(self):
        """Test workflow with run_mode='tool'."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            run_mode="tool",
        )
        assert workflow.run_mode == "tool"

    def test_run_mode_trigger(self):
        """Test workflow with run_mode='trigger'."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            run_mode="trigger",
        )
        assert workflow.run_mode == "trigger"

    def test_run_mode_none(self):
        """Test workflow with run_mode=None."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
        )
        assert workflow.run_mode is None


class TestWorkflowAccessControl:
    """Test suite for workflow access control."""

    def test_access_private_default(self):
        """Test workflow has private access by default."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
        )
        assert workflow.access == "private"

    def test_access_public(self):
        """Test workflow with public access."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            access="public",
        )
        assert workflow.access == "public"


class TestWorkflowMetadata:
    """Test suite for workflow metadata fields."""

    def test_thumbnail_and_thumbnail_url(self):
        """Test workflow thumbnail fields."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            thumbnail="local-thumb.jpg",
            thumbnail_url="https://example.com/thumb.jpg",
        )
        assert workflow.thumbnail == "local-thumb.jpg"
        assert workflow.thumbnail_url == "https://example.com/thumb.jpg"

    def test_html_app_field(self):
        """Test workflow html_app field."""
        html_content = "<html><body>App</body></html>"
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            html_app=html_content,
        )
        assert workflow.html_app == html_content

    def test_tool_name_field(self):
        """Test workflow tool_name field."""
        workflow = Workflow(
            id="test-id",
            user_id="user-123",
            tool_name="my_tool",
        )
        assert workflow.tool_name == "my_tool"
