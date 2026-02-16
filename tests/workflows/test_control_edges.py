"""
Tests for control edge support in the workflow system.

Control edges allow Agent nodes to dynamically set parameters of other nodes.
"""

import pytest

from nodetool.types.api_graph import Edge
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox

# ---------- Test Node Classes ----------


class TestAgentNode(BaseNode):
    """Agent node for testing control edges."""

    prompt: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.agents.TestAgentNode"

    async def process(self, context):
        return {"output": "agent_result"}


class TestProcessingNode(BaseNode):
    """Processing node for testing control edges."""

    threshold: float = 0.5
    mode: str = "normal"

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_control_edges.TestProcessingNode"

    async def process(self, context):
        return {"output": f"{self.mode}:{self.threshold}"}


class TestPlainNode(BaseNode):
    """Non-agent node (should not be a valid control source)."""

    value: str = ""

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.workflows.test_control_edges.TestPlainNode"

    async def process(self, context):
        return {"output": self.value}


# ---------- Phase 1: Edge Model Tests ----------


class TestEdgeModel:
    """Tests for Edge model with control type."""

    def test_control_edge_model(self):
        """Test Edge model with control type."""
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        assert edge.is_control()
        assert edge.edge_type == "control"

    def test_data_edge_model(self):
        """Test Edge model with data type (default)."""
        edge = Edge(
            id="e1",
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="input",
        )
        assert not edge.is_control()
        assert edge.edge_type == "data"

    def test_edge_type_default_is_data(self):
        """Test that edge_type defaults to 'data'."""
        edge = Edge(
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="input",
        )
        assert edge.edge_type == "data"

    def test_edge_serialization_includes_edge_type(self):
        """Test that edge serialization includes edge_type."""
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        data = edge.model_dump()
        assert data["edge_type"] == "control"

    def test_edge_deserialization_with_edge_type(self):
        """Test Edge model deserialization with edge_type."""
        data = {
            "id": "e1",
            "source": "agent1",
            "sourceHandle": "output",
            "target": "node2",
            "targetHandle": "__control__",
            "edge_type": "control",
        }
        edge = Edge(**data)
        assert edge.is_control()

    def test_edge_deserialization_without_edge_type(self):
        """Test Edge model deserialization defaults to data."""
        data = {
            "source": "node1",
            "sourceHandle": "output",
            "target": "node2",
            "targetHandle": "input",
        }
        edge = Edge(**data)
        assert not edge.is_control()


# ---------- Phase 2: Graph Control Edge Tests ----------


class TestGraphControlEdges:
    """Tests for Graph control edge methods."""

    def test_get_control_edges(self):
        """Test getting control edges targeting a node."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e2",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="threshold",
                edge_type="data",
            ),
        ]
        graph = Graph(nodes=[agent, node], edges=edges)
        control_edges = graph.get_control_edges("node2")
        assert len(control_edges) == 1
        assert control_edges[0].id == "e1"

    def test_get_control_edges_none(self):
        """Test getting control edges when none exist."""
        node = TestProcessingNode(id="node1")
        graph = Graph(nodes=[node], edges=[])
        control_edges = graph.get_control_edges("node1")
        assert len(control_edges) == 0

    def test_get_controller_nodes(self):
        """Test getting controller nodes for a target."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent, node], edges=edges)
        controllers = graph.get_controller_nodes("node2")
        assert len(controllers) == 1
        assert controllers[0].id == "agent1"

    def test_get_controlled_nodes(self):
        """Test getting controlled nodes from a source."""
        agent = TestAgentNode(id="agent1")
        node1 = TestProcessingNode(id="node1")
        node2 = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node1",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e2",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent, node1, node2], edges=edges)
        controlled = graph.get_controlled_nodes("agent1")
        assert set(controlled) == {"node1", "node2"}

    def test_validate_control_edge_target_handle(self):
        """Control edge must use __control__ as targetHandle."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node2",
            targetHandle="wrong_handle",
            edge_type="control",
        )
        graph = Graph(nodes=[agent, node], edges=[edge])
        errors = graph.validate_control_edges()
        assert len(errors) > 0
        assert "__control__" in errors[0]

    def test_validate_control_edge_from_non_agent(self):
        """Control edge must originate from Agent node."""
        node1 = TestPlainNode(id="node1")
        node2 = TestProcessingNode(id="node2")
        edge = Edge(
            id="e1",
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        graph = Graph(nodes=[node1, node2], edges=[edge])
        errors = graph.validate_control_edges()
        assert len(errors) > 0
        assert "Agent" in errors[0]

    def test_validate_circular_control_dependency(self):
        """Circular control dependencies should be detected."""
        agent1 = TestAgentNode(id="agent1")
        agent2 = TestAgentNode(id="agent2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="agent2",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e2",
                source="agent2",
                sourceHandle="output",
                target="agent1",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent1, agent2], edges=edges)
        errors = graph.validate_control_edges()
        assert len(errors) > 0
        assert "circular" in errors[0].lower()

    def test_multiple_controllers_allowed(self):
        """Multiple controllers for same node should be allowed."""
        agent1 = TestAgentNode(id="agent1")
        agent2 = TestAgentNode(id="agent2")
        target = TestProcessingNode(id="target")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="target",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e2",
                source="agent2",
                sourceHandle="output",
                target="target",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent1, agent2, target], edges=edges)
        errors = graph.validate_control_edges()
        assert len(errors) == 0

    def test_valid_control_edge(self):
        """Valid control edge should pass validation."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        graph = Graph(nodes=[agent, node], edges=[edge])
        errors = graph.validate_control_edges()
        assert len(errors) == 0

    def test_validate_control_edge_invalid_source(self):
        """Control edge with non-existent source should fail validation."""
        node = TestProcessingNode(id="node2")
        edge = Edge(
            id="e1",
            source="nonexistent",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        graph = Graph(nodes=[node], edges=[edge])
        errors = graph.validate_control_edges()
        assert len(errors) > 0
        assert "invalid source" in errors[0].lower()

    def test_validate_control_edge_invalid_target(self):
        """Control edge with non-existent target should fail validation."""
        agent = TestAgentNode(id="agent1")
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="nonexistent",
            targetHandle="__control__",
            edge_type="control",
        )
        graph = Graph(nodes=[agent], edges=[edge])
        errors = graph.validate_control_edges()
        assert len(errors) > 0
        assert "invalid target" in errors[0].lower()

    def test_validate_edge_types_includes_control_validation(self):
        """validate_edge_types should include control edge validation."""
        node1 = TestPlainNode(id="node1")
        node2 = TestProcessingNode(id="node2")
        edge = Edge(
            id="e1",
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="__control__",
            edge_type="control",
        )
        graph = Graph(nodes=[node1, node2], edges=[edge])
        errors = graph.validate_edge_types()
        # Should contain errors from control edge validation
        assert any("Agent" in e for e in errors)

    def test_validate_edge_types_skips_control_edges_for_type_checking(self):
        """Control edges should be excluded from data type compatibility checks."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent, node], edges=edges)
        errors = graph.validate_edge_types()
        # Control edge should NOT cause type mismatch errors
        type_errors = [e for e in errors if "Type mismatch" in e or "Property" in e]
        assert len(type_errors) == 0

    def test_topological_sort_with_control_edges(self):
        """Topological sort should work with control edges."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
        ]
        graph = Graph(nodes=[agent, node], edges=edges)
        levels = graph.topological_sort()
        # Should produce valid topological order
        assert len(levels) > 0
        # Flatten levels to get execution order
        flat = [nid for level in levels for nid in level]
        assert "agent1" in flat
        assert "node2" in flat
        # agent1 should come before node2
        assert flat.index("agent1") < flat.index("node2")


# ---------- Phase 4: NodeActor Control Parameter Tests ----------


class TestNodeActorControlParams:
    """Tests for NodeActor control parameter methods."""

    @pytest.mark.asyncio
    async def test_validate_control_params_valid(self):
        """Valid control params should pass validation."""
        from unittest.mock import MagicMock

        node = TestProcessingNode(id="node1")
        runner = MagicMock()
        runner._control_edges = {}
        runner.multi_edge_list_inputs = {}
        context = MagicMock()
        inbox = NodeInbox()

        from nodetool.workflows.actor import NodeActor

        actor = NodeActor(runner, node, context, inbox)
        errors = actor._validate_control_params({"threshold": 0.8, "mode": "fast"})
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_control_params_invalid(self):
        """Invalid control params should return errors."""
        from unittest.mock import MagicMock

        node = TestProcessingNode(id="node1")
        runner = MagicMock()
        runner._control_edges = {}
        runner.multi_edge_list_inputs = {}
        context = MagicMock()
        inbox = NodeInbox()

        from nodetool.workflows.actor import NodeActor

        actor = NodeActor(runner, node, context, inbox)
        errors = actor._validate_control_params({"nonexistent_param": "value"})
        assert len(errors) == 1
        assert "nonexistent_param" in errors[0]

    @pytest.mark.asyncio
    async def test_has_control_edges(self):
        """Test _has_control_edges detection."""
        from unittest.mock import MagicMock

        node = TestProcessingNode(id="node1")
        runner = MagicMock()
        runner.multi_edge_list_inputs = {}
        context = MagicMock()
        inbox = NodeInbox()

        from nodetool.workflows.actor import NodeActor

        # No control edges
        runner._control_edges = {}
        actor = NodeActor(runner, node, context, inbox)
        assert not actor._has_control_edges()

        # With control edge
        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node1",
            targetHandle="__control__",
            edge_type="control",
        )
        runner._control_edges = {"node1": [edge]}
        actor2 = NodeActor(runner, node, context, inbox)
        assert actor2._has_control_edges()

    @pytest.mark.asyncio
    async def test_wait_for_control_params(self):
        """Test receiving control params through inbox."""
        from unittest.mock import MagicMock

        node = TestProcessingNode(id="node1")
        runner = MagicMock()
        runner.multi_edge_list_inputs = {}
        context = MagicMock()
        inbox = NodeInbox()
        inbox.add_upstream("__control__", 1)

        edge = Edge(
            id="e1",
            source="agent1",
            sourceHandle="output",
            target="node1",
            targetHandle="__control__",
            edge_type="control",
        )
        runner._control_edges = {"node1": [edge]}

        from nodetool.workflows.actor import NodeActor

        actor = NodeActor(runner, node, context, inbox)

        # Put control params in inbox
        await inbox.put("__control__", {"threshold": 0.9, "mode": "fast"})
        inbox.mark_source_done("__control__")

        # Wait for control params
        params = await actor._wait_for_control_params()
        assert params == {"threshold": 0.9, "mode": "fast"}


# ---------- Phase 5: Inbox Control Handle Tests ----------


class TestInboxControlHandle:
    """Tests verifying inbox supports __control__ handle."""

    @pytest.mark.asyncio
    async def test_inbox_control_handle_put_and_iter(self):
        """Test inbox put and iter on __control__ handle."""
        inbox = NodeInbox()
        inbox.add_upstream("__control__", 1)

        await inbox.put("__control__", {"param1": "value1"})
        inbox.mark_source_done("__control__")

        items = []
        async for item in inbox.iter_input("__control__"):
            items.append(item)

        assert len(items) == 1
        assert items[0] == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_inbox_control_handle_multiple_messages(self):
        """Test inbox with multiple control messages."""
        inbox = NodeInbox()
        inbox.add_upstream("__control__", 2)

        await inbox.put("__control__", {"param1": "value1"})
        await inbox.put("__control__", {"param2": "value2"})
        inbox.mark_source_done("__control__")
        inbox.mark_source_done("__control__")

        items = []
        async for item in inbox.iter_input("__control__"):
            items.append(item)

        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_inbox_control_and_data_handles(self):
        """Test inbox with both __control__ and data handles."""
        inbox = NodeInbox()
        inbox.add_upstream("__control__", 1)
        inbox.add_upstream("input", 1)

        await inbox.put("__control__", {"threshold": 0.9})
        await inbox.put("input", "data_value")
        inbox.mark_source_done("__control__")
        inbox.mark_source_done("input")

        # Read control
        control_items = []
        async for item in inbox.iter_input("__control__"):
            control_items.append(item)
        assert len(control_items) == 1
        assert control_items[0] == {"threshold": 0.9}

        # Read data
        data_items = []
        async for item in inbox.iter_input("input"):
            data_items.append(item)
        assert len(data_items) == 1
        assert data_items[0] == "data_value"


# ---------- WorkflowRunner Control Edge Classification ----------


class TestWorkflowRunnerControlEdges:
    """Tests for WorkflowRunner control edge classification."""

    def test_classify_control_edges(self):
        """Test that _classify_control_edges populates correctly."""
        from nodetool.workflows.workflow_runner import WorkflowRunner

        runner = WorkflowRunner(job_id="test-job")

        agent = TestAgentNode(id="agent1")
        node1 = TestProcessingNode(id="node1")
        node2 = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node1",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e2",
                source="agent1",
                sourceHandle="output",
                target="node2",
                targetHandle="__control__",
                edge_type="control",
            ),
            Edge(
                id="e3",
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="threshold",
                edge_type="data",
            ),
        ]
        graph = Graph(nodes=[agent, node1, node2], edges=edges)

        runner._classify_control_edges(graph)

        assert "node1" in runner._control_edges
        assert "node2" in runner._control_edges
        assert len(runner._control_edges["node1"]) == 1
        assert len(runner._control_edges["node2"]) == 1
        # Data edges should not be in _control_edges
        assert "agent1" not in runner._control_edges

    def test_classify_control_edges_empty(self):
        """Test classification with no control edges."""
        from nodetool.workflows.workflow_runner import WorkflowRunner

        runner = WorkflowRunner(job_id="test-job")

        node1 = TestProcessingNode(id="node1")
        node2 = TestProcessingNode(id="node2")
        edges = [
            Edge(
                id="e1",
                source="node1",
                sourceHandle="output",
                target="node2",
                targetHandle="threshold",
                edge_type="data",
            ),
        ]
        graph = Graph(nodes=[node1, node2], edges=edges)

        runner._classify_control_edges(graph)

        assert len(runner._control_edges) == 0

    def test_build_control_context(self):
        """Test building control context for a node."""
        from nodetool.workflows.workflow_runner import WorkflowRunner

        runner = WorkflowRunner(job_id="test-job")

        node = TestProcessingNode(id="node1")
        graph = Graph(nodes=[node], edges=[])

        ctx = runner._build_control_context(node, graph)

        assert ctx["node_id"] == "node1"
        assert ctx["node_type"] == "tests.workflows.test_control_edges.TestProcessingNode"
        assert "threshold" in ctx["properties"]
        assert "mode" in ctx["properties"]
        assert ctx["properties"]["threshold"]["value"] == 0.5
        assert ctx["properties"]["mode"]["value"] == "normal"
