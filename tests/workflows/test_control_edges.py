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


# ---------- Phase 3: Graph.from_dict Control Edge Tests ----------


class TestGraphFromDictControlEdges:
    """Tests for Graph.from_dict() handling control edges."""

    def test_from_dict_preserves_control_edge_type(self):
        """Control edge_type should be preserved through from_dict()."""
        graph_dict = {
            "nodes": [
                {"id": "agent1", "type": "nodetool.agents.TestAgentNode", "data": {}},
                {"id": "node1", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "agent1",
                    "sourceHandle": "output",
                    "target": "node1",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                }
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].edge_type == "control"
        assert graph.edges[0].is_control()

    def test_from_dict_defaults_missing_edge_type_to_data(self):
        """Missing edge_type should default to 'data'."""
        graph_dict = {
            "nodes": [
                {"id": "node1", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
                {"id": "node2", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "node2",
                    "targetHandle": "threshold",
                    # Note: no edge_type field
                }
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        assert len(graph.edges) == 1
        assert graph.edges[0].edge_type == "data"
        assert not graph.edges[0].is_control()

    def test_from_dict_mixed_edge_types(self):
        """Graph can have both data and control edges."""
        graph_dict = {
            "nodes": [
                {"id": "agent1", "type": "nodetool.agents.TestAgentNode", "data": {}},
                {"id": "node1", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
                {"id": "node2", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "agent1",
                    "sourceHandle": "output",
                    "target": "node2",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                },
                {
                    "id": "e2",
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "node2",
                    "targetHandle": "threshold",
                    "edge_type": "data",
                },
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        assert len(graph.edges) == 2

        control_edges = [e for e in graph.edges if e.edge_type == "control"]
        data_edges = [e for e in graph.edges if e.edge_type == "data"]
        assert len(control_edges) == 1
        assert len(data_edges) == 1

    def test_from_dict_control_edge_does_not_filter_node_data(self):
        """Control edges should not filter node data (unlike data edges)."""
        graph_dict = {
            "nodes": [
                {"id": "agent1", "type": "nodetool.agents.TestAgentNode", "data": {}},
                {
                    "id": "node1",
                    "type": "tests.workflows.test_control_edges.TestProcessingNode",
                    "data": {"threshold": 0.7, "mode": "fast"},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "agent1",
                    "sourceHandle": "output",
                    "target": "node1",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                }
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        node = graph.find_node("node1")
        # __control__ is not a real property, so node data should be preserved
        assert node is not None
        assert node.threshold == 0.7
        assert node.mode == "fast"

    def test_from_dict_data_edge_filters_node_data(self):
        """Data edges should filter connected properties from node data."""
        graph_dict = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "tests.workflows.test_control_edges.TestProcessingNode",
                    "data": {"threshold": 0.7, "mode": "fast"},
                },
                {"id": "node2", "type": "tests.workflows.test_control_edges.TestPlainNode", "data": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "node2",
                    "sourceHandle": "output",
                    "target": "node1",
                    "targetHandle": "threshold",  # This is a real property
                    "edge_type": "data",
                }
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        node = graph.find_node("node1")
        assert node is not None
        # threshold should be filtered (will use default)
        assert node.threshold == 0.5  # default value
        # mode should be preserved
        assert node.mode == "fast"

    def test_from_dict_roundtrip_control_edges(self):
        """Control edges should survive serialization roundtrip."""
        agent = TestAgentNode(id="agent1")
        node = TestProcessingNode(id="node1")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="output",
                target="node1",
                targetHandle="__control__",
                edge_type="control",
            )
        ]
        original = Graph(nodes=[agent, node], edges=edges)

        # Serialize
        data = original.model_dump()

        # Deserialize via from_dict format
        graph_dict = {
            "nodes": [{"id": n.id, "type": n.get_node_type(), "data": {}} for n in original.nodes],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "sourceHandle": e.sourceHandle,
                    "target": e.target,
                    "targetHandle": e.targetHandle,
                    "edge_type": e.edge_type,
                }
                for e in original.edges
            ],
        }
        loaded = Graph.from_dict(graph_dict, skip_errors=False)

        assert len(loaded.edges) == 1
        assert loaded.edges[0].edge_type == "control"
        assert loaded.edges[0].is_control()

    def test_from_dict_chained_control_edges(self):
        """Chained control edges (A->B, B->C) should load correctly."""
        graph_dict = {
            "nodes": [
                {"id": "agent1", "type": "nodetool.agents.TestAgentNode", "data": {}},
                {"id": "agent2", "type": "nodetool.agents.TestAgentNode", "data": {}},
                {"id": "node1", "type": "tests.workflows.test_control_edges.TestProcessingNode", "data": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "agent1",
                    "sourceHandle": "output",
                    "target": "agent2",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                },
                {
                    "id": "e2",
                    "source": "agent2",
                    "sourceHandle": "output",
                    "target": "node1",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                },
            ],
        }
        graph = Graph.from_dict(graph_dict, skip_errors=False)
        assert len(graph.edges) == 2
        assert all(e.edge_type == "control" for e in graph.edges)

        # Validate topological order: agent1 -> agent2 -> node1
        levels = graph.topological_sort()
        flat = [nid for level in levels for nid in level]
        assert flat.index("agent1") < flat.index("agent2")
        assert flat.index("agent2") < flat.index("node1")


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


# ---------- Phase 6: WorkflowRunner.send_messages Control Routing ----------


class TestWorkflowRunnerControlRouting:
    """Tests for WorkflowRunner.send_messages() control output routing."""

    @pytest.mark.asyncio
    async def test_send_messages_routes_control_output(self):
        """Control output should be routed to __control__ inbox handle."""
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from unittest.mock import MagicMock

        runner = WorkflowRunner(job_id="test-job")

        agent = TestAgentNode(id="agent1")
        target = TestProcessingNode(id="target")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="__control_output__",
                target="target",
                targetHandle="__control__",
                edge_type="control",
            )
        ]
        graph = Graph(nodes=[agent, target], edges=edges)

        # Setup context mock
        context = MagicMock()
        context.graph = graph
        context.workflow_id = "test-workflow"

        # Setup inboxes
        target_inbox = NodeInbox()
        target_inbox.add_upstream("__control__", 1)
        runner.node_inboxes = {"target": target_inbox}

        # Send control params
        control_params = {"threshold": 0.9, "mode": "fast"}
        result = {"__control_output__": control_params, "output": "some_data"}

        await runner.send_messages(agent, result, context)

        # Mark source done so iteration can complete
        target_inbox.mark_source_done("__control__")

        # Verify control params were routed to target's __control__ handle
        items = []
        async for item in target_inbox.iter_input("__control__"):
            items.append(item)
        assert len(items) == 1
        assert items[0] == control_params

    @pytest.mark.asyncio
    async def test_send_messages_control_output_missing_target_inbox(self):
        """Control output with missing target inbox should not crash."""
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from unittest.mock import MagicMock

        runner = WorkflowRunner(job_id="test-job")

        agent = TestAgentNode(id="agent1")
        target = TestProcessingNode(id="target")
        edges = [
            Edge(
                id="e1",
                source="agent1",
                sourceHandle="__control_output__",
                target="target",
                targetHandle="__control__",
                edge_type="control",
            )
        ]
        graph = Graph(nodes=[agent, target], edges=edges)

        context = MagicMock()
        context.graph = graph
        context.workflow_id = "test-workflow"

        # No inbox for target
        runner.node_inboxes = {}

        control_params = {"threshold": 0.9}
        result = {"__control_output__": control_params}

        # Should not raise
        await runner.send_messages(agent, result, context)


# ---------- Phase 7: Integration Tests ----------


class TestControlEdgeIntegration:
    """Integration tests for control edge execution flow."""

    @pytest.mark.asyncio
    async def test_control_params_override_node_defaults(self):
        """Control params should override node default values."""
        from nodetool.workflows.actor import NodeActor
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from unittest.mock import MagicMock, AsyncMock

        # Create node with default values
        node = TestProcessingNode(id="node1", threshold=0.5, mode="normal")

        # Setup runner with control edge
        runner = MagicMock(spec=WorkflowRunner)
        runner.multi_edge_list_inputs = {}
        runner._control_edges = {
            "node1": [
                Edge(
                    id="e1",
                    source="agent1",
                    sourceHandle="output",
                    target="node1",
                    targetHandle="__control__",
                    edge_type="control",
                )
            ]
        }
        runner.disable_caching = True
        runner.device = "cpu"
        runner.job_id = "test-job"

        # Setup context
        context = MagicMock()
        context.graph = Graph(nodes=[node], edges=[])
        context.workflow_id = "test-workflow"

        # Setup inbox with control params
        inbox = NodeInbox()
        inbox.add_upstream("__control__", 1)
        await inbox.put("__control__", {"threshold": 0.9, "mode": "fast"})
        inbox.mark_source_done("__control__")

        actor = NodeActor(runner, node, context, inbox)

        # Wait for control params
        params = await actor._wait_for_control_params()
        assert params["threshold"] == 0.9
        assert params["mode"] == "fast"

        # Validate they would apply to node
        errors = actor._validate_control_params(params)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_control_params_mixed_with_data_inputs(self):
        """Control params should merge with data inputs, taking precedence."""
        from nodetool.workflows.actor import NodeActor
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from unittest.mock import MagicMock

        node = TestProcessingNode(id="node1")

        runner = MagicMock(spec=WorkflowRunner)
        runner.multi_edge_list_inputs = {}
        runner._control_edges = {
            "node1": [
                Edge(
                    id="e1",
                    source="agent1",
                    sourceHandle="output",
                    target="node1",
                    targetHandle="__control__",
                    edge_type="control",
                )
            ]
        }

        context = MagicMock()
        context.graph = Graph(nodes=[node], edges=[])
        context.workflow_id = "test-workflow"

        inbox = NodeInbox()
        # Control param for threshold
        inbox.add_upstream("__control__", 1)
        await inbox.put("__control__", {"threshold": 0.95})
        inbox.mark_source_done("__control__")

        # Data input for mode (simulating a data edge)
        inbox.add_upstream("mode", 1)
        await inbox.put("mode", "turbo")
        inbox.mark_source_done("mode")

        actor = NodeActor(runner, node, context, inbox)

        # Get control params
        control_params = await actor._wait_for_control_params()

        # Simulate the merge logic from process_node_with_inputs
        data_inputs = {"mode": "turbo"}  # From data edge
        merged = {**data_inputs, **control_params}  # Control takes precedence

        # Threshold from control, mode from data
        assert merged["threshold"] == 0.95
        assert merged["mode"] == "turbo"

    @pytest.mark.asyncio
    async def test_full_graph_json_roundtrip(self):
        """Test complete graph JSON serialization roundtrip with control edges."""
        # Build a complete graph with both data and control edges
        # Use test-local nodes to avoid dependency on nodetool-base
        graph_dict = {
            "nodes": [
                {
                    "id": "node1",
                    "type": "tests.workflows.test_control_edges.TestPlainNode",
                    "data": {"value": "test input"},
                },
                {
                    "id": "agent1",
                    "type": "nodetool.agents.TestAgentNode",
                    "data": {"prompt": ""},
                },
                {
                    "id": "processor1",
                    "type": "tests.workflows.test_control_edges.TestProcessingNode",
                    "data": {"threshold": 0.5, "mode": "normal"},
                },
            ],
            "edges": [
                {
                    "id": "data1",
                    "source": "node1",
                    "sourceHandle": "output",
                    "target": "processor1",
                    "targetHandle": "mode",
                    "edge_type": "data",
                },
                {
                    "id": "control1",
                    "source": "agent1",
                    "sourceHandle": "__control_output__",
                    "target": "processor1",
                    "targetHandle": "__control__",
                    "edge_type": "control",
                },
            ],
        }

        # Load graph
        graph = Graph.from_dict(graph_dict, skip_errors=False)

        # Verify structure
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

        # Verify edge types
        data_edges = [e for e in graph.edges if e.edge_type == "data"]
        control_edges = [e for e in graph.edges if e.edge_type == "control"]
        assert len(data_edges) == 1
        assert len(control_edges) == 1

        # Verify topological order
        levels = graph.topological_sort()
        flat = [nid for level in levels for nid in level]
        # node1 and agent1 should come before processor1
        assert flat.index("agent1") < flat.index("processor1")
        assert flat.index("node1") < flat.index("processor1")

        # Serialize back
        serialized = graph.model_dump()
        edge_types = {e["id"]: e["edge_type"] for e in serialized["edges"]}
        assert edge_types["data1"] == "data"
        assert edge_types["control1"] == "control"
