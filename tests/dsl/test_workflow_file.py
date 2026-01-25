"""Tests for nodetool.dsl.workflow_file module."""

import tempfile
from pathlib import Path
from typing import ClassVar

import pytest

from nodetool.dsl.workflow_file import (
    WorkflowFile,
    _extract_workflow_metadata_static,
    load_workflow_file,
    workflow_file_to_py,
    workflow_to_workflow_file,
)
from nodetool.types.api_graph import Edge, Graph, Node


def create_simple_graph() -> Graph:
    """Create a simple test graph with two nodes."""
    nodes = [
        Node(
            id="node1",
            type="nodetool.input.StringInput",
            data={"name": "input", "value": "test"},
        ),
        Node(
            id="node2",
            type="nodetool.text.Concat",
            data={"a": "", "b": " suffix"},
        ),
    ]
    edges = [
        Edge(
            id="edge1",
            source="node1",
            sourceHandle="output",
            target="node2",
            targetHandle="a",
        ),
    ]
    return Graph(nodes=nodes, edges=edges)


class TestWorkflowFile:
    """Tests for the WorkflowFile dataclass."""

    def test_default_values(self):
        """Test WorkflowFile with default values."""
        wf = WorkflowFile()
        assert wf.name == ""
        assert wf.description == ""
        assert wf.docstring == ""
        assert wf.tags == []
        assert wf.graph is None
        assert wf.settings == {}
        assert wf.thumbnail is None
        assert wf.path is None

    def test_with_values(self):
        """Test WorkflowFile with provided values."""
        graph = create_simple_graph()
        wf = WorkflowFile(
            name="Test Workflow",
            description="A test workflow",
            docstring="Background info here",
            tags=["test", "example"],
            graph=graph,
            settings={"key": "value"},
            thumbnail="thumb.png",
            path="/path/to/workflow.py",
        )
        assert wf.name == "Test Workflow"
        assert wf.description == "A test workflow"
        assert wf.docstring == "Background info here"
        assert wf.tags == ["test", "example"]
        assert wf.graph == graph
        assert wf.settings == {"key": "value"}
        assert wf.thumbnail == "thumb.png"
        assert wf.path == "/path/to/workflow.py"


class TestWorkflowFileToPy:
    """Tests for workflow_file_to_py function."""

    def test_basic_export(self):
        """Test basic workflow export to Python."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test Workflow",
            description="A test workflow",
        )

        assert "from nodetool.dsl.graph import graph" in result
        assert "name = 'Test Workflow'" in result or 'name = "Test Workflow"' in result
        assert "description = 'A test workflow'" in result or 'description = "A test workflow"' in result
        assert "graph = graph(" in result

    def test_with_docstring(self):
        """Test export with module docstring."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test",
            docstring="This is the workflow background.\n\nMore context here.",
        )

        assert '"""' in result
        assert "This is the workflow background." in result
        assert "More context here." in result

    def test_with_tags(self):
        """Test export with tags."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test",
            tags=["tag1", "tag2"],
        )

        assert "tags = " in result
        assert '"tag1"' in result or "'tag1'" in result

    def test_with_settings(self):
        """Test export with settings."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test",
            settings={"key": "value"},
        )

        assert "settings = " in result

    def test_with_tool_name(self):
        """Test export with tool name."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test",
            tool_name="my_tool",
        )

        assert "tool_name = 'my_tool'" in result or 'tool_name = "my_tool"' in result

    def test_with_run_mode(self):
        """Test export with run mode."""
        graph = create_simple_graph()
        result = workflow_file_to_py(
            graph,
            name="Test",
            run_mode="trigger",
        )

        assert "run_mode = 'trigger'" in result or 'run_mode = "trigger"' in result

    def test_invalid_graph_type(self):
        """Test that non-Graph types raise TypeError."""
        with pytest.raises(TypeError):
            workflow_file_to_py({"nodes": [], "edges": []})  # type: ignore


class TestWorkflowToWorkflowFile:
    """Tests for workflow_to_workflow_file function."""

    def test_from_workflow_file(self):
        """Test conversion from WorkflowFile."""
        graph = create_simple_graph()
        wf = WorkflowFile(
            name="My Workflow",
            description="Description here",
            graph=graph,
            tags=["a", "b"],
        )

        result = workflow_to_workflow_file(wf)

        assert "name = 'My Workflow'" in result or 'name = "My Workflow"' in result
        assert "description = 'Description here'" in result or 'description = "Description here"' in result

    def test_missing_graph_raises_error(self):
        """Test that missing graph raises ValueError."""
        wf = WorkflowFile(name="No Graph")

        with pytest.raises(ValueError, match="must have a graph"):
            workflow_to_workflow_file(wf)

    def test_from_dict_like_object(self):
        """Test conversion from object with dict graph."""

        class MockWorkflow:
            name: ClassVar[str] = "Mock"
            description: ClassVar[str] = "Mock workflow"
            graph: ClassVar[dict] = {
                "nodes": [
                    {"id": "n1", "type": "nodetool.input.StringInput", "data": {}},
                ],
                "edges": [],
            }

        result = workflow_to_workflow_file(MockWorkflow())
        assert "name = 'Mock'" in result or 'name = "Mock"' in result


class TestLoadWorkflowFile:
    """Tests for load_workflow_file function."""

    def test_load_simple_workflow(self):
        """Test loading a simple workflow file."""
        workflow_content = '''
"""
This is a test workflow.

Background:
    Used for testing.
"""

from nodetool.dsl.graph import graph as create_graph
from nodetool.types.api_graph import Graph, Node

name = "Test Workflow"
description = "A test workflow"
tags = ["test"]

# Create a simple graph
graph = Graph(nodes=[], edges=[])
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(workflow_content)
            f.flush()

            wf = load_workflow_file(f.name)

            assert wf.name == "Test Workflow"
            assert wf.description == "A test workflow"
            assert wf.tags == ["test"]
            assert "This is a test workflow" in wf.docstring
            assert wf.graph is not None

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_workflow_file("/nonexistent/path/workflow.py")

    def test_invalid_extension(self):
        """Test error for non-Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("not python")
            f.flush()

            with pytest.raises(ValueError, match="must be a Python file"):
                load_workflow_file(f.name)

    def test_invalid_graph_type(self):
        """Test error when graph is not a Graph object."""
        workflow_content = '''
name = "Bad Workflow"
graph = "not a graph"
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(workflow_content)
            f.flush()

            with pytest.raises(ValueError, match="must be a Graph object"):
                load_workflow_file(f.name)


class TestExtractWorkflowMetadataStatic:
    """Tests for _extract_workflow_metadata_static function."""

    def test_extract_basic_metadata(self):
        """Test static extraction of basic metadata."""
        workflow_content = '''
"""
Module docstring here.
"""

name = "Static Test"
description = "Static description"
tags = ["static", "test"]
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(workflow_content)
            f.flush()

            metadata = _extract_workflow_metadata_static(f.name)

            assert metadata["name"] == "Static Test"
            assert metadata["description"] == "Static description"
            assert metadata["tags"] == ["static", "test"]
            assert "Module docstring here" in metadata["docstring"]

    def test_default_name_from_filename(self):
        """Test that default name comes from filename."""
        workflow_content = '''
description = "No name defined"
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="my_workflow_", delete=False
        ) as f:
            f.write(workflow_content)
            f.flush()

            metadata = _extract_workflow_metadata_static(f.name)

            # Name should be derived from filename
            assert "my_workflow_" in metadata["name"]

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            _extract_workflow_metadata_static("/nonexistent/path/workflow.py")

    def test_extract_settings(self):
        """Test extraction of settings dictionary."""
        workflow_content = '''
name = "Settings Test"
settings = {"key": "value", "num": 42}
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(workflow_content)
            f.flush()

            metadata = _extract_workflow_metadata_static(f.name)

            assert metadata["settings"] == {"key": "value", "num": 42}


class TestRoundTrip:
    """Tests for round-trip export and load."""

    def test_export_and_load(self):
        """Test exporting a workflow and loading it back."""
        graph = create_simple_graph()
        original = WorkflowFile(
            name="Round Trip Test",
            description="Testing round trip",
            docstring="Background information.\n\nContext here.",
            tags=["round", "trip"],
            graph=graph,
        )

        # Export to Python
        py_source = workflow_to_workflow_file(original)

        # The exported source should contain all metadata
        assert "name = 'Round Trip Test'" in py_source or 'name = "Round Trip Test"' in py_source
        assert "description = 'Testing round trip'" in py_source or 'description = "Testing round trip"' in py_source
        assert "Background information." in py_source

        # Static extraction should recover metadata
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(py_source)
            f.flush()

            metadata = _extract_workflow_metadata_static(f.name)

            assert metadata["name"] == "Round Trip Test"
            assert metadata["description"] == "Testing round trip"
            assert metadata["tags"] == ["round", "trip"]
