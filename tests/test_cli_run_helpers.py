
import pytest
import json
import base64
from unittest.mock import MagicMock, patch
from nodetool.cli import (
    _run_json_default,
    _run_is_dsl_file,
    _run_is_dsl_content,
    _run_parse_workflow_arg,
)
from nodetool.types.api_graph import Graph
from nodetool.workflows.run_job_request import RunJobRequest

class TestRunHelpers:
    """Tests for CLI run helper functions."""

    def test_run_json_default_serialization(self):
        """Test _run_json_default handles different types correctly."""
        # Test bytes
        data = b"hello"
        result = _run_json_default(data)
        assert result["__type__"] == "bytes"
        assert result["base64"] == base64.b64encode(data).decode("utf-8")

        # Test simple types fallback
        assert _run_json_default("test") == "test"
        assert _run_json_default(123) == "123"

        # Test object with model_dump
        class MockModel:
            def model_dump(self):
                return {"key": "value"}

        assert _run_json_default(MockModel()) == {"key": "value"}

    def test_run_is_dsl_file(self):
        """Test file extension check."""
        assert _run_is_dsl_file("workflow.py") is True
        assert _run_is_dsl_file("path/to/workflow.py") is True
        assert _run_is_dsl_file("workflow.pyc") is False
        assert _run_is_dsl_file("workflow.json") is False

    def test_run_is_dsl_content(self):
        """Test DSL content heuristics."""
        assert _run_is_dsl_content("graph = Graph()") is True
        assert _run_is_dsl_content("from nodetool.dsl import graph") is True
        assert _run_is_dsl_content("{}") is False  # JSON object
        assert _run_is_dsl_content("   ") is False # Empty

        # Test imports check
        assert _run_is_dsl_content("import os\n") is True

    @patch("os.path.isfile")
    @patch("nodetool.cli._run_is_dsl_file")
    @patch("nodetool.cli._run_load_dsl_file")
    def test_parse_workflow_arg_dsl_file(self, mock_load_dsl, mock_is_dsl, mock_isfile):
        """Test parsing a DSL file argument."""
        mock_isfile.return_value = True
        mock_is_dsl.return_value = True
        mock_graph = Graph(nodes=[], edges=[])
        mock_load_dsl.return_value = mock_graph

        req = _run_parse_workflow_arg("workflow.py", "user1", "token1")

        assert isinstance(req, RunJobRequest)
        assert req.user_id == "user1"
        assert req.auth_token == "token1"
        assert req.graph == mock_graph
        mock_load_dsl.assert_called_with("workflow.py")

    @patch("os.path.isfile")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.load")
    def test_parse_workflow_arg_json_file(self, mock_json_load, mock_open, mock_isfile):
        """Test parsing a JSON file argument."""
        mock_isfile.return_value = True
        mock_open.return_value.__enter__.return_value = MagicMock()

        # Case 1: Full RunJobRequest JSON
        mock_json_load.return_value = {
            "workflow_id": "test_wf",
            "user_id": "json_user",
            "auth_token": "json_token"
        }
        req = _run_parse_workflow_arg("request.json", "cli_user", "cli_token")
        assert req.workflow_id == "test_wf"
        assert req.user_id == "json_user" # Should use value from JSON if present

        # Case 2: Workflow definition (graph only)
        mock_json_load.return_value = {
            "graph": {"nodes": [], "edges": []}
        }
        req = _run_parse_workflow_arg("graph.json", "cli_user", "cli_token")
        assert req.user_id == "cli_user"
        assert req.auth_token == "cli_token"
        assert isinstance(req.graph, Graph)

    def test_parse_workflow_arg_inline_json(self):
        """Test parsing inline JSON string."""
        json_str = '{"workflow_id": "inline", "params": {"a": 1}}'
        req = _run_parse_workflow_arg(json_str, "user1", "token1")
        assert req.workflow_id == "inline"
        assert req.params == {"a": 1}

    def test_parse_workflow_arg_id(self):
        """Test treating argument as simple workflow ID."""
        req = _run_parse_workflow_arg("my-workflow-id", "user1", "token1")
        assert req.workflow_id == "my-workflow-id"
        assert req.user_id == "user1"
        assert req.auth_token == "token1"
