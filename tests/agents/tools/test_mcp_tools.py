"""Tests for MCP agent tools."""

import pytest
from nodetool.agents.tools.mcp_tools import (
    get_all_mcp_tools,
    ListWorkflowsTool,
    GetWorkflowTool,
    RunWorkflowTool,
    CreateWorkflowTool,
    ListNodesTool,
    SearchNodesTool,
    GetNodeInfoTool,
    ListJobsTool,
    GetJobTool,
    ListAssetsTool,
    ListModelsTool,
)
from nodetool.agents.tools.base import Tool


class TestMCPToolsBasics:
    """Test basic MCP tool functionality."""

    def test_get_all_mcp_tools_returns_list(self):
        """Test that get_all_mcp_tools returns a list of tools."""
        tools = get_all_mcp_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_all_tools_are_tool_instances(self):
        """Test that all returned tools are Tool instances."""
        tools = get_all_mcp_tools()
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_all_tools_have_required_attributes(self):
        """Test that all tools have required name, description, and input_schema."""
        tools = get_all_mcp_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "input_schema")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0
            assert isinstance(tool.input_schema, dict)

    def test_all_tools_have_process_method(self):
        """Test that all tools have a process method."""
        tools = get_all_mcp_tools()
        for tool in tools:
            assert hasattr(tool, "process")
            assert callable(tool.process)

    def test_all_tools_have_user_message_method(self):
        """Test that all tools have a user_message method."""
        tools = get_all_mcp_tools()
        for tool in tools:
            assert hasattr(tool, "user_message")
            assert callable(tool.user_message)


class TestWorkflowTools:
    """Test workflow-related MCP tools."""

    def test_list_workflows_tool_definition(self):
        """Test ListWorkflowsTool has correct definition."""
        tool = ListWorkflowsTool()
        assert tool.name == "list_workflows"
        assert "workflow_type" in tool.input_schema.get("properties", {})
        assert "query" in tool.input_schema.get("properties", {})

    def test_get_workflow_tool_definition(self):
        """Test GetWorkflowTool has correct definition."""
        tool = GetWorkflowTool()
        assert tool.name == "get_workflow"
        assert "workflow_id" in tool.input_schema.get("properties", {})
        assert "workflow_id" in tool.input_schema.get("required", [])

    def test_run_workflow_tool_definition(self):
        """Test RunWorkflowTool has correct definition."""
        tool = RunWorkflowTool()
        assert tool.name == "run_workflow"
        assert "workflow_id" in tool.input_schema.get("properties", {})
        assert "params" in tool.input_schema.get("properties", {})

    def test_create_workflow_tool_definition(self):
        """Test CreateWorkflowTool has correct definition."""
        tool = CreateWorkflowTool()
        assert tool.name == "create_workflow"
        assert "name" in tool.input_schema.get("properties", {})
        assert "graph" in tool.input_schema.get("properties", {})
        assert "name" in tool.input_schema.get("required", [])
        assert "graph" in tool.input_schema.get("required", [])


class TestNodeTools:
    """Test node-related MCP tools."""

    def test_list_nodes_tool_definition(self):
        """Test ListNodesTool has correct definition."""
        tool = ListNodesTool()
        assert tool.name == "list_nodes"
        assert "namespace" in tool.input_schema.get("properties", {})
        assert "limit" in tool.input_schema.get("properties", {})

    def test_search_nodes_tool_definition(self):
        """Test SearchNodesTool has correct definition."""
        tool = SearchNodesTool()
        assert tool.name == "search_nodes"
        assert "query" in tool.input_schema.get("properties", {})
        assert "query" in tool.input_schema.get("required", [])

    def test_get_node_info_tool_definition(self):
        """Test GetNodeInfoTool has correct definition."""
        tool = GetNodeInfoTool()
        assert tool.name == "get_node_info"
        assert "node_type" in tool.input_schema.get("properties", {})
        assert "node_type" in tool.input_schema.get("required", [])


class TestJobTools:
    """Test job-related MCP tools."""

    def test_list_jobs_tool_definition(self):
        """Test ListJobsTool has correct definition."""
        tool = ListJobsTool()
        assert tool.name == "list_jobs"
        assert "workflow_id" in tool.input_schema.get("properties", {})

    def test_get_job_tool_definition(self):
        """Test GetJobTool has correct definition."""
        tool = GetJobTool()
        assert tool.name == "get_job"
        assert "job_id" in tool.input_schema.get("properties", {})
        assert "job_id" in tool.input_schema.get("required", [])


class TestOtherTools:
    """Test other MCP tools."""

    def test_list_assets_tool_definition(self):
        """Test ListAssetsTool has correct definition."""
        tool = ListAssetsTool()
        assert tool.name == "list_assets"
        assert "source" in tool.input_schema.get("properties", {})

    def test_list_models_tool_definition(self):
        """Test ListModelsTool has correct definition."""
        tool = ListModelsTool()
        assert tool.name == "list_models"
        assert "provider" in tool.input_schema.get("properties", {})


class TestToolParam:
    """Test tool_param generation for LLM function calling."""

    def test_tool_param_structure(self):
        """Test that tool_param returns correct structure for LLM."""
        tool = ListWorkflowsTool()
        param = tool.tool_param()

        assert "type" in param
        assert param["type"] == "function"
        assert "function" in param
        assert "name" in param["function"]
        assert "description" in param["function"]
        assert "parameters" in param["function"]
        assert param["function"]["name"] == "list_workflows"


class TestUserMessage:
    """Test user_message generation."""

    def test_list_workflows_user_message(self):
        """Test ListWorkflowsTool user_message."""
        tool = ListWorkflowsTool()
        msg = tool.user_message({"workflow_type": "example", "query": "test"})
        assert "example" in msg
        assert "test" in msg

    def test_run_workflow_user_message(self):
        """Test RunWorkflowTool user_message."""
        tool = RunWorkflowTool()
        msg = tool.user_message({"workflow_id": "abc123"})
        assert "abc123" in msg
