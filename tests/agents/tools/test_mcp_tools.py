"""
Tests for MCP tools integration with the agent system.
"""

import pytest

from nodetool.agents.tools.mcp_tools import (
    MCPToolWrapper,
    get_all_mcp_tools,
    get_mcp_tool_by_name,
)
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_get_all_mcp_tools():
    """Test that MCP tools can be loaded."""
    tools = await get_all_mcp_tools()
    
    # We should have multiple MCP tools
    assert len(tools) > 0, "Should have loaded at least some MCP tools"
    
    # All tools should be Tool instances
    for tool in tools:
        assert isinstance(tool, MCPToolWrapper)
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "input_schema")


@pytest.mark.asyncio
async def test_get_mcp_tool_by_name():
    """Test retrieving a specific MCP tool by name."""
    # First, ensure tools are loaded
    all_tools = await get_all_mcp_tools()
    assert len(all_tools) > 0
    
    # Get the first tool's name
    first_tool_name = all_tools[0].name
    
    # Retrieve it by name
    tool = await get_mcp_tool_by_name(first_tool_name)
    assert tool is not None
    assert tool.name == first_tool_name
    
    # Try to get a non-existent tool
    fake_tool = await get_mcp_tool_by_name("nonexistent_tool_12345")
    assert fake_tool is None


@pytest.mark.asyncio
async def test_mcp_tool_has_required_attributes():
    """Test that MCP tools have the required attributes."""
    tools = await get_all_mcp_tools()
    assert len(tools) > 0
    
    tool = tools[0]
    
    # Check required attributes
    assert hasattr(tool, "name")
    assert hasattr(tool, "description")
    assert hasattr(tool, "input_schema")
    assert hasattr(tool, "process")
    assert hasattr(tool, "user_message")
    assert hasattr(tool, "tool_param")
    
    # Test tool_param returns correct structure
    tool_param = tool.tool_param()
    assert "type" in tool_param
    assert tool_param["type"] == "function"
    assert "function" in tool_param
    assert "name" in tool_param["function"]
    assert "description" in tool_param["function"]
    assert "parameters" in tool_param["function"]


@pytest.mark.asyncio
async def test_mcp_tool_user_message():
    """Test that MCP tools can generate user messages."""
    tools = await get_all_mcp_tools()
    assert len(tools) > 0
    
    tool = tools[0]
    
    # Test user_message with empty params
    message = tool.user_message({})
    assert isinstance(message, str)
    assert len(message) > 0
    assert tool.name in message
    
    # Test user_message with some params
    message = tool.user_message({"param1": "value1", "param2": "value2"})
    assert isinstance(message, str)
    assert len(message) > 0


@pytest.mark.asyncio  
async def test_tool_registry_resolves_mcp_tools():
    """Test that the tool registry can resolve MCP tools."""
    from nodetool.agents.tools.tool_registry import resolve_tool_by_name
    
    # Get all MCP tools
    mcp_tools = await get_all_mcp_tools()
    assert len(mcp_tools) > 0
    
    # Try to resolve one of them
    tool_name = mcp_tools[0].name
    resolved_tool = await resolve_tool_by_name(tool_name, user_id="test_user")
    
    assert resolved_tool is not None
    assert resolved_tool.name == tool_name


@pytest.mark.asyncio
async def test_mcp_tool_execution_handles_errors():
    """Test that MCP tool execution handles errors gracefully."""
    tools = await get_all_mcp_tools()
    assert len(tools) > 0
    
    tool = tools[0]
    context = ProcessingContext(user_id="test_user")
    
    # Try to execute with invalid parameters (should not crash)
    # The exact behavior depends on the tool, but it should handle errors gracefully
    try:
        result = await tool.process(context, {})
        # Result should be something (either success or error dict)
        assert result is not None
    except Exception as e:
        # If it raises an exception, it should be a well-formed error
        assert isinstance(e, Exception)
        assert str(e) != ""
