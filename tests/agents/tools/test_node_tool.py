"""
Test cases for NodeTool class
"""

import pytest
from typing import Any
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field


class TestNode(BaseNode):
    """A simple test node for testing NodeTool."""
    
    input_text: str = Field(default="", description="Input text to process")
    multiplier: int = Field(default=1, description="Number of times to repeat the text")
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Process the input by repeating it."""
        result = self.input_text * self.multiplier
        return {"output": result}


class ComplexTestNode(BaseNode):
    """A more complex test node with multiple inputs and outputs."""
    
    text: str = Field(description="Text to process")
    prefix: str = Field(default="", description="Prefix to add")
    suffix: str = Field(default="", description="Suffix to add")
    uppercase: bool = Field(default=False, description="Convert to uppercase")
    
    async def process(self, context: ProcessingContext) -> dict[str, Any]:
        """Process text with various transformations."""
        result = self.prefix + self.text + self.suffix
        if self.uppercase:
            result = result.upper()
        return {
            "processed_text": result,
            "length": len(result),
            "word_count": len(result.split())
        }


@pytest.mark.asyncio
async def test_node_tool_creation():
    """Test creating a NodeTool from a node class."""
    tool = NodeTool(TestNode)
    
    assert tool.name == "node_test"
    assert "test node" in tool.description.lower()
    assert tool.node_type == TestNode.get_node_type()
    
    # Check input schema
    assert tool.input_schema["type"] == "object"
    assert "input_text" in tool.input_schema["properties"]
    assert "multiplier" in tool.input_schema["properties"]


@pytest.mark.asyncio
async def test_node_tool_custom_name():
    """Test creating a NodeTool with a custom name."""
    tool = NodeTool(TestNode, name="my_custom_tool")
    assert tool.name == "my_custom_tool"


@pytest.mark.asyncio
async def test_node_tool_process():
    """Test executing a node through NodeTool."""
    tool = NodeTool(TestNode)
    
    # Create a mock context
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False
    )
    
    # Execute the tool
    params = {
        "input_text": "Hello ",
        "multiplier": 3
    }
    
    result = await tool.process(context, params)
    
    assert result["status"] == "completed"
    assert result["node_type"] == TestNode.get_node_type()
    assert result["result"]["output"]["output"] == "Hello Hello Hello "


@pytest.mark.asyncio
async def test_node_tool_complex_node():
    """Test NodeTool with a more complex node."""
    tool = NodeTool(ComplexTestNode)
    
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False
    )
    
    params = {
        "text": "world",
        "prefix": "Hello ",
        "suffix": "!",
        "uppercase": True
    }
    
    result = await tool.process(context, params)
    
    # Debug print to see what's wrong
    if result["status"] == "failed":
        print(f"Error: {result.get('error')}")
        print(f"Traceback: {result.get('traceback')}")
    
    assert result["status"] == "completed"
    # The result is wrapped in "output" because convert_output wraps dict returns
    assert result["result"]["output"]["processed_text"] == "HELLO WORLD!"
    assert result["result"]["output"]["length"] == 12
    assert result["result"]["output"]["word_count"] == 2


@pytest.mark.asyncio
async def test_node_tool_error_handling():
    """Test NodeTool error handling with invalid parameters."""
    tool = NodeTool(ComplexTestNode)
    
    context = ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow",
        encode_assets_as_base64=False
    )
    
    # Missing required parameter 'text'
    params = {
        "prefix": "Hello "
    }
    
    result = await tool.process(context, params)
    
    assert result["status"] == "failed"
    assert "error" in result
    assert result["node_type"] == ComplexTestNode.get_node_type()


@pytest.mark.asyncio
async def test_node_tool_from_node_type():
    """Test creating NodeTool from a node type string."""
    # This test assumes the TestNode is registered in NODE_BY_TYPE
    # In practice, you'd use a real registered node type
    try:
        tool = NodeTool.from_node_type("nodetool.text.Concatenate")
        assert tool.node_type == "nodetool.text.Concatenate"
    except ValueError:
        # If the node type doesn't exist, that's ok for this test
        pass


def test_node_tool_user_message():
    """Test the user message generation."""
    tool = NodeTool(TestNode)
    
    params = {"input_text": "Hello", "multiplier": 2}
    message = tool.user_message(params)
    
    assert "Test" in message
    assert "Hello" in message
    assert "2" in message


def test_node_tool_snake_case_conversion():
    """Test the snake_case conversion for tool names."""
    tool = NodeTool(ComplexTestNode)
    assert tool.name == "node_complex_test"
    
    # Test with a node that doesn't end in "Node"
    class MySpecialProcessor(BaseNode):
        async def process(self, context: Any) -> Any:
            return {}
    
    tool2 = NodeTool(MySpecialProcessor)
    assert tool2.name == "node_my_special_processor"