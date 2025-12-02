#!/usr/bin/env python3
"""
Example: Using NodeTool to wrap workflow nodes as agent tools

This example demonstrates how to use the NodeTool class to make
individual workflow nodes available as tools for agents.
"""

import asyncio

from pydantic import Field

from nodetool.agents.agent import Agent
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


# Example 1: Simple text processing node
class TextProcessorNode(BaseNode):
    """Process text with various transformations."""

    text: str = Field(description="Text to process")
    uppercase: bool = Field(default=False, description="Convert to uppercase")
    reverse: bool = Field(default=False, description="Reverse the text")

    async def process(self, context: ProcessingContext) -> dict[str, str]:
        result = self.text

        if self.uppercase:
            result = result.upper()

        if self.reverse:
            result = result[::-1]

        return {"processed": result}


# Example 2: Math operation node
class MathOperationNode(BaseNode):
    """Perform mathematical operations."""

    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(
        default="add", description="Operation: add, subtract, multiply, divide"
    )

    async def process(self, context: ProcessingContext) -> dict[str, float]:
        if self.operation == "add":
            result = self.a + self.b
        elif self.operation == "subtract":
            result = self.a - self.b
        elif self.operation == "multiply":
            result = self.a * self.b
        elif self.operation == "divide":
            if self.b == 0:
                raise ValueError("Division by zero")
            result = self.a / self.b
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        return {"result": result}


async def run_node_tool_example():
    """Demonstrate using NodeTool with agents."""

    # Create a processing context
    context = ProcessingContext(
        user_id="example_user",
        auth_token="example_token",
        workflow_id="example_workflow",
        encode_assets_as_base64=False,
    )

    print("=== NodeTool with Agent Example ===\n")

    # Example 1: Direct usage of NodeTool
    print("1. Direct NodeTool Usage:")
    text_tool = NodeTool(TextProcessorNode)
    print(f"Tool name: {text_tool.name}")
    print(f"Tool description: {text_tool.description}")

    # Execute the text processing tool directly
    result = await text_tool.process(
        context, {"text": "Hello World", "uppercase": True, "reverse": False}
    )
    print(f"Direct result: {result}\n")

    # Example 2: Create tools for the agent
    print("2. Creating Agent with NodeTools:")

    # Set up provider and model
    provider = await get_provider(Provider.HuggingFaceCerebras)
    model = "openai/gpt-oss-120b"

    # Create NodeTools for the agent
    tools = [
        NodeTool(TextProcessorNode),
        NodeTool(MathOperationNode),
    ]

    # Create an agent with custom node tools
    agent = Agent(
        name="Data Processing Agent",
        objective="""
        Demonstrate the use of custom workflow nodes as tools:
        1. Use the text_processor tool to convert 'artificial intelligence' to uppercase
        2. Use the math_calculator tool to calculate 42 * 3.14159
        3. Use the text_processor tool to reverse the text 'Machine Learning'
        4. Use the math_calculator tool to divide 100 by 7
        5. Summarize all the results in a clear format
        """,
        provider=provider,
        model=model,
        tools=tools,
        enable_analysis_phase=True,
    )

    print("\n3. Agent Execution:")
    print("Agent is working on the tasks...\n")

    # Execute the agent
    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\n\nWorkspace: {context.workspace_dir}")
    print("\nThe agent has completed all tasks using the custom NodeTools.")


async def main():
    async with ResourceScope():
        await run_node_tool_example()


if __name__ == "__main__":
    asyncio.run(main())
