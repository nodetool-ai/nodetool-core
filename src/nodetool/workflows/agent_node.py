"""
Agent Node for Nodetool Workflows
=================================

This module provides an AgentNode class that wraps the SubTaskContext,
allowing agents to be executed as nodes within workflows.
"""

import asyncio
from typing import Any, Dict, List, Optional

from nodetool.agents.tools.google_tools import (
    GoogleGroundedSearchTool,
    GoogleImageGenerationTool,
)
from nodetool.agents.tools.math_tools import CalculatorTool, StatisticsTool
from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.agents.agent_executor import AgentExecutor
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import LanguageModel, ToolName
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider as ProviderEnum
from nodetool.common.environment import Environment

from nodetool.agents.tools import (
    GoogleImageGenerationTool,
    GoogleGroundedSearchTool,
    GoogleNewsTool,
    GoogleImagesTool,
    GoogleSearchTool,
    GoogleLensTool,
    GoogleMapsTool,
    GoogleShoppingTool,
    GoogleFinanceTool,
    GoogleJobsTool,
    BrowserTool,
    ChromaHybridSearchTool,
    SearchEmailTool,
    OpenAIImageGenerationTool,
    OpenAITextToSpeechTool,
    StatisticsTool,
    CalculatorTool,
    ConversionTool,
    GeometryTool,
    TrigonometryTool,
)


log = Environment.get_logger()

TOOLS = {
    tool.name: tool
    for tool in [
        GoogleImageGenerationTool,
        GoogleGroundedSearchTool,
        GoogleNewsTool,
        GoogleImagesTool,
        GoogleSearchTool,
        GoogleLensTool,
        GoogleMapsTool,
        GoogleShoppingTool,
        GoogleFinanceTool,
        GoogleJobsTool,
        BrowserTool,
        ChromaHybridSearchTool,
        SearchEmailTool,
        OpenAIImageGenerationTool,
        OpenAITextToSpeechTool,
        StatisticsTool,
        CalculatorTool,
        ConversionTool,
        GeometryTool,
        TrigonometryTool,
    ]
}


def init_tool(tool: ToolName) -> Tool | None:
    if tool.name:
        tool_class = TOOLS.get(tool.name)
        if tool_class:
            return tool_class()
        else:
            raise ValueError(f"Tool {tool.name} not found")
    else:
        return None


class AgentNode(BaseNode):
    """
    Execute an AI agent with dynamic input values.  This node allows you to run agent tasks within workflows. You can add any number of input values by setting dynamic properties. The agent will have access to all provided values and will execute according to the
    given instructions.
    agent, execution, tasks, simple

    Use cases:
    - Simple, focused tasks with a clear objective
    - Tasks that don't require complex planning
    - Quick responses with tool calling capabilities
    """

    # Node properties
    objective: str = Field(
        default="", description="The objective or problem to create a plan for"
    )
    output_type: str = Field(
        default="string",
        description="The expected output type (string, json, markdown, etc.)",
    )
    model: LanguageModel = Field(
        default=LanguageModel(
            provider=ProviderEnum.OpenAI,
            id="gpt-4o-mini",
        ),
        description="The AI model to use",
    )
    tools: List[ToolName] = Field(
        default=[],
        description="The tools to use",
    )
    max_iterations: int = Field(
        default=10, description="Maximum iterations for the agent"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON schema for the output format",
    )

    # Enable dynamic properties for input values
    _is_dynamic: bool = True

    @classmethod
    def get_title(cls) -> str:
        return "Agent"

    @classmethod
    def get_description(cls) -> str:
        return """"""

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """Execute the agent and return its result."""

        # Collect input values from dynamic properties
        input_values = {}
        for name, value in self._dynamic_properties.items():
            if value is not None:  # Skip None values
                input_values[name] = value

        log.debug(f"AgentNode: Collected {len(input_values)} input values")

        tools = [init_tool(tool) for tool in self.tools]
        tools = [tool for tool in tools if tool is not None]

        # Create AgentExecutor with simplified interface
        agent_executor = AgentExecutor(
            objective=self.objective,
            output_type=self.output_type,
            processing_context=context,
            provider=get_provider(self.model.provider),
            model=self.model.id,
            tools=tools,
            input_values=input_values,
            max_iterations=self.max_iterations,
            output_schema=self.output_schema,
        )

        # Execute the agent
        log.debug(f"AgentNode: Starting agent execution")
        async for update in agent_executor.execute():
            print(update)

        # Get the result
        result = agent_executor.get_result()
        metadata = agent_executor.get_metadata()
        log.debug(f"AgentNode: Got result: {result}")

        if result is None:
            raise ValueError("Agent did not produce a result")

        # Return the result with metadata
        return {"output": result, "metadata": metadata}


# Smoke test
if __name__ == "__main__":

    async def test_agent_node():
        """Simple smoke test for AgentNode."""
        from nodetool.workflows.processing_context import ProcessingContext

        # Create processing context
        context = ProcessingContext()

        # Create and configure agent node
        node = AgentNode(
            objective="Analyze the provided data and extract insights",
            output_type="json",
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
            model=LanguageModel(
                provider=ProviderEnum.OpenAI,
                id="gpt-4o-mini",
            ),
            tools=[
                ToolName(name="statistics"),
                ToolName(name="calculator"),
            ],
        )

        # Add input values as dynamic properties
        node._dynamic_properties["customer_data"] = [
            {"name": "Alice", "age": 25, "purchases": 3},
            {"name": "Bob", "age": 32, "purchases": 7},
            {"name": "Charlie", "age": 28, "purchases": 2},
        ]
        node._dynamic_properties["analysis_type"] = "customer_insights"

        try:
            # Execute the node
            print("Executing AgentNode...")
            result = await node.process(context)
            print(f"Result: {result}")

            if "output" in result:
                print(f"Agent output: {result['output']}")
            if "metadata" in result:
                print(f"Agent metadata: {result['metadata']}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    # Run the test
    asyncio.run(test_agent_node())
