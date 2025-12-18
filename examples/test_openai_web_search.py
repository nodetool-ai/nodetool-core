#!/usr/bin/env python3
"""
Test script for Multi-Agent Coordination using specialized agents.

This script demonstrates the use of MultiAgentCoordinator to orchestrate two specialized agents:
1. A Research Agent: Responsible for retrieving information from the web
2. A Summarization Agent: Processes and summarizes the retrieved information

This example shows how to:
1. Set up multiple specialized agents with different capabilities
2. Define their roles and coordination in a task plan
3. Have the MultiAgentCoordinator manage task dependencies and execution flow
4. Generate comprehensive research with information retrieval and summarization
"""

import asyncio
import json

from nodetool.agents.agent import Agent
from nodetool.agents.tools.openai_tools import OpenAIWebSearchTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def run_web_search_example():
    # Configure test parameters
    context = ProcessingContext()

    # Initialize chat provider
    provider = await get_provider(Provider.HuggingFaceCerebras)
    model = "openai/gpt-oss-120b"

    # Create test tools
    tools = [
        OpenAIWebSearchTool(),
    ]

    agent = Agent(
        name="Research Agent",
        objective="Research and summarize the competitive landscape of AI code tools in 2025. Use web search to identify AI code assistant tools and summarize findings",
        provider=provider,
        model=model,
        tools=tools,
        display_manager=AgentConsole(),
        output_schema={
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "features": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    )

    # Execute the agent
    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    
    # Check if results are available
    if agent.results:
        print("\n[bold green]Agent Execution Successful![/bold green]")
        print("\n[bold]Output Results:[/bold]")
        print(json.dumps(agent.results, indent=2))
    else:
        print("\n[bold red]No results returned![/bold red]")


async def main():
    async with ResourceScope():
        await run_web_search_example()


if __name__ == "__main__":
    asyncio.run(main())
