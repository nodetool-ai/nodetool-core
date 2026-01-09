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

from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleGroundedSearchTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

SUMMARIZER_SYSTEM_PROMPT = """You are a specialized Summarization Agent for AI industry intelligence. Your role is to:
"""


async def run_google_grounded_agent():
    context = ProcessingContext()

    provider = await get_provider(Provider.HuggingFaceCerebras)
    model = "openai/gpt-oss-120b"

    retrieval_tools = [
        GoogleGroundedSearchTool(),
        BrowserTool(),
    ]

    agent = Agent(
        name="Research Agent",
        objective="""
        1. Use google to identify a list of recipes for chicken wings
        2. Browse to the recipe websites and extract the ingredients and instructions for each recipe
        Do not use remote browser.
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
        display_manager=AgentConsole(),
        output_schema={
            "type": "object",
            "properties": {
                "recipes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "image_url": {"type": "string"},
                            "ingredients": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "quantity": {"type": "string"},
                                        "unit": {"type": "string"},
                                    },
                                },
                            },
                            "instructions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    )
    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print(f"Results: {agent.results}")


async def main():
    async with ResourceScope():
        await run_google_grounded_agent()


if __name__ == "__main__":
    asyncio.run(main())
