#!/usr/bin/env python3
"""
Test script for the Nodetool Agent class using web search tools.

This script demonstrates the use of a single Agent instance configured with
web browsing capabilities (BrowserTool) plus GoogleSearch for finding candidate URLs.

This example shows how to:
1. Set up a single agent with specific tools and an objective.
2. Define an output schema for the desired results.
3. Execute the agent and process its streaming output.
4. Retrieve structured data (chicken wing recipes) from the web based on the objective.
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools import GoogleSearchTool, BrowserTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, PlanningUpdate
from nodetool.runtime.resources import ResourceScope

import dotenv

dotenv.load_dotenv()


async def test_google_agent(provider: BaseProvider, model: str):
    context = ProcessingContext()

    agent = Agent(
        name="Research Agent",
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        objective="""
        1. Identify a list of chicken wing recipe websites during planning phase
        2. Crawl one website per subtask
        3. Extract the ingredients and instructions for each recipe
        4. Return the results in the format specified by the output_schema
        Do not use remote browser.
        """,
        provider=provider,
        model=model,
        tools=[BrowserTool(), GoogleSearchTool()],
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
        await test_google_agent(
            provider=await get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
        )


if __name__ == "__main__":
    asyncio.run(main())
