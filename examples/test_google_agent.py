#!/usr/bin/env python3
"""
Test script for the Nodetool Agent class using web search tools.

This script demonstrates the use of a single Agent instance configured with
web browsing capabilities (GoogleSearchTool, BrowserTool) to perform a specific task.

This example shows how to:
1. Set up a single agent with specific tools and an objective.
2. Define an output schema for the desired results.
3. Execute the agent and process its streaming output.
4. Retrieve structured data (chicken wing recipes) from the web based on the objective.
"""

import asyncio
import datetime
from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import GoogleSearchTool, BrowserTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

import dotenv

dotenv.load_dotenv()


async def test_google_agent(provider: ChatProvider, model: str):
    context = ProcessingContext()

    retrieval_tools = [
        GoogleSearchTool(),
        BrowserTool(),
    ]

    agent = Agent(
        name="Research Agent",
        docker_image="nodetool",
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        objective="""
        1. Identify a list of chicken wing recipe websites during planning phase
        2. Crawl one website per subtask
        3. Extract the ingredients and instructions for each recipe
        4. Return the results in the format specified by the output_schema
        Do not use remote browser.
        """,
        provider=provider,
        model=model,
        tools=retrieval_tools,
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


if __name__ == "__main__":
    # asyncio.run(
    #     test_google_agent(provider=get_provider(Provider.Ollama), model="qwen3:14b")
    # )
    # asyncio.run(
    #     test_google_agent(
    #         provider=get_provider(Provider.Gemini), model="gemini-2.0-flash"
    #     )
    # )
    # asyncio.run(
    #     test_google_agent(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #     )
    # )
    asyncio.run(
        test_google_agent(provider=get_provider(Provider.OpenAI), model="gpt-4o-mini")
    )
    # asyncio.run(
    #     test_google_agent(
    #         provider=get_provider(Provider.Ollama),
    #         model="gemma3:12b",
    #     )
    # )
