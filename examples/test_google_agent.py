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
import datetime
from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.browser import GoogleSearchTool, BrowserTool
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
        BrowserTool(use_readability=True),
    ]

    agent = Agent(
        name="Research Agent",
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
    asyncio.run(
        test_google_agent(provider=get_provider(Provider.Ollama), model="qwen3:14b")
    )
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
    # asyncio.run(
    #     test_google_agent(provider=get_provider(Provider.OpenAI), model="gpt-4o-mini")
    # )
    # asyncio.run(
    #     test_google_agent(
    #         provider=get_provider(Provider.Ollama),
    #         model="gemma3:12b",
    #     )
    # )
