#!/usr/bin/env python3
"""
Test script for the Nodetool Agent class using the GoogleNewsTool.

This script demonstrates the use of a single Agent instance configured with
the GoogleNewsTool to perform a news search task.

This example shows how to:
1. Set up a single agent with the GoogleNewsTool and an objective.
2. Define an output schema for the desired news results.
3. Execute the agent and process its streaming output.
4. Retrieve structured news data from Google News based on the objective.
"""

import asyncio
import json

import dotenv

from nodetool.agents.agent import Agent
from nodetool.agents.tools.serp_tools import GoogleNewsTool
from nodetool.metadata.types import Provider
from nodetool.providers import BaseProvider, get_provider
from nodetool.runtime.resources import ResourceScope
from nodetool.ui.console import AgentConsole
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

dotenv.load_dotenv()


async def test_google_news_agent(provider: BaseProvider, model: str):
    """
    Tests the Agent's ability to use GoogleNewsTool.

    Args:
        provider: The chat provider instance.
        model: The model name to use.
    """
    context = ProcessingContext()

    # Only include the GoogleNewsTool
    news_tool = GoogleNewsTool()

    # Define the search parameters within the objective or let the agent decide
    search_keyword = (
        "AI, OpenAI, ChatGPT, Anthropic, Gemini, Agents, LLMs, RAG, Vector Databases"
    )
    search_location = "United States"

    agent = Agent(
        name="Google News Agent",
        objective=f"""
        Search Google News for the keywords '{search_keyword}'.
        Group common news items together.
        Return a list of news items with exciting titles and summaries.
        """,
        provider=provider,
        display_manager=AgentConsole(),
        model=model,
        tools=[news_tool],  # Pass the tool instance
        # Define an output schema based on expected GoogleNewsTool results
        output_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "link": {"type": "string"},
                            "date": {"type": "string"},
                            "summary": {"type": "string"},
                        },
                        "required": ["title", "link", "date", "summary"],
                    },
                }
            },
            "required": ["results"],
        },
    )

    print(f"Executing Google News Agent with model: {model}")
    print(f"Objective: Search Google News for '{search_keyword}' in {search_location}")

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("=== Agent Execution Complete ===")
    print(f"Workspace: {context.workspace_dir}")

    # Pretty print the results if available
    if agent.results:
        print("=== Agent Results ===")
        # Check if results conform to the expected structure
        if isinstance(agent.results, dict) and "results" in agent.results:
            print(json.dumps(agent.results, indent=2))
            print(
                f"Successfully retrieved {len(agent.results.get('results', []))} news items."
            )
        else:
            print("Results structure might not match the expected schema:")
            print(json.dumps(agent.results, indent=2))
    else:
        print("No results returned by the agent.")


async def main():
    async with ResourceScope():
        await test_google_news_agent(
            provider=await get_provider(Provider.HuggingFaceCerebras),
            model="openai/gpt-oss-120b",
        )


if __name__ == "__main__":
    asyncio.run(main())
