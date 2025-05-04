#!/usr/bin/env python3
"""
Script for a Twitter/X trends agent using BrowserTool with TaskPlanner.

This script creates a Twitter/X agent with the BrowserTool and GoogleSearchTool
and uses a TaskPlanner to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent to collect information about trending content on Twitter/X.
"""

import asyncio
import os
import json
from pathlib import Path

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools.browser_tools import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider, Task
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk


async def main():
    context = ProcessingContext()

    # 2. Initialize provider and model
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # Alternatively, you can use Anthropic:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(context.workspace_dir),
        GoogleSearchTool(context.workspace_dir),
    ]

    # 4. Create Twitter/X trends collector agent
    trends_agent = Agent(
        name="Twitter/X Trends Collector",
        objective="""
        Identify viral accounts by browsing https://x.com/explore/tabs/trending.
        Collect the top 10 trending topics and analyze tweets.
        """,
        provider=provider,
        model=model,
        tools=tools,
        output_type="json",
        output_schema={
            "type": "object",
            "properties": {
                "trends": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                        },
                    },
                },
            },
        },
    )

    async for item in trends_agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # 7. Print result summary
    print("\n\nTask execution completed.")
    trends_results = trends_agent.get_results()

    print(f"\nTrends: {json.dumps(trends_results, indent=2)}")
    print(f"\nWorkspace: {context.workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
