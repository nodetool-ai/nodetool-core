#!/usr/bin/env python3
"""
Script for an Instagram trends agent using BrowserTool with TaskPlanner.

This script creates an Instagram agent with the BrowserTool and GoogleSearchTool
and uses a TaskPlanner to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent to collect information about trending content on Instagram.
"""

import asyncio
from nodetool.chat.agent import Agent
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()

    # 2. Initialize provider and model
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"
    # Alternatively, you can use Anthropic:
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(workspace_dir=str(workspace_dir)),
        GoogleSearchTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create Instagram trends collector agent
    agent = Agent(
        name="Instagram Trends Collector",
        provider=provider,
        model=model,
        tools=tools,
        objective="""
        Collect current trends, hashtags, and viral content from Instagram
        You can use the BrowserTool to search for trends on Instagram.
        You can also use the GoogleSearchTool to search for trends on Google.
        Create well-structured reports with clear categorization of trends by topic, audience demographics, and content formats.
        Include actionable insights that could be valuable for content creators or marketers.
        Use markdown formatting to format your reports and include images, videos, and other media.
        """,
    )

    processing_context = ProcessingContext()

    # 6. Execute each task in the plan
    async for item in agent.execute(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
