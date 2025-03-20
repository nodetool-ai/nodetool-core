#!/usr/bin/env python3
"""
Script for an Instagram trends agent using BrowserTool with TaskPlanner.

This script creates an Instagram agent with the BrowserTool and GoogleSearchTool
and uses a TaskPlanner to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent to collect information about trending content on Instagram.
"""

import asyncio
import os
import json
from pathlib import Path

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import BrowserTool, GoogleSearchTool
from nodetool.metadata.types import Provider, Task
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Created workspace at: {workspace_dir}")

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
    trends_agent = Agent(
        name="Instagram Trends Collector",
        objective="Collect current trends, hashtags, and viral content from Instagram.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        description="""
        You are a social media trends collector specializing in Instagram.
        Your expertise is in identifying trending topics, hashtags, and viral content on Instagram.
        You can use the BrowserTool to search for trends on Instagram.
        You can also use the GoogleSearchTool to search for trends on Google.
        """,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=15,
        max_subtask_iterations=4,
    )

    analyzer_agent = Agent(
        name="Instagram Trends Analyzer",
        objective="Analyze trends collected from Instagram and create comprehensive reports.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[],
        description="""
        You are a social media trends analyst specializing in Instagram.
        Your expertise is in identifying patterns, emerging trends, and explaining why certain content becomes popular.
        Create well-structured reports with clear categorization of trends by topic, audience demographics, and content formats.
        Include actionable insights that could be valuable for content creators or marketers.
        """,
        max_steps=12,
        max_subtask_iterations=3,
    )

    # 5. Create and use the TaskPlanner
    objective = """
        Research and identify current trending topics, hashtags, and viral content on Instagram.
        Focus on identifying:
        1. Popular hashtags across different categories (fashion, food, technology, etc.)
        2. Trending video formats, styles, and transitions
        3. Emerging influencers and content creators gaining popularity
        4. Brand campaigns that are gaining traction
        5. New Instagram features being widely adopted by users
        Save all retrieved information as organized files in the /workspace directory.
    """

    task_planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=str(workspace_dir),
        tools=tools,
        agents=[trends_agent, analyzer_agent],
        max_research_iterations=3,
    )

    coordinator = MultiAgentCoordinator(
        agents=[trends_agent, analyzer_agent],
        workspace_dir=str(workspace_dir),
        provider=provider,
        planner=task_planner,
    )

    processing_context = ProcessingContext()

    # 6. Execute each task in the plan
    async for item in coordinator.solve_problem(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
