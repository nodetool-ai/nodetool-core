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

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
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

    # 4. Create Twitter/X trends collector agent
    trends_agent = Agent(
        name="Twitter/X Trends Collector",
        objective="Collect current trends, hashtags, and viral content from Twitter/X.",
        description="An agent that identifies and collects information about trending topics on Twitter/X.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=15,
        max_subtask_iterations=4,
    )

    analyzer_agent = Agent(
        name="Twitter/X Trends Analyzer",
        objective="Analyze trends collected from Twitter/X and create comprehensive reports.",
        description="An agent that analyzes Twitter/X trends data and produces insightful summaries.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[],
        system_prompt="""
        You are a social media trends analyst specializing in Twitter/X.
        Your expertise is in identifying patterns, emerging trends, and explaining why certain content becomes popular.
        Create well-structured reports with clear categorization of trends by topic, audience demographics, and content formats.
        Include actionable insights that could be valuable for content creators, marketers, or businesses.
        Pay special attention to viral tweets, influential accounts, and trending hashtags.
        """,
        max_steps=12,
        max_subtask_iterations=3,
    )

    # 5. Create and use the TaskPlanner
    objective = """
        Research and identify current trending topics, hashtags, and viral content on Twitter/X.
        Focus on identifying:
        1. Popular hashtags across different categories (news, politics, entertainment, technology, etc.)
        2. Trending tweet formats, memes, and conversation styles
        3. Emerging influential accounts and content creators gaining popularity
        4. Brand campaigns and marketing strategies that are gaining traction
        5. Current events and news topics driving conversations
        6. New Twitter/X features being widely adopted by users
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

    print(f"\nCreating task plan for objective: {objective}")

    # Generate the task plan
    async for item in task_planner.create_plan():
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # Get the generated task plan
    task_plan = task_planner.task_plan
    if not task_plan:
        print("\nFailed to create task plan")
        return

    print(f"\n\nGenerated task plan: {task_plan.title}")
    processing_context = ProcessingContext()

    # 6. Execute each task in the plan
    for i, task in enumerate(task_plan.tasks):
        print(f"\nExecuting task {i+1}/{len(task_plan.tasks)}: {task.title}")

        # Select the appropriate agent based on the task
        agent = (
            analyzer_agent
            if "analyze" in task.title.lower() or "report" in task.title.lower()
            else trends_agent
        )
        print(f"Using agent: {agent.name}")

        # Execute the task
        async for item in agent.execute_task(task, processing_context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)

    # 7. Print result summary
    print("\n\nTask execution completed.")
    trends_results = trends_agent.get_results()
    analyzer_results = analyzer_agent.get_results()

    print(f"\nTrends Collection Results: {json.dumps(trends_results, indent=2)}")
    print(f"\nAnalysis Results: {json.dumps(analyzer_results, indent=2)}")
    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
