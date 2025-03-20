#!/usr/bin/env python3
"""
Test script for a Reddit agent using RedditScraperTool with TaskPlanner.

This script creates a Reddit agent with the RedditScraperTool and uses a TaskPlanner
to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent with Reddit-based retrieval tools and
use the TaskPlanner for dynamic task generation.
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
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o"

    # 3. Set up Reddit scraper tool
    tools = [
        BrowserTool(workspace_dir=str(workspace_dir)),
        GoogleSearchTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create Reddit agent
    agent = Agent(
        name="Reddit Downloader",
        objective="Download information from Reddit and save results to files.",
        description="A research agent that retrieves information from Reddit and saves it to files.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=10,
        max_subtask_iterations=3,
    )

    writer_agent = Agent(
        name="Reddit Writer",
        objective="Write a summary of the information gathered from Reddit.",
        description="A research agent that writes a summary of the information gathered from Reddit.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=[],
        system_prompt="""
        You are a world-class writer that writes a summary of the information gathered from Reddit.
        You have an interesting writing style that is engaging and informative.
        You are precise and concise, and you write in a way that is easy to understand.
        """,
        max_steps=10,
        max_subtask_iterations=3,
    )

    # 5. Create and use the TaskPlanner
    objective = """
        Research current discussions about AI and large language models on Reddit. 
        Search and gather information about the latest trends, opinions, and developments in the field.
        Save all retrieved information as files in the /workspace directory.
    """

    task_planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=str(workspace_dir),
        tools=tools,
        agents=[agent],
        max_research_iterations=2,
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

        # Execute the task
        async for item in agent.execute_task(task, processing_context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)

    # 7. Print result summary
    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
