#!/usr/bin/env python3
"""
Test script for an Email Retrieval agent using email tools with TaskPlanner.

This script creates an Email Retrieval agent with email tools and uses a TaskPlanner
to automatically generate and execute a task plan based on the objective.
It demonstrates how to set up an agent for email processing and summarization.
"""

import asyncio
import os
from pathlib import Path
from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.multi_agent import MultiAgentCoordinator
from nodetool.chat.providers import get_provider, Chunk
from nodetool.metadata.types import Provider, Task
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.chat.task_planner import TaskPlanner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools.email import SearchEmailTool


async def main():
    # 1. Set up workspace directory
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()
    print(f"Created workspace at: {workspace_dir}")

    # 2. Initialize provider and model
    # provider = get_provider(Provider.Anthropic)
    # model = "claude-3-7-sonnet-20250219"
    provider = get_provider(Provider.OpenAI)
    model = "gpt-4o-mini"

    # 3. Set up email tools
    email_tools = [
        SearchEmailTool(workspace_dir=str(workspace_dir)),
    ]

    # 4. Create Email Retrieval agent
    retrieval_agent = Agent(
        name="Email Retriever",
        objective="Search for emails from AINews in subject from last 24 hours and summarize them.",
        description="""
        You are a world-class email agent
        """,
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=email_tools,
        max_steps=10,
        max_subtask_iterations=10,
    )

    # 6. Create and use the TaskPlanner
    objective = """
    Search for AINews in subject from last 7 days and summarize the top news.
    """

    task_planner = TaskPlanner(
        provider=provider,
        model=model,
        objective=objective,
        workspace_dir=str(workspace_dir),
        tools=[],
        agents=[retrieval_agent],
        max_research_iterations=1,
    )
    # 7. Create a multi-agent coordinator
    coordinator = MultiAgentCoordinator(
        provider=provider,
        planner=task_planner,
        workspace_dir=str(workspace_dir),
        agents=[retrieval_agent],
        max_steps=30,
    )

    # 8. Solve the problem using the multi-agent coordinator
    processing_context = ProcessingContext()
    async for item in coordinator.solve_problem(processing_context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {workspace_dir}")


if __name__ == "__main__":
    asyncio.run(main())
