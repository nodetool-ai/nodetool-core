#!/usr/bin/env python3
"""
Simple test script for a retrieval agent using browser tools.

This script creates a retrieval agent with browser tools and executes a hard-coded
task plan without using the task planner. It demonstrates how to set up an agent
with browser-based retrieval tools and track its execution.
"""

import asyncio
import os
import json
from pathlib import Path

from nodetool.chat.agent import Agent, RETRIEVAL_SYSTEM_PROMPT
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools.browser import GoogleSearchTool, WebFetchTool, BrowserTool
from nodetool.metadata.types import Provider, Task, TaskPlan, SubTask
from nodetool.chat.workspace_manager import WorkspaceManager


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

    # 3. Set up tools for retrieval
    tools = [
        GoogleSearchTool(str(workspace_dir)),
        # WebFetchTool(str(workspace_dir)),
        BrowserTool(str(workspace_dir)),
    ]

    # 4. Create retrieval agent
    agent = Agent(
        name="Research Agent",
        objective="Research information about AI models and save results to files.",
        description="A research agent that retrieves information from the web and saves it to files.",
        provider=provider,
        model=model,
        workspace_dir=str(workspace_dir),
        tools=tools,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        max_steps=10,
        max_subtask_iterations=3,
    )

    # 5. Create a hard-coded task plan with a single task
    task_plan = TaskPlan(
        title="Research AI Models",
        tasks=[
            Task(
                title="Research recent LLM developments",
                agent_name="Research Agent",
                subtasks=[
                    SubTask(
                        id="search_ai_models",
                        content="Search and list the latest AI language models and their capabilities.",
                        task_type="multi_step",
                        max_tool_calls=1,
                        output_type="md",
                        output_file="/workspace/ai_models_search.md",
                        file_dependencies=[],
                    ),
                    SubTask(
                        id="fetch_anthropic_rate_limits",
                        content="Fetch content from https://docs.anthropic.com/en/api/rate-limits.",
                        task_type="multi_step",
                        max_tool_calls=1,
                        output_type="md",
                        output_file="/workspace/anthropic_rate_limits.md",
                        file_dependencies=[],
                    ),
                    SubTask(
                        id="fetch_anthropic_messages_api",
                        content="Fetch content from https://docs.anthropic.com/en/api/messages",
                        task_type="multi_step",
                        max_tool_calls=1,
                        output_type="md",
                        output_file="/workspace/anthropic_messages_api.md",
                        file_dependencies=[],
                    ),
                ],
            )
        ],
    )

    # 6. Execute the task and print results
    print(f"\nExecuting task plan: {task_plan.title}")

    # Get the first task (we only have one in this example)
    task = task_plan.tasks[0]
    print(f"\nExecuting task: {task.title}")

    # Execute the task
    async for item in agent.execute_task(task_plan, task):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    # 7. Print result summary
    print("\n\nTask execution completed.")
    results = agent.get_results()
    print(f"\nResults: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
