#!/usr/bin/env python3
"""
Example script demonstrating how to use TaskPlanner and TaskExecutor separately.

This script shows how to:
1. Create a plan using TaskPlanner
2. Modify or inspect the plan if needed
3. Execute the plan using TaskExecutor
4. Save and load plans between sessions

This approach allows for greater flexibility in automated workflows
where you might want to review or modify plans before execution.
"""

import asyncio
import json
import os
from typing import List

from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk, get_provider
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools.browser import BrowserTool
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.common.settings import get_system_data_path
from nodetool.metadata.types import TaskPlan, ToolCall

# Import a provider (this example uses Anthropic)
from nodetool.chat.providers.anthropic import AnthropicProvider
from nodetool.metadata.types import FunctionModel, Provider

# Import some tools
from nodetool.chat.tools import (
    GoogleSearchTool,
)


async def create_plan(
    provider: ChatProvider,
    model: str,
    objective: str,
    workspace_dir: str,
    tools: List,
    agents: List[Agent],
) -> TaskPlan:
    """
    Create a task plan for a given problem.

    Args:
        provider: The chat provider to use
        model: The model to use
        objective: The objective to solve
        tools: List of available tools

    Returns:
        TaskPlan: The generated plan
    """
    planner = TaskPlanner(
        provider,
        model,
        objective=objective,
        workspace_dir=workspace_dir,
        tools=tools,
        agents=agents,
    )
    async for chunk in planner.create_plan():
        print(chunk.content, end="")

    assert (
        planner.task_plan is not None
    ), "Task plan was not created, check the objective and tools"

    print("\n=== Generated Plan ===")
    print(planner.task_plan.to_markdown())

    return planner.task_plan


def save_plan(task_plan: TaskPlan, filename: str) -> None:
    """
    Save a task list to a JSON file.

    Args:
        task_plan: The task list to save
        filename: The file to save to
    """
    # Convert to dict and save as JSON
    with open(filename, "w") as f:
        json.dump(task_plan.model_dump(), f, indent=2)

    print(f"Plan saved to {filename}")


def load_plan(filename: str) -> TaskPlan:
    """
    Load a task list from a JSON file.

    Args:
        filename: The file to load from

    Returns:
        TaskPlan: The loaded task list
    """
    with open(filename, "r") as f:
        data = json.load(f)

    task_plan = TaskPlan.model_validate(data)
    print(f"Plan loaded from {filename}")

    return task_plan


async def main():
    workspace_manager = WorkspaceManager()
    workspace_dir = workspace_manager.get_current_directory()

    provider = get_provider(Provider.OpenAI)
    planning_model = "gpt-4o"
    # provider = get_provider(Provider.Anthropic)
    # planning_model = "claude-3-7-sonnet-20250219"

    # Define tools
    tools = [
        GoogleSearchTool(str(workspace_dir)),
        BrowserTool(str(workspace_dir)),
    ]

    # Define the problem
    problem = "Research the latest trends in AI and machine learning."

    agent = Agent(
        name="Research Agent",
        objective="Research the latest trends in AI and machine learning.",
        description="A research agent that retrieves information from the web and saves it to files.",
        provider=provider,
        model=planning_model,
        workspace_dir=str(workspace_dir),
        tools=tools,
    )

    # Step 1: Create a plan
    task_plan = await create_plan(
        provider, planning_model, problem, str(workspace_dir), tools, agents=[agent]
    )

    # Step 2: Save the plan (optional)
    save_plan(task_plan, os.path.join(workspace_dir, "tasks.json"))

    # Step 3: Load the plan (optional, demonstrating how plans can be saved/loaded)
    task_plan = load_plan(os.path.join(workspace_dir, "tasks.json"))

    print("\n=== Loaded Plan ===")
    print(task_plan.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
