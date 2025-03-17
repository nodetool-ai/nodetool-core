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
from typing import List

from nodetool.chat.cot_agent import TaskPlanner, TaskExecutor
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.common.settings import get_system_data_path
from nodetool.metadata.types import TaskPlan, ToolCall

# Import a provider (this example uses Anthropic)
from nodetool.chat.providers.anthropic import AnthropicProvider
from nodetool.metadata.types import FunctionModel, Provider

# Import some tools
from nodetool.chat.tools import (
    GoogleSearchTool,
    CreateWorkspaceFileTool,
    ReadWorkspaceFileTool,
    UpdateWorkspaceFileTool,
    ListWorkspaceContentsTool,
)


async def create_plan(
    provider: ChatProvider, model: FunctionModel, objective: str, tools: List
) -> TaskPlan:
    """
    Create a task plan for a given problem.

    Args:
        provider: The chat provider to use
        model: The model to use
        problem: The problem to solve
        tools: List of available tools

    Returns:
        TaskPlan: The generated plan
    """
    planner = TaskPlanner(provider, model, objective=objective)
    task_plan = await planner.create_plan()

    print("\n=== Generated Plan ===")
    print(task_plan.to_markdown())

    return task_plan


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


def modify_plan(task_plan: TaskPlan) -> TaskPlan:
    """
    Example of how to modify a plan before execution.

    Args:
        task_plan: The original task list

    Returns:
        TaskPlan: The modified task list
    """
    # Example: Add a new task
    if len(task_plan.tasks) > 0:
        # Add a verification subtask to the first task
        first_task = task_plan.tasks[0]

        # Create a verification subtask
        verify_id = f"verify_{first_task.subtasks[-1].id}"
        first_task.add_subtask(
            subtask_id=verify_id,
            content=f"Verify the results of previous steps and summarize progress",
            dependencies=[
                s.id for s in first_task.subtasks
            ],  # Depends on all previous subtasks
        )

        print("\n=== Modified Plan ===")
        print(task_plan.to_markdown())

    return task_plan


async def execute_plan(
    provider: ChatProvider,
    model: FunctionModel,
    task_plan: TaskPlan,
    workspace_dir: str,
    tools: List,
) -> None:
    """
    Execute a task plan.

    Args:
        provider: The chat provider to use
        model: The model to use
        task_plan: The task list to execute
        problem: The original problem statement
        workspace_dir: Directory for workspace files
        tools: List of available tools
    """
    executor = TaskExecutor(provider, model, workspace_dir, tools, task_plan)

    print("\n=== Executing Plan ===")

    async for result in executor.execute_tasks():
        if isinstance(result, Chunk):
            print(result.content, end="")
        elif isinstance(result, ToolCall):
            print(f"\n[Tool Call: {result.name}]\n")


async def main():
    workspace_root = get_system_data_path("workspaces")
    workspace_root.mkdir(exist_ok=True)
    workspace_name = f"workspace-{int(asyncio.get_event_loop().time())}"
    workspace_dir = workspace_root / workspace_name
    provider = AnthropicProvider()

    model = FunctionModel(
        provider=Provider.Anthropic, name="claude-3-7-sonnet-20250219"
    )

    # Define tools
    tools = [
        GoogleSearchTool(),
        CreateWorkspaceFileTool(str(workspace_dir)),
        ReadWorkspaceFileTool(str(workspace_dir)),
        UpdateWorkspaceFileTool(str(workspace_dir)),
        ListWorkspaceContentsTool(str(workspace_dir)),
    ]

    # Define the problem
    problem = "Create a simple Python weather app that fetches current weather data for a given city using the OpenWeatherMap API. The app should have a README and a requirements.txt file."

    # Step 1: Create a plan
    task_plan = await create_plan(provider, model, problem, tools)

    # Step 2: Save the plan (optional)
    save_plan(task_plan, "weather_app_plan.json")

    # Step 3: Load the plan (optional, demonstrating how plans can be saved/loaded)
    # task_plan = load_plan("weather_app_plan.json")

    # Step 4: Modify the plan (optional)
    # task_plan = modify_plan(task_plan)

    # Step 5: Execute the plan
    await execute_plan(provider, model, task_plan, str(workspace_dir), tools)


if __name__ == "__main__":
    asyncio.run(main())
