from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools import Tool
from nodetool.metadata.types import Message, Task, TaskPlan, ToolCall


import os
from pathlib import Path
from typing import AsyncGenerator, List, Union

from nodetool.workflows.processing_context import ProcessingContext


class MultiAgentCoordinator:
    """
    🎯 The Mission Control - Coordinates specialized agents to solve complex problems

    This class is the command center that orchestrates the entire agent ecosystem,
    from planning to execution. It:

    1. Plans the approach using TaskPlanner
    2. Assigns tasks to appropriate specialized agents
    3. Tracks overall progress and dependencies
    4. Ensures the full objective is completed efficiently

    Think of it as a project manager who oversees multiple teams with different
    specialties, ensuring everyone works in harmony toward a common goal.

    Features:
    - Planning with specialized research capabilities
    - Agent specialization based on task types
    - Task dependency management
    - Progress reporting and persistence
    """

    def __init__(
        self,
        provider: ChatProvider,
        planner: TaskPlanner,
        workspace_dir: str,
        agents: List[Agent],
        max_steps: int = 30,
    ):
        """
        Initialize the multi-agent coordinator.
        """
        self.provider = provider
        self.planner = planner
        self.workspace_dir = workspace_dir
        self.max_steps = max_steps
        self.agents = agents
        self.tasks_file_path = Path(workspace_dir) / "tasks.json"

    async def solve_problem(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        🧩 The Grand Solution - Solves the entire objective using specialized agents

        This method is the entry point for solving a complex problem end-to-end,
        orchestrating the entire process from planning to execution:

        1. Creates or loads a task plan
        2. Displays the plan for transparency
        3. Identifies executable tasks based on dependencies
        4. Assigns each task to the appropriate specialized agent
        5. Tracks progress and updates the saved plan
        6. Continues until completion or maximum steps reached
        7. Provides a detailed summary of results

        It's like conducting a symphony where each musician plays their part
        at exactly the right time, creating a harmonious solution.

        Args:
            print_usage (bool): Whether to print token usage statistics

        Yields:
            Union[Message, Chunk, ToolCall]: Live updates during problem solving
        """
        # Phase 1: Planning or loading existing plan
        if self.tasks_file_path.exists():
            yield Chunk(
                content=f"Existing task plan found at {self.tasks_file_path}. Loading...\n",
                done=False,
            )
        else:
            yield Chunk(content="Phase 1: Planning the approach...\n", done=False)

        async for item in self.planner.create_plan():
            yield item

        # Get the task plan
        task_plan = self.planner.task_plan
        if not task_plan:
            raise ValueError("Failed to create or load a valid task plan")

        # Display the plan
        yield Chunk(content="\nTask Plan:\n", done=False)
        yield Chunk(content=task_plan.to_markdown(), done=False)

        # Phase 2: Execution with specialized agents
        yield Chunk(
            content="\nPhase 2: Executing with specialized agents...\n", done=False
        )

        for task in task_plan.tasks:
            yield Chunk(
                content=f"\nExecuting task: {task.title}\n",
                done=False,
            )
            agent = self._get_agent_for_task(task)

            assert task is not None, f"Task not found for agent: {agent.name}"
            async for item in agent.execute_task(task, processing_context):
                yield item
                if isinstance(item, ToolCall) and item.name == "finish_subtask":
                    yield Chunk(
                        content="\n" + task_plan.to_markdown() + "\n",
                        done=False,
                    )

    def _get_agent_for_task(self, task: Task) -> Agent:
        """
        Get the appropriate agent for a task based on its agent_name.

        Args:
            task: The task to get an agent for

        Returns:
            Agent: The appropriate specialized agent
        """
        for agent in self.agents:
            if agent.name == task.agent_name:
                return agent
        raise ValueError(f"No agent found for task type: {task.agent_name}")

    def _all_tasks_complete(self, task_plan: TaskPlan) -> bool:
        """
        Check if all tasks in the task plan are complete.

        A task is complete when all its subtasks are completed.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        for task in task_plan.tasks:
            if not all(subtask.completed for subtask in task.subtasks):
                return False
        return True
