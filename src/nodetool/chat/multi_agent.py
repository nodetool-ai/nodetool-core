from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.sub_task_context import TaskUpdate, TaskUpdateEvent
from nodetool.chat.task_planner import PlanUpdate, TaskPlanner
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
        self.tasks_file_path = Path(workspace_dir) / "tasks.yaml"

    async def solve_problem(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Union[TaskUpdate, Chunk, ToolCall, PlanUpdate], None]:
        """
        🧩 The Grand Solution - Solves the entire objective using specialized agents

        This method is the entry point for solving a complex problem end-to-end,
        orchestrating the entire process from planning to execution:

        1. Creates or loads a task plan
        2. Assigns each task to the appropriate specialized agent
        3. Tracks progress and updates the saved plan
        4. Continues until completion or maximum steps reached
        5. Provides a detailed summary of results

        It's like conducting a symphony where each musician plays their part
        at exactly the right time, creating a harmonious solution.

        Args:
            print_usage (bool): Whether to print token usage statistics

        Yields:
            Union[Message, Chunk, ToolCall]: Live updates during problem solving
        """
        # Phase 1: Planning
        async for item in self.planner.create_plan():
            yield item
            if isinstance(item, PlanUpdate):
                print(
                    f"{item.event} - "
                    + (f"{item.subtask.content}" if item.subtask else "")
                    + (f" - {item.retry_count}" if item.retry_count else "")
                    + (f" - {item.error_message}" if item.error_message else "")
                )
            if isinstance(item, Chunk):
                print(item.content, end="")

        task_plan = self.planner.task_plan
        if not task_plan:
            raise ValueError("Failed to create or load a valid task plan")

        print(task_plan.to_markdown())

        # Phase 2: Execution
        accumulated_results = []  # Track all results across agents

        for task in task_plan.tasks:
            agent = self._get_agent_for_task(task)

            # Pass accumulated results as input files to the next agent
            if accumulated_results:
                agent.input_files = accumulated_results

            assert task is not None, f"Task not found for agent: {agent.name}"
            async for item in agent.execute_task(task, processing_context):
                yield item

            # Collect results from this agent for the next agent
            agent_results = agent.get_results()
            if agent_results:
                accumulated_results.extend(agent_results)

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
