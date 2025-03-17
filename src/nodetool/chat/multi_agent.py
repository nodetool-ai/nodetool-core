from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.task_planner import TaskPlanner
from nodetool.chat.tools import Tool
from nodetool.metadata.types import FunctionModel, Message, Task, TaskPlan, ToolCall


import os
from pathlib import Path
from typing import AsyncGenerator, List, Union


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
        planner_model: FunctionModel,
        objective: str,
        workspace_dir: str,
        agents: List[Agent],
        max_steps: int = 30,
        planning_steps: int = 3,
        planning_tools: List[Tool] = [],
        max_subtask_iterations: int = 3,
        max_token_limit: int = 8000,
    ):
        """
        Initialize the multi-agent coordinator.
        """
        self.provider = provider
        self.planner_model = planner_model
        self.objective = objective
        self.workspace_dir = workspace_dir
        self.max_steps = max_steps
        self.planning_steps = planning_steps
        self.planning_tools = planning_tools
        self.max_subtask_iterations = max_subtask_iterations
        self.max_token_limit = max_token_limit
        self.agents = agents
        self.tasks_file_path = Path(workspace_dir) / "tasks.json"

    async def solve_problem(
        self,
        print_usage: bool = False,
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

        # Create planner with retrieval tools
        self.planner = TaskPlanner(
            provider=self.provider,
            model=self.planner_model,
            objective=self.objective,
            workspace_dir=self.workspace_dir,
            tools=self.planning_tools,
            agents=self.agents,
            max_research_iterations=self.planning_steps,
        )

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

        steps_taken = 0
        all_tasks_complete = False

        # Continue until all tasks are complete or max steps reached
        while not all_tasks_complete and steps_taken < self.max_steps:
            steps_taken += 1

            # Find all executable tasks (at the task level)
            executable_tasks = self._get_executable_tasks(task_plan)

            if not executable_tasks:
                # Check if all tasks are complete
                all_tasks_complete = self._all_tasks_complete(task_plan)

                if all_tasks_complete:
                    break
                else:
                    yield Chunk(
                        content="\nNo executable tasks but not all complete. Possible file dependency issues.\n",
                        done=False,
                    )
                    break

            # Execute the first executable task
            task = executable_tasks[0]

            # Get the appropriate agent
            agent = self._get_agent_for_task(task)

            # Execute the entire task with the appropriate agent
            # The agent will handle all subtasks
            async for item in agent.execute_task(task_plan, task):
                yield item

            # Update the task plan file after each task is executed
            try:
                with open(self.tasks_file_path, "w") as f:
                    f.write(task_plan.model_dump_json(indent=2))
            except Exception as e:
                print(f"Error updating task plan file: {e}")

        # Summary of results
        if all_tasks_complete:
            yield Chunk(
                content="\nAll tasks completed successfully! Final results:\n",
                done=False,
            )
        else:
            yield Chunk(
                content="\nTask execution paused. Progress saved in tasks.json. Final results so far:\n",
                done=False,
            )

        for task in task_plan.tasks:
            # Calculate completion percentage
            completed_subtasks = sum(
                1 for subtask in task.subtasks if subtask.completed
            )
            total_subtasks = len(task.subtasks)
            completion_pct = (
                (completed_subtasks / total_subtasks) * 100 if total_subtasks > 0 else 0
            )

            yield Chunk(
                content=f"\nAgent: {task.agent_name} "
                f"({completed_subtasks}/{total_subtasks} - {completion_pct:.1f}% complete)\n",
                done=False,
            )
            for subtask in task.subtasks:
                status = "✓ DONE" if subtask.completed else "➤ PENDING"
                yield Chunk(
                    content=f"- [{status}] {subtask.id}: {subtask.output_file}\n",
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

    def _get_executable_tasks(self, task_plan: TaskPlan) -> List[Task]:
        """
        Get all executable tasks from the task plan.

        A task is considered executable when:
        1. Not all of its subtasks are completed yet
        2. All file dependencies for at least one of its incomplete subtasks are met

        Returns:
            List[Task]: List of executable tasks
        """
        executable_tasks = []

        for task in task_plan.tasks:
            # Skip if all subtasks in this task are already complete
            if all(subtask.completed for subtask in task.subtasks):
                continue

            # Check if at least one subtask is executable
            for subtask in task.subtasks:
                if subtask.completed:
                    continue

                # Check if all file dependencies exist for this subtask
                all_dependencies_met = True
                for file_path in subtask.file_dependencies:
                    # Strip /workspace prefix for file system operations
                    if file_path.startswith("/workspace/"):
                        relative_path = os.path.relpath(file_path, "/workspace")
                        full_path = os.path.join(self.workspace_dir, relative_path)
                    else:
                        full_path = os.path.join(self.workspace_dir, file_path)

                    if not os.path.exists(full_path):
                        all_dependencies_met = False
                        break

                if all_dependencies_met:
                    # This task has at least one executable subtask
                    executable_tasks.append(task)
                    break

        return executable_tasks

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
