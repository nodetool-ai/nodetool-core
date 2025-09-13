"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A TaskPlanner class that creates a task list with dependencies
2. A TaskExecutor class that executes tasks in the correct order
3. An Agent class that combines planning and execution
4. Integration with the existing provider and tool system
5. Support for streaming results during reasoning
"""

import datetime
from nodetool.config.logging_config import get_logger
import json
import os
import asyncio
from typing import AsyncGenerator, List, Sequence, Union, Any, Optional

from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.config.settings import get_log_path
from nodetool.workflows.types import (
    Chunk,
    PlanningUpdate,
    SubTaskResult,
    TaskUpdate,
    TaskUpdateEvent,
)
from nodetool.agents.task_executor import TaskExecutor
from nodetool.chat.providers import ChatProvider
from nodetool.agents.task_planner import TaskPlanner
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole
from nodetool.agents.base_agent import BaseAgent


log = get_logger(__name__)


def sanitize_file_path(file_path: str) -> str:
    """
    Sanitize a file path by replacing spaces and slashes with underscores.

    Args:
        file_path (str): The file path to sanitize.

    Returns:
        str: The sanitized file path.
    """
    return file_path.replace(" ", "_").replace("/", "_").replace("\\", "_")


class Agent(BaseAgent):
    """
    🤖 Orchestrates AI-driven task execution using Language Models and Tools.

    The Agent class acts as a high-level controller that takes a complex objective,
    breaks it down into a step-by-step plan, and then executes that plan using
    a specified Language Model (LLM) and a set of available tools.

    Think of it as an intelligent assistant that can understand your goal, figure out
    the necessary actions (like searching the web, reading files, performing calculations,
    or running code), and carry them out autonomously to achieve the objective.

    Key Capabilities:
    - **Planning:** Decomposes complex objectives into manageable subtasks.
    - **Execution:** Runs the subtasks in the correct order, handling dependencies.
    - **Tool Integration:** Leverages specialized tools to interact with external
      systems or perform specific actions (e.g., file operations, web browsing,
      code execution).
    - **LLM Agnostic:** Works with different LLM providers (OpenAI, Anthropic, Ollama).
    - **Progress Tracking:** Can stream updates as the task progresses.
    - **Input/Output Management:** Handles input files and collects final results.

    Use this class to automate workflows that require reasoning, planning, and
    interaction with various data sources or tools.
    """

    def __init__(
        self,
        name: str,
        objective: str,
        provider: ChatProvider,
        model: str,
        planning_model: str | None = None,
        reasoning_model: str | None = None,
        tools: Optional[Sequence[Tool]] = None,
        description: str = "",
        inputs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        max_subtasks: int = 10,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int | None = None,
        output_schema: dict | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
        task: Task | None = None,  # Add optional task parameter
        verbose: bool = True,  # Add verbose flag
        docker_image: str | None = None,
        display_manager: AgentConsole | None = None,
    ):
        """
        Initialize the base agent.

        Args:
            name (str): The name of the agent
            objective (str): The objective of the agent
            description (str): The description of the agent
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            reasoning_model (str, optional): The model to use for reasoning, defaults to the same as the provider model
            planning_model (str, optional): The model to use for planning, defaults to the same as the provider model
            tools (List[Tool]): List of tools available for this agent
            inputs (dict[str, Any], optional): Inputs to use for the agent
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum reasoning steps
            max_subtask_iterations (int, optional): Maximum iterations per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
            max_subtasks (int, optional): Maximum number of subtasks to be created
            output_schema (dict, optional): JSON schema for the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 2)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 3)
            task (Task, optional): Pre-defined task to execute, skipping planning
            verbose (bool, optional): Enable/disable console output (default: True)
            docker_image (str, optional): If set, execute the agent inside this Docker image.
        """
        super().__init__(
            name=name,
            objective=objective,
            provider=provider,
            model=model,
            tools=tools or [],
            inputs=inputs or {},
            system_prompt=system_prompt,
            max_token_limit=max_token_limit,
        )
        self.description = description
        self.planning_model = planning_model or model
        self.reasoning_model = reasoning_model or model
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_subtasks = max_subtasks
        self.output_schema = output_schema
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase
        self.initial_task = task
        if self.initial_task:
            self.task = self.initial_task
        self.verbose = verbose
        self.docker_image = docker_image
        self.display_manager = display_manager

    async def execute(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[
        Union[TaskUpdate, Chunk, ToolCall, PlanningUpdate, SubTaskResult], None
    ]:
        """
        Execute the agent using the task plan.

        Args:
            processing_context (ProcessingContext): The processing context

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        # Copy input files to the workspace directory if they are not already there
        if self.docker_image:
            async for item in self._execute_in_docker(processing_context):
                yield item
            return

        tools = list(self.tools)
        task_planner_instance: Optional[TaskPlanner] = (
            None  # Keep track of planner instance
        )

        if self.task:  # If self.task is already set (e.g. by initial_task in __init__)
            # If self.task was set by initial_task, we skip planning.
            # We need to ensure it passes the None check for subsequent operations.
            pass
        else:
            if self.display_manager:
                log.debug(
                    "Agent '%s' planning task for objective: %s",
                    self.name,
                    self.objective,
                )
            self.provider.log_file = str(
                get_log_path(
                    sanitize_file_path(
                        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}__{self.name}__planner.jsonl"
                    )
                )
            )

            task_planner_instance = TaskPlanner(
                provider=self.provider,
                model=self.planning_model,
                reasoning_model=self.reasoning_model,
                objective=self.objective,
                workspace_dir=processing_context.workspace_dir,
                execution_tools=tools,
                inputs=self.inputs,
                output_schema=self.output_schema,
                enable_analysis_phase=self.enable_analysis_phase,
                enable_data_contracts_phase=self.enable_data_contracts_phase,
                verbose=self.verbose,
                display_manager=self.display_manager,
            )

            async for chunk in task_planner_instance.create_task(
                processing_context, self.objective
            ):
                yield chunk

            if (
                task_planner_instance.task_plan
                and task_planner_instance.task_plan.tasks
            ):
                self.task = task_planner_instance.task_plan.tasks[0]

            assert (
                self.task is not None
            ), "Task was not created by planner and was not provided initially."

            yield TaskUpdate(
                task=self.task,
                event=TaskUpdateEvent.TASK_CREATED,
            )

        if self.output_schema and len(self.task.subtasks) > 0:
            self.task.subtasks[-1].output_schema = json.dumps(self.output_schema)

        tool_calls: List[ToolCall] = []

        # Start live display managed by AgentConsole
        if self.display_manager:
            self.display_manager.start_live(
                self.display_manager.create_execution_tree(
                    title=self.name, task=self.task, tool_calls=tool_calls
                )
            )

        try:
            executor = TaskExecutor(
                provider=self.provider,
                model=self.model,
                processing_context=processing_context,
                tools=list(self.tools),  # Ensure it's a list of Tool
                task=self.task,
                system_prompt=self.system_prompt,
                inputs=self.inputs,
                max_steps=self.max_steps,
                max_subtask_iterations=self.max_subtask_iterations,
                max_token_limit=self.max_token_limit,
            )

            # Execute all subtasks within this task and yield results
            async for item in executor.execute_tasks(processing_context):
                # Update tool_calls list if item is a ToolCall
                if isinstance(item, ToolCall):
                    tool_calls.append(item)

                # Create the updated table and update the live display
                if self.display_manager:
                    new_table = self.display_manager.create_execution_tree(
                        title=f"Task:\\n{self.objective}",
                        task=self.task,
                        tool_calls=tool_calls,
                    )
                    self.display_manager.update_live(new_table)

                # Yield the item
                if isinstance(item, ToolCall):
                    if item.name == "finish_task":
                        self.results = item.args["result"]
                        yield TaskUpdate(
                            task=self.task,
                            event=TaskUpdateEvent.TASK_COMPLETED,
                        )
                    if item.name == "finish_subtask" or item.name == "finish_task":
                        for subtask in self.task.subtasks:
                            if subtask.id == item.subtask_id and "result" in item.args:
                                yield SubTaskResult(
                                    subtask=subtask,
                                    result=item.args["result"],
                                    is_task_result=item.name == "finish_task",
                                )
                elif isinstance(item, TaskUpdate):
                    yield item
                    # Update provider log file when a subtask starts/completes
                    if item.event in [
                        TaskUpdateEvent.SUBTASK_STARTED,
                        TaskUpdateEvent.SUBTASK_COMPLETED,
                    ]:
                        timestamp = datetime.datetime.now().strftime(
                            "%Y-%m-%d_%H-%M-%S"
                        )
                        assert item.subtask is not None
                        self.provider.log_file = str(
                            get_log_path(
                                sanitize_file_path(
                                    f"{timestamp}__{self.name}__{item.subtask.id}.jsonl"
                                )
                            )
                        )
                elif isinstance(
                    item, (Chunk, ToolCall)
                ):  # Yield chunks and other tool calls too
                    yield item

        finally:
            # Ensure live display is stopped
            if self.display_manager:
                self.display_manager.stop_live()

        if self.display_manager:
            log.debug("Provider usage: %s", self.provider.usage)

    def get_results(self) -> Any:
        """
        Get the results produced by this agent.
        If a final result exists from finish_task, return that.
        Otherwise, return all collected results.

        Returns:
            List[Any]: Results with priority given to finish_task output
        """
        return self.results

    async def _execute_in_docker(
        self,
        processing_context: ProcessingContext,
    ) -> AsyncGenerator[Chunk, None]:
        """Run the agent inside a Docker container."""
        workspace = processing_context.workspace_dir
        config = {
            "name": self.name,
            "objective": self.objective,
            "provider": self.provider.provider.name,
            "model": self.model,
            "planning_model": self.planning_model,
            "reasoning_model": self.reasoning_model,
            "tools": [t.__class__.name for t in self.tools],
            "description": self.description,
            "inputs": self.inputs,
            "system_prompt": self.system_prompt,
            "max_subtasks": self.max_subtasks,
            "max_steps": self.max_steps,
            "max_subtask_iterations": self.max_subtask_iterations,
            "max_token_limit": self.max_token_limit,
            "output_schema": self.output_schema,
            "enable_analysis_phase": self.enable_analysis_phase,
            "enable_data_contracts_phase": self.enable_data_contracts_phase,
            "verbose": self.verbose,
            "workspace_dir": "/workspace",
            "result_path": "/workspace/docker_result.json",
        }

        host_config = os.path.join(workspace, "docker_agent_config.json")
        with open(host_config, "w") as f:
            json.dump(config, f)

        env_vars: dict[str, str] = {}
        env_vars.update(self.provider.get_container_env())
        for tool in self.tools:
            env_vars.update(tool.get_container_env())

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{workspace}:/workspace",
        ]

        for k, v in env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        assert self.docker_image is not None, "Docker image is not set"
        cmd.extend(
            [
                self.docker_image,
                "python",
                "-m",
                "nodetool.agents.docker_runner",
                "/workspace/docker_agent_config.json",
            ]
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout
        async for line in proc.stdout:
            yield Chunk(content=line.decode())
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker run failed with code {proc.returncode}")

        result_file = os.path.join(workspace, "docker_result.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                self.results = json.load(f)
        yield Chunk(content="\n[docker completed]\n", done=True)


async def test_docker_feature():
    """
    Smoke test for the Docker feature in Agent.
    Tests that an Agent can be initialized with a docker_image parameter.
    """
    from nodetool.chat.providers.openai_provider import OpenAIProvider

    # Create a mock provider
    provider = OpenAIProvider()

    # Test that Agent can be initialized with docker_image parameter
    agent = Agent(
        name="test-docker-agent",
        objective="Write python code to ",
        provider=provider,
        model="gpt-4o-mini",
        enable_analysis_phase=False,
        enable_data_contracts_phase=False,
        docker_image="nodetool",
        tools=[ExecutePythonTool()],
    )

    context = ProcessingContext()

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print(f"\nWorkspace: {context.workspace_dir}")
    print(f"Results: {agent.results}")
    print("✓ Docker feature smoke test passed")


if __name__ == "__main__":
    # Run the smoke test when the module is executed directly
    asyncio.run(test_docker_feature())
