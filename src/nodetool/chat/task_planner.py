import asyncio
from enum import Enum
from pydantic import BaseModel
from nodetool.chat.providers import ChatProvider
from nodetool.chat.sub_task_context import FinishTaskTool
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    Message,
    SubTask,
    Task,
    TaskPlan,
)

import json
import yaml
import os
from pathlib import Path
from typing import List, Sequence, Dict, Set

from nodetool.workflows.processing_context import ProcessingContext
import time
import networkx as nx


# Simplify the DEFAULT_PLANNING_SYSTEM_PROMPT
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are a task planning agent that creates optimized, executable plans.

RESPOND WITH TOOL CALLS TO CREATE TASKS.

KEY PLANNING PRINCIPLES:
1. Break complex goals into clear subtasks
2. Optimize for parallel execution
3. Create self-contained tasks with minimal coupling
4. Define dependencies between tasks using the input_files field
5. Provide clear instructions and all necessary information for each subtask
6. The LAST subtask MUST use the finish_task tool to complete the entire task

DEPENDENCY GRAPH:
- The dependency graph is a directed graph of dependencies between subtasks
- SUBTASKS must not have circular dependencies
- SUBTASKS must depend on existent input files
- SUBTASKS must not have duplicate output_files
- SUBTASKS must depend on input files or other subtask outputs
- The LAST subtask must collect and synthesize all results using finish_task
"""


class CreateTaskTool(Tool):
    """
    Task Creator - Tool for generating a task with subtasks
    """

    name = "create_task"
    description = "Create a single task with subtasks"

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir=workspace_dir)
        self.workspace_dir = workspace_dir
        self.input_schema = {
            "type": "object",
            "required": ["title", "subtasks"],
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The objective of the task",
                },
                "subtasks": {
                    "type": "array",
                    "description": "The subtasks of the task",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Instructions for the subtask to complete",
                            },
                            "output_file": {
                                "type": "string",
                                "description": "The file path where the subtask will save its output",
                            },
                            "input_files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "The input files for the subtask, must be a list of output files of other subtasks, or input files of the task",
                                },
                            },
                            "output_schema": {
                                "type": "object",
                                "description": "The JSON schema of the output of the subtask. REQUIRED when output_type is 'json'. Must be a valid JSON schema object.",
                            },
                            "output_type": {
                                "type": "string",
                                "description": "The file format of the output of the subtask. When using 'json', you MUST provide a valid JSON schema in output_schema.",
                                "enum": [
                                    "string",
                                    "markdown",
                                    "json",
                                    "yaml",
                                    "csv",
                                    "html",
                                    "xml",
                                    "jsonl",
                                    "python",
                                    "py",
                                    "javascript",
                                    "js",
                                    "typescript",
                                    "ts",
                                    "java",
                                    "cpp",
                                    "c++",
                                    "go",
                                    "rust",
                                    "diff",
                                    "shell",
                                    "sql",
                                    "dockerfile",
                                    "css",
                                    "svg",
                                ],
                            },
                        },
                        "required": [
                            "content",
                            "output_file",
                            "input_files",
                            "output_type",
                        ],
                    },
                },
            },
        }

    async def process(self, context: ProcessingContext, params: dict):
        pass


# Remove research-related prompts and simplify agent task prompt
DEFAULT_AGENT_TASK_PROMPT = """
Objective: {objective}

Subtasks will have access to the following tools:
{tools_info}

Use these files as input (input_files) BUT NOT AS output_file:
{input_files_info}

Think carefully about:
1. How to structure subtasks to make them clear and executable
2. How to effectively process the provided input files (batch processing when appropriate)
3. What data to read from the input files
4. How to organize dependencies between subtasks
5. Ensure the LAST subtask uses the finish_task tool to complete the entire task

Create subtasks that are clear, executable, and leverage the appropriate tools when needed.
The final subtask must synthesize all previous results and use finish_task to complete the task.
"""


class TaskPlanner:
    """
    🧩 The Master Planner - Breaks complex problems into executable chunks

    This strategic component divides large objectives into smaller, manageable tasks
    with dependencies between them. It's like a project manager breaking down a large
    project into sprints and tickets, identifying which tasks depend on others.

    The planner can also conduct research before planning to gather relevant information,
    ensuring the plan is well-informed and realistic. Plans are saved to enable
    persistence across sessions.

    Features:
    - Research capabilities to gather information before planning
    - Dependency tracking between subtasks
    - Parallel execution optimization
    - Plan persistence through JSON storage
    - Detailed LLM trace logging for debugging and analysis
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        tools: Sequence[Tool],
        input_files: Sequence[str] = [],
        system_prompt: str | None = None,
        agent_task_prompt: str | None = None,
        enable_tracing: bool = True,
        output_schema: dict | None = None,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            objective (str): The objective to solve
            workspace_dir (str): The workspace directory path
            input_files (list[str]): The input files to use for planning
            tools (List[Tool]): Tools available for research during planning
            system_prompt (str, optional): Custom system prompt
            agent_task_prompt (str, optional): Custom agent task prompt
            enable_tracing (bool, optional): Whether to enable LLM trace logging
            output_schema (dict, optional): JSON schema for the final task output
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.workspace_dir = workspace_dir
        self.task_plan = None
        self.input_files = input_files
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_PLANNING_SYSTEM_PROMPT
        )
        self.agent_task_prompt = (
            agent_task_prompt if agent_task_prompt else DEFAULT_AGENT_TASK_PROMPT
        )
        self.tools = tools or []
        self.enable_tracing = enable_tracing
        self.output_schema = output_schema

        # Setup tracing
        if self.enable_tracing:
            self.traces_dir = os.path.join(self.workspace_dir, "traces")
            os.makedirs(self.traces_dir, exist_ok=True)
            sanitized_objective = "".join(
                c if c.isalnum() else "_" for c in self.objective[:40]
            )
            self.trace_file_path = os.path.join(
                self.traces_dir, f"trace_planner_{sanitized_objective}.jsonl"
            )
            self._log_trace_event(
                "planner_initialized",
                {"objective": self.objective, "model": self.model},
            )

        self.tasks_file_path = Path(workspace_dir) / "tasks.yaml"

    def _log_trace_event(self, event_type: str, data: dict) -> None:
        """
        Log an event to the trace file.

        Args:
            event_type (str): Type of event (message, tool_call, research, etc.)
            data (dict): Event data to log
        """
        if not self.enable_tracing:
            return

        trace_entry = {"timestamp": time.time(), "event": event_type, "data": data}

        with open(self.trace_file_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

    async def _load_existing_plan(self):
        """
        Try to load an existing task plan from the workspace.

        Returns:
            bool: True if plan was loaded successfully, False otherwise
        """
        if self.tasks_file_path.exists():
            try:
                with open(self.tasks_file_path, "r") as f:
                    task_plan_data = yaml.safe_load(f)
                    self.task_plan = TaskPlan(**task_plan_data)
                    return True
            except Exception as e:
                return False
        return False

    async def _build_agent_task_prompt(self) -> str:
        if self.input_files and len(self.input_files) > 0:
            input_files_info = "\n".join(self.input_files)
        else:
            input_files_info = ""

        if self.output_schema:
            output_schema_info = "Output schema of the high-level task:\n"
            output_schema_info += json.dumps(self.output_schema, indent=2)
        else:
            output_schema_info = ""

        if self.tools:
            tools_info = "Available tools for task execution:\n"
            for tool in self.tools:
                tools_info += f"- {tool.name}: {tool.description}\n"
        else:
            tools_info = ""

        return self.agent_task_prompt.format(
            objective=self.objective,
            tools_info=tools_info,
            input_files_info=input_files_info,
        )

    def _build_dependency_graph(self, subtasks: List[SubTask]) -> nx.DiGraph:
        """
        Build a directed graph of dependencies between subtasks.

        Args:
            subtasks: List of subtasks to analyze

        Returns:
            nx.DiGraph: Directed graph representing dependencies
        """
        # Create mapping of output files to their subtasks
        output_to_subtask: Dict[str, SubTask] = {
            subtask.output_file: subtask for subtask in subtasks
        }

        # Create graph
        G = nx.DiGraph()

        # Add all subtasks as nodes
        for subtask in subtasks:
            G.add_node(subtask.output_file)

        # Add edges for dependencies
        for subtask in subtasks:
            if subtask.input_files:
                for input_file in subtask.input_files:
                    if input_file in output_to_subtask:
                        # Add edge from input to output, showing dependency
                        G.add_edge(input_file, subtask.output_file)

        return G

    def _validate_dependencies(
        self,
        subtasks: List[SubTask],
    ) -> List[str]:
        """
        Validate dependencies for a list of subtasks using DAG analysis.

        Args:
            subtasks: List of subtasks to validate dependencies for

        Returns:
            List[str]: List of validation error messages
        """
        validation_errors = []

        # Track all available input files
        available_files = set(self.input_files)

        # Check for duplicate output files and validate output schemas
        output_files = {}
        for subtask in subtasks:
            if subtask.output_file in output_files:
                validation_errors.append(
                    f"Multiple subtasks trying to write to '{subtask.output_file}'"
                )
            output_files[subtask.output_file] = subtask

            # Validate that JSON output type has a schema
            if subtask.output_type == "json":
                if not subtask.output_schema:
                    validation_errors.append(
                        f"Subtask '{subtask.content}' has JSON output_type but no output_schema"
                    )

        # Build and analyze dependency graph
        G = self._build_dependency_graph(subtasks)

        # Check for cycles
        try:
            nx.find_cycle(G)
            validation_errors.append("Circular dependency detected in subtasks")
        except nx.NetworkXNoCycle:
            pass  # No cycles found, which is good

        # Validate all input files exist
        for subtask in subtasks:
            if subtask.input_files:
                for file_path in subtask.input_files:
                    if (
                        file_path not in available_files
                        and file_path not in output_files
                    ):
                        validation_errors.append(
                            f"Subtask '{subtask.content}' depends on missing file '{file_path}'"
                        )

        # Get execution order (topological sort)
        if not validation_errors:
            try:
                execution_order = list(nx.topological_sort(G))
                if self.enable_tracing:
                    self._log_trace_event(
                        "dependency_analysis",
                        {
                            "execution_order": execution_order,
                            "node_count": G.number_of_nodes(),
                            "edge_count": G.number_of_edges(),
                        },
                    )
            except nx.NetworkXUnfeasible:
                validation_errors.append(
                    "Cannot determine valid execution order due to dependency issues"
                )

        return validation_errors

    async def _create_task_for_objective(
        self,
        objective: str,
        input_files: List[str],
        max_retries: int = 3,
    ) -> Task:
        """
        Create subtasks all at once for a specific objective using JSON format.

        Args:
            objective: The objective to create subtasks for
            input_files: List of all available files
            max_retries: Maximum number of retry attempts per subtask

        Returns:
            Task: The created task
        """
        # Build the initial prompt
        agent_task_prompt = await self._build_agent_task_prompt()
        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=agent_task_prompt),
        ]
        # Track retry attempts
        current_retry = 0

        subtasks = []

        # Main loop for creating subtasks with retries
        while current_retry < max_retries:
            message = await self.provider.generate_message(
                messages=history,
                model=self.model,
                tools=[
                    CreateTaskTool(
                        self.workspace_dir,
                    )
                ],
            )

            if not message.tool_calls:
                raise ValueError("No tool calls found in the message")

            subtasks = []
            for tool_call in message.tool_calls:
                for subtask_params in tool_call.args.get("subtasks", []):
                    subtask = SubTask(**subtask_params)
                    subtasks.append(subtask)

            validation_errors = self._validate_dependencies(subtasks)
            # If we have validation errors, retry with feedback
            if validation_errors and current_retry < max_retries - 1:
                print(f"Validation errors: {validation_errors}")
                current_retry += 1
                retry_prompt = f"Please fix following errors:\n"
                for error in validation_errors:
                    retry_prompt += f"- {error}\n"

                history.append(Message(role="user", content=retry_prompt))

                # Log retry attempt
                if self.enable_tracing:
                    self._log_trace_event(
                        "subtask_creation_retry",
                        {
                            "retry_number": current_retry,
                            "max_retries": max_retries,
                            "validation_errors": validation_errors,
                        },
                    )
            else:
                # All subtasks valid or max retries reached
                break

        # Create the task if we have at least one subtask
        if subtasks:
            return Task(
                title=objective,
                subtasks=subtasks,
            )
        else:
            raise ValueError("No subtasks created")

    async def create_task(self, objective: str) -> Task:
        """
        🏗️ Blueprint Designer - Creates or loads a task execution plan

        Creates a high-level task with subtasks, ensuring proper dependencies and organization.
        If a plan already exists in the workspace, it will load that instead of creating a new one.

        Args:
            objective: The objective to create a task for

        Returns:
            Task: The created task with its subtasks
        """
        task = await self._create_task_for_objective(objective, list(self.input_files))
        return task

    async def save_task_plan(self) -> None:
        """
        Save the current task plan to tasks.yaml in the workspace directory.
        """
        if self.task_plan:
            task_dict = self.task_plan.model_dump()
            with open(self.tasks_file_path, "w") as f:
                yaml.dump(task_dict, f, indent=2, sort_keys=False)
