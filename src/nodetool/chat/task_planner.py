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


# Simplified and phase-agnostic system prompt
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are an expert task planning agent that creates highly optimized, executable plans.
You excel at breaking down complex tasks into logical subtasks with clear dependencies.

CRITICAL CAPABILITIES:
- Breaking complex tasks into optimal components
- Identifying parallel execution opportunities
- Defining clear data contracts between components
- Creating efficient, executable subtask sequences
- Ensuring type safety throughout the workflow

SUBTASK REQUIREMENTS:
- Self-contained with clear instructions
- Properly sequenced with minimal dependencies
- Correctly specified input and output formats
- Type-safe data handling

FINAL SUBTASK MUST:
- Use finish_task tool
- Match the task's output schema
- Include comprehensive metadata
- Validate the complete result

RESPOND WITH TOOL CALLS TO CREATE TASKS.
"""

# Agent task prompt used in the final planning stage
DEFAULT_AGENT_TASK_PROMPT = """
OBJECTIVE: {objective}

AVAILABLE TOOLS:
{tools_info}

INPUT FILES:
{input_files_info}
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
        self.example = """
        Example:
        create_task(
            title="Analyze customer feedback and create a summary report",
            subtasks=[
                {
                    "content": "Extract key themes and sentiment from customer feedback data",
                    "output_file": "feedback_analysis.json",
                    "input_files": ["customer_feedback.csv"],
                    "output_type": "json",
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "themes": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "sentiment": {
                                "type": "object",
                                "properties": {
                                    "positive": {"type": "number"},
                                    "neutral": {"type": "number"},
                                    "negative": {"type": "number"}
                                }
                            }
                        }
                    },
                    "use_code_interpreter": True
                },
                {
                    "content": "Generate data visualizations based on the feedback analysis",
                    "output_file": "feedback_charts.md",
                    "input_files": ["feedback_analysis.json"],
                    "output_type": "markdown",
                    "use_code_interpreter": True
                },
                {
                    "content": "Write a comprehensive summary report with insights and recommendations",
                    "output_file": "feedback_report.md",
                    "input_files": ["feedback_analysis.json", "feedback_charts.md"],
                    "output_type": "markdown",
                    "use_code_interpreter": False
                }
            ],
        )
        """
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
                            "use_code_interpreter": {
                                "type": "boolean",
                                "description": "Whether the subtask should use code interpreter for execution. Set to true if the subtask involves data analysis, calculations, or code execution.",
                                "default": False,
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


# Make sure the phase prompts are concrete and focused
ANALYSIS_PROMPT = """
ANALYSIS PHASE

First, assess if this is a simple task that could be completed in 1-2 steps.
If so, you can create a streamlined plan immediately using the create_task tool.

COMPLEXITY ASSESSMENT:
1. Is this a straightforward, single-purpose task?
2. Can it be completed in 1-2 steps?
3. Are there minimal dependencies?
4. Is the data flow simple and linear?

If YES to most of these questions, create a simplified plan immediately.
If NO, proceed with detailed analysis:

1. Core components and their relationships
2. Opportunities for parallel execution
3. Potential bottlenecks or dependencies
4. Patterns that can be consolidated

Current Objective: {objective}

Available Resources:
{tools_info}
{input_files_info}

If this is a simple task, USE THE create_task TOOL NOW.
Otherwise, provide detailed analysis for complex planning.
"""

DATA_CONTRACTS_PROMPT = """
DATA CONTRACTS PHASE

Analyze the data requirements:
1. Input/output schemas for each identified component
2. Data validation requirements
3. Type safety and consistency
4. Minimizing data transformations

If you determine the data contracts are simple enough,
you may create a streamlined plan now using the create_task tool.

Otherwise, define specific data contracts covering:
1. Schema definitions
2. Data validation rules
3. Type requirements
4. Format standardization

If complexity is low, USE THE create_task TOOL NOW.
Otherwise, continue with detailed contract design.
"""

PLAN_CREATION_PROMPT = """
PLAN CREATION PHASE

Create an optimized task plan that:
1. Implements the identified components as concrete subtasks
2. Follows the defined data contracts
3. Maximizes parallel execution
4. Ensures proper dependency management

USE THE create_task TOOL to implement the plan.
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
        enable_tracing: bool = True,
        output_schema: dict | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
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
            enable_tracing (bool, optional): Whether to enable LLM trace logging
            output_schema (dict, optional): JSON schema for the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 1)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 2)
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.workspace_dir = workspace_dir
        self.task_plan = None
        self.input_files = input_files

        # Check if the provider has code interpreter capabilities
        self.has_code_interpreter = (
            hasattr(provider, "has_code_interpreter") and provider.has_code_interpreter
        )

        # Customize system prompt based on provider capabilities
        self.system_prompt = self._customize_system_prompt(system_prompt)

        self.tools = tools or []
        self.enable_tracing = enable_tracing
        self.output_schema = output_schema
        self.enable_analysis_phase = enable_analysis_phase
        self.enable_data_contracts_phase = enable_data_contracts_phase

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

    def _customize_system_prompt(self, system_prompt: str | None) -> str:
        """
        Customize the system prompt based on provider capabilities.

        Args:
            system_prompt: Optional custom system prompt

        Returns:
            str: The customized system prompt
        """
        if system_prompt:
            base_prompt = system_prompt
        else:
            base_prompt = DEFAULT_PLANNING_SYSTEM_PROMPT

        # If code interpreter is not available, remove the related rule
        if self.has_code_interpreter:
            base_prompt += """
            SET use_code_interpreter TO TRUE when:
            - Subtask involves data analysis or calculations
            - Subtask requires executing code
            - Subtask processes or transforms data programmatically
            - Mathematical operations or statistical analysis is needed
            """

        return base_prompt

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

        return DEFAULT_AGENT_TASK_PROMPT.format(
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
        max_retries: int = 3,
    ) -> Task:
        """
        Create subtasks using the configured planning process, allowing for early shortcuts.
        """
        history = [
            Message(role="system", content=self.system_prompt),
        ]

        # Phase 1: Analysis with potential shortcut (if enabled)
        if self.enable_analysis_phase:
            analysis_prompt = ANALYSIS_PROMPT.format(
                objective=self.objective,
                tools_info=self._get_tools_info(),
                input_files_info=self._get_input_files_info(),
            )
            history.append(Message(role="user", content=analysis_prompt))
            task = await self._generate_with_retry(history)

            # Check if a plan was created in Phase 1
            if task:
                return task

        # Phase 2: Data Contracts with potential shortcut (if enabled)
        if self.enable_data_contracts_phase:
            history.append(Message(role="user", content=DATA_CONTRACTS_PROMPT))
            task = await self._generate_with_retry(history)

            # Check if a plan was created in Phase 2
            if task:
                return task

        # Phase 3: Final Plan Creation (always enabled)
        history.append(
            Message(
                role="user",
                content=PLAN_CREATION_PROMPT + await self._build_agent_task_prompt(),
            )
        )
        task = await self._generate_with_retry(history, max_retries)
        if task:
            return task
        raise ValueError("Failed to create valid task after maximum retries")

    async def _process_tool_calls(
        self, message: Message, history: List[Message]
    ) -> Task:
        """Helper method to process tool calls and create task"""
        if not message.tool_calls:
            raise ValueError(f"No tool calls found in the message: {message.content}")

        # Add tool response messages
        for tool_call in message.tool_calls:
            history.append(
                Message(
                    role="tool",
                    content="Task created successfully",
                    tool_call_id=tool_call.id,
                )
            )

        subtasks = []
        validation_errors = []
        for tool_call in message.tool_calls:
            for subtask_params in tool_call.args.get("subtasks", []):
                try:
                    # Set default value for use_code_interpreter if not present,
                    # or force it to False if provider doesn't support it
                    if "use_code_interpreter" not in subtask_params:
                        subtask_params["use_code_interpreter"] = False
                    elif not self.has_code_interpreter:
                        # Force to False if provider doesn't have code interpreter
                        subtask_params["use_code_interpreter"] = False

                    subtask = SubTask(**subtask_params)
                    subtasks.append(subtask)
                except Exception as e:
                    validation_errors.append(
                        f"Error creating subtask: {subtask_params} with error: {e}"
                    )

        validation_errors.extend(self._validate_dependencies(subtasks))
        if validation_errors:
            raise ValueError(f"Validation errors in created task: {validation_errors}")

        return Task(
            title=self.objective,
            subtasks=subtasks,
        )

    async def _generate_with_retry(
        self, history: List[Message], max_retries: int = 3
    ) -> Task | None:
        """Helper method to process tool calls with retry logic"""
        current_retry = 0
        while current_retry < max_retries:
            message = await self.provider.generate_message(
                messages=history,
                model=self.model,
                tools=[CreateTaskTool(self.workspace_dir)],
            )
            history.append(message)

            if not message.tool_calls:
                return None

            try:
                return await self._process_tool_calls(message, history)
            except ValueError as e:
                if current_retry < max_retries - 1:
                    current_retry += 1
                    retry_prompt = f"Please fix the following errors:\n{str(e)}"
                    print(retry_prompt)
                    history.append(Message(role="user", content=retry_prompt))
                else:
                    raise

        raise ValueError("Failed to create valid task after maximum retries")

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
        task = await self._create_task_for_objective(objective)
        return task

    async def save_task_plan(self) -> None:
        """
        Save the current task plan to tasks.yaml in the workspace directory.
        """
        if self.task_plan:
            task_dict = self.task_plan.model_dump()
            with open(self.tasks_file_path, "w") as f:
                yaml.dump(task_dict, f, indent=2, sort_keys=False)

    def _get_tools_info(self) -> str:
        """
        Get formatted information about available tools.

        Returns:
            str: Formatted string containing tool information
        """
        if not self.tools:
            return "No tools available"

        tools_info = "Available tools for task execution:\n"
        for tool in self.tools:
            tools_info += f"- {tool.name}: {tool.description}\n"
        return tools_info

    def _get_input_files_info(self) -> str:
        """
        Get formatted information about input files.

        Returns:
            str: Formatted string containing input files information
        """
        if not self.input_files:
            return "No input files available"

        input_files_info = "Input files:\n"
        for file_path in self.input_files:
            input_files_info += f"- {file_path}\n"
        return input_files_info
