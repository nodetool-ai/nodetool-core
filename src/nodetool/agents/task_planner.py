"""Manages the breakdown of complex objectives into executable task plans.

The TaskPlanner is responsible for taking a high-level user objective and
transforming it into a structured `TaskPlan`. This plan consists of
interdependent `SubTask` instances. The planning process can involve
multiple phases, including self-reflection for complexity assessment,
objective analysis, data flow definition, and final plan creation.

The planner interacts with an LLM provider to generate and refine the plan,
ensuring that subtasks are well-defined, dependencies are clear (forming a
Directed Acyclic Graph - DAG), and file paths are correctly managed within
a specified workspace. Validation is a key aspect of the planner's role to
ensure the generated plan is robust and executable.
"""

from nodetool.config.logging_config import get_logger
import traceback
from nodetool.providers import BaseProvider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import (
    Message,
    SubTask,
    Task,
    TaskPlan,
    ToolCall,
)

import json

# Add jsonschema import for validation
from jsonschema import validate, ValidationError
import yaml
import re  # Added import re
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    List,
    Sequence,
    Dict,
    Set,
    Optional,
)  # Add Optional

from nodetool.workflows.processing_context import ProcessingContext
import networkx as nx

# Removed rich imports - Console, Table, Text, Live
from nodetool.ui.console import AgentConsole  # Import the new display manager
from rich.text import Text  # Re-add Text import

# Add Jinja2 imports
from jinja2 import Environment, BaseLoader

from nodetool.workflows.types import Chunk, PlanningUpdate

log = get_logger(__name__)

COMPACT_SUBTASK_NOTATION_DESCRIPTION = """
--- Data Flow Representation (DOT/Graphviz Syntax) ---
This format helps visualize the flow of data between subtasks.
Example DOT for a pipeline:
digraph DataPipeline {
  "read_data" -> "process_data";
  "process_data" -> "generate_report";
}
Used for concise subtask representation.
"""

# --- Task Naming Convention ---
# Use short names, prefixed with $, for conceptual tasks or process steps (e.g., $read_data, $process_logs, $generate_report).
# These names should be unique and descriptive of the step.


class CreateTaskTool(Tool):
    """
    Task Creator - Tool for generating a task with subtasks
    """

    name = "create_task"
    description = "Create a single task with subtasks"

    input_schema = {
        "type": "object",
        "required": ["title", "subtasks"],
        "additionalProperties": False,
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
                    "additionalProperties": False,
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "The unique identifier for the subtask.",
                        },
                        "content": {
                            "type": "string",
                            "description": "High-level natural language instructions for the agent executing this subtask.",
                        },
                        "input_tasks": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "The ID of an upstream subtask or input key that this subtask depends on.",
                            },
                            "description": "List of subtask IDs or input keys from the inputs dictionary that this subtask depends on for input data.",
                        },
                        "output_schema": {
                            "type": "string",
                            "description": 'Output schema for the subtask as a JSON string. Use \'{"type": "string"}\' for unstructured output types.',
                        }
                    },
                    "required": [
                        "id",
                        "content",
                        "output_schema",
                        "input_tasks"
                    ],
                },
            },
        },
    }

    async def process(self, context: ProcessingContext, params: dict):
        """Process the CreateTaskTool call.

        This method is called when the LLM uses the 'create_task' tool.
        However, the actual validation and processing of the task and subtask
        data are handled by the `TaskPlanner`.

        Args:
            context: The processing context.
            params: The parameters provided by the LLM for the tool call,
                    matching the `input_schema`.
        """
        pass


# Simplified and phase-agnostic system prompt (GPT-5 aligned)
DEFAULT_PLANNING_SYSTEM_PROMPT = """
<role>
You are a TaskArchitect system that transforms user objectives into executable multi-phase plans.
</role>

<goal>
Transform the user's objective into an executable plan through three phases:
1. Analysis - Understand objective and break into conceptual subtasks (using two-tier planning: discovery vs execution)
2. Data Flow - Define dependencies and data contracts
3. Plan Creation - Generate final executable task plan with planning tasks for dynamic work discovery

**Dynamic Planning Capability:**
Agents executing subtasks have access to the `add_subtask` tool, enabling adaptive workflows. Use "planning tasks" for work that requires discovery (e.g., finding URLs, identifying items to process). Planning tasks should explicitly instruct agents to use `add_subtask` to create focused subtasks for each discovered item.
</goal>

<operating_constraints>
- Complete all phases without stopping early
- Resolve ambiguities through reasonable assumptions (document in notes)
- Use tool calls exclusively; emit `create_task` tool call as final action
- Stop immediately after emitting the final tool call
</operating_constraints>

<output_requirements>
- Begin each phase with one-sentence goal restatement and 1-3 step outline
- Provide concise justification before final tool call
- Use structured outputs over prose
- Emit only requested fields and tool calls (no chain-of-thought)
- Use concise JSON with exact identifiers (no markdown unless specified)
</output_requirements>

<validation_checklist>
Before emitting `create_task` tool call, verify:
- All subtask IDs are unique and descriptive
- Dependencies form a valid DAG (Directed Acyclic Graph)
- All referenced task IDs and input keys exist
- Subtasks are atomic (smallest executable units)
- Planning tasks explicitly instruct agents to use `add_subtask` for discovered items
- Execution tasks have clear, direct implementation instructions
- Final output conforms to required schema
</validation_checklist>
"""

# Phase prompts optimized using GPT-5 best practices
ANALYSIS_PHASE_TEMPLATE = """
<phase>PHASE 0: OBJECTIVE ANALYSIS & CONCEPTUAL SUBTASK BREAKDOWN</phase>

<goal>
Deeply understand the user's objective and decompose it into atomic conceptual subtasks using a two-tier planning strategy: planning tasks for discovery and execution tasks for known work.
</goal>

<instructions>
1. Interpret the objective:
   - State your interpretation of the user's core goal
   - Identify ambiguities and document assumptions
   - Consider output requirements and available inputs

2. Devise strategic plan using two-tier task decomposition:

   **Planning Tasks (Discovery & Dynamic Expansion):**
   Use planning tasks when:
   - The exact set of work items is unknown upfront (e.g., "find URLs", "discover products")
   - Multiple similar items need processing (URLs, files, data points)
   - Work quantity depends on search/discovery results

   Planning tasks should:
   - Use tools to discover what needs to be done
   - Use the `add_subtask` tool to create focused subtasks for each discovered item
   - Have output_schema describing the discovery results

   Example: Instead of creating 10 URL processing subtasks upfront, create:
   - $discover_urls: "Use GoogleSearch to find relevant URLs, then use add_subtask to create one processing task per URL found"

   **Execution Tasks (Direct Implementation):**
   Use execution tasks when:
   - Work is clearly defined and scope is known
   - No discovery phase needed
   - Task is truly atomic (single focused action)

   Example: $compile_results: "Aggregate all processing results into final report format"

   **Key Principle:** Prefer planning tasks for iterative/multiple-item work. This enables adaptive planning based on actual discoveries.
</instructions>

<output_format>
1. Objective Understanding & Strategic Reasoning (concise, <120 tokens):
   <objective_interpretation>
   - Core goal interpretation
   - Key aspects, constraints, desired outcomes
   - Assumptions made to resolve ambiguities
   </objective_interpretation>

   <input_analysis>
   - How inputs will be used
   - Critical inputs per conceptual subtask
   </input_analysis>

   <subtask_rationale>
   - How interpretation led to subtask breakdown
   - Which tasks are planning tasks (discovery) vs execution tasks (direct)
   - Why planning tasks are used for dynamic work discovery
   - Parallel processing considerations
   </subtask_rationale>

   Use '$short_name' convention (e.g., $discover_urls, $compile_results)

2. Task List with Type Classification:
   - Format: `$task_name [PLANNING|EXECUTION] - Brief description`
   - Planning Task Example: `$discover_urls [PLANNING] - Search for Reddit AI workflow posts, create subtask per URL using add_subtask`
   - Execution Task Example: `$compile_report [EXECUTION] - Aggregate all results into markdown report`
</output_format>

<constraints>
- Focus on conceptual subtasks only (no detailed schemas yet)
- Ensure unique, descriptive '$short_name' for each task
- Output only requested sections, no extra commentary
</constraints>

<context>
User's Objective: {{ objective }}
Available Inputs: {{ inputs_info }}
Output Schema: {{ output_schema }}
Execution Tools: {{ execution_tools_info }}
</context>
"""

DATA_FLOW_ANALYSIS_TEMPLATE = """
<phase>PHASE 1: DATA FLOW ANALYSIS - Defining Dependencies and Contracts</phase>

<goal>
Refine conceptual subtask plan from Phase 0 by defining precise data flow and dependencies.
</goal>

<instructions>
1. Review conceptual subtask list from Phase 0 (in conversation history)
2. For each subtask:
   - Determine dependencies on other subtasks (via task IDs)
   - Identify dependencies on input keys from inputs dictionary
   - Refine high-level instructions (keep concise)
3. Ensure final result subtask(s) match overall output requirements
4. Validate all dependencies form a DAG (no cycles)
</instructions>

<output_format>
1. Reasoning & Data Flow Analysis (concise, <150 tokens):
   <data_flow_design>
   - Dependency determination for each Phase 0 subtask
   - Subtasks using input keys directly
   - Refinements to subtask instructions
   - Use '$name' convention for Phase 0 task references
   </data_flow_design>

2. Data Flow (DOT/Graphviz):
   ```dot
   digraph DataPipeline {
     "input:data_file" -> "$process_data";
     "$fetch_data" -> "$process_data";
     "$process_data" -> "$analyze_results";
     "$analyze_results" -> "$generate_report";
   }
   ```
   - Include input keys as source nodes (format: "input:key_name")
   - Process steps use '$short_name' from Phase 0
</output_format>

<constraints>
- Dependencies MUST form a DAG (Directed Acyclic Graph)
- Input keys reference names from inputs dictionary
- Clearly link Phase 0 tasks to concrete data flow
- No chain-of-thought in output
</constraints>

<context>
User's Objective: {{ objective }}
Available Inputs: {{ inputs_info }}
Conceptual Subtask Plan: (see conversation history)
Output Schema: {{ output_schema }}
Execution Tools: {{ execution_tools_info }}
DOT/Graphviz Guide: {{ COMPACT_SUBTASK_NOTATION_DESCRIPTION }}
</context>
"""

PLAN_CREATION_TEMPLATE = """
<phase>PHASE 2: PLAN CREATION - Generating the Executable Task</phase>

<goal>
Transform conceptual subtask plan and data flow from previous phases into concrete, executable task plan via single `create_task` tool call.
</goal>

<output_format>
1. Brief Justification (<200 tokens, no chain-of-thought):
   <plan_construction>
   - How Phase 1 subtasks/data flow translate to `create_task` arguments
   - Decisions for `content`, `input_tasks`, `output_schema` per subtask
   - Validation criteria met (DAG, final output conformance)
   - Data flow summary based on `input_tasks` dependencies
   - Use '$name' convention for Phase 0 task references
   </plan_construction>

2. `create_task` Tool Call:
   - Make exactly ONE call immediately after justification
   - No additional text after tool call
</output_format>

<create_task_guidelines>
General Structure:
- title: Appropriate title for overall task
- subtasks: Array of subtask objects

Per Subtask:
- id: Unique identifier (e.g., "discover_urls", "analyze_results", "compile_report")
- content: High-level natural language instructions (distinguish planning vs execution)

  **For PLANNING tasks (discovery/dynamic expansion):**
  * Explicitly instruct agent to use `add_subtask` tool after discovery
  * Example: "Use GoogleSearch to find 3-5 Reddit posts about AI workflows. For each URL found, use the add_subtask tool to create a new subtask with:
    - content: 'Fetch {{url}}.json with BrowserTool and extract post title, summary, and top 3 comments'
    - input_tasks: [] (no dependencies)
    - output_schema describing the extracted data"
  * Be specific about what data to pass to dynamically created subtasks
  * Mention which input keys to use if applicable

  **For EXECUTION tasks (direct implementation):**
  * Focus on WHAT to achieve, not HOW
  * Example: "Aggregate results from all URL processing subtasks into markdown report with sections per workflow"
  * Mention which input keys to use if applicable

- input_tasks: Array of subtask IDs or input keys this depends on
  * From Phase 1 data flow graph
  * Empty array [] for initial subtasks with no dependencies
  * Can reference input dictionary keys (e.g., "data_file")
- output_schema: Valid JSON string describing result structure
  * Example: '{"type": "string"}' for unstructured
  * Use concise, canonical JSON (no markdown)

Pre-Tool-Call Validation Checklist:
✓ All subtask IDs are unique
✓ Dependencies form valid DAG (no cycles)
✓ All referenced task IDs and input keys exist
✓ Subtasks are focused and atomic
✓ Terminal subtask output matches overall output schema
✓ Schemas use canonical JSON format

Agent Execution Model (for awareness):
- Agents call `finish_subtask` with result objects
- Results auto-pass between subtasks via `input_tasks`
- `read_result` tool fetches upstream task results or input keys
- Input dictionary keys accessible same way as task results
- **Dynamic Planning**: Agents have access to `add_subtask` tool to create new subtasks during execution
  * Planning tasks should explicitly instruct agent to use this tool
  * Dynamically added subtasks execute after the planning task completes
  * This enables adaptive workflows based on discovered data
</create_task_guidelines>

<context>
User's Objective: {{ objective }}
Available Inputs: {{ inputs_info }}
Conceptual Subtasks & Data Flow: (see Phase 1 in conversation history)
Output Schema: {{ output_schema }}
LLM Models: Primary={{ model }}, Reasoning={{ reasoning_model }}
Execution Tools: {{ execution_tools_info }}
</context>
"""

# Final validation checklist for plan creation
DEFAULT_AGENT_TASK_TEMPLATE = """
<final_check>
Before emitting `create_task` tool call, verify each subtask:

1. Clarity of Purpose: `content` is crystal-clear, high-level objective for autonomous agent
   - Planning tasks: Explicitly instruct to use `add_subtask` after discovery
   - Execution tasks: Clear direct implementation instructions
2. Self-Containment: `input_tasks` lists all upstream dependencies needed
3. Output Precision: `output_schema` accurately describes result structure
4. DAG Integrity: All dependencies exist and form acyclic graph (no orphans)
5. Naming: Subtask `id`s unique, descriptive, consistent with `input_tasks` references
6. Output Fit: Terminal subtask output matches overall task `output_schema`
7. Dynamic Planning: Planning tasks explicitly mention using `add_subtask` for discovered items

Verify plan addresses: {{ objective }}

After checklist, immediately emit single `create_task` tool call. No chain-of-thought. No extra commentary.
</final_check>
"""


class TaskPlanner:
    """
    Orchestrates the breakdown of a complex objective into a validated, executable
    workflow plan (`TaskPlan`) composed of interdependent subtasks (`SubTask`).

    Think of this as the lead architect for an AI agent system. It doesn't execute
    the subtasks itself, but meticulously designs the blueprint. Given a high-level
    objective (e.g., "Analyze market trends for product X"), the TaskPlanner uses
    an LLM to generate a structured plan detailing:

    1.  **Decomposition:** Breaking the objective into smaller, logical, and ideally
        atomic units of work (subtasks).
    2.  **Task Typing:** Determining if each subtask is a straightforward,
        deterministic call to a specific `Tool` (e.g., download a file) or if it
        requires more complex reasoning or multiple steps better handled by a
        probabilistic `Agent` executor (e.g., summarize analysis findings).
    3.  **Data Flow & Dependencies:** Explicitly defining the inputs (`input_files`)
        and `output_file` for each subtask. Crucially, it
        establishes the dependency graph, ensuring subtasks run only after their
        required inputs are available. This forms a Directed Acyclic Graph (DAG).
    4.  **Contracts:** Defining the expected data format (`output_schema`) for each subtask's output, promoting type safety and
        predictable integration between steps.
    5.  **Workspace Management:** Enforcing the use of *relative* file paths within
        a defined workspace, preventing dangerous absolute path manipulations and
        ensuring plan portability. Paths like `/tmp/foo` or `C:\\Users\\...` are
        strictly forbidden; only paths like `data/interim_results.csv` are valid.

    The planning process itself can involve multiple phases (configurable):
    - **Analysis Phase:** High-level strategic breakdown and identification of
      subtask types (Tool vs. Agent).
    - **Data Flow Analysis:** Refining dependencies, inputs/outputs, and data schemas.
    - **Plan Creation:** Generating the final, concrete `Task` object using the LLM,
      typically by invoking an internal `CreateTaskTool` or leveraging structured
      output capabilities if supported by the LLM provider.

    **Core Responsibility:** To transform an ambiguous user objective into an unambiguous,
    validated, and machine-executable plan. It prioritizes structure, clear contracts,
    and dependency management over monolithic, error-prone LLM interactions. This
    structured approach is essential for reliable, efficient, and debuggable AI workflows.

    **Validation is paramount:** The planner rigorously checks the generated plan for:
    - Cyclic dependencies (fatal).
    - Missing input files.
    - Correct `tool_name` usage and valid JSON arguments for Tool tasks.
    - Correct `content` format (natural language instructions) for Agent tasks.
    - Valid and relative file paths.
    - Schema consistency.

    A plan that fails validation is rejected, forcing the LLM to correct its mistakes.
    This avoids garbage-in, garbage-out execution downstream.

    Attributes:
        provider (ChatProvider): The LLM provider instance used for generation.
        model (str): The specific LLM model identifier.
        objective (str): The high-level goal the plan aims to achieve.
        workspace_dir (str): The root directory for all relative file paths.
        input_files (List[str]): Initial files available at the start of the plan.
        execution_tools (Sequence[Tool]): Tools available for subtasks designated as Tool tasks
                                         during the Plan Creation phase.
        task_plan (Optional[TaskPlan]): The generated plan (populated after creation).
        system_prompt (str): The core instructions guiding the LLM planner.
        output_schema (Optional[dict]): Optional schema for the *final* output of the
                                        overall task (not individual subtasks).
        enable_analysis_phase (bool): Controls whether the analysis phase runs.
        enable_data_contracts_phase (bool): Controls whether the data contract phase runs.
        verbose (bool): Enables detailed logging and progress display during planning.
        display_manager (AgentConsole): Handles Rich display output.
        jinja_env (Environment): Jinja2 environment for rendering prompts.
        tasks_file_path (Path): Path where the plan might be saved/loaded (`tasks.yaml`).
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        execution_tools: Sequence[Tool],
        reasoning_model: str | None = None,
        inputs: dict[str, Any] = {},
        system_prompt: str | None = None,
        output_schema: dict | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
        display_manager: AgentConsole | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            reasoning_model (str | None): The model to use for reasoning
            objective (str): The objective to solve
            workspace_dir (str): The workspace directory path
            execution_tools (List[Tool]): Tools available for subtask execution.
            inputs (dict[str, Any]): The inputs to use for planning
            system_prompt (str, optional): Custom system prompt
            output_schema (dict, optional): JSON schema for the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 0)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 1)
            verbose (bool, optional): Whether to print planning progress table (default: True)
        """
        # Note: debug logging will be handled by display_manager initialization

        self.provider: BaseProvider = provider
        self.model: str = model
        self.reasoning_model: str = reasoning_model or model
        self.objective: str = objective
        self.workspace_dir: str = workspace_dir
        self.task_plan: TaskPlan = TaskPlan()
        self.inputs: dict[str, Any] = inputs
        self.system_prompt: str = system_prompt or DEFAULT_PLANNING_SYSTEM_PROMPT
        self.execution_tools: Sequence[Tool] = execution_tools or []
        self.output_schema: Optional[dict] = output_schema
        self.enable_analysis_phase: bool = enable_analysis_phase
        self.enable_data_contracts_phase: bool = enable_data_contracts_phase
        self.verbose: bool = verbose
        self.tasks_file_path: Path = Path(workspace_dir) / "tasks.yaml"
        self.display_manager = display_manager
        self.jinja_env: Environment = Environment(loader=BaseLoader())

    async def _load_existing_plan(self) -> bool:
        """
        Try to load an existing task plan from the workspace.

        Returns:
            bool: True if plan was loaded successfully, False otherwise
        """
        if self.tasks_file_path.exists():
            try:
                with open(self.tasks_file_path, "r") as f:
                    task_plan_data: dict = yaml.safe_load(f)
                    self.task_plan = TaskPlan(**task_plan_data)
                    if self.display_manager:
                        log.debug(
                            "Loaded existing task plan from %s", self.tasks_file_path
                        )
                    return True
            except (
                Exception
            ) as e:  # Keep general exception for file I/O or parsing issues
                if self.display_manager:
                    log.debug(
                        "Could not load or parse existing task plan from %s: %s",
                        self.tasks_file_path,
                        e,
                    )
                return False
        return False

    def _get_prompt_context(self) -> Dict[str, str]:
        """Helper to build the context for Jinja2 prompt rendering.

        This method assembles a dictionary of common variables required by
        the Jinja2 prompt templates used in different planning phases.

        Returns:
            A dictionary containing key-value pairs for prompt templating,
            such as objective, model names, tool information, input file info,
            and overall output requirements.
        """
        # Provide default string representation if schema/type are None or not set
        overall_output_schema_str = (
            json.dumps(self.output_schema)
            if self.output_schema
            else "Not specified (default: string)"
        )

        # Format inputs information
        if self.inputs:
            inputs_info = "The following inputs are available:\n"
            for key, value in self.inputs.items():
                # Truncate long values for display
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                inputs_info += f"- {key}: {type(value).__name__} = {value_str}\n"
        else:
            inputs_info = "No inputs provided."

        return {
            "objective": self.objective,
            "model": self.model,  # Add planner's primary model
            "reasoning_model": self.reasoning_model,  # Add planner's reasoning model
            "execution_tools_info": self._get_execution_tools_info(),
            "output_schema": overall_output_schema_str,
            "inputs_info": inputs_info,
            "COMPACT_SUBTASK_NOTATION_DESCRIPTION": COMPACT_SUBTASK_NOTATION_DESCRIPTION,
        }

    def _render_prompt(
        self, template_string: str, context: Optional[Dict[str, str]] = None
    ) -> str:
        """Renders a prompt template using Jinja2.

        Args:
            template_string: The Jinja2 template string.
            context: An optional dictionary of context variables. If None,
                     `_get_prompt_context()` is used to get default context.

        Returns:
            The rendered prompt string.
        """
        if context is None:
            context = self._get_prompt_context()
        template = self.jinja_env.from_string(template_string)
        return template.render(context)

    async def _build_agent_task_prompt_content(self) -> str:
        """Builds the content for the agent task prompt using Jinja2.

        This prompt (`DEFAULT_AGENT_TASK_TEMPLATE`) provides final check
        guidelines to the LLM when it's constructing the `create_task` tool call.

        Returns:
            The rendered agent task prompt string.
        """
        return self._render_prompt(DEFAULT_AGENT_TASK_TEMPLATE)

    def _build_dependency_graph(self, subtasks: List[SubTask]) -> nx.DiGraph:
        """
            Build a directed graph of dependencies between subtasks.

        The graph nodes represent subtasks (identified by their `id`) and input keys.
            An edge from node A to subtask B means B depends on the output of A
            (i.e., one of B's `input_tasks` is A's `id` or an input key).

            Args:
                subtasks: A list of `SubTask` objects.

            Returns:
                A `networkx.DiGraph` representing the dependencies.
        """
        G = nx.DiGraph()

        # Add nodes representing subtasks
        for subtask in subtasks:
            G.add_node(subtask.id)  # Node represents the subtask completion

        # Add nodes representing input keys
        for input_key in self.inputs:
            G.add_node(input_key)  # Node represents an input

        # Add edges for dependencies
        for subtask in subtasks:
            if subtask.input_tasks:
                for dependency in subtask.input_tasks:
                    # Only add edge if the dependency exists (either as subtask or input)
                    if (
                        dependency in [t.id for t in subtasks]
                        or dependency in self.inputs
                    ):
                        G.add_edge(dependency, subtask.id)

        return G

    def _check_inputs(
        self,
        subtasks: List[SubTask],
    ) -> List[str]:
        """Checks if all input task dependencies for subtasks are available.

        An input task dependency is considered available if it:
        1. References a valid subtask ID within the current task plan, OR
        2. References a valid input key from the inputs dictionary

        Args:
            subtasks: A list of `SubTask` objects.

        Returns:
            A list of string error messages for any missing task dependencies.
        """
        validation_errors: List[str] = []
        tasks_by_id = {task.id: task for task in subtasks}

        for subtask in subtasks:
            if subtask.input_tasks:
                for dependency in subtask.input_tasks:
                    # Check if it's a valid subtask ID or an input key
                    if dependency not in tasks_by_id and dependency not in self.inputs:
                        validation_errors.append(
                            f"Subtask '{subtask.content}' depends on missing subtask or input '{dependency}'"
                        )
        return validation_errors

    def _validate_dependencies(self, subtasks: List[SubTask]) -> List[str]:
        """
        Validate task dependencies and DAG structure for subtasks.

        This method performs the following checks:
        1.  Cycle detection: Builds a dependency graph and checks for circular
            dependencies, which would make execution impossible.
        2.  Task availability: Verifies that all `input_tasks` for each
            subtask reference valid task IDs in the plan.
        3.  Topological sort feasibility: Checks if a valid linear execution
            order for the subtasks can be determined.

        Args:
            subtasks: A list of `SubTask` objects to validate.

        Returns:
            A list of strings, where each string is an error message
            describing a validation failure. An empty list indicates
            all dependency checks passed.
        """
        log.debug("Starting dependency validation for %d subtasks", len(subtasks))
        validation_errors: List[str] = []

        # Log subtask summary for debugging
        subtask_ids = [task.id for task in subtasks]
        log.debug("Subtask IDs to validate: %s", subtask_ids)

        for i, task in enumerate(subtasks):
            log.debug(
                "Subtask %d: id='%s', input_tasks=%s", i, task.id, task.input_tasks
            )

        # Build dependency graph
        log.debug("Building dependency graph")
        G = self._build_dependency_graph(subtasks)
        log.debug(
            "Dependency graph built with %d nodes and %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        # Log graph structure for debugging
        if G.number_of_edges() > 0:
            edges = list(G.edges())
            log.debug("Graph edges: %s", edges)

        # Check for cycles
        log.debug("Checking for circular dependencies")
        try:
            cycle = nx.find_cycle(G)
            cycle_error = f"Circular dependency detected: {cycle}"
            validation_errors.append(cycle_error)
            log.error("CRITICAL: Circular dependency found: %s", cycle)
        except nx.NetworkXNoCycle:
            log.debug("✓ No circular dependencies found")
            pass  # No cycles found, which is good

        # Check that all input task dependencies exist
        log.debug("Checking input task availability")
        input_errors = self._check_inputs(subtasks)
        validation_errors.extend(input_errors)
        if input_errors:
            log.error(
                "Input task dependency check found %d errors: %s",
                len(input_errors),
                input_errors,
            )
        else:
            log.debug("✓ All input task dependencies are valid")

        # Check if a valid execution order exists (topological sort)
        if not validation_errors:  # Only check if no other critical errors found
            log.debug("Checking topological sort feasibility")
            try:
                topo_order = list(nx.topological_sort(G))
                log.debug("✓ Valid execution order found: %s", topo_order)
            except nx.NetworkXUnfeasible:
                # This might be redundant if cycle check or input availability check failed,
                # but provides an extra layer of verification.
                error_msg = "Cannot determine valid execution order due to unresolved dependency issues (potentially complex cycle or missing input)."
                validation_errors.append(error_msg)
                log.error("CRITICAL: Topological sort failed: %s", error_msg)
        else:
            log.warning(
                "Skipping topological sort check due to %d existing validation errors",
                len(validation_errors),
            )

        if validation_errors:
            log.error(
                "❌ Dependency validation FAILED with %d total errors: %s",
                len(validation_errors),
                validation_errors,
            )
        else:
            log.debug(
                "✅ Dependency validation PASSED - all %d subtasks validated successfully",
                len(subtasks),
            )
        return validation_errors

    async def _run_phase(
        self,
        history: List[Message],
        phase_name: str,
        phase_display_name: str,
        is_enabled: bool,
        prompt_template: str,
        phase_result_name: str,
    ) -> tuple[List[Message], Optional[PlanningUpdate]]:
        """Generic method to run a planning phase.

        This method handles the common pattern for executing planning phases:
        - Checking if phase is enabled
        - Generating prompt using template
        - Updating display before/after LLM call
        - Calling the LLM
        - Formatting the response
        - Creating PlanningUpdate

        Args:
            history: The current list of messages in the planning conversation.
            phase_name: The name of the phase for logging (e.g., "analysis", "data flow").
            phase_display_name: The display name for UI (e.g., "0. Analysis", "1. Data Contracts").
            is_enabled: Whether this phase is enabled.
            prompt_template: The template string to render for the prompt.
            phase_result_name: The name to use in the PlanningUpdate (e.g., "Analysis", "Data Flow").

        Returns:
            A tuple containing:
                - The updated history with messages from this phase.
                - A `PlanningUpdate` object summarizing the outcome, or None if skipped.
        """
        if self.display_manager:
            self.display_manager.set_current_phase(phase_result_name)
        log.debug("Starting %s phase", phase_name)

        if not is_enabled:
            log.debug("Skipping %s phase: disabled by global flag", phase_name)
            if self.display_manager:
                self.display_manager.update_planning_display(
                    phase_display_name, "Skipped", "Phase disabled by global flag."
                )
            return history, None

        log.debug("Generating %s prompt", phase_name)
        prompt_content: str = self._render_prompt(prompt_template)
        log.debug(
            "%s prompt generated, length: %d chars",
            phase_name.capitalize(),
            len(prompt_content),
        )
        history.append(Message(role="user", content=prompt_content))

        # Update display before LLM call
        if self.display_manager:
            self.display_manager.update_planning_display(
                phase_display_name, "Running", f"Generating {phase_name}..."
            )

        log.debug("Calling LLM for %s using model: %s", phase_name, self.model)
        response_message: Message = await self.provider.generate_message(
            messages=history, model=self.model, tools=[]  # Explicitly empty list
        )
        history.append(response_message)
        log.debug(
            "%s phase LLM response received, content length: %d chars",
            phase_name.capitalize(),
            (len(str(response_message.content)) if response_message.content else 0),
        )

        phase_status: str = "Completed"
        phase_content: str | Text = self._format_message_content(response_message)
        log.debug(
            "%s phase completed with status: %s", phase_name.capitalize(), phase_status
        )

        if self.display_manager:
            self.display_manager.update_planning_display(
                phase_display_name, phase_status, phase_content
            )

        planning_update = PlanningUpdate(
            phase=phase_result_name,
            status=phase_status,
            content=str(phase_content),
        )

        return history, planning_update

    async def _run_analysis_phase(
        self, history: List[Message]
    ) -> tuple[List[Message], Optional[PlanningUpdate]]:
        """Handles Phase 0: Analysis.

        In this phase, the LLM interprets the user's objective, clarifies
        understanding, and devises a high-level strategic plan by breaking
        down the objective into conceptual subtasks.

        This phase can be skipped if:
        - `self.enable_analysis_phase` is False.
        # Tier-based skipping is removed as Analysis runs for both Low and High complexity.

        Args:
            history: The current list of messages in the planning conversation.
                     The LLM's prompt and response for this phase are appended.

        Returns:
            A tuple containing:
                - The updated history with messages from this phase.
                - A `PlanningUpdate` object summarizing the outcome, or None if skipped.
        """
        return await self._run_phase(
            history=history,
            phase_name="analysis",
            phase_display_name="0. Analysis",
            is_enabled=self.enable_analysis_phase,
            prompt_template=ANALYSIS_PHASE_TEMPLATE,
            phase_result_name="Analysis",
        )

    async def _run_data_flow_phase(
        self, history: List[Message]
    ) -> tuple[List[Message], Optional[PlanningUpdate]]:
        """Handles Phase 1: Data Flow Analysis.

        This phase refines the conceptual subtask plan from Phase 0 by
        defining precise data flow, dependencies (input/output files),
        and contracts for each subtask. It may also involve the LLM
        generating a DOT/Graphviz representation of the data flow.

        This phase can be skipped if:
        - `self.enable_data_contracts_phase` is False.

        Args:
            history: The current list of messages in the planning conversation.
                     The LLM's prompt and response for this phase are appended.

        Returns:
            A tuple containing:
                - The updated history with messages from this phase.
                - A `PlanningUpdate` object summarizing the outcome, or None if skipped.
        """
        return await self._run_phase(
            history=history,
            phase_name="data flow",
            phase_display_name="1. Data Contracts",
            is_enabled=self.enable_data_contracts_phase,
            prompt_template=DATA_FLOW_ANALYSIS_TEMPLATE,
            phase_result_name="Data Flow",
        )

    async def _run_plan_creation_phase(
        self,
        history: List[Message],
        objective: str,
        max_retries: int,
    ) -> tuple[Optional[Task], Optional[Exception], Optional[PlanningUpdate]]:
        """Handles Phase 2: Plan Creation.

        This is the final planning phase where the LLM generates the concrete,
        executable task plan. It does this by making a call to the
        `create_task` tool.

        The complexity of the plan generated depends on the information
        gathered in preceding phases (Analysis, Data Flow).

        Args:
            history: The list of messages from previous planning phases.
                     The LLM's prompt and response for this phase are appended.
            objective: The original user objective, used as a fallback title.
            max_retries: The maximum number of retries for LLM generation if
                         initial attempts fail validation or don't produce
                         the expected output.

        Returns:
            A tuple containing:
                - The generated `Task` object if successful, otherwise None.
                - An `Exception` object if an error occurred during plan creation,
                  otherwise None.
                - A `PlanningUpdate` object summarizing the outcome of this phase.
        """
        if self.display_manager:
            self.display_manager.debug(
                f"Starting plan creation phase with max_retries={max_retries}"
            )

        task: Optional[Task] = None
        final_message: Optional[Message] = None
        plan_creation_error: Optional[Exception] = None
        phase_status: str = "Failed"
        phase_content: str | Text = "N/A"
        current_phase_name: str = "2. Plan Creation"

        # Adjust prompt/logic for Tier 1 if necessary, e.g., by using a simpler template
        # or adding specific instructions to the existing template for single-task plans.
        # For now, we use the same template but the LLM will have less preceding context from skipped phases.
        # The main effect of Tier 1 will be the lack of Analysis and Data Flow history.

        # Use tool-based generation
        if self.display_manager:
            self.display_manager.debug("Using tool-based generation for plan creation")
        plan_creation_prompt_content = self._render_prompt(PLAN_CREATION_TEMPLATE)
        agent_task_prompt_content = await self._build_agent_task_prompt_content()
        if self.display_manager:
            self.display_manager.debug(
                f"Plan creation prompt length: {len(plan_creation_prompt_content)} chars"
            )
        if self.display_manager:
            self.display_manager.debug(
                f"Agent task prompt length: {len(agent_task_prompt_content)} chars"
            )

        history.append(
            Message(
                role="user",
                content=f"{plan_creation_prompt_content}\n{agent_task_prompt_content}",
            )
        )
        if self.display_manager:
            self.display_manager.update_planning_display(
                current_phase_name,
                "Running",
                "Attempting plan creation using the 'create_task' tool...",
            )
        try:
            log.debug("Starting tool-based plan generation with retry logic")
            task, final_message = await self._generate_with_retry(
                history,
                tools=[CreateTaskTool()],
                max_retries=max_retries,
            )

            if task:
                phase_status = "Success"
                phase_content = (
                    self._format_message_content(final_message)
                    if final_message
                    else "Plan created using tool calls."
                )
                log.debug(
                    "Tool-based plan creation successful: %d subtasks",
                    len(task.subtasks),
                )
            else:
                failure_reason = "Unknown failure after retries."
                if final_message and (
                    final_message.content or final_message.tool_calls
                ):
                    formatted_content = self._format_message_content(final_message)
                    if (
                        "error" in str(formatted_content).lower()
                        or "fail" in str(formatted_content).lower()
                    ):
                        failure_reason = f"LLM indicated failure: {formatted_content}"
                    else:
                        failure_reason = f"LLM did not produce a valid 'create_task' tool call. Last message: {formatted_content}"
                elif (
                    plan_creation_error
                ):  # Check if _generate_with_retry raised an error internally
                    failure_reason = (
                        f"Tool call generation failed internally: {plan_creation_error}"
                    )

                log.warning("Tool-based plan creation failed: %s", failure_reason)
                plan_creation_error = ValueError(
                    f"Tool call generation failed: {failure_reason}"
                )
                phase_content = f"Tool call generation failed: {failure_reason}"
                phase_status = "Failed"
        except Exception as e:
            log.error("Tool-based plan creation failed: %s", e, exc_info=True)
            plan_creation_error = e
            phase_status = "Failed"
            phase_content = f"Tool call generation failed: {str(e)}\n{traceback.format_exc()}"  # Keep traceback for display

        # Update Table for Phase 2
        log.debug("Plan creation phase completed with status: %s", phase_status)
        if self.display_manager:
            self.display_manager.update_planning_display(
                current_phase_name,
                phase_status,
                Text(
                    str(phase_content),
                    style=("bold red" if phase_status == "Failed" else "default"),
                ),  # Use Text with style
                is_error=(phase_status == "Failed"),
            )

        planning_update = PlanningUpdate(
            phase="Plan Creation",
            status=phase_status,
            content=str(phase_content),  # Update uses plain string
        )

        return task, plan_creation_error, planning_update

    async def create_task(
        self,
        context: ProcessingContext,
        objective: str,
        max_retries: int = 3,
    ) -> AsyncGenerator[Chunk | PlanningUpdate, None]:
        """
        Create subtasks using the configured planning process, allowing for early shortcuts.
        Yields PlanningUpdate events during the process.
        Displays a live table summarizing the planning process if verbose mode is enabled.
        """
        log.debug(
            "Starting task creation for objective: '%s...' with max_retries=%d",
            objective[:100],
            max_retries,
        )

        # Start the live display using the display manager
        if self.display_manager:
            self.display_manager.start_live(
                self.display_manager.create_planning_tree("Task Planner")
            )

        history: List[Message] = [
            Message(role="system", content=self.system_prompt),
        ]

        error_message: Optional[str] = None
        plan_creation_error: Optional[Exception] = None
        task: Optional[Task] = None
        current_phase = "Initialization"

        try:
            if self.display_manager:
                self.display_manager.set_current_phase("Initialization")
            log.debug("Starting planning phases")

            # Phase 0: Analysis
            current_phase = "Analysis"
            if self.display_manager:
                self.display_manager.set_current_phase(current_phase)
            log.debug("Entering phase: %s", current_phase)
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            history, planning_update = await self._run_analysis_phase(history)
            if planning_update:
                yield planning_update

            # Phase 1: Data Flow Analysis
            current_phase = "Data Flow"
            if self.display_manager:
                self.display_manager.set_current_phase(current_phase)
            log.debug("Entering phase: %s", current_phase)
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            history, planning_update = await self._run_data_flow_phase(history)
            if planning_update:
                yield planning_update

            # Phase 2: Plan Creation
            current_phase = "Plan Creation"
            if self.display_manager:
                self.display_manager.set_current_phase(current_phase)
            log.debug("Entering phase: %s", current_phase)
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            task, plan_creation_error, planning_update = (
                await self._run_plan_creation_phase(history, objective, max_retries)
            )
            if planning_update:
                yield planning_update  # Yield the update from the phase itself

            # --- Final Outcome ---
            if task:
                log.debug(
                    "Plan created successfully with %d subtasks", len(task.subtasks)
                )
                if self.display_manager:
                    log.debug("Plan created successfully.")
                else:
                    log.debug("Plan created successfully.")
                self.task_plan.tasks.append(task)
            else:
                # Construct error message based on plan_creation_error or last message
                if plan_creation_error:
                    error_message = f"Failed to create valid task during Plan Creation phase. Original error: {str(plan_creation_error)}"
                    full_error_message = (
                        f"{error_message}\n{traceback.format_exc()}"
                        if self.verbose
                        else error_message
                    )
                    log.error("Task creation failed: %s", error_message)
                    # Yield failure update before raising
                    yield PlanningUpdate(
                        phase=current_phase, status="Failed", content=error_message
                    )
                    # Update display for overall failure
                    if self.display_manager:
                        self.display_manager.update_planning_display(
                            "Overall Status",
                            "Failed",
                            Text(full_error_message, style="bold red"),
                            is_error=True,
                        )
                    raise ValueError(full_error_message) from plan_creation_error
                else:
                    error_message = "Failed to create valid task after maximum retries in Plan Creation phase for an unknown reason."
                    log.error("Task creation failed: %s", error_message)
                    # Yield failure update before raising
                    yield PlanningUpdate(
                        phase=current_phase, status="Failed", content=error_message
                    )
                    # Update display for overall failure
                    if self.display_manager:
                        self.display_manager.update_planning_display(
                            "Overall Status",
                            "Failed",
                            Text(error_message, style="bold red"),
                            is_error=True,
                        )
                    raise ValueError(error_message)

        except Exception as e:
            # Capture the original exception type and message
            error_message = f"Planning failed during phase '{current_phase}': {type(e).__name__}: {str(e)}"
            log.error(
                "Task creation failed during %s: %s",
                current_phase,
                e,
                exc_info=True,
            )

            # Log traceback if verbose
            if self.verbose:
                if self.display_manager:
                    self.display_manager.print_exception(show_locals=False)
                else:
                    log.exception("Planning exception (show_locals=False)")

            # Add error row to table via display manager
            if error_message:
                if self.display_manager:
                    self.display_manager.update_planning_display(
                        "Overall Status",
                        "Failed",
                        Text(
                            f"{error_message}\n{traceback.format_exc() if self.verbose else ''}",
                            style="bold red",
                        ),
                        is_error=True,
                    )
            # Print error to console otherwise (handled by display_manager if verbose is off)
            if self.display_manager:
                log.debug("Planning Error: %s", error_message)
            else:
                log.error("Planning Error: %s", error_message)

            # Yield failure update before re-raising
            yield PlanningUpdate(
                phase=current_phase, status="Failed", content=error_message
            )
            raise  # Re-raise the caught exception

        finally:
            log.debug("Stopping live display and completing task creation")
            # Stop the live display using the display manager
            if self.display_manager:
                self.display_manager.stop_live()

    def _remove_think_tags(self, text_content: Optional[str]) -> Optional[str]:
        """Removes <think>...</think> blocks from a string.

        Args:
            text_content: The string to process.

        Returns:
            The string with <think> blocks removed, or None if input was None.
        """
        if text_content is None:
            return None
        # Use regex to remove <think>...</think> blocks, including newlines within them.
        # re.DOTALL makes . match newlines.
        # We also strip leading/trailing whitespace from the result.
        return re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()

    def _format_message_content(self, message: Optional[Message]) -> str | Text:
        """Formats message content for table display.

        Handles `None` messages, summarizes tool calls, and cleans `<think>`
        tags from textual content.

        Args:
            message: The `Message` object to format. Can be None.

        Returns:
            A `rich.text.Text` object suitable for display, or a plain string
            for tool call summaries (though Text is preferred).
            Returns a default Text object if the message is None or content is empty.
        """
        if not message:
            return Text("No response message.", style="dim")

        if message.tool_calls:
            # Summarize tool calls
            calls_summary: List[str] = []
            for tc in message.tool_calls:
                # Truncate args if too long for table display
                args_str = json.dumps(tc.args)
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                calls_summary.append(f"- Tool Call: {tc.name}\\n  Args: {args_str}")
            # Use Text object for potential future styling
            return Text(
                "\\n".join(calls_summary)
            )  # No <think> tag removal for tool call summaries

        raw_content_str: Optional[str] = None
        if message.content:
            if isinstance(message.content, list):
                # Attempt to join list items; handle potential non-string items
                try:
                    raw_content_str = "\\n".join(str(item) for item in message.content)
                except Exception:
                    raw_content_str = str(
                        message.content
                    )  # Fallback to string representation of the list
            elif isinstance(message.content, str):
                raw_content_str = message.content
            else:
                # Handle other unexpected content types
                raw_content_str = (
                    f"Unexpected content type: {type(message.content).__name__}"
                )

        cleaned_content: Optional[str] = self._remove_think_tags(raw_content_str)

        if cleaned_content:  # If cleaned_content is not None and not empty
            return Text(cleaned_content)
        elif (
            raw_content_str is not None
        ):  # Original content existed but was all <think> tags or whitespace
            return Text("")  # Display as empty, not "Empty message content"
        else:  # No message.content to begin with
            return Text("Empty message content.", style="dim")

    def _format_message_content_for_update(
        self, message: Optional[Message]
    ) -> Optional[str]:
        """Formats message content into a simple string for PlanningUpdate.

        This method is similar to `_format_message_content` but specifically
        targets the `content` attribute of a `Message` and returns a plain
        string, primarily for use in `PlanningUpdate.content`. It removes
        `<think>` tags.

        Args:
            message: The `Message` object. Can be None.

        Returns:
            A string representation of the message content after cleaning,
            or None if the message or its content is None/empty.
        """
        if not message:
            return None

        raw_str_content: Optional[str] = None
        if message.content:  # This method primarily processes .content
            if isinstance(message.content, list):
                try:
                    raw_str_content = "\\n".join(str(item) for item in message.content)
                except Exception:
                    raw_str_content = str(message.content)  # Fallback
            elif isinstance(message.content, str):
                raw_str_content = message.content
            else:
                raw_str_content = (
                    f"Unexpected content type: {type(message.content).__name__}"
                )

        return self._remove_think_tags(raw_str_content)

    def _validate_tool_task(
        self,
        subtask_data: dict,
        tool_name: str,
        content: Any,
        available_execution_tools: Dict[str, Tool],
        sub_context: str,
    ) -> tuple[Optional[dict], List[str]]:
        """Validates a subtask intended as a direct tool call.

        This involves:
        1.  Checking if the specified `tool_name` is among the available
            `execution_tools`.
        2.  Ensuring the `content` (expected to be tool arguments) is valid JSON.
        3.  Validating the parsed JSON arguments against the tool's `input_schema`
            using `jsonschema.validate`.

        Args:
            subtask_data: The raw dictionary data for the subtask.
            tool_name: The name of the tool to be called.
            content: The content for the subtask, expected to be JSON arguments
                     for the tool.
            available_execution_tools: A dictionary mapping tool names to `Tool` objects.
            sub_context: A string prefix for error messages (e.g., "Subtask 1").

        Returns:
            A tuple containing:
                - A dictionary of the parsed and validated tool arguments if
                  validation is successful, otherwise None.
                - A list of string error messages encountered during validation.
        """
        log.debug(
            "%s: Starting tool task validation for tool '%s'", sub_context, tool_name
        )
        validation_errors: List[str] = []
        parsed_content: Optional[dict] = None

        # Check if tool exists
        if tool_name not in available_execution_tools:
            error_msg = f"Specified tool_name '{tool_name}' is not in the list of available execution tools: {list(available_execution_tools.keys())}."
            log.error("%s: %s", sub_context, error_msg)
            validation_errors.append(f"{sub_context}: {error_msg}")
            return None, validation_errors

        tool_to_use = available_execution_tools[tool_name]
        log.debug(
            "%s: Tool '%s' found, validating content type: %s",
            sub_context,
            tool_name,
            type(content),
        )

        # Validate content is JSON and parse it
        if isinstance(content, dict):
            parsed_content = content  # Already a dict
            log.debug(
                "%s: Content is already a dict with %d keys", sub_context, len(content)
            )
        elif isinstance(content, str):
            log.debug(
                "%s: Parsing JSON string content of length %d",
                sub_context,
                len(content),
            )
            try:
                parsed_content = json.loads(content) if content.strip() else {}
                if parsed_content is not None:
                    log.debug(
                        "%s: Successfully parsed JSON with %d keys: %s",
                        sub_context,
                        len(parsed_content),
                        list(parsed_content.keys()),
                    )
                else:
                    log.debug("%s: Parsed content is None", sub_context)
            except json.JSONDecodeError as e:
                error_msg = (
                    f"'content' is not valid JSON. Error: {e}. Content: '{content}'"
                )
                log.error("%s (tool: %s): %s", sub_context, tool_name, error_msg)
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): {error_msg}"
                )
                return None, validation_errors
        else:
            error_msg = f"Expected JSON string or object for tool arguments, but got {type(content)}. Content: '{content}'"
            log.error("%s (tool: %s): %s", sub_context, tool_name, error_msg)
            validation_errors.append(f"{sub_context} (tool: {tool_name}): {error_msg}")
            return None, validation_errors

        # Validate the parsed content against the tool's input schema
        if tool_to_use.input_schema and parsed_content is not None:
            log.debug("%s: Validating parsed content against tool schema", sub_context)
            try:
                validate(instance=parsed_content, schema=tool_to_use.input_schema)
                log.debug(
                    "%s: Tool arguments validation successful for '%s'",
                    sub_context,
                    tool_name,
                )
            except ValidationError as e:
                error_msg = f"JSON arguments in 'content' do not match the tool's input schema. Error: {e.message}. Path: {'/'.join(map(str, e.path))}. Schema: {e.schema}. Args: {parsed_content}"
                log.error(
                    "%s (tool: %s): Schema validation failed: %s",
                    sub_context,
                    tool_name,
                    error_msg,
                )
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): {error_msg}"
                )
                return None, validation_errors
            except Exception as e:  # Catch other potential validation errors
                error_msg = f"Error validating arguments against tool schema. Error: {e}. Args: {parsed_content}"
                log.error(
                    "%s (tool: %s): Unexpected validation error: %s",
                    sub_context,
                    tool_name,
                    error_msg,
                )
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): {error_msg}"
                )
                return None, validation_errors
        else:
            log.debug(
                "%s: No input schema defined for tool '%s', skipping schema validation",
                sub_context,
                tool_name,
            )

        log.debug(
            "%s: Tool task validation completed successfully for '%s'",
            sub_context,
            tool_name,
        )
        return parsed_content, validation_errors

    def _validate_agent_task(self, content: Any, sub_context: str) -> List[str]:
        """Validates a subtask intended for agent execution.

        Ensures that the `content` for an agent task (which represents
        natural language instructions) is a non-empty string.

        Args:
            content: The content of the subtask (instructions for the agent).
            sub_context: A string prefix for error messages (e.g., "Subtask 1").

        Returns:
            A list of string error messages. An empty list means validation passed.
        """
        log.debug("%s: Starting agent task validation", sub_context)
        validation_errors: List[str] = []

        log.debug(
            "%s: Validating content type: %s, length: %d",
            sub_context,
            type(content),
            (len(str(content)) if content else 0),
        )

        if not isinstance(content, str) or not content.strip():
            error_msg = f"'content' must be a non-empty string containing instructions when 'tool_name' is not provided, but got: '{content}' (type: {type(content)})."
            log.error("%s: Agent task validation failed: %s", sub_context, error_msg)
            validation_errors.append(f"{sub_context}: {error_msg}")
        else:
            log.debug(
                "%s: Agent task validation successful - content is valid string with %d characters",
                sub_context,
                len(content),
            )

        return validation_errors

    def _process_subtask_schema(
        self, subtask_data: dict, sub_context: str
    ) -> tuple[Optional[str], List[str]]:
        """Processes and validates the output_schema for a subtask.

        Args:
            subtask_data: The raw dictionary data for the subtask
            sub_context: A string prefix for error messages.

        Returns:
            A tuple containing:
                - The processed and validated `output_schema` as a JSON string,
                  or None if a fatal error occurred.
                - A list of string error messages encountered.
        """
        log.debug("%s: Starting schema processing", sub_context)
        validation_errors: List[str] = []
        current_schema_str: Any = subtask_data.get("output_schema")
        final_schema_str: Optional[str] = None
        schema_dict: Optional[dict] = None

        # Add logging for the input schema string
        log.debug(
            "%s: schema_input='%s' (type: %s)",
            sub_context,
            current_schema_str,
            type(current_schema_str),
        )
        log.debug(
            "%s: Attempting to process output_schema: '%s' of type %s",
            sub_context,
            current_schema_str,
            type(current_schema_str),
        )

        try:
            if isinstance(current_schema_str, str) and current_schema_str.strip():
                log.debug("%s: Parsing string schema", sub_context)
                log.debug(
                    "%s: Parsing string schema: '%s'", sub_context, current_schema_str
                )
                schema_dict = json.loads(current_schema_str)
                log.debug(
                    "%s: Successfully parsed schema dict: %s", sub_context, schema_dict
                )
            else:  # Invalid type for schema string
                error_msg = f"Output schema must be a JSON string or None, got {type(current_schema_str)}"
                log.error("%s: %s", sub_context, error_msg)
                raise ValueError(error_msg)

            # Apply defaults if schema_dict was successfully loaded or generated
            if schema_dict is not None:
                log.debug(
                    "%s: Applying additionalProperties constraints to schema",
                    sub_context,
                )
                schema_dict = self._ensure_additional_properties_false(schema_dict)
                final_schema_str = json.dumps(schema_dict)
                log.debug(
                    "%s: Final schema prepared, length=%d: %s",
                    sub_context,
                    len(final_schema_str),
                    final_schema_str,
                )

        except (ValueError, json.JSONDecodeError) as e:
            error_msg = f"Invalid output_schema provided: '{current_schema_str}'. Error: {e}. Using default string schema."
            validation_errors.append(f"{sub_context}: {error_msg}")
            log.warning("%s: Schema parsing failed: %s", sub_context, e)
            # Log the specific error
            log.debug(
                "%s: JSONDecodeError or ValueError for schema '%s': %s",
                sub_context,
                current_schema_str,
                e,
            )

        log.debug(
            "%s: Schema processing completed, errors=%d, final_schema_str=%s",
            sub_context,
            len(validation_errors),
            final_schema_str is not None,
        )
        return final_schema_str, validation_errors

    def _prepare_subtask_data(
        self,
        subtask_data: dict,
        final_schema_str: str,
        parsed_tool_content: Optional[dict],
        sub_context: str,
    ) -> tuple[Optional[dict], List[str]]:
        """Prepares the final data dictionary for SubTask creation.

        This method takes the raw subtask data, the validated `final_schema_str`,
        and potentially parsed tool content, then performs:
        1.  Validates input_tasks references to ensure they exist.
        2.  Ensures `tool_name` is None if it was an empty string.
        3.  Handles default model assignment for the subtask.
        4.  Filters the data dictionary to include only fields recognized by the
            `SubTask` Pydantic model.
        5.  Stringifies `parsed_tool_content` back into the `content` field if
            it was a tool task.

        Args:
            subtask_data: The raw dictionary data for the subtask.
            final_schema_str: The validated output schema as a JSON string.
            parsed_tool_content: Parsed JSON arguments if it's a tool task, else None.
            sub_context: A string prefix for error messages.

        Returns:
            A tuple containing:
                - A dictionary ready for `SubTask` model instantiation, or None
                  if a fatal error occurred.
                - A list of string error messages.
        """
        validation_errors: List[str] = []
        processed_data = subtask_data.copy()  # Work on a copy

        try:
            processed_data["output_schema"] = (
                final_schema_str  # Already validated/generated
            )

            # Validate input_tasks
            raw_input_tasks = processed_data.get("input_tasks", [])
            if not isinstance(raw_input_tasks, list):
                validation_errors.append(
                    f"{sub_context}: input_tasks must be a list, got {type(raw_input_tasks)}."
                )
                processed_data["input_tasks"] = []
            else:
                # Ensure all items in input_tasks are strings
                validated_input_tasks = []
                for task_id in raw_input_tasks:
                    if not isinstance(task_id, str):
                        validation_errors.append(
                            f"{sub_context}: Input task ID '{task_id}' must be a string, got {type(task_id)}."
                        )
                    else:
                        validated_input_tasks.append(task_id)
                processed_data["input_tasks"] = validated_input_tasks

            # Ensure tool_name is None if empty string or missing
            processed_data["tool_name"] = processed_data.get("tool_name") or None

            # Handle optional model assignment
            subtask_model = processed_data.get("model")
            if not isinstance(subtask_model, str) or not subtask_model.strip():
                processed_data["model"] = (
                    self.model
                )  # Default to planner's primary model
            else:
                processed_data["model"] = subtask_model.strip()

            # Filter args based on SubTask model fields
            subtask_model_fields = SubTask.model_fields.keys()
            filtered_data = {
                k: v for k, v in processed_data.items() if k in subtask_model_fields
            }

            # Stringify content if it was a parsed JSON object (for tool args)
            if isinstance(parsed_tool_content, dict):
                # Ensure the original content key exists before assignment
                if "content" in filtered_data:
                    filtered_data["content"] = json.dumps(parsed_tool_content)
                else:
                    # This case might indicate an issue if content was expected but not provided/filtered
                    validation_errors.append(
                        f"{sub_context}: Content field missing after filtering, cannot stringify tool arguments."
                    )
                    return None, validation_errors
            elif isinstance(filtered_data.get("content"), str):
                pass  # Keep agent task content as string

            return filtered_data, validation_errors

        except Exception as e:  # Catch unexpected errors during preparation
            validation_errors.append(
                f"{sub_context}: Unexpected error preparing subtask data: {e}"
            )
            return None, validation_errors

    async def _process_single_subtask(
        self,
        subtask_data: dict,
        index: int,
        context_prefix: str,
        available_execution_tools: Dict[str, Tool],
    ) -> tuple[Optional[SubTask], List[str]]:
        """
        Processes and validates data for a single subtask by delegating steps.
        """
        sub_context = f"{context_prefix} subtask {index}"
        log.debug("Processing %s", sub_context)
        all_validation_errors: List[str] = []
        parsed_tool_content: Optional[dict] = (
            None  # To store parsed JSON for tool tasks
        )

        try:
            # --- Validate Tool Call vs Agent Instruction ---
            tool_name = subtask_data.get("tool_name")
            content = subtask_data.get("content")
            log.debug(
                "%s: tool_name='%s', content_length=%d",
                sub_context,
                tool_name,
                (len(str(content)) if content else 0),
            )

            if tool_name:
                # --- Deterministic Tool Task Validation ---
                log.debug("%s: Validating as tool task", sub_context)
                parsed_tool_content, tool_errors = self._validate_tool_task(
                    subtask_data,
                    tool_name,
                    content,
                    available_execution_tools,
                    sub_context,
                )
                all_validation_errors.extend(tool_errors)
                if (
                    parsed_tool_content is None and tool_errors
                ):  # Fatal error during tool validation
                    log.error(
                        "%s: Tool validation failed with %d errors",
                        sub_context,
                        len(tool_errors),
                    )
                    return None, all_validation_errors
                else:
                    log.debug("%s: Tool validation successful", sub_context)
            else:
                # --- Probabilistic Agent Task Validation ---
                log.debug("%s: Validating as agent task", sub_context)
                agent_errors = self._validate_agent_task(content, sub_context)
                all_validation_errors.extend(agent_errors)
                if agent_errors:  # Fatal error during agent validation
                    log.error(
                        "%s: Agent validation failed with %d errors",
                        sub_context,
                        len(agent_errors),
                    )
                    return None, all_validation_errors
                else:
                    log.debug("%s: Agent validation successful", sub_context)

            # --- Process schema ---
            log.debug("%s: Processing output schema", sub_context)
            final_schema_str, schema_errors = self._process_subtask_schema(
                subtask_data, sub_context
            )
            all_validation_errors.extend(schema_errors)
            # Continue even if there were schema errors, as a default might be used.
            # Need final_schema_str for data preparation. If None, indicates a fatal schema issue.
            if final_schema_str is None:
                log.error("%s: Fatal schema processing error", sub_context)
                all_validation_errors.append(
                    f"{sub_context}: Fatal error processing output schema."
                )
                return None, all_validation_errors
            else:
                log.debug("%s: Schema processing successful", sub_context)

            # --- Prepare data for SubTask creation (Paths, Filtering, Stringify Tool Args) ---
            log.debug("%s: Preparing data for SubTask creation", sub_context)
            filtered_data, preparation_errors = self._prepare_subtask_data(
                subtask_data, final_schema_str, parsed_tool_content, sub_context
            )
            all_validation_errors.extend(preparation_errors)
            if filtered_data is None:  # Fatal error during data preparation
                log.error(
                    "%s: Data preparation failed with %d errors",
                    sub_context,
                    len(preparation_errors),
                )
                return None, all_validation_errors
            else:
                log.debug("%s: Data preparation successful", sub_context)

            # --- Create SubTask object ---
            # Pydantic validation happens here
            log.debug("%s: Creating SubTask object", sub_context)
            subtask = SubTask(**filtered_data)
            log.debug(
                "%s: SubTask created successfully with id='%s'", sub_context, subtask.id
            )
            # Return successful subtask and any *non-fatal* validation errors collected
            return subtask, all_validation_errors

        except (
            ValidationError
        ) as e:  # Catch Pydantic validation errors during SubTask(**filtered_data)
            error_msg = f"{sub_context}: Invalid data for SubTask model: {e}"
            log.error("%s: Pydantic validation error: %s", sub_context, e)
            all_validation_errors.append(error_msg)
            return None, all_validation_errors
        except Exception as e:  # Catch any other unexpected errors
            error_msg = f"{sub_context}: Unexpected error processing subtask: {e}\n{traceback.format_exc()}"
            log.error(
                "%s: Unexpected processing error: %s", sub_context, e, exc_info=True
            )
            all_validation_errors.append(error_msg)
            return None, all_validation_errors

    async def _process_subtask_list(
        self, raw_subtasks: list, context_prefix: str
    ) -> tuple[List[SubTask], List[str]]:
        """
        Processes a list of raw subtask data dictionaries using the helper method.
        """
        processed_subtasks: List[SubTask] = []
        all_validation_errors: List[str] = []
        # Build tool map once
        available_execution_tools: Dict[str, Tool] = {
            tool.name: tool for tool in self.execution_tools
        }

        for i, subtask_data in enumerate(raw_subtasks):
            sub_context = f"{context_prefix} subtask {i}"
            if not isinstance(subtask_data, dict):
                all_validation_errors.append(
                    f"{sub_context}: Expected subtask item to be a dict, but got {type(subtask_data)}. Data: {subtask_data}"
                )
                continue  # Skip this item

            # Call the helper to process this single subtask
            subtask, single_errors = await self._process_single_subtask(
                subtask_data, i, context_prefix, available_execution_tools
            )

            # Extend the list of errors collected (includes fatal and non-fatal)
            all_validation_errors.extend(single_errors)

            # Add the subtask ONLY if processing was successful (subtask is not None)
            if subtask:
                processed_subtasks.append(subtask)
            # If subtask is None, it means a fatal validation error occurred,
            # and the errors have already been added to all_validation_errors.

        return processed_subtasks, all_validation_errors

    async def _validate_structured_output_plan(
        self, task_data: dict, objective: str
    ) -> tuple[Optional[Task], List[str]]:
        """Validates the plan data received from structured output.

        This method is used when the LLM generates the plan as direct JSON
        output rather than through a tool call. It involves:
        1.  Processing the list of subtasks using `_process_subtask_list`.
        2.  Validating dependencies between the processed subtasks using
            `_validate_dependencies`.

        Args:
            task_data: A dictionary representing the entire task plan, typically
                       with "title" and "subtasks" keys, as generated by the LLM.
            objective: The original user objective, used as a fallback title.

        Returns:
            A tuple containing:
                - A `Task` object if the plan is valid, otherwise None.
                - A list of all validation error messages encountered.
        """
        all_validation_errors: List[str] = []

        # Validate the subtasks first
        subtasks, subtask_validation_errors = await self._process_subtask_list(
            task_data.get("subtasks", []), "structured output"
        )
        all_validation_errors.extend(subtask_validation_errors)

        log.debug("Subtasks processed: %s", subtasks)
        log.debug("Subtask validation errors: %s", subtask_validation_errors)

        # If subtask processing had fatal errors, don't proceed to dependency check
        if not subtasks and task_data.get(
            "subtasks"
        ):  # Check if subtasks were provided but processing failed
            # Errors are already in all_validation_errors
            return None, all_validation_errors

        # Validate dependencies only if subtasks were processed successfully
        if subtasks:
            dependency_errors = self._validate_dependencies(subtasks)
            all_validation_errors.extend(dependency_errors)

        # If any fatal errors occurred anywhere, return None
        if any(
            err for err in all_validation_errors
        ):  # Check if there are any errors at all
            # Check specifically for fatal errors that would prevent Task creation
            # (e.g., invalid structure from _process_subtask_list, dependency errors)
            # For simplicity here, consider *any* validation error as potentially blocking.
            # A more nuanced check could differentiate warnings from fatal errors if needed.
            is_fatal = True  # Assume any error is fatal for now
            if is_fatal:
                return None, all_validation_errors

        # If validation passed (or only non-fatal errors occurred)
        return (
            Task(
                title=task_data.get("title", objective),
                subtasks=subtasks,  # Use the processed and validated subtasks
            ),
            all_validation_errors,
        )  # Return task and any non-fatal errors

    async def _validate_and_build_task_from_tool_calls(
        self, tool_calls: List[ToolCall], history: List[Message]
    ) -> tuple[Optional[Task], List[str]]:
        """Processes 'create_task' tool calls, validates subtasks and dependencies.

        This method handles the arguments from one or more `create_task` tool
        calls made by the LLM. For each such call, it:
        1.  Extracts the task title and the list of raw subtask data.
        2.  Processes the raw subtasks using `_process_subtask_list` to convert
            them into `SubTask` objects and collect validation errors.
        3.  Appends a "tool" role message to the history acknowledging the call
            and summarizing any validation issues for that specific call.

        After processing all `create_task` calls, it:
        4.  Validates dependencies across *all* collected subtasks from *all* calls
            using `_validate_dependencies`.

        If any validation errors occur at any stage (either within a single
        subtask, a tool call's subtask list, or in the final dependency check),
        it raises a `ValueError` to trigger retry logic in the calling function
        (`_generate_with_retry`).

        Args:
            tool_calls: A list of `ToolCall` objects from the LLM's message.
            history: The current planning conversation history. Tool response
                     messages will be appended to this list.

        Returns:
            A tuple containing:
                - A `Task` object if all validations pass and subtasks exist.
                - An empty list of validation errors (as errors trigger an exception).

        Raises:
            ValueError: If any validation errors are found during the processing
                        of subtasks or overall plan dependencies, or if a
                        `create_task` call results in no valid subtasks.
        """
        all_subtasks: List[SubTask] = []
        all_validation_errors: List[str] = []
        task_title: str = self.objective  # Default title
        tool_responses_added: Set[str] = set()  # Track processed tool call IDs

        for tool_call in tool_calls:
            if tool_call.id in tool_responses_added:
                continue  # Already processed

            if tool_call.name == "create_task":
                context_prefix = f"processing tool call {tool_call.id}"
                # Extract title safely
                task_title = tool_call.args.get("title", task_title)

                # --- Process Subtasks using helper ---
                raw_subtasks_list = tool_call.args.get("subtasks", [])
                if not isinstance(raw_subtasks_list, list):
                    all_validation_errors.append(
                        f"{context_prefix}: 'subtasks' field must be a list, but got {type(raw_subtasks_list)}."
                    )
                    # Skip processing this call's subtasks if field is invalid, but record error
                    subtasks, validation_errors = [], []  # Ensure these are empty lists
                else:
                    # Use the updated _process_subtask_list
                    subtasks, validation_errors = await self._process_subtask_list(
                        raw_subtasks_list, context_prefix
                    )

                all_subtasks.extend(subtasks)
                all_validation_errors.extend(validation_errors)
                # --- End Subtask Processing ---

                # Add tool response message *after* processing its args
                response_content = "Task parameters received and processed."
                call_specific_errors = [
                    e for e in validation_errors if context_prefix in e
                ]
                if call_specific_errors:
                    response_content = f"Task parameters received, but validation errors occurred: {'; '.join(call_specific_errors)}"

                history.append(
                    Message(
                        role="tool", content=response_content, tool_call_id=tool_call.id
                    )
                )
                tool_responses_added.add(tool_call.id)  # Mark as processed

            else:
                # Handle unexpected tool calls
                error_msg = f"Unexpected tool call received: {tool_call.name}"
                all_validation_errors.append(error_msg)
                history.append(
                    Message(
                        role="tool",
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call.id,
                    )
                )
                tool_responses_added.add(tool_call.id)

        # --- Validate Dependencies *after* collecting all subtasks ---
        # Only run if there were subtasks and no fundamental structural errors earlier
        if all_subtasks and not any(
            "must be a list" in e for e in all_validation_errors
        ):
            dependency_errors = self._validate_dependencies(all_subtasks)
            all_validation_errors.extend(dependency_errors)
        # --- End Dependency Validation ---

        # Check if any errors occurred
        if all_validation_errors:
            # Raise ValueError to trigger retry logic in _generate_with_retry
            error_string = "\n".join(all_validation_errors)
            raise ValueError(f"Validation errors in created task:\n{error_string}")

        # Ensure we actually created subtasks if the call succeeded validation
        if not all_subtasks and any(tc.name == "create_task" for tc in tool_calls):
            # Check if create_task was called but resulted in no valid subtasks
            # This might happen if the subtasks list was empty or all items failed validation
            raise ValueError(
                "Task creation tool call processed, but resulted in zero valid subtasks."
            )

        # If validation passed and subtasks exist
        return (
            Task(title=task_title, subtasks=all_subtasks),
            all_validation_errors,
        )  # Return task and empty error list

    async def _process_tool_calls(
        self, message: Message, history: List[Message]
    ) -> Task:
        """
        Helper method to process tool calls, create task, and handle validation.
        Delegates validation logic to _validate_and_build_task_from_tool_calls.
        """
        if not message.tool_calls:
            raise ValueError(f"No tool calls found in the message: {message.content}")

        # Delegate the core processing and validation
        task, validation_errors = await self._validate_and_build_task_from_tool_calls(
            message.tool_calls, history
        )

        # _validate_and_build_task_from_tool_calls raises ValueError on failure,
        # so if we reach here, validation passed.
        if task is None:
            # This case should technically be handled by the exception in the helper,
            # but added for defensive programming.
            raise ValueError(
                "Task validation passed but task object is unexpectedly None."
            )

        return task

    async def _generate_with_retry(
        self, history: List[Message], tools: List[Tool], max_retries: int = 3
    ) -> tuple[Optional[Task], Optional[Message]]:
        """
        Generates response, processes tool calls with validation and retry logic.
        """
        log.debug(
            "Starting generation with retry, max_retries=%d, tools=%s",
            max_retries,
            [t.name for t in tools],
        )
        current_retry: int = 0
        last_message: Optional[Message] = None

        while current_retry < max_retries:
            attempt = current_retry + 1
            log.debug("Generation attempt %d/%d", attempt, max_retries)

            # Generate response using current history
            log.debug("Calling LLM with %d messages in history", len(history))
            message = await self.provider.generate_message(
                messages=history,
                model=self.model,
                tools=tools,
            )
            history.append(
                message
            )  # Add assistant's response to history *before* processing
            last_message = message
            log.debug(
                "LLM response received, has_tool_calls=%s", bool(message.tool_calls)
            )

            if not message.tool_calls:
                # LLM didn't use the expected tool
                log.warning("LLM did not use required tools on attempt %d", attempt)
                if tools and current_retry < max_retries - 1:
                    current_retry += 1
                    tool_names = ", ".join([t.name for t in tools])
                    retry_prompt = f"Please use one of the available tools ({tool_names}) to define the task based on the previous analysis and requirements."
                    history.append(Message(role="user", content=retry_prompt))
                    log.debug(
                        "Added retry prompt for tool usage, attempt %d",
                        current_retry + 1,
                    )
                    if self.display_manager:
                        log.debug(
                            "Retry %d/%d: Asking LLM to use required tool(s).",
                            attempt,
                            max_retries,
                        )
                    else:
                        log.warning(
                            "Retry %d/%d: Asking LLM to use required tool(s).",
                            attempt,
                            max_retries,
                        )
                    continue  # Go to next iteration
                else:
                    # Max retries reached without tool use
                    log.error(
                        "Max retries reached without tool usage after %d attempts",
                        max_retries,
                    )
                    if self.display_manager:
                        log.debug(
                            "Failed after %d retries: LLM did not use the required tool(s). Last message: %s",
                            max_retries,
                            self._format_message_content(message),
                        )
                    else:
                        log.error(
                            "Failed after %d retries: LLM did not use the required tool(s). Last message: %s",
                            max_retries,
                            self._format_message_content(message),
                        )
                    return None, last_message

            # Tool call exists, try to process it
            try:
                log.debug("Processing %d tool call(s)", len(message.tool_calls))
                # Process the tool call(s). This adds 'tool' role messages to history
                # and raises ValueError on validation failure.
                task = await self._process_tool_calls(message, history)
                # If _process_tool_calls returns without error, success!
                log.debug(
                    "Tool calls processed successfully, created task with %d subtasks",
                    len(task.subtasks),
                )
                if self.display_manager:
                    log.debug("Tool call processed successfully.")
                else:
                    log.debug("Tool call processed successfully.")
                return (
                    task,
                    last_message,
                )  # Return created task and the assistant message
            except ValueError as e:  # Catch validation errors from _process_tool_calls
                log.warning("Tool call validation failed on attempt %d: %s", attempt, e)
                if self.display_manager:
                    log.debug(
                        "Validation Error (Retry %d/%d): %s",
                        attempt,
                        max_retries,
                        str(e),
                    )
                else:
                    log.warning(
                        "Validation Error (Retry %d/%d): %s",
                        attempt,
                        max_retries,
                        str(e),
                    )
                if current_retry < max_retries - 1:
                    current_retry += 1
                    # Add a user message asking LLM to fix errors. The 'tool' message with error details
                    # should already be in history from _process_tool_calls failing.
                    retry_prompt = f"The previous attempt failed validation. Please review the errors detailed in the tool response and call the tool again correctly:\n{str(e)}"
                    history.append(Message(role="user", content=retry_prompt))
                    log.debug(
                        "Added validation error retry prompt, attempt %d",
                        current_retry + 1,
                    )
                    if self.display_manager:
                        log.debug(
                            "Retry %d/%d: Asking LLM to fix validation errors.",
                            attempt + 1,
                            max_retries,
                        )
                    else:
                        log.warning(
                            "Retry %d/%d: Asking LLM to fix validation errors.",
                            attempt + 1,
                            max_retries,
                        )
                    # Optional: Add a small delay before retrying
                    # await asyncio.sleep(1)
                    continue  # Go to next iteration
                else:
                    # Max retries reached after validation errors
                    log.error(
                        "Max retries reached due to persistent validation errors after %d attempts",
                        max_retries,
                    )
                    if self.display_manager:
                        log.debug(
                            "Failed after %d retries due to persistent validation errors.",
                            max_retries,
                        )
                    else:
                        log.error(
                            "Failed after %d retries due to persistent validation errors.",
                            max_retries,
                        )
                    return (
                        None,
                        last_message,
                    )  # Return no task and the last assistant message

        # Should only be reached if max_retries is 0 or loop finishes unexpectedly
        log.error("Generation with retry exited unexpectedly")
        return None, last_message

    def _get_execution_tools_info(self) -> str:
        """
        Get formatted string information about available execution tools.

        This information is used in prompts to inform the LLM about the tools
        that agents can use later to execute subtasks. It includes the tool's
        name, description, and a summary of its arguments (if an input schema
        is defined).

        Returns:
            A string detailing the available execution tools, or
            "No execution tools available" if none are configured.
        """
        if not self.execution_tools:
            return "No execution tools available"

        tools_info = "Available execution tools for subtasks:\n"
        for tool in self.execution_tools:
            # Add schema if available, keep it concise
            schema_info = ""
            if tool.input_schema and tool.input_schema.get("properties"):
                props = list(tool.input_schema["properties"].keys())
                req = tool.input_schema.get("required", [])
                prop_details = []
                for p in props:
                    is_req = " (required)" if p in req else ""
                    prop_details.append(f"{p}{is_req}")
                schema_info = f" | Args: {', '.join(prop_details)}"
            tools_info += f"- {tool.name}: {tool.description}{schema_info}\n"
        return tools_info.strip()  # Remove trailing newline

    def _get_input_files_info(self) -> str:
        """
        Get formatted string information about initial input files.

        Since we're now using object-based data flow instead of files,
        this always returns a message indicating no file dependencies.

        Returns:
            A string indicating no file-based inputs are used.
        """
        return "No file-based inputs (data flows through task dependencies)"

    def _detect_list_processing(
        self, objective: str, input_files: List[str]
    ) -> tuple[bool, int]:
        """
        Detects if the objective involves processing a list of items.

        Args:
            objective: The user's objective
            input_files: List of input files

        Returns:
            Tuple of (is_list_processing, estimated_item_count)
        """
        # List processing indicators
        list_indicators = [
            "for each",
            "all items",
            "every",
            "multiple",
            "list of",
            "batch",
            "process all",
            "analyze each",
            "extract from all",
            "urls",
            "documents",
            "files",
            "entries",
            "records",
        ]

        objective_lower = objective.lower()
        is_list = any(indicator in objective_lower for indicator in list_indicators)

        # Try to estimate count from objective
        estimated_count = 0
        numbers = re.findall(r"\b(\d+)\b", objective)
        if numbers:
            # Take the largest number as potential item count
            estimated_count = max(int(n) for n in numbers)

        # Check if input files suggest list processing
        if not is_list and input_files:
            for file in input_files:
                if any(
                    ext in file.lower() for ext in [".csv", ".jsonl", "list", "urls"]
                ):
                    is_list = True
                    if estimated_count == 0:
                        estimated_count = 50  # Default assumption
                    break

        return is_list, estimated_count

    def _ensure_additional_properties_false(self, schema: dict) -> dict:
        """
        Recursively ensures `additionalProperties: false` on object schemas.

        It also adds default `items: {"type": "string"}` for array schemas
        if `items` is not defined, and makes all properties of an object
        required by default if a `required` list is not already present.

        This is used to make JSON schemas generated or processed by the planner
        stricter, reducing ambiguity for LLM interactions.

        Args:
            schema: The JSON schema dictionary to process.

        Returns:
            The modified schema dictionary.
        """
        # Handle the current level if it's an object schema
        if isinstance(schema, dict) and schema.get("type") == "object":
            # Set additionalProperties if not explicitly defined
            schema.setdefault("additionalProperties", False)

            # Make all defined properties required if 'required' is not already present
            if (
                "properties" in schema
                and isinstance(schema["properties"], dict)
                and "required" not in schema
            ):
                prop_names = list(schema["properties"].keys())
                if prop_names:  # Only add 'required' if there are properties
                    schema["required"] = prop_names

        # Handle arrays - add a default items field if 'items' is missing
        elif (
            isinstance(schema, dict)
            and schema.get("type") == "array"
            and "items" not in schema
        ):
            schema["items"] = {"type": "string"}  # Default to string items

        # Recursively process nested schemas within properties
        if (
            isinstance(schema, dict)
            and "properties" in schema
            and isinstance(schema["properties"], dict)
        ):
            for prop_name, prop_schema in schema["properties"].items():
                # Ensure prop_schema is a dict before recursing
                if isinstance(prop_schema, dict):
                    schema["properties"][prop_name] = (
                        self._ensure_additional_properties_false(prop_schema)
                    )

        # Recursively process nested schemas within array items
        if isinstance(schema, dict) and "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, dict):
                schema["items"] = self._ensure_additional_properties_false(items_schema)
            elif isinstance(items_schema, list):  # Handle tuple schemas in 'items'
                schema["items"] = [
                    (
                        self._ensure_additional_properties_false(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in items_schema
                ]

        # Recursively process anyOf, allOf, oneOf schemas
        for key in ["anyOf", "allOf", "oneOf"]:
            if (
                isinstance(schema, dict)
                and key in schema
                and isinstance(schema[key], list)
            ):
                schema[key] = [
                    (
                        self._ensure_additional_properties_false(item)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in schema[key]
                ]

        return schema
