"""Manages the breakdown of complex objectives into executable task plans.

The TaskPlanner is responsible for taking a high-level user objective and
transforming it into a structured `TaskPlan`. This plan consists of
interdependent `Step` instances. The planning process can involve
multiple phases, including self-reflection for complexity assessment,
objective analysis, data flow definition, and final plan creation.

The planner interacts with an LLM provider to generate and refine the plan,
ensuring that steps are well-defined, dependencies are clear (forming a
Directed Acyclic Graph - DAG), and file paths are correctly managed within
a specified workspace. Validation is a key aspect of the planner's role to
ensure the generated plan is robust and executable.
"""

import json
import re  # Added import re
import traceback
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
)  # Add Optional

import networkx as nx
import yaml

# Add Jinja2 imports
from jinja2 import BaseLoader, Environment

# Add jsonschema import for validation
from jsonschema import ValidationError, validate
from rich.text import Text  # Re-add Text import

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    Step,
    Task,
    TaskPlan,
    ToolCall,
)
from nodetool.providers import BaseProvider

# Removed rich imports - Console, Table, Text, Live
from nodetool.ui.console import AgentConsole  # Import the new display manager
from nodetool.utils.message_parsing import (
    extract_json_from_message,
    lenient_json_parse,
    remove_think_tags,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, PlanningUpdate

log = get_logger(__name__)

STEP_JSON_SCHEMA = json.dumps(Step.model_json_schema(), indent=2)
PLAN_CREATION_MAX_TOKENS = 20000
PLANNING_PHASE_MAX_TOKENS = 20000  # Max tokens for planning phase

# Single-phase planning system prompt
DEFAULT_PLANNING_SYSTEM_PROMPT = """
<role>
You are a TaskArchitect that transforms user objectives into executable Task plans.
</role>

<terminology>
- **Task**: A container with a title and a list of Steps.
- **Step**: An individual unit of work with dependencies, instructions, and output schema.
</terminology>

<execution_patterns>
Choose the pattern that best fits the objective:

**Sequential** - Steps execute one after another
  Example: "Download file → Parse content → Generate summary"

**Fan-out (Discover → Process → Aggregate)** - Find items, process each, combine
  Example: "Find repos → Analyze each → Create comparison"

**Parallel** - Independent Steps run concurrently, then merge
  Example: "Research A | Research B → Combine findings"

**Single Step** - One atomic operation for simple objectives
</execution_patterns>

<step_fields>
Required:
- **id** (string): Unique snake_case identifier
- **instructions** (string): Clear, actionable instructions
- **depends_on** (array): List of Step IDs this depends on ([] for none)

Optional:
- **mode**: "discover" | "process" | "aggregate"
- **output_schema** (string): JSON schema string for Step output
- **per_item_instructions** (string): For mode="process", template with {field} placeholders
- **per_item_schema** (string): For mode="process", JSON schema for each item
- **tools** (array): Restrict which tools this Step can use
</step_fields>

<validation_rules>
- All Step IDs unique and descriptive
- Dependencies form a valid DAG (no cycles)
- All referenced Step IDs exist
- Steps are atomic (smallest executable units)
- All output_schema fields are valid JSON strings
</validation_rules>
"""

# User prompt template for plan creation
PLAN_CREATION_PROMPT = """
Create an executable Task for this objective using the create_task tool.

<objective>{{ objective }}</objective>

<available_inputs>{{ inputs_info }}</available_inputs>

<output_schema>{{ output_schema }}</output_schema>

<available_tools>{{ execution_tools_info }}</available_tools>

Choose the best execution pattern and call the create_task tool with your task plan.

<patterns>
- Sequential: step1 → step2 → step3
- Fan-out: discover → process each → aggregate
- Parallel: step1 | step2 | step3 → combine
- Single: one step for simple tasks
</patterns>

<step_fields>
Required: id (snake_case), instructions (clear actionable text), depends_on (list of step IDs)
Optional: mode (discover|process|aggregate), output_schema (JSON schema string), per_item_instructions (for process mode with {field} placeholders), per_item_schema, tools
</step_fields>
"""


class CreateTaskTool(Tool):
    name = "create_task"
    description = """Create an executable task with a list of steps to accomplish the given objective.

    The task should include:
    - title: A clear, descriptive title for the task
    - steps: A list of Step objects that form a valid DAG (Directed Acyclic Graph)

    Each step should have:
    - id: Unique snake_case identifier
    - instructions: Clear, actionable instructions for what to accomplish
    - depends_on: List of step IDs this step depends on (empty array for no dependencies)

    Optional step fields:
    - mode: "discover", "process", or "aggregate" for fan-out patterns
    - output_schema: JSON schema string defining expected output format
    - per_item_instructions: Template with {field} placeholders for process mode
    - per_item_schema: JSON schema string for each item in process mode
    - tools: List of specific tools this step should use

    Validation requirements:
    - All step IDs must be unique
    - Dependencies must form a valid DAG (no circular dependencies)
    - All referenced dependency IDs must exist as step IDs or input keys
    - All output_schema and per_item_schema fields must be valid JSON strings
    """
    input_schema = Task.model_json_schema()
    example = ""


class TaskPlanner:
    """
    Orchestrates the breakdown of a complex objective into a validated, executable
    workflow plan (`TaskPlan`) composed of interdependent steps (`Step`).

    Think of this as the lead architect for an AI agent system. It doesn't execute
    the steps itself, but meticulously designs the blueprint. Given a high-level
    objective (e.g., "Analyze market trends for product X"), the TaskPlanner uses
    an LLM to generate a structured plan detailing:

    1.  **Decomposition:** Breaking the objective into smaller, logical, and ideally
        atomic units of work (steps).
    2.  **Task Typing:** Determining if each step is a straightforward,
        deterministic call to a specific `Tool` (e.g., download a file) or if it
        requires more complex reasoning or multiple steps better handled by a
        probabilistic `Agent` executor (e.g., summarize analysis findings).
    3.  **Data Flow & Dependencies:** Explicitly defining the inputs (`input_files`)
        and `output_file` for each step. Crucially, it
        establishes the dependency graph, ensuring steps run only after their
        required inputs are available. This forms a Directed Acyclic Graph (DAG).
    4.  **Contracts:** Defining the expected data format (`output_schema`) for each step's output, promoting type safety and
        predictable integration between steps.
    5.  **Workspace Management:** Enforcing the use of *relative* file paths within
        a defined workspace, preventing dangerous absolute path manipulations and
        ensuring plan portability. Paths like `/tmp/foo` or `C:\\Users\\...` are
        strictly forbidden; only paths like `data/interim_results.csv` are valid.

    The planning process itself can involve multiple phases (configurable):
    - **Analysis Phase:** High-level strategic breakdown and identification of
      step types (Tool vs. Agent).
    - **Data Flow Analysis:** Refining dependencies, inputs/outputs, and data schemas.
    - **Plan Creation:** Generating the final, concrete `Task` object by instructing
      the LLM to output a JSON structure, which is then extracted and validated.

    **Core Responsibility:** To transform an ambiguous user objective into an unambiguous,
    validated, and machine-executable plan. It prioritizes structure, clear contracts,
    and dependency management over monolithic, error-prone LLM interactions. This
    structured approach is essential for reliable, efficient, and debuggable AI workflows.

    **Validation is paramount:** The planner rigorously checks the generated plan for:
    - Cyclic dependencies (fatal).
    - Missing input files.
    - Correct `tool_name` usage and valid JSON arguments for Tool tasks.
    - Correct `instructions` format (natural language instructions) for Agent tasks.
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
        execution_tools (Sequence[Tool]): Tools available for steps designated as Tool tasks
                                         during the Plan Creation phase.
        task_plan (Optional[TaskPlan]): The generated plan (populated after creation).
        system_prompt (str): The core instructions guiding the LLM planner.
        output_schema (Optional[dict]): Optional schema for the *final* output of the
                                        overall task (not individual steps).
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
        inputs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        output_schema: dict | None = None,
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
            execution_tools (List[Tool]): Tools available for step execution.
            inputs (dict[str, Any]): The inputs to use for planning
            system_prompt (str, optional): Custom system prompt
            output_schema (dict, optional): JSON schema for the final task output
            verbose (bool, optional): Whether to print planning progress table (default: True)
        """
        # Note: debug logging will be handled by display_manager initialization

        self.provider: BaseProvider = provider
        self.model: str = model
        self.reasoning_model: str = reasoning_model or model
        self.objective: str = objective
        self.workspace_dir: str = workspace_dir
        self.task_plan: TaskPlan = TaskPlan()
        self.inputs: dict[str, Any] = inputs or {}
        self.system_prompt: str = system_prompt or DEFAULT_PLANNING_SYSTEM_PROMPT
        self.execution_tools: Sequence[Tool] = execution_tools or []
        self._planning_context: ProcessingContext | None = None
        self.output_schema: Optional[dict] = output_schema
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
                with open(self.tasks_file_path) as f:
                    task_plan_data: dict = yaml.safe_load(f)
                    self.task_plan = TaskPlan(**task_plan_data)
                    if self.display_manager:
                        log.debug("Loaded existing task plan from %s", self.tasks_file_path)
                    return True
            except Exception as e:  # Keep general exception for file I/O or parsing issues
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
            json.dumps(self.output_schema) if self.output_schema else "Not specified (default: string)"
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
            "model": self.model,
            "reasoning_model": self.reasoning_model,
            "execution_tools_info": self._get_execution_tools_info(),
            "output_schema": overall_output_schema_str,
            "inputs_info": inputs_info,
            "step_schema": STEP_JSON_SCHEMA,
        }

    def _render_prompt(self, template_string: str, context: Optional[Dict[str, str]] = None) -> str:
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

    def _build_dependency_graph(self, steps: List[Step]) -> nx.DiGraph:
        """
            Build a directed graph of dependencies between steps.

        The graph nodes represent steps (identified by their `id`) and input keys.
            An edge from node A to step B means B depends on the output of A
            (i.e., one of B's `depends_on` is A's `id` or an input key).

            Args:
                steps: A list of `Step` objects.

            Returns:
                A `networkx.DiGraph` representing the dependencies.
        """
        G = nx.DiGraph()

        # Add nodes representing steps
        for step in steps:
            G.add_node(step.id)  # Node represents the step completion

        # Add nodes representing input keys
        for input_key in self.inputs:
            G.add_node(input_key)  # Node represents an input

        # Add edges for dependencies
        for step in steps:
            if step.depends_on:
                for dependency in step.depends_on:
                    # Only add edge if the dependency exists (either as step or input)
                    if dependency in [t.id for t in steps] or dependency in self.inputs:
                        G.add_edge(dependency, step.id)

        return G

    def _check_inputs(
        self,
        steps: List[Step],
    ) -> List[str]:
        """Checks if all input task dependencies for steps are available.

        An input task dependency is considered available if it:
        1. References a valid step ID within the current task plan, OR
        2. References a valid input key from the inputs dictionary

        Args:
            steps: A list of `Step` objects.

        Returns:
            A list of string error messages for any missing task dependencies.
        """
        validation_errors: List[str] = []
        tasks_by_id = {task.id: task for task in steps}

        for step in steps:
            if step.depends_on:
                for dependency in step.depends_on:
                    # Check if it's a valid step ID or an input key
                    if dependency not in tasks_by_id and dependency not in self.inputs:
                        validation_errors.append(
                            f"Subtask '{step.instructions}' depends on missing step or input '{dependency}'"
                        )
        return validation_errors

    def _validate_dependencies(self, steps: List[Step]) -> List[str]:
        """
        Validate task dependencies and DAG structure for steps.

        This method performs the following checks:
        1.  Cycle detection: Builds a dependency graph and checks for circular
            dependencies, which would make execution impossible.
        2.  Task availability: Verifies that all `depends_on` for each
            step reference valid task IDs in the plan.
        3.  Topological sort feasibility: Checks if a valid linear execution
            order for the steps can be determined.

        Args:
            steps: A list of `Step` objects to validate.

        Returns:
            A list of strings, where each string is an error message
            describing a validation failure. An empty list indicates
            all dependency checks passed.
        """
        log.debug("Starting dependency validation for %d steps", len(steps))
        validation_errors: List[str] = []

        # Log step summary for debugging
        step_ids = [task.id for task in steps]
        log.debug("Subtask IDs to validate: %s", step_ids)

        for i, task in enumerate(steps):
            log.debug("Subtask %d: id='%s', depends_on=%s", i, task.id, task.depends_on)

        # Build dependency graph
        log.debug("Building dependency graph")
        G = self._build_dependency_graph(steps)
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
        input_errors = self._check_inputs(steps)
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
                "✅ Dependency validation PASSED - all %d steps validated successfully",
                len(steps),
            )
        return validation_errors

    def _validate_plan_semantics(self, steps: List[Step]) -> List[str]:
        """
        Enforce semantic rules beyond DAG validation.

        We now rely on the planner prompts to enforce semantics (modes, schemas, etc).
        """
        return []

    def _apply_schema_overrides(self, steps: List[Step]) -> None:
        """
        Normalize output schemas for steps.

        NOTE: This legacy implementation relied on `st.mode` and `st.per_item_schema`
        which are no longer part of the `Step` model. Disabling for now to prevent crashes.
        The LLM should generate the correct output_schema directly.
        """
        pass
        # Original implementation commented out to fix AttributeError:
        # process_steps = [st for st in steps if st.mode == "process"]
        # aggregate_steps = [st for st in steps if st.mode == "aggregate"]

        # # 1. Process Schema Wrapping (Apply to ALL process steps)
        # for process_step in process_steps:
        #     item_schema_str = (process_step.per_item_schema or "").strip()
        #     if item_schema_str:
        #         try:
        #             item_schema = json.loads(item_schema_str)
        #             process_step.output_schema = json.dumps(
        #                 {"type": "array", "items": item_schema}
        #             )
        #         except Exception as exc:
        #             # We log/ignore here because validation will catch it later
        #             log.warning(
        #                 "Process step '%s' has invalid per_item_schema JSON: %s",
        #                 process_step.id,
        #                 exc,
        #             )

        # # 2. Aggregate Schema Override (Apply to ALL aggregate steps if they lack schema)
        # if self.output_schema:
        #     output_schema_str = json.dumps(self.output_schema)
        #     for agg_step in aggregate_steps:
        #         if not agg_step.output_schema:
        #             agg_step.output_schema = output_schema_str

    def _validate_legacy_plan_semantics(self, steps: List[Step]) -> List[str]:
        """Legacy validation for plans that enumerate per-item steps."""
        errors: list[str] = []

        # Helper: parse JSON schema string into dict (best-effort)
        def _parse_schema(schema_str: str | None) -> dict | None:
            if not schema_str or not isinstance(schema_str, str):
                return None
            try:
                return json.loads(schema_str)
            except Exception:
                try:
                    return yaml.safe_load(schema_str)  # permissive fallback
                except Exception:
                    return None

        # 1) No discovery/search/find steps by id or by content
        discovery_id_patterns = (
            "discover",
            "search",
            "find",
        )
        discovery_content_triggers = (
            "use google search",
            "use the google_search tool",
            "google search",
            "serp",
            "search engine",
            "perform a search",
            "query google",
            "site:",
        )

        for st in steps:
            sid = (st.id or "").lower()
            scontent = (st.instructions or "").lower()
            if any(p in sid for p in discovery_id_patterns):
                errors.append(
                    f"Final plan must not include discovery steps (found id '{st.id}'). Move discovery to planning phase and fan-out execution steps instead."
                )
            if any(trigger in scontent for trigger in discovery_content_triggers):
                errors.append(
                    f"Subtask '{st.id}' contains discovery/search instructions in content. Discovery must be done during planning; remove runtime discovery."
                )

        # 2) No looping phrasing in execution steps (except aggregator)
        looping_phrases = (
            "for each",
            "for every",
            "iterate over",
            "loop over",
            "for all",
            "all urls",
            "list of urls",
            "urls list",
            "all discovered",
        )

        # Identify aggregator candidate(s): schema matches overall output schema or id indicates aggregation
        overall_schema = self.output_schema or None
        aggregator_ids: set[str] = set()
        for st in steps:
            schema_dict = _parse_schema(getattr(st, "output_schema", None))
            if overall_schema and schema_dict == overall_schema:
                aggregator_ids.add(st.id)
            else:
                sid = (st.id or "").lower()
                if any(
                    x in sid
                    for x in (
                        "aggregate",
                        "compile",
                        "combine",
                        "merge",
                        "final",
                        "report",
                    )
                ):
                    aggregator_ids.add(st.id)

        for st in steps:
            if st.id in aggregator_ids:
                continue
            scontent = (st.instructions or "").lower()
            if any(p in scontent for p in looping_phrases):
                errors.append(
                    f"Subtask '{st.id}' appears to loop over a collection (e.g., '{scontent[:60]}...'). Emit one step per discovered item (fan-out) instead."
                )

        # 3) Aggregator wiring: if there is an aggregator and extractor-like steps, ensure aggregator depends on all
        extractor_like_ids: list[str] = []
        for st in steps:
            sid = (st.id or "").lower()
            if any(x in sid for x in ("extract", "fetch", "scrape", "crawl", "parse", "process")):
                extractor_like_ids.append(st.id)

        if aggregator_ids and extractor_like_ids:
            for agg_id in aggregator_ids:
                agg = next((t for t in steps if t.id == agg_id), None)
                if agg is None:
                    continue
                declared_inputs = set(agg.depends_on or [])
                missing = [eid for eid in extractor_like_ids if eid not in declared_inputs]
                if missing:
                    errors.append(
                        f"Aggregator '{agg_id}' must depend on all extractor steps. Missing dependencies: {missing}"
                    )

        return errors

    async def _run_phase(
        self,
        history: List[Message],
        phase_name: str,
        phase_display_name: str,
        is_enabled: bool,
        prompt_template: str,
        phase_result_name: str,
        skip_reason: str | None = None,
    ) -> AsyncGenerator[
        Chunk | ToolCall | PlanningUpdate | tuple[List[Message], Optional[PlanningUpdate]],
        None,
    ]:
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
            skip_reason: Optional message to surface when a phase is skipped.

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
                    phase_display_name,
                    "Skipped",
                    skip_reason or "Phase disabled by global flag.",
                )
            planning_update = PlanningUpdate(
                phase=phase_result_name,
                status="Skipped",
                content=skip_reason or "Phase disabled by global flag.",
            )
            yield history, planning_update
            return

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
            self.display_manager.update_planning_display(phase_display_name, "Running", f"Generating {phase_name}...")

        log.debug("Calling LLM for %s using model: %s", phase_name, self.model)
        available_tools: Sequence[Tool] = self.execution_tools
        response_message: Message | None = None
        tool_iterations = 0
        max_tool_iterations = 6

        while True:
            response_content = ""
            tool_calls = []

            async for chunk in self.provider.generate_messages(
                messages=history,
                model=self.model,
                tools=available_tools,
                max_tokens=PLANNING_PHASE_MAX_TOKENS,
            ):
                if isinstance(chunk, Chunk):
                    if chunk.content:
                        response_content += chunk.content
                    yield chunk
                elif isinstance(chunk, ToolCall):
                    tool_calls.append(chunk)
                    yield chunk

            response_message = Message(
                role="assistant",
                content=response_content if response_content else None,
                tool_calls=tool_calls if tool_calls else None,
            )
            history.append(response_message)

            if not response_message.tool_calls or not available_tools:
                break

            if tool_iterations >= max_tool_iterations:
                log.warning(
                    "%s phase exceeded tool call limit (%d). Continuing without additional tool runs.",
                    phase_name.capitalize(),
                    max_tool_iterations,
                )
                break

            tool_iterations += 1
            tool_messages = await self._execute_planning_tool_calls(
                response_message.tool_calls,
                phase_name,
            )
            history.extend(tool_messages)
            log.debug(
                "%s phase processed %d planning tool call(s)",
                phase_name.capitalize(),
                len(tool_messages),
            )

        assert response_message is not None
        log.debug(
            "%s phase LLM response received, content length: %d chars",
            phase_name.capitalize(),
            (len(str(response_message.content)) if response_message.content else 0),
        )

        phase_status: str = "Completed"
        phase_content: str | Text = self._format_message_content(response_message)
        log.debug("%s phase completed with status: %s", phase_name.capitalize(), phase_status)

        if self.display_manager:
            self.display_manager.update_planning_display(phase_display_name, phase_status, phase_content)

        log.info(f"{phase_result_name} phase completed with status: {phase_status}")
        log.info(f"{phase_result_name} phase content: {phase_content}")

        planning_update = PlanningUpdate(
            phase=phase_result_name,
            status=phase_status,
            content=str(phase_content),
        )

        yield history, planning_update

    async def _run_plan_creation_phase(
        self,
        history: List[Message],
        objective: str,
        max_retries: int,
    ) -> AsyncGenerator[
        Chunk | ToolCall | PlanningUpdate | tuple[Optional[Task], Optional[Exception], Optional[PlanningUpdate]],
        None,
    ]:
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
            self.display_manager.debug(f"Starting plan creation phase with max_retries={max_retries}")

        task: Optional[Task] = None
        plan_creation_error: Optional[Exception] = None
        phase_status: str = "Failed"
        phase_content: str | Text = "N/A"
        current_phase_name: str = "3. Plan Creation"

        # Generate plan using JSON output instead of tool calls
        if self.display_manager:
            self.display_manager.debug("Using JSON-based generation for plan creation")

        # Use the same consolidated prompt for both structured and non-structured output
        plan_creation_prompt_content = self._render_prompt(PLAN_CREATION_PROMPT)
        agent_task_prompt_content = ""

        if self.display_manager:
            self.display_manager.debug(f"Plan creation prompt length: {len(plan_creation_prompt_content)} chars")
        if self.display_manager:
            self.display_manager.debug(f"Agent task prompt length: {len(agent_task_prompt_content)} chars")

        history.append(
            Message(
                role="user",
                # Concatenate only if agent_task_prompt_content is not empty
                content=f"{plan_creation_prompt_content}\n{agent_task_prompt_content}".strip(),
            )
        )
        if self.display_manager:
            self.display_manager.update_planning_display(
                current_phase_name,
                "Running",
                "Generating task plan...",
            )

        create_task_tool = CreateTaskTool()

        # Retry loop for tool-based plan generation
        for attempt in range(max_retries):
            try:
                # Call LLM with the create_task tool
                response_content = ""
                tool_calls = []
                async for chunk in self.provider.generate_messages(
                    messages=history,
                    model=self.model,
                    tools=[create_task_tool],
                    max_tokens=PLAN_CREATION_MAX_TOKENS,
                ):
                    if isinstance(chunk, Chunk):
                        if chunk.content:
                            response_content += chunk.content
                        yield chunk
                    elif isinstance(chunk, ToolCall):
                        tool_calls.append(chunk)
                        yield chunk

                response_message = Message(
                    role="assistant",
                    content=response_content if response_content else None,
                    tool_calls=tool_calls if tool_calls else None,
                )
                history.append(response_message)

                # Extract task data from tool calls
                task_data = None
                if response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.name == "create_task":
                            task_data = tool_call.args
                            log.debug(
                                "Extracted task from tool call: %s",
                                (list(task_data.keys()) if isinstance(task_data, dict) else type(task_data)),
                            )
                            break

                if not task_data:
                    # Fallback: try to extract JSON from content (for providers that might not use tool calls)
                    task_data = extract_json_from_message(response_message)
                    if task_data:
                        log.debug("Extracted task from message content as fallback")

                if not task_data:
                    failure_reason = f"LLM did not call create_task tool on attempt {attempt + 1}/{max_retries}"
                    log.warning(f"{failure_reason}. Response: {response_message.content}")

                    # Add error feedback to history for retry
                    if attempt < max_retries - 1:
                        error_feedback = Message(
                            role="user",
                            content=f"Error: {failure_reason}. Please call the create_task tool with your task plan.",
                        )
                        history.append(error_feedback)
                        continue
                    else:
                        plan_creation_error = ValueError(failure_reason)
                        phase_content = (
                            f"{failure_reason}. Last message: {self._format_message_content(response_message)}"
                        )
                        phase_status = "Failed"
                        break

                # Validate and build task from extracted JSON
                log.debug(
                    "Validating extracted JSON: %s",
                    (list(task_data.keys()) if isinstance(task_data, dict) else type(task_data)),
                )
                validated_task, validation_errors = await self._validate_structured_output_plan(task_data, objective)

                if validated_task and not validation_errors:
                    task = validated_task
                    phase_status = "Success"
                    # Don't show raw JSON content in the UI update, show a summary
                    phase_content = f"Created plan '{task.title}' with {len(task.steps)} steps."
                    log.debug(
                        "JSON-based plan creation successful: %d steps",
                        len(task.steps),
                    )
                    break
                else:
                    # Validation failed, provide feedback for retry
                    failure_reason = (
                        f"Task validation failed on attempt {attempt + 1}/{max_retries}: {'; '.join(validation_errors)}"
                    )
                    log.warning(failure_reason)

                    if attempt < max_retries - 1:
                        error_msg = (
                            "Error: The task plan has validation errors:\n"
                            + "\n".join(f"- {err}" for err in validation_errors)
                            + "\n\nPlease fix these issues and output the corrected task plan as JSON."
                        )

                        if response_message.tool_calls:
                            for tc in response_message.tool_calls:
                                history.append(
                                    Message(
                                        role="tool",
                                        tool_call_id=tc.id,
                                        content=error_msg,
                                    )
                                )
                        else:
                            history.append(Message(role="user", content=error_msg))

                        continue
                    else:
                        plan_creation_error = ValueError(failure_reason)
                        phase_content = f"{failure_reason}"
                        phase_status = "Failed"
                        break

            except Exception as e:
                log.error(
                    "JSON-based plan creation failed on attempt %d: %s",
                    attempt + 1,
                    e,
                    exc_info=True,
                )

                if attempt < max_retries - 1:
                    last_msg = history[-1] if history else None
                    if last_msg and last_msg.role == "assistant" and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            history.append(
                                Message(
                                    role="tool",
                                    tool_call_id=tc.id,
                                    content=f"Error during plan generation: {str(e)}",
                                )
                            )
                    else:
                        history.append(
                            Message(
                                role="user",
                                content=f"Error during plan generation: {str(e)}. Please try again and output a valid JSON task plan.",
                            )
                        )
                    continue
                else:
                    plan_creation_error = e
                    phase_status = "Failed"
                    phase_content = f"JSON generation failed: {str(e)}\n{traceback.format_exc()}"
                    break

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
            content=str(phase_content),  # Update uses summary string now
        )

        yield task, plan_creation_error, planning_update

    async def create_task(
        self,
        context: ProcessingContext,
        objective: str,
        max_retries: int = 3,
    ) -> AsyncGenerator[Chunk | ToolCall | PlanningUpdate, None]:
        """
        Create steps using the configured planning process, allowing for early shortcuts.
        Yields PlanningUpdate events during the process.
        Displays a live table summarizing the planning process if verbose mode is enabled.
        """
        log.debug(
            "Starting task creation for objective: '%s...' with max_retries=%d",
            objective[:100],
            max_retries,
        )

        # Preserve processing context for tool execution during planning
        self._planning_context = context

        # Start the live display using the display manager
        if self.display_manager:
            self.display_manager.start_live(self.display_manager.create_planning_tree("Task Planner"))

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
            log.debug("Starting planning")

            # Single Phase: Planning (directly creates the Task)
            current_phase = "Planning"
            if self.display_manager:
                self.display_manager.set_current_phase(current_phase)
            log.debug("Entering phase: %s", current_phase)
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)

            task = None
            plan_creation_error = None
            planning_update = None

            async for update in self._run_plan_creation_phase(history, objective, max_retries):
                if isinstance(update, tuple):
                    task, plan_creation_error, planning_update = update
                else:
                    yield update

            if planning_update:
                yield planning_update  # Yield the update from the phase itself

            # --- Final Outcome ---
            if task:
                log.debug("Plan created successfully with %d steps", len(task.steps))
                if self.display_manager:
                    log.debug("Plan created successfully.")
                else:
                    log.debug("Plan created successfully.")
                self.task_plan.tasks.append(task)
                log.debug(f"Task created: \n{task.to_markdown()}")
            else:
                # Construct error message based on plan_creation_error or last message
                if plan_creation_error:
                    error_message = f"Failed to create valid task during Plan Creation phase. Original error: {str(plan_creation_error)}"
                    full_error_message = f"{error_message}\n{traceback.format_exc()}" if self.verbose else error_message
                    log.error("Task creation failed: %s", error_message)
                    # Yield failure update before raising
                    yield PlanningUpdate(phase=current_phase, status="Failed", content=error_message)
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
                    yield PlanningUpdate(phase=current_phase, status="Failed", content=error_message)
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
            if error_message and self.display_manager:
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
            yield PlanningUpdate(phase=current_phase, status="Failed", content=error_message)
            raise  # Re-raise the caught exception

        finally:
            log.debug("Stopping live display and completing task creation")
            # Stop the live display using the display manager
            if self.display_manager:
                self.display_manager.stop_live()
            self._planning_context = None

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
                calls_summary.append(f"- Tool Call: {tc.name}\\n  Args: {args_str}")
            # Use Text object for potential future styling
            return Text("\\n".join(calls_summary))  # No <think> tag removal for tool call summaries

        raw_content_str: Optional[str] = None
        if message.content:
            if isinstance(message.content, list):
                # Attempt to join list items; handle potential non-string items
                try:
                    raw_content_str = "\\n".join(str(item) for item in message.content)
                except Exception:
                    raw_content_str = str(message.content)  # Fallback to string representation of the list
            elif isinstance(message.content, str):
                raw_content_str = message.content
            else:
                # Handle other unexpected content types
                raw_content_str = f"Unexpected content type: {type(message.content).__name__}"

        cleaned_content: Optional[str] = remove_think_tags(raw_content_str)

        if cleaned_content:  # If cleaned_content is not None and not empty
            return Text(cleaned_content)
        elif raw_content_str is not None:  # Original content existed but was all <think> tags or whitespace
            return Text("")  # Display as empty, not "Empty message content"
        else:  # No message.content to begin with
            return Text("Empty message content.", style="dim")

    def _format_message_content_for_update(self, message: Optional[Message]) -> Optional[str]:
        """Formats message content into a simple string for PlanningUpdate.

        This method is similar to `_format_message_content` but specifically
        targets the `instructions` attribute of a `Message` and returns a plain
        string, primarily for use in `PlanningUpdate.instructions`. It removes
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
        if message.content:  # This method primarily processes .instructions
            if isinstance(message.content, list):
                try:
                    raw_str_content = "\\n".join(str(item) for item in message.content)
                except Exception:
                    raw_str_content = str(message.content)  # Fallback
            elif isinstance(message.content, str):
                raw_str_content = message.content
            else:
                raw_str_content = f"Unexpected content type: {type(message.content).__name__}"

        return remove_think_tags(raw_str_content)

    async def _execute_planning_tool_calls(
        self,
        tool_calls: Sequence[ToolCall],
        phase_name: str,
    ) -> List[Message]:
        """Executes tool calls issued during planning phases."""

        tool_messages: List[Message] = []
        context = self._planning_context
        if not context:
            log.warning(
                "%s phase received tool calls but planning context is unavailable",
                phase_name.capitalize(),
            )
            return tool_messages

        tool_lookup = {tool.name: tool for tool in self.execution_tools}

        for tool_call in tool_calls:
            tool = tool_lookup.get(tool_call.name)
            if not tool:
                content = json.dumps(
                    {
                        "error": f"Tool '{tool_call.name}' is not available during planning.",
                    }
                )
            else:
                try:
                    raw_result = await tool.process(context, tool_call.args)
                    normalized = self._normalize_tool_output(raw_result)
                    content = json.dumps(normalized, ensure_ascii=False)
                except Exception as exc:  # pragma: no cover - best effort
                    log.warning(
                        "Planning tool '%s' failed during %s phase: %s",
                        tool_call.name,
                        phase_name,
                        exc,
                    )
                    content = json.dumps(
                        {
                            "error": f"Tool '{tool_call.name}' failed: {exc}",
                        }
                    )

            tool_messages.append(
                Message(
                    role="tool",
                    name=tool_call.name,
                    tool_call_id=tool_call.id,
                    content=content,
                )
            )

        return tool_messages

    def _normalize_tool_output(self, value: Any) -> Any:
        """Convert tool results into JSON-serializable primitives."""

        if hasattr(value, "model_dump"):
            return self._normalize_tool_output(value.model_dump())
        if isinstance(value, dict):
            return {key: self._normalize_tool_output(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._normalize_tool_output(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    def _validate_tool_task(
        self,
        step_data: dict,
        tool_name: str,
        content: Any,
        available_execution_tools: Dict[str, Tool],
        sub_context: str,
    ) -> tuple[Optional[dict], List[str]]:
        """Validates a step intended as a direct tool call.

        This involves:
        1.  Checking if the specified `tool_name` is among the available
            `execution_tools`.
        2.  Ensuring the `instructions` (expected to be tool arguments) is valid JSON.
        3.  Validating the parsed JSON arguments against the tool's `input_schema`
            using `jsonschema.validate`.

        Args:
            step_data: The raw dictionary data for the step.
            tool_name: The name of the tool to be called.
            content: The content for the step, expected to be JSON arguments
                     for the tool.
            available_execution_tools: A dictionary mapping tool names to `Tool` objects.
            sub_context: A string prefix for error messages (e.g., "Subtask 1").

        Returns:
            A tuple containing:
                - A dictionary of the parsed and validated tool arguments if
                  validation is successful, otherwise None.
                - A list of string error messages encountered during validation.
        """
        log.debug("%s: Starting tool task validation for tool '%s'", sub_context, tool_name)
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
            log.debug("%s: Content is already a dict with %d keys", sub_context, len(content))
        elif isinstance(content, str):
            log.debug(
                "%s: Parsing JSON string content of length %d",
                sub_context,
                len(content),
            )
            parsed_content = {} if not content.strip() else lenient_json_parse(content)

            if parsed_content is not None:
                log.debug(
                    "%s: Successfully parsed JSON with %d keys: %s",
                    sub_context,
                    len(parsed_content),
                    list(parsed_content.keys()),
                )
            else:
                error_msg = f"'content' is not valid JSON. Content: '{content}'"
                log.error("%s (tool: %s): %s", sub_context, tool_name, error_msg)
                validation_errors.append(f"{sub_context} (tool: {tool_name}): {error_msg}")
                return None, validation_errors
        else:
            error_msg = (
                f"Expected JSON string or object for tool arguments, but got {type(content)}. Content: '{content}'"
            )
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
                error_msg = f"JSON arguments in 'content' do not match the tool's input schema. Error: {e.message}. Path: {'/'.join([str(p) for p in e.path])}. Schema: {e.schema}. Args: {parsed_content}"
                log.error(
                    "%s (tool: %s): Schema validation failed: %s",
                    sub_context,
                    tool_name,
                    error_msg,
                )
                validation_errors.append(f"{sub_context} (tool: {tool_name}): {error_msg}")
                return None, validation_errors
            except Exception as e:  # Catch other potential validation errors
                error_msg = f"Error validating arguments against tool schema. Error: {e}. Args: {parsed_content}"
                log.error(
                    "%s (tool: %s): Unexpected validation error: %s",
                    sub_context,
                    tool_name,
                    error_msg,
                )
                validation_errors.append(f"{sub_context} (tool: {tool_name}): {error_msg}")
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
        """Validates a step intended for agent execution.

        Ensures that the `instructions` for an agent task (which represents
            natural language instructions) is a non-empty string.

        Args:
            content: The content of the step (instructions for the agent).
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

    def _process_step_schema(self, step_data: dict, sub_context: str) -> tuple[Optional[str], List[str]]:
        """Processes and validates the output_schema for a step.

        Accepts schema definitions provided as JSON strings or already-parsed
        dictionaries. When a schema string cannot be parsed as JSON, the parser
        will attempt a YAML fallback before defaulting to a generic string
        schema. Only fatal situations (where no schema can be produced) are
        returned as validation errors so that non-fatal issues do not halt plan
        creation.

        Args:
            step_data: The raw dictionary data for the step
            sub_context: A string prefix for error messages.

        Returns:
            A tuple containing:
                - The processed and validated `output_schema` as a JSON string,
                  or None if a fatal error occurred.
                - A list of string error messages encountered.
        """
        log.debug("%s: Starting schema processing", sub_context)
        validation_errors: List[str] = []
        raw_schema: Any = step_data.get("output_schema")
        final_schema_str: Optional[str] = None
        schema_dict: Optional[dict] = None

        def default_schema(reason: str) -> dict:
            """Return a default schema and log why it was needed."""
            log.warning(
                "%s: %s. Using default string schema.",
                sub_context,
                reason,
            )
            return {
                "type": "string",
                "description": "Subtask result",
            }

        log.debug(
            "%s: schema_input='%s' (type: %s)",
            sub_context,
            raw_schema,
            type(raw_schema),
        )

        try:
            if isinstance(raw_schema, dict):
                log.debug("%s: Schema already provided as dict", sub_context)
                schema_dict = deepcopy(raw_schema)
            elif isinstance(raw_schema, str):
                stripped_schema = raw_schema.strip()
                if not stripped_schema:
                    log.debug(
                        "%s: Empty schema string provided, using default",
                        sub_context,
                    )
                    schema_dict = default_schema("Empty output_schema string")
                else:
                    log.debug(
                        "%s: Parsing string schema: '%s'",
                        sub_context,
                        stripped_schema,
                    )
                    try:
                        schema_dict = json.loads(stripped_schema)
                        log.debug(
                            "%s: Successfully parsed schema via JSON",
                            sub_context,
                        )
                    except json.JSONDecodeError as json_error:
                        log.debug(
                            "%s: JSON parsing failed (%s), attempting YAML",
                            sub_context,
                            json_error,
                        )
                        try:
                            yaml_candidate = yaml.safe_load(stripped_schema)
                        except yaml.YAMLError as yaml_error:
                            raise ValueError("Unable to parse output_schema as JSON or YAML") from yaml_error

                        if not isinstance(yaml_candidate, dict):
                            raise ValueError("Output schema string must describe an object structure") from json_error

                        schema_dict = yaml_candidate
                        log.warning(
                            "%s: Parsed schema using YAML fallback due to JSON error: %s",
                            sub_context,
                            json_error,
                        )
            elif raw_schema is None:
                log.debug("%s: No output_schema provided; using default", sub_context)
                schema_dict = default_schema("Missing output_schema")
            else:
                raise ValueError(f"Output schema must be a JSON string or dict, got {type(raw_schema)}")

            if schema_dict is None:
                raise ValueError("Schema parsing produced no result")

            log.debug(
                "%s: Applying additionalProperties constraints to schema",
                sub_context,
            )
            schema_dict = self._ensure_additional_properties_false(schema_dict)
            final_schema_str = json.dumps(schema_dict)
            log.debug(
                "%s: Final schema prepared, length=%d",
                sub_context,
                len(final_schema_str),
            )

        except ValueError as e:
            schema_dict = default_schema(str(e))
            final_schema_str = json.dumps(schema_dict)

        log.debug(
            "%s: Schema processing completed, errors=%d, final_schema_str=%s",
            sub_context,
            len(validation_errors),
            final_schema_str is not None,
        )
        return final_schema_str, validation_errors

    def _prepare_step_data(
        self,
        step_data: dict,
        final_schema_str: str,
        parsed_tool_content: Optional[dict],
        sub_context: str,
        available_execution_tools: Dict[str, Tool],
    ) -> tuple[Optional[dict], List[str]]:
        """Prepares the final data dictionary for Step creation.

        This method takes the raw step data, the validated `final_schema_str`,
        and potentially parsed tool content, then performs:
        1.  Validates depends_on references to ensure they exist.
        2.  Ensures `tool_name` is None if it was an empty string.
        3.  Handles default model assignment for the step.
        4.  Filters the data dictionary to include only fields recognized by the
            `Step` Pydantic model.
        5.  Stringifies `parsed_tool_content` back into the `instructions` field if
            it was a tool task.

        Args:
            step_data: The raw dictionary data for the step.
            final_schema_str: The validated output schema as a JSON string.
            parsed_tool_content: Parsed JSON arguments if it's a tool task, else None.
            sub_context: A string prefix for error messages.

        Returns:
            A tuple containing:
                - A dictionary ready for `Step` model instantiation, or None
                  if a fatal error occurred.
                - A list of string error messages.
        """
        validation_errors: List[str] = []
        processed_data = step_data.copy()  # Work on a copy

        try:
            processed_data["output_schema"] = final_schema_str  # Already validated/generated

            # Map 'depends_on' to 'depends_on' if LLM used wrong field name
            if "depends_on" in processed_data and "depends_on" not in processed_data:
                processed_data["depends_on"] = processed_data.pop("depends_on")
                log.debug(
                    "%s: Mapped 'depends_on' to 'depends_on': %s",
                    sub_context,
                    processed_data["depends_on"],
                )

            # Validate depends_on
            raw_depends_on = processed_data.get("depends_on", [])
            if not isinstance(raw_depends_on, list):
                validation_errors.append(f"{sub_context}: depends_on must be a list, got {type(raw_depends_on)}.")
                processed_data["depends_on"] = []
            else:
                # Ensure all items in depends_on are strings
                validated_depends_on = []
                for task_id in raw_depends_on:
                    if not isinstance(task_id, str):
                        validation_errors.append(
                            f"{sub_context}: Input task ID '{task_id}' must be a string, got {type(task_id)}."
                        )
                    else:
                        validated_depends_on.append(task_id)
                processed_data["depends_on"] = validated_depends_on

            # Ensure tool_name is None if empty string or missing
            processed_data["tool_name"] = processed_data.get("tool_name") or None

            # Handle optional model assignment
            step_model = processed_data.get("model")
            if not isinstance(step_model, str) or not step_model.strip():
                processed_data["model"] = self.model  # Default to planner's primary model
            else:
                processed_data["model"] = step_model.strip()

            allowed_tools = self._sanitize_tools_list(
                processed_data.get("tools"),
                available_execution_tools,
                sub_context,
            )
            if allowed_tools is not None:
                processed_data["tools"] = allowed_tools
            elif "tools" in processed_data:
                processed_data.pop("tools", None)

            # Filter args based on Step model fields
            step_model_fields = Step.model_fields.keys()
            filtered_data = {k: v for k, v in processed_data.items() if k in step_model_fields}

            # Stringify content if it was a parsed JSON object (for tool args)
            if isinstance(parsed_tool_content, dict):
                # Ensure the original content key exists before assignment
                if "instructions" in filtered_data:
                    filtered_data["instructions"] = json.dumps(parsed_tool_content)
                else:
                    # This case might indicate an issue if content was expected but not provided/filtered
                    validation_errors.append(
                        f"{sub_context}: Content field missing after filtering, cannot stringify tool arguments."
                    )
                    return None, validation_errors
            elif isinstance(filtered_data.get("instructions"), str):
                pass  # Keep agent task content as string

            # Stringify per_item_schema if it was provided as a dict
            raw_item_schema = filtered_data.get("per_item_schema")
            if isinstance(raw_item_schema, dict):
                filtered_data["per_item_schema"] = json.dumps(raw_item_schema)
            elif raw_item_schema is not None and not isinstance(raw_item_schema, str):
                validation_errors.append(
                    f"{sub_context}: per_item_schema must be a dict or JSON string, got {type(raw_item_schema)}."
                )

            return filtered_data, validation_errors

        except Exception as e:  # Catch unexpected errors during preparation
            validation_errors.append(f"{sub_context}: Unexpected error preparing step data: {e}")
            return None, validation_errors

    def _sanitize_tools_list(
        self,
        requested_tools: Any,
        available_execution_tools: Dict[str, Tool],
        sub_context: str,
    ) -> list[str] | None:
        """Validate the optional `tools` list for a step."""

        if requested_tools in (None, [], ()):  # No restriction requested
            return None

        if not isinstance(requested_tools, list):
            log.warning(
                "%s: Ignoring non-list value for tools (%s)",
                sub_context,
                type(requested_tools),
            )
            return None

        normalized: list[str] = []
        for tool_name in requested_tools:
            if not isinstance(tool_name, str) or not tool_name.strip():
                log.warning(
                    "%s: Ignoring invalid tool entry in tools list: %s",
                    sub_context,
                    tool_name,
                )
                continue
            tool_name = tool_name.strip()
            if tool_name not in available_execution_tools:
                log.warning(
                    "%s: Requested tool '%s' is not available and will be ignored",
                    sub_context,
                    tool_name,
                )
                continue
            if tool_name not in normalized:
                normalized.append(tool_name)

        return normalized or None

    async def _process_single_step(
        self,
        step_data: dict,
        index: int,
        context_prefix: str,
        available_execution_tools: Dict[str, Tool],
    ) -> tuple[Optional[Step], List[str]]:
        """
        Processes and validates data for a single step by delegating steps.
        """
        sub_context = f"{context_prefix} step {index}"
        log.debug("Processing %s", sub_context)
        all_validation_errors: List[str] = []
        parsed_tool_content: Optional[dict] = None  # To store parsed JSON for tool tasks

        try:
            # --- Validate Tool Call vs Agent Instruction ---
            tool_name = step_data.get("tool_name")
            content = step_data.get("instructions")
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
                    step_data,
                    tool_name,
                    content,
                    available_execution_tools,
                    sub_context,
                )
                all_validation_errors.extend(tool_errors)
                if parsed_tool_content is None and tool_errors:  # Fatal error during tool validation
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
            final_schema_str, schema_errors = self._process_step_schema(step_data, sub_context)
            all_validation_errors.extend(schema_errors)
            # Continue even if there were schema errors, as a default might be used.
            # Need final_schema_str for data preparation. If None, indicates a fatal schema issue.
            if final_schema_str is None:
                log.error("%s: Fatal schema processing error", sub_context)
                all_validation_errors.append(f"{sub_context}: Fatal error processing output schema.")
                return None, all_validation_errors
            else:
                log.debug("%s: Schema processing successful", sub_context)

            # --- Prepare data for Step creation (Paths, Filtering, Stringify Tool Args) ---
            log.debug("%s: Preparing data for Step creation", sub_context)
            filtered_data, preparation_errors = self._prepare_step_data(
                step_data,
                final_schema_str,
                parsed_tool_content,
                sub_context,
                available_execution_tools,
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

            # --- Create Step object ---
            # Pydantic validation happens here
            log.debug("%s: Creating Step object", sub_context)
            step = Step(**filtered_data)
            log.debug("%s: Step created successfully with id='%s'", sub_context, step.id)
            # Return successful step and any *non-fatal* validation errors collected
            return step, all_validation_errors

        except ValidationError as e:  # Catch Pydantic validation errors during Step(**filtered_data)
            error_msg = f"{sub_context}: Invalid data for Step model: {e}"
            log.error("%s: Pydantic validation error: %s", sub_context, e)
            all_validation_errors.append(error_msg)
            return None, all_validation_errors
        except Exception as e:  # Catch any other unexpected errors
            error_msg = f"{sub_context}: Unexpected error processing step: {e}\n{traceback.format_exc()}"
            log.error("%s: Unexpected processing error: %s", sub_context, e, exc_info=True)
            all_validation_errors.append(error_msg)
            return None, all_validation_errors

    async def _process_step_list(self, raw_steps: list, context_prefix: str) -> tuple[List[Step], List[str]]:
        """
        Processes a list of raw step data dictionaries using the helper method.
        """
        processed_steps: List[Step] = []
        all_validation_errors: List[str] = []
        # Build tool map once
        available_execution_tools: Dict[str, Tool] = {tool.name: tool for tool in self.execution_tools}

        for i, step_data in enumerate(raw_steps):
            sub_context = f"{context_prefix} step {i}"
            if not isinstance(step_data, dict):
                all_validation_errors.append(
                    f"{sub_context}: Expected step item to be a dict, but got {type(step_data)}. Data: {step_data}"
                )
                continue  # Skip this item

            # Call the helper to process this single step
            step, single_errors = await self._process_single_step(
                step_data, i, context_prefix, available_execution_tools
            )

            # Extend the list of errors collected (includes fatal and non-fatal)
            all_validation_errors.extend(single_errors)

            # Add the step ONLY if processing was successful (step is not None)
            if step:
                processed_steps.append(step)
            # If step is None, it means a fatal validation error occurred,
            # and the errors have already been added to all_validation_errors.

        return processed_steps, all_validation_errors

    async def _validate_structured_output_plan(
        self, task_data: dict, objective: str
    ) -> tuple[Optional[Task], List[str]]:
        """Validates the plan data received from structured output.

        This method is used when the LLM generates the plan as direct JSON
        output rather than through a tool call. It involves:
        1.  Processing the list of steps using `_process_step_list`.
        2.  Validating dependencies between the processed steps using
            `_validate_dependencies`.

        Args:
            task_data: A dictionary representing the entire task plan, typically
                       with "title" and "steps" keys, as generated by the LLM.
            objective: The original user objective, used as a fallback title.

        Returns:
            A tuple containing:
                - A `Task` object if the plan is valid, otherwise None.
                - A list of all validation error messages encountered.
        """
        all_validation_errors: List[str] = []

        # Validate the steps first
        steps, step_validation_errors = await self._process_step_list(task_data.get("steps", []), "structured output")
        all_validation_errors.extend(step_validation_errors)

        log.debug("Subtasks processed: %s", steps)
        log.debug("Subtask validation errors: %s", step_validation_errors)

        # If step processing had fatal errors, don't proceed to dependency check
        if not steps and task_data.get("steps"):  # Check if steps were provided but processing failed
            # Errors are already in all_validation_errors
            return None, all_validation_errors

        # Validate dependencies only if steps were processed successfully
        if steps:
            # Log the plan details before auto-wiring
            log.info("=== Plan received from LLM (before auto-wiring) ===")
            for st in steps:
                log.info(
                    "  Subtask: id=%s, depends_on=%s, content=%s...",
                    st.id,
                    st.depends_on,
                    (st.instructions or "")[:50],
                )

            # Apply schema overrides and auto-wire dependencies before validation
            try:
                self._apply_schema_overrides(steps)
            except ValueError as exc:
                all_validation_errors.append(str(exc))

            # Log the plan details after auto-wiring
            log.info("=== Plan after auto-wiring ===")
            for st in steps:
                log.info(
                    "  Subtask: id=%s, depends_on=%s",
                    st.id,
                    st.depends_on,
                )

            dependency_errors = self._validate_dependencies(steps)
            all_validation_errors.extend(dependency_errors)
            # Enforce semantic rules (no discovery in final plan, fan-out, aggregator wiring)
            semantic_errors = self._validate_plan_semantics(steps)
            all_validation_errors.extend(semantic_errors)

        # If any fatal errors occurred anywhere, return None
        if any(err for err in all_validation_errors):  # Check if there are any errors at all
            # Check specifically for fatal errors that would prevent Task creation
            # (e.g., invalid structure from _process_step_list, dependency errors)
            # For simplicity here, consider *any* validation error as potentially blocking.
            # A more nuanced check could differentiate warnings from fatal errors if needed.
            is_fatal = True  # Assume any error is fatal for now
            if is_fatal:
                return None, all_validation_errors

        # If validation passed (or only non-fatal errors occurred)
        return (
            Task(
                title=task_data.get("title", objective),
                steps=steps,  # Use the processed and validated steps
            ),
            all_validation_errors,
        )  # Return task and any non-fatal errors

    async def _validate_and_build_task_from_tool_calls(
        self, tool_calls: List[ToolCall], history: List[Message]
    ) -> tuple[Optional[Task], List[str]]:
        """Processes 'create_task' tool calls, validates steps and dependencies.

        This method handles the arguments from one or more `create_task` tool
        calls made by the LLM. For each such call, it:
        1.  Extracts the task title and the list of raw step data.
        2.  Processes the raw steps using `_process_step_list` to convert
            them into `Step` objects and collect validation errors.
        3.  Appends a "tool" role message to the history acknowledging the call
            and summarizing any validation issues for that specific call.

        After processing all `create_task` calls, it:
        4.  Validates dependencies across *all* collected steps from *all* calls
            using `_validate_dependencies`, and applies semantic validation with
            `_validate_plan_semantics`.

        If any validation errors occur at any stage (either within a single
        step, a tool call's step list, or in the final dependency check),
        it raises a `ValueError` to trigger retry logic in the calling function
        (`_generate_with_retry`).

        Args:
            tool_calls: A list of `ToolCall` objects from the LLM's message.
            history: The current planning conversation history. Tool response
                     messages will be appended to this list.

        Returns:
            A tuple containing:
                - A `Task` object if all validations pass and steps exist.
                - An empty list of validation errors (as errors trigger an exception).

        Raises:
            ValueError: If any validation errors are found during the processing
                        of steps or overall plan dependencies, or if a
                        `create_task` call results in no valid steps.
        """
        all_steps: List[Step] = []
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
                raw_steps_list = tool_call.args.get("steps", [])
                if not isinstance(raw_steps_list, list):
                    all_validation_errors.append(
                        f"{context_prefix}: 'steps' field must be a list, but got {type(raw_steps_list)}."
                    )
                    # Skip processing this call's steps if field is invalid, but record error
                    steps, validation_errors = [], []  # Ensure these are empty lists
                else:
                    # Use the updated _process_step_list
                    steps, validation_errors = await self._process_step_list(raw_steps_list, context_prefix)

                all_steps.extend(steps)
                all_validation_errors.extend(validation_errors)
                # --- End Subtask Processing ---

                # Add tool response message *after* processing its args
                response_content = "Task parameters received and processed."
                call_specific_errors = [e for e in validation_errors if context_prefix in e]
                if call_specific_errors:
                    response_content = (
                        f"Task parameters received, but validation errors occurred: {'; '.join(call_specific_errors)}"
                    )

                history.append(Message(role="tool", content=response_content, tool_call_id=tool_call.id))
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

        # --- Validate Dependencies *after* collecting all steps ---
        # Only run if there were steps and no fundamental structural errors earlier
        if all_steps:
            try:
                self._apply_schema_overrides(all_steps)
            except ValueError as exc:
                all_validation_errors.append(str(exc))

        if all_steps and not any("must be a list" in e for e in all_validation_errors):
            dependency_errors = self._validate_dependencies(all_steps)
            all_validation_errors.extend(dependency_errors)
            semantic_errors = self._validate_plan_semantics(all_steps)
            all_validation_errors.extend(semantic_errors)
        # --- End Dependency Validation ---

        # Check if any errors occurred
        if all_validation_errors:
            # Raise ValueError to trigger retry logic in _generate_with_retry
            error_string = "\n".join(all_validation_errors)
            raise ValueError(f"Validation errors in created task:\n{error_string}")

        # Ensure we actually created steps if the call succeeded validation
        if not all_steps and any(tc.name == "create_task" for tc in tool_calls):
            # Check if create_task was called but resulted in no valid steps
            # This might happen if the steps list was empty or all items failed validation
            raise ValueError("Task creation tool call processed, but resulted in zero valid steps.")

        # If validation passed and steps exist
        return (
            Task(title=task_title, steps=all_steps),
            all_validation_errors,
        )  # Return task and empty error list

    async def _process_tool_calls(self, message: Message, history: List[Message]) -> Task:
        """
        Helper method to process tool calls, create task, and handle validation.
        Delegates validation logic to _validate_and_build_task_from_tool_calls.
        """
        if not message.tool_calls:
            raise ValueError(f"No tool calls found in the message: {message.content}")

        # Delegate the core processing and validation
        task, _validation_errors = await self._validate_and_build_task_from_tool_calls(message.tool_calls, history)

        # _validate_and_build_task_from_tool_calls raises ValueError on failure,
        # so if we reach here, validation passed.
        if task is None:
            # This case should technically be handled by the exception in the helper,
            # but added for defensive programming.
            raise ValueError("Task validation passed but task object is unexpectedly None.")

        return task

    async def _generate_with_retry(
        self, history: List[Message], tools: List[Tool], max_retries: int = 3
    ) -> tuple[Optional[Task], Optional[Message]]:
        """
        Generates response, processes tool calls with validation and retry logic.

        DEPRECATED: This method is no longer used by the TaskPlanner.
        The planner now uses JSON output extraction in _run_plan_creation_phase.
        This method is kept for backward compatibility only.
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
                max_tokens=PLAN_CREATION_MAX_TOKENS,
            )
            history.append(message)  # Add assistant's response to history *before* processing
            last_message = message
            log.debug("LLM response received, has_tool_calls=%s", bool(message.tool_calls))

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
                    "Tool calls processed successfully, created task with %d steps",
                    len(task.steps),
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

    def _format_tools_info(
        self,
        tools: Sequence[Tool],
        heading: str,
        empty_message: str,
    ) -> str:
        """
        Shared formatter for tool descriptions.
        """
        if not tools:
            return empty_message

        tools_info = f"{heading}\n"
        for tool in tools:
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
        return tools_info.strip()

    def _get_execution_tools_info(self) -> str:
        """
        Describe the tools available to steps during execution.
        """
        return self._format_tools_info(
            self.execution_tools,
            "Available execution tools for steps:",
            "No execution tools available",
        )

    def _get_input_files_info(self) -> str:
        """
        Get formatted string information about initial input files.

        Since we're now using object-based data flow instead of files,
        this always returns a message indicating no file dependencies.

        Returns:
            A string indicating no file-based inputs are used.
        """
        return "No file-based inputs (data flows through task dependencies)"

    def _detect_list_processing(self, objective: str, input_files: List[str]) -> tuple[bool, int]:
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
                if any(ext in file.lower() for ext in [".csv", ".jsonl", "list", "urls"]):
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
            if "properties" in schema and isinstance(schema["properties"], dict) and "required" not in schema:
                prop_names = list(schema["properties"].keys())
                if prop_names:  # Only add 'required' if there are properties
                    schema["required"] = prop_names

        # Handle arrays - add a default items field if 'items' is missing
        elif isinstance(schema, dict) and schema.get("type") == "array" and "items" not in schema:
            schema["items"] = {"type": "string"}  # Default to string items

        # Recursively process nested schemas within properties
        if isinstance(schema, dict) and "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                # Ensure prop_schema is a dict before recursing
                if isinstance(prop_schema, dict):
                    schema["properties"][prop_name] = self._ensure_additional_properties_false(prop_schema)

        # Recursively process nested schemas within array items
        if isinstance(schema, dict) and "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, dict):
                schema["items"] = self._ensure_additional_properties_false(items_schema)
            elif isinstance(items_schema, list):  # Handle tuple schemas in 'items'
                schema["items"] = [
                    (self._ensure_additional_properties_false(item) if isinstance(item, dict) else item)
                    for item in items_schema
                ]

        # Recursively process anyOf, allOf, oneOf schemas
        for key in ["anyOf", "allOf", "oneOf"]:
            if isinstance(schema, dict) and key in schema and isinstance(schema[key], list):
                schema[key] = [
                    (self._ensure_additional_properties_false(item) if isinstance(item, dict) else item)
                    for item in schema[key]
                ]

        return schema
