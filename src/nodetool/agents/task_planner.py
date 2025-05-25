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

import asyncio
import logging
import traceback
from nodetool.chat.providers import ChatProvider
from nodetool.agents.sub_task_context import (
    FILE_POINTER_SCHEMA,
    is_binary_output_type,
    json_schema_for_output_type,
)
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
import os
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

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COMPACT_SUBTASK_NOTATION_DESCRIPTION = """
--- Data Flow Representation (DOT/Graphviz Syntax) ---
This format helps visualize the flow of data between files and process steps.
Focus on file nodes and process step nodes, connected by arrows showing data movement.
**Note: All file paths (`input_files`, `output_file`) can refer to either files or directories.**
Example DOT for a pipeline:
digraph DataPipeline {
  "users.json" -> "merge_data";
  "logs.csv" -> "merge_data";
  "merge_data" -> "filter_activity";
  "filter_activity" -> "filtered_logs.csv";
  "filtered_logs.csv" -> "generate_report";
  "generate_report" -> "summary.pdf";
}
Used for concise subtask representation.
"""

# --- Task Naming Convention ---
# Use short names, prefixed with $, for conceptual tasks or process steps (e.g., $read_data, $process_logs, $generate_report).
# These names should be unique and descriptive of the step.


def clean_and_validate_path(workspace_dir: str, path: Any, context: str) -> str:
    """
    Cleans workspace prefix and validates a path is relative and safe.
    """
    # Stricter initial type check
    if not isinstance(path, str):
        raise ValueError(
            f"Path must be a string, but got type {type(path)} ('{path}') in {context}."
        )
    if not path.strip():
        raise ValueError(f"Path must be a non-empty string in {context}.")

    # Ensure the path is not absolute
    if Path(path).is_absolute():  # Use pathlib for robustness
        raise ValueError(
            f"Path must be relative, but got absolute path '{path}' in {context}."
        )

    cleaned_path: str = path

    # Remove leading workspace prefixes more robustly
    # Convert potential backslashes for consistency before prefix check
    normalized_prefix_path = cleaned_path.replace("\\", "/")
    if normalized_prefix_path.startswith("workspace/"):
        cleaned_path = cleaned_path[len("workspace/") :]
    # Check for absolute /workspace/ only if it wasn't caught by is_absolute earlier (edge case)
    elif normalized_prefix_path.startswith("/workspace/"):
        cleaned_path = cleaned_path[len("/workspace/") :]

    # After cleaning, double-check it didn't somehow become absolute or empty
    if not cleaned_path:
        raise ValueError(
            f"Path became empty after cleaning prefix: '{path}' in {context}."
        )
    # Re-check absoluteness after potential prefix stripping
    if Path(cleaned_path).is_absolute():
        raise ValueError(
            f"Path became absolute after cleaning prefix: '{path}' -> '{cleaned_path}' in {context}."
        )

    # --- Path Traversal Check ---
    try:
        workspace_root = Path(workspace_dir).resolve(
            strict=True
        )  # Ensure workspace exists
        # Create the full path by joining workspace root and the relative path
        full_path = (workspace_root / cleaned_path).resolve()

        # Check if the resolved path is within the workspace directory
        # Using Path.is_relative_to (Python 3.9+) or common path check
        is_within = False
        try:
            # Preferred method if available
            is_within = full_path.is_relative_to(workspace_root)
        except AttributeError:  # Fallback for Python < 3.9
            is_within = (
                workspace_root == full_path or workspace_root in full_path.parents
            )

        if not is_within:
            raise ValueError(
                f"Path validation failed: Resolved path '{full_path}' is outside the workspace root '{workspace_root}' for input '{path}' in {context}."
            )

    except FileNotFoundError:
        # This occurs if self.workspace_dir doesn't exist during validation
        raise ValueError(
            f"Workspace directory '{workspace_dir}' not found during path validation for '{path}' in {context}."
        )
    except Exception as e:  # Catch other potential resolution or validation errors
        raise ValueError(f"Error validating path safety for '{path}' in {context}: {e}")

    # Return the cleaned, validated relative path
    return cleaned_path


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
                        "content": {
                            "type": "string",
                            "description": "High-level natural language instructions for the agent executing this subtask.",
                        },
                        "output_file": {
                            "type": "string",
                            "description": "The file path where the subtask will save its output. MUST be relative to the workspace root and at the top level (e.g., 'result.json'). Do NOT use subdirectories, absolute paths, or '/workspace/' prefix.",
                        },
                        "input_files": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "description": "An input file for the subtask. MUST be a relative path at the top level of the workspace (e.g., 'source.txt' or 'another_output.json'). Paths must not be in subdirectories. Corresponds to an initial input file or the output_file of another subtask.",
                            },
                        },
                        "output_schema": {
                            "type": "string",
                            "description": 'Output schema for the subtask as a JSON string. Use \'{"type": "string"}\' for unstructured output types.',
                        },
                        "output_type": {
                            "type": "string",
                            "description": "The file format of the output of the subtask, e.g. 'json', 'markdown', 'csv', 'html'",
                        },
                        "max_iterations": {
                            "type": "integer",
                            "description": "Maximum number of iterations allowed for the agent executing this subtask. Adjust based on complexity (e.g., 5 for simple tasks, 10-15 for complex analysis). Default is 10.",
                        },
                        "model": {
                            "type": "string",
                            "description": "Specific LLM model to use for this subtask (e.g., 'gpt-4-turbo' or 'claude-3-opus-20240229'). Defaults to the planner's primary model if not specified or invalid.",
                        },
                        "is_intermediate_result": {
                            "type": "boolean",
                            "description": "Whether the subtask is an intermediate result of a task",
                        },
                        "batch_processing": {
                            "type": "object",
                            "description": "Configuration for batch processing of list items",
                            "properties": {
                                "enabled": {"type": "boolean"},
                                "batch_size": {"type": "integer"},
                                "start_index": {"type": "integer"},
                                "end_index": {"type": "integer"},
                                "total_items": {"type": "integer"},
                            },
                            "required": [
                                "enabled",
                                "batch_size",
                                "start_index",
                                "end_index",
                                "total_items",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required": [
                        "content",
                        "output_file",
                        "output_type",
                        "output_schema",
                        "input_files",
                        "max_iterations",
                        "model",
                        "is_intermediate_result",
                        "batch_processing",
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


# Simplified and phase-agnostic system prompt
DEFAULT_PLANNING_SYSTEM_PROMPT = """
# TaskArchitect System Core Directives

## Goal
As TaskArchitect, your primary goal is to transform complex user objectives 
into executable, multi-phase task plans. You will guide the LLM through 
distinct phases (Analysis, Data Flow, Plan Creation) to produce a valid 
and optimal plan.

## Return Format
This system prompt establishes your operational context. For each 
subsequent phase, you will receive specific instructions detailing the 
required output format. Your cumulative output across all phases will be 
a well-structured task plan, ultimately generated via the `create_task` 
tool.

## Warnings & Core Principles
Adhere strictly to these overarching principles and warnings throughout 
all planning activities:

1.  **Multi-Phase Process:** Follow the structured, multi-phase approach 
    (Analysis -> Data Flow -> Plan Creation). Each phase has specific 
    objectives.
2.  **Atomic Subtasks:** Decompose objectives into the smallest logical, 
    executable units (subtasks).
3.  **Clear Data Flow:** Define explicit data dependencies between 
    subtasks using `input_files` and `output_file`. Ensure this forms a 
    Directed Acyclic Graph (DAG – no cycles).
4.  **Relative & Unique Paths:** ALL file paths MUST be relative to the 
    workspace root (e.g., `file.txt`) and `output_file` paths MUST 
    be unique. Absolute paths or `workspace/` prefixes are forbidden. 
    All file paths MUST be flat at the workspace root (e.g., 
    `interim_file.json`, `final_output.pdf`), not in subdirectories. 
    Paths *refer to files only*, not directories, unless they are initial 
    input directories.
5.  **Tool Usage:** When a phase requires generating the plan (Phase 2), 
    use the `create_task` tool as instructed.
6.  **Optimization:** Design plans that minimize unnecessary context.
7.  **Result Passing:** Plan for subtasks to pass results efficiently, 
    preferring file pointers (`{"path": "relative/path"}`) for non-trivial 
    data.
"""

# Make sure the phase prompts are concrete and focused
ANALYSIS_PHASE_TEMPLATE = """
# PHASE 0: OBJECTIVE ANALYSIS & CONCEPTUAL SUBTASK BREAKDOWN

## Goal
Your primary goal in this phase is to deeply understand the user's objective.
Based on this understanding, and considering available input files and overall
output requirements, you will then devise a strategic execution plan. This plan
will consist of a list of conceptual subtasks designed to achieve the objective.

Specifically, you must:
- **Clarify Objective Understanding:**
    - Articulate your interpretation of the user's core goal.
    - Identify any ambiguities in the objective and state your assumptions.
    - Consider how the desired final output (type and schema) informs the objective.
- **Devise Strategic Plan:**
    - Based on your understanding, decompose the objective into distinct, atomic
      subtasks suitable for agent execution.
    - If the objective involves processing multiple similar items independently
      (e.g., a list of files or URLs), plan for a separate subtask for each
      item where appropriate.
    - Design subtasks with minimal execution context in mind.

## Return Format
Provide your output in the following structure:

1.  **Objective Understanding & Strategic Reasoning:**
    *   **Objective Interpretation:**
        *   Clearly state your interpretation of the user's objective.
        *   Explain any key aspects, constraints, or desired outcomes you've
            inferred from the objective and the overall task output requirements.
        *   If there were ambiguities in the objective, explicitly state the
            assumptions you made to resolve them.
    *   **Subtask Rationale:**
        *   Explain how your interpretation of the objective led to the
            proposed conceptual subtask breakdown.
        *   Detail your thought process for identifying each conceptual subtask.
        *   Describe how you considered parallel processing and planned the
            initial data flow at a high level.
    *   When referring to conceptual tasks in your reasoning, use the
        '$short_name' convention (e.g., $read_data, $process_item).

2.  **Task List:**
    *   List each conceptual subtask.
    *   For each subtask, provide its '$short_name' and a brief, clear
        description on a single line.
    *   Example: `$process_file - Processes the content of a single input
        file.`

## Warnings
- Focus on *conceptual* subtasks here. Detailed file paths and schemas
  will be defined in the next phase.
- Ensure your reasoning clearly justifies the proposed subtask breakdown based
  on your understanding of the objective.
- The '$short_name' for each task should be unique and descriptive.

## Context Dump
**User's Objective:**
{{ objective }}

**Overall Task Output Requirements:**
- Type: {{ overall_output_type }}
- Schema: {{ overall_output_schema }}

**Execution Tools Available (for the Agent to potentially use later):**
{{ execution_tools_info }}

**Initial Input Files:**
{{ input_files_info }}
"""

DATA_FLOW_ANALYSIS_TEMPLATE = """
# PHASE 1: DATA FLOW ANALYSIS - Defining Dependencies and Contracts

## Goal
Your goal in this phase is to refine the conceptual subtask plan from Phase 0 
by defining the precise data flow, dependencies, and contracts (input/output 
files) for each subtask. You should also visualize this data flow.

Specifically, you must:
- Review the conceptual subtask list generated in Phase 0 (available in the 
  conversation history).
- For each conceptual subtask, determine its `input_files` and a unique 
  `output_file`. These paths can refer to files or directories and MUST be 
  relative.
- Establish the dependencies between subtasks based on these file inputs 
  and outputs.
- Refine the high-level instructions for each subtask if necessary, keeping 
  them concise.
- Ensure the subtask(s) producing the final result will generate output 
  matching the overall task requirements.

## Return Format
Provide your output in the following structure:

1.  **Reasoning & Data Flow Analysis:**
    *   Explain your data flow design choices, detailing how `input_files` 
        and `output_file` were determined for each subtask from Phase 0.
    *   Describe any refinements made to subtask instructions.
    *   Use the '$name' convention when referring to tasks from Phase 0 in 
        your reasoning.

2.  **Data Flow (DOT/Graphviz Representation):**
    *   Describe the data flow between all subtasks and files using the 
        DOT/Graphviz syntax (see "Data Flow Representation (DOT/Graphviz 
        Syntax) Guide" in Context Dump below).
    *   The process steps in your DOT graph should correspond to the 
        '$short_name' of the conceptual subtasks.
    *   Example:
        ```dot
        digraph DataPipeline {
          "input_file1.json" -> "$process_step_A";
          "input_file2.csv" -> "$process_step_A";
          "$process_step_A" -> "intermediate_A.json"; // Output of 
          $process_step_A
          "intermediate_A.json" -> "$process_step_B";
          "$process_step_B" -> "final_output.pdf";      // Output of 
          $process_step_B
        }
        ```

## Warnings
- **DAG Requirement:** The defined dependencies MUST form a Directed Acyclic Graph (DAG). No circular dependencies.
- **Relative & Unique Paths:** All file paths (`input_files`, `output_file`) MUST be relative to the workspace root and at the top level (e.g., `temp_file.csv`, `report.pdf`). Each `output_file` MUST be unique. No subdirectories. Do NOT use absolute paths or the `/workspace/` prefix. Paths refer to files only.
- **Clarity:** Your reasoning should clearly link the conceptual tasks from 
  Phase 0 to the concrete data flow defined here.
- **Final Output:** Ensure the final subtask(s) produce output matching the 
  overall task requirements (Type: `{{ overall_output_type }}`, Schema: 
  `{{ overall_output_schema }}`).

## Context Dump
**User's Objective:**
{{ objective }}

**Conceptual Subtask Plan from Phase 0:**
(This will be available in the preceding conversation history. Please refer to 
it.)

**Overall Task Output Requirements:**
- Type: {{ overall_output_type }}
- Schema: {{ overall_output_schema }}

**Execution Tools Available (for the Agent to potentially use later):**
{{ execution_tools_info }}

**Initial Input Files (available to the first subtasks):**
{{ input_files_info }}

**Data Flow Representation (DOT/Graphviz Syntax) Guide:**
{{ COMPACT_SUBTASK_NOTATION_DESCRIPTION }}
"""

PLAN_CREATION_TEMPLATE = """
# PHASE 2: PLAN CREATION - Generating the Executable Task

## Goal
Your primary goal in this phase is to transform the conceptual subtask plan 
and data flow analysis from the previous phases into a concrete, executable 
task plan. This is achieved by making a single, valid call to the `create_task` 
tool.

You must also provide detailed reasoning for the construction of this final 
plan *before* making the tool call.

## Return Format
Your response MUST be structured as follows:

1.  **Detailed Reasoning and Final Plan Construction:**
    *   Provide your comprehensive reasoning for constructing the final 
        executable task plan.
    *   Explain how you translated the refined conceptual subtasks and data 
        flow (from Phase 1, including DOT graph and CSN strings if applicable) 
        into the specific arguments for the `create_task` tool.
    *   Detail your decisions for `content`, `input_files`, `output_file`, 
        `output_type`, `output_schema`, `max_iterations`, `model`, and `is_intermediate_result` for 
        each subtask.
    *   Confirm how you ensured all validation criteria (see Warnings section) 
        were met, including dependency management (DAG), path correctness, 
        and final output conformance.
    *   If helpful, mentally construct or briefly describe the final data flow 
        (as a DOT concept) based on the `input_files` and `output_file` you 
        are defining for the `create_task` call.
    *   Refer to conceptual tasks using the '$name' convention where 
        appropriate in your reasoning.

2.  **`create_task` Tool Call:**
    *   Immediately following your reasoning, make exactly one call to the 
        `create_task` tool to generate the entire plan.
    *   Populate the tool call arguments meticulously based on your reasoning 
        and the guidelines below.

## Warnings: `create_task` Tool Call Guidelines
Adhere strictly to the following when constructing the arguments for the 
`create_task` tool:

**General Structure:**
*   **Overall Task `title`:** Set an appropriate title for the overall task.
*   **Subtasks Array:** Construct the `subtasks` array, where each item is 
    an object representing a subtask.

**For Each Subtask:**
*   **`content` (Agent Instructions):**
    *   MUST be high-level natural language instructions for the agent.
    *   Focus on *what* the agent should achieve, not *how*.
*   **`input_files`:**
    *   Determine from the data flow graph (Phase 1).
    *   List all relative paths to files/directories this subtask depends on.
*   **`output_file`:**
    *   Determine from the data flow graph (Phase 1).
    *   MUST be a unique, relative file path at the workspace root (e.g., `data.json`, `final_report.pdf`). No subdirectories.
*   **`output_type`:** Specify the correct file format of the output (e.g., 
    'json', 'markdown', 'csv').
*   **`output_schema`:**
    *   Provide as a valid JSON string (e.g., `'{"type": "string"}'` for 
        unstructured, or a more detailed schema).
    *   Derive from `output_type` or use a default if appropriate.
*   **`max_iterations`:**
    *   Set an integer value based on the subtask's complexity.
    *   Use values like 5 for simple tasks, 10-15 for complex analysis. 
        Default to 10 if unsure.
*   **`model` (LLM Assignment):**
    *   Assign the appropriate LLM:
        *   Use `{{ model }}` (primary model) for standard tasks.
        *   Use `{{ reasoning_model }}` (reasoning model) for subtasks 
            requiring complex analysis, code generation, or significant reasoning.
    *   If unsure, default to `{{ model }}`.
*   **`is_intermediate_result` (Boolean):**
    *   Set to `true` if the subtask's `output_file` is primarily consumed by another subtask in the plan and is NOT a primary final output of the overall objective.
    *   Set to `false` if the subtask's `output_file` represents a final deliverable for the user, or if it's a terminal output that directly contributes to the overall task goal (even if it's not the *only* final output).
    *   **Crucially for multiple outputs:** If the objective requires several distinct final files (e.g., a text report and a separate image), ensure that each subtask producing one such final file has `is_intermediate_result: false`.
    *   Consider the `Overall Task Output Requirements` when deciding this for terminal subtasks.

**Plan-Wide Validation (Perform these checks *before* calling the tool):**
*   **Path Correctness:** All file paths (`input_files`, `output_file`) 
    MUST be relative to the workspace root and at the top level (e.g., `data.txt` or `final_report.csv`). No subdirectories. No absolute paths or `/workspace/` prefix. Paths refer to files only.
*   **Unique Output Files:** Ensure ALL `output_file` paths are unique across 
    the entire plan.
*   **DAG Dependencies:** Subtask dependencies, as defined by `input_files` 
    referencing `output_file` of other tasks, MUST form a Directed Acyclic 
    Graph (DAG). No cycles.
*   **Input Availability:** Ensure inputs for a subtask are produced by 
    preceding tasks or are part of the initial `Input Files`.
*   **Context Minimization:** Keep subtasks focused. Prefer passing results 
    via file pointers (`{"path": "..."}`) within the agent's `finish_subtask` 
    call to keep context small, especially for larger data.
*   **Final Output Conformance:** The plan's final output (from the terminal 
    subtask(s)) MUST conform to the overall task requirements (Type: 
    `{{ overall_output_type }}`, Schema: `{{ overall_output_schema }}`).

**Agent Execution Notes (for your planning awareness):**
*   Agents executing these subtasks will call `finish_subtask`.
*   They will pass small result content directly or, preferably, a file 
    pointer `{"path": "relative/path/..."}` for larger data/files.
*   If a subtask requires significant reasoning, the *agent* might use a 
    reasoning tool; this is distinct from your `model` selection for the 
    subtask itself.

## Context Dump
**User's Objective:**
{{ objective }}

**Conceptual Subtasks & Data Flow (from Phase 1):**
(Refer to the conversation history for the refined subtask list and DOT 
graph from Phase 1. You should parse any CSN strings or the DOT graph to 
inform your `create_task` call.)

**Overall Task Output Requirements:**
- Type: {{ overall_output_type }}
- Schema: {{ overall_output_schema }}

**Planner's LLM Models Available:**
- Primary Model: `{{ model }}`
- Reasoning Model: `{{ reasoning_model }}`

**Execution Tools Available (for the Agent to potentially use within a subtask):**
{{ execution_tools_info }}

**Initial Input Files (available to the first subtasks):**
{{ input_files_info }}
"""

# Agent task prompt used in the final planning stage
DEFAULT_AGENT_TASK_TEMPLATE = """
---
## Final Check: Subtask Design within `create_task`

As you finalize the `create_task` call, please double-check these crucial 
aspects for *each* subtask:

1.  **Clarity of Purpose:** Is the `content` a crystal-clear, high-level 
    objective for an autonomous agent?
2.  **Self-Containment (Inputs):** Does `input_files` correctly list *all* 
    necessary data for the subtask to run independently once those files 
    are available?
3.  **Output Precision:** Is the `output_file` path unique? Does 
    `output_type` and `output_schema` accurately describe what will be 
    produced?

Ensure your overall plan effectively addresses the original `{{ objective }}` 
using the available resources and adheres to the final output requirements.
"""


# --- Tiered Planning ---
# Added a new enum-like structure for planning tiers
class PlanningTier:
    """Defines the complexity tiers for planning strategies."""

    TIER_LOW_COMPLEXITY = "Low Complexity"  # Analysis -> Plan Creation
    TIER_HIGH_COMPLEXITY = "High Complexity"  # Analysis -> Data Flow -> Plan Creation
    DEFAULT = TIER_HIGH_COMPLEXITY  # Default to full planning if reflection fails

    @classmethod
    def from_string(cls, s: str) -> str:
        """Parse a string to determine the corresponding planning tier.

        Args:
            s: The string to parse, expected to contain "Low Complexity" or "High Complexity".

        Returns:
            The identified planning tier string
            or the default tier if no match is found.
        """
        if "Low Complexity" in s:  # Match the exact string for robustness
            return cls.TIER_LOW_COMPLEXITY
        elif "High Complexity" in s:  # Match the exact string
            return cls.TIER_HIGH_COMPLEXITY
        return cls.DEFAULT


SELF_REFLECTION_PROMPT_TEMPLATE = """
# Objective Complexity Assessment

## Context:
You are an expert at assessing the complexity of a given objective to determine the most efficient planning strategy.

**Objective:**
{{ objective }}

**Initial Input Files:**
{{ input_files_info }}

**Overall Task Output Requirements:**
- Type: {{ overall_output_type }}
- Schema: {{ overall_output_schema }}

## Task:
Based on the objective, inputs, and desired outputs, classify the objective's complexity into one of the following tiers.
Respond with ONLY the tier name (e.g., "Low Complexity" or "High Complexity"). Your response must contain nothing else.

*   **Low Complexity:** The planning process involves an initial analysis of the objective to define conceptual subtasks, followed directly by the creation of the executable plan. This tier bypasses the explicit data flow definition phase. (Phases: Analysis -> Plan Creation)
*   **High Complexity:** The planning process is comprehensive, including an initial analysis to define conceptual subtasks, a dedicated phase to define detailed data flows and dependencies between these subtasks, and finally, the creation of the executable plan. (Phases: Analysis -> Data Flow Analysis -> Plan Creation)

**Your Classification (Low Complexity or High Complexity):**
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
    4.  **Contracts:** Defining the expected data format (`output_type`,
        `output_schema`) for each subtask's output, promoting type safety and
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
        use_structured_output (bool): If True, attempts to use LLM's structured output
                                      features for plan generation, otherwise uses tool calls.
        verbose (bool): Enables detailed logging and progress display during planning.
        display_manager (AgentConsole): Handles Rich display output.
        jinja_env (Environment): Jinja2 environment for rendering prompts.
        tasks_file_path (Path): Path where the plan might be saved/loaded (`tasks.yaml`).
        current_planning_tier (str): The current planning tier determined by self-reflection.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        execution_tools: Sequence[Tool],
        reasoning_model: str | None = None,
        input_files: Sequence[str] = [],
        system_prompt: str | None = None,
        output_schema: dict | None = None,
        output_type: str | None = None,
        enable_analysis_phase: bool = True,
        enable_data_contracts_phase: bool = True,
        use_structured_output: bool = False,
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
            input_files (list[str]): The input files to use for planning
            system_prompt (str, optional): Custom system prompt
            output_schema (dict, optional): JSON schema for the final task output
            output_type (str, optional): The type of the final task output
            enable_analysis_phase (bool, optional): Whether to run the analysis phase (PHASE 0)
            enable_data_contracts_phase (bool, optional): Whether to run the data contracts phase (PHASE 1)
            use_structured_output (bool, optional): Whether to use structured output for plan creation
            verbose (bool, optional): Whether to print planning progress table (default: True)
        """
        logger.debug(
            f"Initializing TaskPlanner with model={model}, reasoning_model={reasoning_model}, "
            f"objective='{objective[:100]}...', workspace_dir={workspace_dir}, "
            f"enable_analysis_phase={enable_analysis_phase}, "
            f"enable_data_contracts_phase={enable_data_contracts_phase}, "
            f"use_structured_output={use_structured_output}, verbose={verbose}"
        )

        self.provider: ChatProvider = provider
        self.model: str = model
        self.reasoning_model: str = reasoning_model or model
        self.objective: str = objective
        self.workspace_dir: str = workspace_dir
        self.task_plan: TaskPlan = TaskPlan()
        # Clean and validate initial input files relative to workspace
        logger.debug(f"Processing {len(input_files)} initial input files")
        self.input_files: List[str] = [
            self._clean_and_validate_path(f, "initial input files") for f in input_files
        ]
        logger.debug(f"Cleaned input files: {self.input_files}")

        self.system_prompt: str = self._customize_system_prompt(system_prompt)
        logger.debug(
            f"System prompt customized, length: {len(self.system_prompt)} chars"
        )

        self.execution_tools: Sequence[Tool] = execution_tools or []
        logger.debug(
            f"Available execution tools: {[tool.name for tool in self.execution_tools]}"
        )

        self.output_schema: Optional[dict] = output_schema
        self.output_type: Optional[str] = output_type
        self.enable_analysis_phase: bool = enable_analysis_phase
        self.enable_data_contracts_phase: bool = enable_data_contracts_phase
        self.use_structured_output: bool = use_structured_output
        self.verbose: bool = verbose
        self.tasks_file_path: Path = Path(workspace_dir) / "tasks.yaml"
        self.display_manager = AgentConsole(verbose=self.verbose)
        self.current_planning_tier: str = PlanningTier.DEFAULT  # Added

        # Initialize Jinja2 environment
        self.jinja_env: Environment = Environment(loader=BaseLoader())

        logger.debug("TaskPlanner initialization completed successfully")

    def _clean_and_validate_path(self, path: Any, context: str) -> str:
        """Clean and validate a path relative to the workspace directory.

        This is a wrapper around the global `clean_and_validate_path` function,
        using the instance's `self.workspace_dir`.

        Args:
            path: The path to clean and validate.
            context: A string describing the context of the validation.

        Returns:
            The cleaned and validated relative path.

        Raises:
            ValueError: If the path is invalid.
        """
        logger.debug(f"Cleaning and validating path '{path}' in context: {context}")
        result = clean_and_validate_path(self.workspace_dir, path, context)
        logger.debug(f"Path validation result: '{result}'")
        return result

    def _customize_system_prompt(self, system_prompt: str | None) -> str:
        """Customize the system prompt based on provider capabilities.

        If a custom `system_prompt` is provided, it's used directly.
        Otherwise, `DEFAULT_PLANNING_SYSTEM_PROMPT` is used.

        Args:
            system_prompt: Optional custom system prompt string.

        Returns:
            The system prompt string to be used.
        """
        if system_prompt:
            base_prompt = system_prompt
        else:
            base_prompt = DEFAULT_PLANNING_SYSTEM_PROMPT

        return str(base_prompt)

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
                    self.display_manager.print(
                        f"[cyan]Loaded existing task plan from {self.tasks_file_path}[/cyan]"
                    )
                    return True
            except (
                Exception
            ) as e:  # Keep general exception for file I/O or parsing issues
                self.display_manager.print(  # Use display manager print
                    f"[yellow]Could not load or parse existing task plan from {self.tasks_file_path}: {e}[/yellow]"
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
        overall_output_type_str = (
            self.output_type if self.output_type else "Not specified (default: string)"
        )

        return {
            "objective": self.objective,
            "model": self.model,  # Add planner's primary model
            "reasoning_model": self.reasoning_model,  # Add planner's reasoning model
            "execution_tools_info": self._get_execution_tools_info(),
            "input_files_info": self._get_input_files_info(),
            "overall_output_type": overall_output_type_str,
            "overall_output_schema": overall_output_schema_str,
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

        The graph nodes represent subtasks (identified by their `output_file`).
        An edge from subtask A to subtask B means B depends on the output of A
        (i.e., one of B's `input_files` is A's `output_file`).

        Args:
            subtasks: A list of `SubTask` objects.

        Returns:
            A `networkx.DiGraph` representing the dependencies.
        """
        # Create mapping of output files to their subtasks
        output_to_subtask: Dict[str, SubTask] = {}
        for subtask in subtasks:
            output_to_subtask[subtask.output_file] = subtask

        G = nx.DiGraph()

        # Add nodes representing subtasks (using output_file as a unique ID proxy for the node)
        for subtask in subtasks:
            G.add_node(subtask.output_file)  # Node represents the subtask completion

        # Add edges for dependencies: from the *output file* of the dependency task to the *output file* of the current task
        for subtask in subtasks:
            if subtask.input_files:
                for input_file in subtask.input_files:
                    if input_file in output_to_subtask:
                        # Get the subtask that produces this input file
                        producer_subtask = output_to_subtask[input_file]
                        # Add edge from the producer's node to the current subtask's node
                        G.add_edge(producer_subtask.output_file, subtask.output_file)

        return G

    def _check_output_file_conflicts(
        self, subtasks: List[SubTask]
    ) -> tuple[List[str], Set[str]]:
        """Checks for duplicate output files among subtasks.

        Ensures that each subtask defines a unique `output_file`.
        It also checks that a subtask's primary `output_file` does not
        conflict with any artifact files if those were to be tracked separately
        (currently, artifacts are not explicitly handled here beyond primary outputs).

        Args:
            subtasks: A list of `SubTask` objects.

        Returns:
            A tuple containing:
                - A list of string error messages describing any conflicts found.
                - A set of all unique `output_file` paths generated by the subtasks.
        """
        validation_errors: List[str] = []
        output_files: Dict[str, SubTask] = {}
        all_generated_files: Set[str] = set()

        for i, subtask in enumerate(subtasks):
            sub_context = f"Subtask {i} ('{subtask.content[:30]}...')"
            # Check primary output file
            if subtask.output_file in output_files:
                validation_errors.append(
                    f"{sub_context}: Multiple subtasks trying to write primary output to '{subtask.output_file}'"
                )
            elif subtask.output_file in all_generated_files:
                validation_errors.append(
                    f"{sub_context}: Primary output file '{subtask.output_file}' conflicts with an artifact from another task."
                )
            else:
                output_files[subtask.output_file] = subtask
                all_generated_files.add(subtask.output_file)

        return validation_errors, all_generated_files

    def _check_input_file_availability(
        self, subtasks: List[SubTask], all_generated_files: Set[str]
    ) -> List[str]:
        """Checks if all input files for subtasks are available.

        An input file is considered available if it's one of the initial
        input files provided to the planner or if it's an `output_file`
        of another subtask in the plan.

        Args:
            subtasks: A list of `SubTask` objects.
            all_generated_files: A set of all `output_file` paths from the subtasks.

        Returns:
            A list of string error messages for any missing input file dependencies.
        """
        validation_errors: List[str] = []
        available_files: Set[str] = set(self.input_files)

        for subtask in subtasks:
            if subtask.input_files:
                for file_path in subtask.input_files:
                    if (
                        file_path not in available_files
                        and file_path not in all_generated_files
                    ):
                        validation_errors.append(
                            f"Subtask '{subtask.content}' depends on missing file '{file_path}'"
                        )
        return validation_errors

    def _validate_dependencies(self, subtasks: List[SubTask]) -> List[str]:
        """
        Validate dependencies, file conflicts, and DAG structure for subtasks.

        This method performs several checks:
        1.  Output file conflicts: Ensures no two subtasks write to the same
            primary output file.
        2.  Cycle detection: Builds a dependency graph and checks for circular
            dependencies, which would make execution impossible.
        3.  Input file availability: Verifies that all `input_files` for each
            subtask are either part of the initial task inputs or are generated
            by a preceding subtask.
        4.  Topological sort feasibility: Checks if a valid linear execution
            order for the subtasks can be determined.

        Args:
            subtasks: A list of `SubTask` objects to validate.

        Returns:
            A list of strings, where each string is an error message
            describing a validation failure. An empty list indicates
            all dependency checks passed.
        """
        logger.debug(f"Starting dependency validation for {len(subtasks)} subtasks")
        validation_errors: List[str] = []

        # 1. Check for output file conflicts
        logger.debug("Checking for output file conflicts")
        output_conflict_errors, all_generated_files = self._check_output_file_conflicts(
            subtasks
        )
        validation_errors.extend(output_conflict_errors)
        logger.debug(
            f"Output conflict check found {len(output_conflict_errors)} errors"
        )

        # 2. Build dependency graph
        logger.debug("Building dependency graph")
        G = self._build_dependency_graph(subtasks)
        logger.debug(
            f"Dependency graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        # 3. Check for cycles
        logger.debug("Checking for circular dependencies")
        try:
            cycle = nx.find_cycle(G)
            cycle_error = f"Circular dependency detected: {cycle}"
            validation_errors.append(cycle_error)
            logger.warning(f"Circular dependency found: {cycle}")
        except nx.NetworkXNoCycle:
            logger.debug("No circular dependencies found")
            pass  # No cycles found, which is good

        # 4. Validate all input files exist before their use
        logger.debug("Checking input file availability")
        input_availability_errors = self._check_input_file_availability(
            subtasks, all_generated_files
        )
        validation_errors.extend(input_availability_errors)
        logger.debug(
            f"Input availability check found {len(input_availability_errors)} errors"
        )

        # 5. Check if a valid execution order exists (topological sort)
        if not validation_errors:  # Only check if no other critical errors found
            logger.debug("Checking topological sort feasibility")
            try:
                topo_order = list(nx.topological_sort(G))
                logger.debug(f"Valid execution order found: {topo_order}")
            except nx.NetworkXUnfeasible:
                # This might be redundant if cycle check or input availability check failed,
                # but provides an extra layer of verification.
                error_msg = "Cannot determine valid execution order due to unresolved dependency issues (potentially complex cycle or missing input)."
                validation_errors.append(error_msg)
                logger.error(f"Topological sort failed: {error_msg}")
        else:
            logger.debug(
                "Skipping topological sort check due to existing validation errors"
            )

        logger.debug(
            f"Dependency validation completed with {len(validation_errors)} total errors"
        )
        return validation_errors

    async def _run_self_reflection_phase(
        self, history: List[Message]
    ) -> tuple[List[Message], Optional[PlanningUpdate]]:
        """Runs self-reflection to determine planning complexity tier.

        This phase uses the LLM to assess the user's objective and classify
        its complexity into high or low. The result dictates which subsequent planning
        phases (Analysis, Data Flow) are executed.

        The prompt used for self-reflection (`SELF_REFLECTION_PROMPT_TEMPLATE`)
        is not added to the main planning history.

        Args:
            history: The current list of messages in the planning conversation.
                     This list is not modified by this method.

        Returns:
            A tuple containing:
                - The original, unmodified history.
                - A `PlanningUpdate` object summarizing the outcome of this phase,
                  or None if an error occurs that prevents update generation.
        """
        logger.debug("Starting self-reflection phase")

        # Skip self-reflection if data contracts phase is disabled globally
        if not self.enable_data_contracts_phase:
            logger.debug("Skipping self-reflection: data contracts phase disabled")
            self.display_manager.update_planning_display(
                "-1. Self-Reflection",
                "Skipped",
                Text(
                    "Skipped as Data Contracts phase is disabled. Tier defaulted to Low Complexity."
                ),
            )
            self.current_planning_tier = PlanningTier.TIER_LOW_COMPLEXITY
            planning_update = PlanningUpdate(
                phase="Self-Reflection",
                status="Skipped",
                content="Skipped as Data Contracts phase is disabled. Tier defaulted to Low Complexity.",
            )
            return history, planning_update

        reflection_prompt_content: str = self._render_prompt(
            SELF_REFLECTION_PROMPT_TEMPLATE
        )
        logger.debug(
            f"Generated reflection prompt, length: {len(reflection_prompt_content)} chars"
        )

        # We don't add the prompt to history for this special call,
        # as it's a meta-instruction for the planner itself.
        # However, the response from the LLM (the tier) can be logged.

        self.display_manager.update_planning_display(
            "-1. Self-Reflection", "Running", "Assessing objective complexity..."
        )

        try:
            logger.debug(
                f"Calling LLM for self-reflection using model: {self.reasoning_model}"
            )
            # Use a separate call, not part of the main history chain for planning
            # Use the reasoning_model for this critical decision.
            reflection_message: Message = await self.provider.generate_message(
                messages=[Message(role="user", content=reflection_prompt_content)],
                model=self.reasoning_model,  # Use reasoning model for this step
                tools=[],
                max_tokens=10,  # Expecting a very short response like "Tier X"
            )

            tier_response_content = str(reflection_message.content).strip()
            logger.debug(f"LLM self-reflection response: '{tier_response_content}'")

            self.current_planning_tier = PlanningTier.from_string(tier_response_content)
            logger.debug(f"Determined planning tier: {self.current_planning_tier}")

            phase_status = "Completed"
            phase_content = f"Determined Planning Tier: {self.current_planning_tier}"
            if (
                self.current_planning_tier == PlanningTier.DEFAULT
                and tier_response_content
                not in [
                    PlanningTier.TIER_LOW_COMPLEXITY,
                    PlanningTier.TIER_HIGH_COMPLEXITY,
                ]
            ):
                phase_status = "Warning"
                phase_content += f" (LLM response: '{tier_response_content}', defaulted to {PlanningTier.DEFAULT})"
                logger.warning(
                    f"Unexpected tier response, defaulted: {tier_response_content}"
                )

        except Exception as e:
            logger.error(f"Error during self-reflection: {e}", exc_info=True)
            self.current_planning_tier = PlanningTier.DEFAULT
            phase_status = "Failed"
            phase_content = f"Error during self-reflection: {e}. Defaulting to {self.current_planning_tier}"
            # Log the error if verbose or always for critical failures
            self.display_manager.print(f"[red]Self-reflection error: {e}[/red]")

        logger.debug(f"Self-reflection phase completed with status: {phase_status}")
        self.display_manager.update_planning_display(
            "-1. Self-Reflection", phase_status, Text(str(phase_content))
        )
        planning_update = PlanningUpdate(
            phase="Self-Reflection",
            status=phase_status,
            content=str(phase_content),
        )
        # Do not append reflection messages to the main planning history
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
        logger.debug("Starting analysis phase")

        if not self.enable_analysis_phase:
            logger.debug("Skipping analysis phase: disabled by global flag")
            self.display_manager.update_planning_display(
                "0. Analysis", "Skipped", "Phase disabled by global flag."
            )
            return history, None

        # Skip based on determined tier - REMOVED
        # Analysis phase runs for both Low and High complexity tiers,
        # so tier-based skipping is removed from here.
        # It is only skipped if self.enable_analysis_phase is False (handled above).

        logger.debug("Generating analysis prompt")
        analysis_prompt_content: str = self._render_prompt(ANALYSIS_PHASE_TEMPLATE)
        logger.debug(
            f"Analysis prompt generated, length: {len(analysis_prompt_content)} chars"
        )
        history.append(Message(role="user", content=analysis_prompt_content))

        # Update display before LLM call
        self.display_manager.update_planning_display(
            "0. Analysis", "Running", "Generating analysis..."
        )

        logger.debug(f"Calling LLM for analysis using model: {self.model}")
        analysis_message: Message = await self.provider.generate_message(
            messages=history, model=self.model, tools=[]  # Explicitly empty list
        )
        history.append(analysis_message)
        logger.debug(
            f"Analysis phase LLM response received, content length: {len(str(analysis_message.content)) if analysis_message.content else 0} chars"
        )

        phase_status: str = "Completed"
        phase_content: str | Text = self._format_message_content(analysis_message)
        logger.debug(f"Analysis phase completed with status: {phase_status}")

        self.display_manager.update_planning_display(
            "0. Analysis", phase_status, phase_content
        )

        planning_update = PlanningUpdate(
            phase="Analysis",
            status=phase_status,
            content=str(phase_content),
        )

        return history, planning_update

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
        - The `self.current_planning_tier` is `PlanningTier.TIER_1_SIMPLE` or
          `PlanningTier.TIER_2_MODERATE`.

        Args:
            history: The current list of messages in the planning conversation.
                     The LLM's prompt and response for this phase are appended.

        Returns:
            A tuple containing:
                - The updated history with messages from this phase.
                - A `PlanningUpdate` object summarizing the outcome, or None if skipped.
        """
        logger.debug("Starting data flow phase")

        if not self.enable_data_contracts_phase:
            logger.debug("Skipping data flow phase: disabled by global flag")
            self.display_manager.update_planning_display(
                "1. Data Contracts", "Skipped", "Phase disabled by global flag."
            )
            return history, None

        # Skip based on determined tier
        if self.current_planning_tier == PlanningTier.TIER_LOW_COMPLEXITY:
            logger.debug(
                f"Skipping data flow phase: tier is {self.current_planning_tier}"
            )
            self.display_manager.update_planning_display(
                "1. Data Contracts",
                "Skipped",
                f"Skipped due to {self.current_planning_tier} (Low Complexity).",
            )
            return history, None

        logger.debug("Generating data flow prompt")
        data_flow_prompt_content: str = self._render_prompt(DATA_FLOW_ANALYSIS_TEMPLATE)
        logger.debug(
            f"Data flow prompt generated, length: {len(data_flow_prompt_content)} chars"
        )
        history.append(Message(role="user", content=data_flow_prompt_content))

        # Update display before LLM call
        self.display_manager.update_planning_display(
            "1. Data Contracts", "Running", "Generating data flow..."
        )

        logger.debug(f"Calling LLM for data flow analysis using model: {self.model}")
        data_contracts_message: Message = await self.provider.generate_message(
            messages=history, model=self.model, tools=[]  # Explicitly empty list
        )
        history.append(data_contracts_message)
        logger.debug(
            f"Data flow phase LLM response received, content length: {len(str(data_contracts_message.content)) if data_contracts_message.content else 0} chars"
        )

        phase_status: str = "Completed"
        phase_content: str | Text = self._format_message_content(data_contracts_message)
        logger.debug(f"Data flow phase completed with status: {phase_status}")

        self.display_manager.update_planning_display(
            "1. Data Contracts", phase_status, phase_content
        )

        planning_update = PlanningUpdate(
            phase="Data Flow",
            status=phase_status,
            content=str(phase_content),
        )

        return history, planning_update

    async def _run_plan_creation_phase(
        self,
        history: List[Message],
        objective: str,
        max_retries: int,
    ) -> tuple[Optional[Task], Optional[Exception], Optional[PlanningUpdate]]:
        """Handles Phase 2: Plan Creation.

        This is the final planning phase where the LLM generates the concrete,
        executable task plan. It does this either by making a call to the
        `create_task` tool (if `self.use_structured_output` is False) or
        by generating structured JSON output directly if the LLM and provider
        support it.

        The complexity of the plan generated depends on the information
        gathered in preceding phases (Analysis, Data Flow), which are themselves
        conditional on the `current_planning_tier`.

        Args:
            history: The list of messages from previous planning phases.
                     The LLM's prompt and response for this phase are appended if
                     using tool-based generation.
            objective: The original user objective, used as a fallback title
                       and for context in structured output.
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
        logger.debug(
            f"Starting plan creation phase with max_retries={max_retries}, use_structured_output={self.use_structured_output}"
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

        if self.use_structured_output:
            logger.debug("Using structured output for plan creation")
            self.display_manager.update_planning_display(
                current_phase_name,
                "Running",
                "Attempting plan creation using structured output...",
            )
            try:
                task = await self._create_task_with_structured_output(
                    objective, max_retries
                )
                phase_status = "Success"
                phase_content = f"Plan created with {len(task.subtasks)} subtasks using structured output."
                logger.debug(
                    f"Structured output plan creation successful: {len(task.subtasks)} subtasks"
                )
                final_message = None  # Not applicable for structured output path
            except Exception as e:
                logger.error(
                    f"Structured output plan creation failed: {e}", exc_info=True
                )
                plan_creation_error = e
                phase_status = "Failed"
                phase_content = f"Structured output failed: {str(e)}\n{traceback.format_exc()}"  # Keep traceback for display

        else:  # Use tool-based generation
            logger.debug("Using tool-based generation for plan creation")
            plan_creation_prompt_content = self._render_prompt(PLAN_CREATION_TEMPLATE)
            agent_task_prompt_content = await self._build_agent_task_prompt_content()
            logger.debug(
                f"Plan creation prompt length: {len(plan_creation_prompt_content)} chars"
            )
            logger.debug(
                f"Agent task prompt length: {len(agent_task_prompt_content)} chars"
            )

            history.append(
                Message(
                    role="user",
                    content=f"{plan_creation_prompt_content}\n{agent_task_prompt_content}",
                )
            )
            self.display_manager.update_planning_display(
                current_phase_name,
                "Running",
                "Attempting plan creation using the 'create_task' tool...",
            )
            try:
                logger.debug("Starting tool-based plan generation with retry logic")
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
                    logger.debug(
                        f"Tool-based plan creation successful: {len(task.subtasks)} subtasks"
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
                            failure_reason = (
                                f"LLM indicated failure: {formatted_content}"
                            )
                        else:
                            failure_reason = f"LLM did not produce a valid 'create_task' tool call. Last message: {formatted_content}"
                    elif (
                        plan_creation_error
                    ):  # Check if _generate_with_retry raised an error internally
                        failure_reason = f"Tool call generation failed internally: {plan_creation_error}"

                    logger.warning(f"Tool-based plan creation failed: {failure_reason}")
                    plan_creation_error = ValueError(
                        f"Tool call generation failed: {failure_reason}"
                    )
                    phase_content = f"Tool call generation failed: {failure_reason}"
                    phase_status = "Failed"
            except Exception as e:
                logger.error(f"Tool-based plan creation failed: {e}", exc_info=True)
                plan_creation_error = e
                phase_status = "Failed"
                phase_content = f"Tool call generation failed: {str(e)}\n{traceback.format_exc()}"  # Keep traceback for display

        # Update Table for Phase 2
        logger.debug(f"Plan creation phase completed with status: {phase_status}")
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
        logger.info(
            f"Starting task creation for objective: '{objective[:100]}...' with max_retries={max_retries}"
        )

        # Start the live display using the display manager
        self.display_manager.start_live(
            self.display_manager.create_planning_table("Task Planner")
        )

        history: List[Message] = [
            Message(role="system", content=self.system_prompt),
        ]

        error_message: Optional[str] = None
        plan_creation_error: Optional[Exception] = None
        task: Optional[Task] = None
        current_phase = "Initialization"

        try:
            logger.debug("Starting planning phases")

            # Phase -1: Self-Reflection
            current_phase = "Self-Reflection"
            logger.debug(f"Entering phase: {current_phase}")
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            # Run self-reflection. Note: history is not modified by this special phase call.
            _, planning_update = await self._run_self_reflection_phase(history)
            if planning_update:
                yield planning_update

            # Phase 0: Analysis
            current_phase = "Analysis"
            logger.debug(f"Entering phase: {current_phase}")
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            history, planning_update = await self._run_analysis_phase(history)
            if planning_update:
                yield planning_update

            # Phase 1: Data Flow Analysis
            current_phase = "Data Flow"
            logger.debug(f"Entering phase: {current_phase}")
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            history, planning_update = await self._run_data_flow_phase(history)
            if planning_update:
                yield planning_update

            # Phase 2: Plan Creation
            current_phase = "Plan Creation"
            logger.debug(f"Entering phase: {current_phase}")
            yield PlanningUpdate(phase=current_phase, status="Starting", content=None)
            task, plan_creation_error, planning_update = (
                await self._run_plan_creation_phase(history, objective, max_retries)
            )
            if planning_update:
                yield planning_update  # Yield the update from the phase itself

            # --- Final Outcome ---
            if task:
                logger.info(
                    f"Plan created successfully with {len(task.subtasks)} subtasks"
                )
                self.display_manager.print(  # Use display manager print
                    "[bold green]Plan created successfully.[/bold green]"
                )
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
                    logger.error(f"Task creation failed: {error_message}")
                    # Yield failure update before raising
                    yield PlanningUpdate(
                        phase=current_phase, status="Failed", content=error_message
                    )
                    # Update display for overall failure
                    self.display_manager.update_planning_display(
                        "Overall Status",
                        "Failed",
                        Text(full_error_message, style="bold red"),
                        is_error=True,
                    )
                    raise ValueError(full_error_message) from plan_creation_error
                else:
                    error_message = "Failed to create valid task after maximum retries in Plan Creation phase for an unknown reason."
                    logger.error(f"Task creation failed: {error_message}")
                    # Yield failure update before raising
                    yield PlanningUpdate(
                        phase=current_phase, status="Failed", content=error_message
                    )
                    # Update display for overall failure
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
            logger.error(
                f"Task creation failed during {current_phase}: {e}", exc_info=True
            )

            # Log traceback if verbose
            if self.verbose:
                self.display_manager.print_exception(show_locals=False)

            # Add error row to table via display manager
            if error_message:
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
            self.display_manager.print(  # Use display manager print
                f"[bold red]Planning Error:[/bold red] {error_message}"
            )

            # Yield failure update before re-raising
            yield PlanningUpdate(
                phase=current_phase, status="Failed", content=error_message
            )
            raise  # Re-raise the caught exception

        finally:
            logger.debug("Stopping live display and completing task creation")
            # Stop the live display using the display manager
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
        validation_errors: List[str] = []
        parsed_content: Optional[dict] = None

        if tool_name not in available_execution_tools:
            validation_errors.append(
                f"{sub_context}: Specified tool_name '{tool_name}' is not in the list of available execution tools: {list(available_execution_tools.keys())}."
            )
            return None, validation_errors

        tool_to_use = available_execution_tools[tool_name]

        # Validate content is JSON and parse it
        if isinstance(content, dict):
            parsed_content = content  # Already a dict
        elif isinstance(content, str):
            try:
                parsed_content = json.loads(content) if content.strip() else {}
            except json.JSONDecodeError as e:
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): 'content' is not valid JSON. Error: {e}. Content: '{content}'"
                )
                return None, validation_errors
        else:
            validation_errors.append(
                f"{sub_context} (tool: {tool_name}): Expected JSON string or object for tool arguments, but got {type(content)}. Content: '{content}'"
            )
            return None, validation_errors

        # Validate the parsed content against the tool's input schema
        if tool_to_use.input_schema and parsed_content is not None:
            try:
                validate(instance=parsed_content, schema=tool_to_use.input_schema)
            except ValidationError as e:
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): JSON arguments in 'content' do not match the tool's input schema. Error: {e.message}. Path: {'/'.join(map(str, e.path))}. Schema: {e.schema}. Args: {parsed_content}"
                )
                return None, validation_errors
            except Exception as e:  # Catch other potential validation errors
                validation_errors.append(
                    f"{sub_context} (tool: {tool_name}): Error validating arguments against tool schema. Error: {e}. Args: {parsed_content}"
                )
                return None, validation_errors

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
        validation_errors: List[str] = []
        if not isinstance(content, str) or not content.strip():
            validation_errors.append(
                f"{sub_context}: 'content' must be a non-empty string containing instructions when 'tool_name' is not provided, but got: '{content}' (type: {type(content)})."
            )
        return validation_errors

    def _process_subtask_schema(
        self, subtask_data: dict, sub_context: str
    ) -> tuple[Optional[str], List[str]]:
        """Processes and validates the output_schema for a subtask.

        Handles several cases for `output_schema`:
        - If `output_type` is binary, uses `FILE_POINTER_SCHEMA`.
        - If `output_schema` is a valid JSON string, it's parsed.
        - If `output_schema` is None or an empty string, a default schema is
          generated based on `output_type` using `json_schema_for_output_type`.
        - Ensures `additionalProperties: false` is set on object schemas.

        Args:
            subtask_data: The raw dictionary data for the subtask, containing
                          `output_type` and `output_schema`.
            sub_context: A string prefix for error messages.

        Returns:
            A tuple containing:
                - The processed and validated `output_schema` as a JSON string,
                  or None if a fatal error occurred.
                - A list of string error messages encountered.
        """
        logger.debug(f"{sub_context}: Starting schema processing")
        validation_errors: List[str] = []
        output_type: str = subtask_data.get("output_type", "string")
        current_schema_str: Any = subtask_data.get("output_schema")
        final_schema_str: Optional[str] = None
        schema_dict: Optional[dict] = None

        # Add logging for the input schema string
        logger.debug(
            f"{sub_context}: output_type='{output_type}', schema_input='{current_schema_str}' (type: {type(current_schema_str)})"
        )
        self.display_manager.print(
            f"{sub_context}: Attempting to process output_schema: '{current_schema_str}' of type {type(current_schema_str)}"
        )

        if is_binary_output_type(output_type):
            logger.debug(
                f"{sub_context}: Using FILE_POINTER_SCHEMA for binary output type"
            )
            final_schema_str = json.dumps(FILE_POINTER_SCHEMA)
        else:
            try:
                if isinstance(current_schema_str, str) and current_schema_str.strip():
                    logger.debug(f"{sub_context}: Parsing string schema")
                    self.display_manager.print(
                        f"{sub_context}: Parsing string schema: '{current_schema_str}'"
                    )  # Log before loads
                    schema_dict = json.loads(current_schema_str)
                    logger.debug(f"{sub_context}: Successfully parsed schema dict")
                elif current_schema_str is None or (
                    isinstance(current_schema_str, str)
                    and not current_schema_str.strip()
                ):
                    # If schema is None or empty string, generate default based on type
                    logger.debug(
                        f"{sub_context}: Generating default schema for type '{output_type}'"
                    )
                    schema_dict = json_schema_for_output_type(output_type)
                    logger.debug(f"{sub_context}: Generated default schema")
                else:  # Invalid type for schema string
                    error_msg = f"Output schema must be a JSON string or None, got {type(current_schema_str)}"
                    logger.error(f"{sub_context}: {error_msg}")
                    raise ValueError(error_msg)

                # Apply defaults if schema_dict was successfully loaded or generated
                if schema_dict is not None:
                    logger.debug(
                        f"{sub_context}: Applying additionalProperties constraints"
                    )
                    schema_dict = self._ensure_additional_properties_false(schema_dict)
                    final_schema_str = json.dumps(schema_dict)
                    logger.debug(
                        f"{sub_context}: Final schema prepared, length={len(final_schema_str)}"
                    )
                # If schema_dict is still None here, it means default generation failed (unlikely with current json_schema_for_output_type)
                # or input was invalid type that didn't parse

            except (ValueError, json.JSONDecodeError) as e:
                error_msg = f"Invalid output_schema provided: '{current_schema_str}'. Error: {e}. Using default for type '{output_type}'."
                validation_errors.append(f"{sub_context}: {error_msg}")
                logger.warning(f"{sub_context}: Schema parsing failed: {e}")
                # Log the specific error
                self.display_manager.print(
                    f"{sub_context}: JSONDecodeError or ValueError for schema '{current_schema_str}': {e}"
                )
                # Attempt to generate default schema as fallback
                try:
                    logger.debug(f"{sub_context}: Generating fallback default schema")
                    schema_dict = json_schema_for_output_type(output_type)
                    schema_dict = self._ensure_additional_properties_false(schema_dict)
                    final_schema_str = json.dumps(schema_dict)
                    logger.debug(
                        f"{sub_context}: Fallback schema generated successfully"
                    )
                except Exception as default_e:
                    fallback_error = f"Failed to generate default schema for type '{output_type}' after error. Defaulting to string schema. Error: {default_e}"
                    validation_errors.append(f"{sub_context}: {fallback_error}")
                    logger.error(
                        f"{sub_context}: Fallback schema generation failed: {default_e}"
                    )
                    # Final fallback: simple string schema
                    final_schema_str = json.dumps(
                        {"type": "string", "additionalProperties": False}
                    )
                    logger.debug(f"{sub_context}: Using final fallback string schema")

        logger.debug(
            f"{sub_context}: Schema processing completed, errors={len(validation_errors)}"
        )
        return final_schema_str, validation_errors

    def _prepare_subtask_data(
        self,
        subtask_data: dict,
        final_schema_str: str,
        parsed_tool_content: Optional[dict],
        sub_context: str,
    ) -> tuple[Optional[dict], List[str]]:
        """Cleans paths and prepares the final data dictionary for SubTask creation.

        This method takes the raw subtask data, the validated `final_schema_str`,
        and potentially parsed tool content, then performs:
        1.  Path cleaning and validation for `output_file` and `input_files`
            using `_clean_and_validate_path`. It also checks that output files
            are at the workspace root.
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
            raw_output_file = processed_data.get("output_file", "")
            cleaned_output_file = self._clean_and_validate_path(
                raw_output_file, f"{sub_context} output_file"
            )
            if os.path.dirname(cleaned_output_file):
                validation_errors.append(
                    f"{sub_context}: Output file '{cleaned_output_file}' must be at the workspace root, not in a subdirectory."
                )
            processed_data["output_file"] = cleaned_output_file

            raw_input_files = processed_data.get("input_files", [])
            cleaned_input_files_list = []
            for f_path in raw_input_files:
                if not isinstance(f_path, str):
                    validation_errors.append(
                        f"{sub_context}: Input file path item '{f_path}' must be a string, got {type(f_path)}."
                    )
                    continue
                cleaned_f_path = self._clean_and_validate_path(
                    f_path, f"{sub_context} input_file '{f_path}'"
                )
                if os.path.dirname(cleaned_f_path):
                    # If the cleaned path is in a subdirectory, it must be one of the initial input files.
                    # self.input_files contains paths already cleaned by _clean_and_validate_path.
                    if cleaned_f_path not in self.input_files:
                        validation_errors.append(
                            f"{sub_context}: Input file '{cleaned_f_path}' is in a subdirectory. Only initial input files are allowed in subdirectories, and this file is not part of the initial inputs."
                        )
                cleaned_input_files_list.append(cleaned_f_path)
            processed_data["input_files"] = cleaned_input_files_list

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
            # else: # Handle cases where content might be missing or wrong type after filtering
            #    if filtered_data.get("tool_name"): # Tool tasks expect content
            #        validation_errors.append(f"{sub_context}: Missing or invalid 'content' for tool task.")
            #        return None, validation_errors
            # Agent tasks are validated earlier to ensure content is a string

            return filtered_data, validation_errors

        except ValueError as e:  # Catch path validation errors
            validation_errors.append(f"{sub_context}: Error processing paths: {e}")
            return None, validation_errors
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
        logger.debug(f"Processing {sub_context}")
        all_validation_errors: List[str] = []
        parsed_tool_content: Optional[dict] = (
            None  # To store parsed JSON for tool tasks
        )

        try:
            # --- Validate Tool Call vs Agent Instruction ---
            tool_name = subtask_data.get("tool_name")
            content = subtask_data.get("content")
            logger.debug(
                f"{sub_context}: tool_name='{tool_name}', content_length={len(str(content)) if content else 0}"
            )

            if tool_name:
                # --- Deterministic Tool Task Validation ---
                logger.debug(f"{sub_context}: Validating as tool task")
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
                    logger.error(
                        f"{sub_context}: Tool validation failed with {len(tool_errors)} errors"
                    )
                    return None, all_validation_errors
                else:
                    logger.debug(f"{sub_context}: Tool validation successful")
            else:
                # --- Probabilistic Agent Task Validation ---
                logger.debug(f"{sub_context}: Validating as agent task")
                agent_errors = self._validate_agent_task(content, sub_context)
                all_validation_errors.extend(agent_errors)
                if agent_errors:  # Fatal error during agent validation
                    logger.error(
                        f"{sub_context}: Agent validation failed with {len(agent_errors)} errors"
                    )
                    return None, all_validation_errors
                else:
                    logger.debug(f"{sub_context}: Agent validation successful")

            # --- Process schema ---
            logger.debug(f"{sub_context}: Processing output schema")
            final_schema_str, schema_errors = self._process_subtask_schema(
                subtask_data, sub_context
            )
            all_validation_errors.extend(schema_errors)
            # Continue even if there were schema errors, as a default might be used.
            # Need final_schema_str for data preparation. If None, indicates a fatal schema issue.
            if final_schema_str is None:
                logger.error(f"{sub_context}: Fatal schema processing error")
                all_validation_errors.append(
                    f"{sub_context}: Fatal error processing output schema."
                )
                return None, all_validation_errors
            else:
                logger.debug(f"{sub_context}: Schema processing successful")

            # --- Prepare data for SubTask creation (Paths, Filtering, Stringify Tool Args) ---
            logger.debug(f"{sub_context}: Preparing data for SubTask creation")
            filtered_data, preparation_errors = self._prepare_subtask_data(
                subtask_data, final_schema_str, parsed_tool_content, sub_context
            )
            all_validation_errors.extend(preparation_errors)
            if filtered_data is None:  # Fatal error during data preparation
                logger.error(
                    f"{sub_context}: Data preparation failed with {len(preparation_errors)} errors"
                )
                return None, all_validation_errors
            else:
                logger.debug(f"{sub_context}: Data preparation successful")

            # --- Create SubTask object ---
            # Pydantic validation happens here
            logger.debug(f"{sub_context}: Creating SubTask object")
            subtask = SubTask(**filtered_data)
            logger.debug(
                f"{sub_context}: SubTask created successfully with output_file='{subtask.output_file}'"
            )
            # Return successful subtask and any *non-fatal* validation errors collected
            return subtask, all_validation_errors

        except (
            ValidationError
        ) as e:  # Catch Pydantic validation errors during SubTask(**filtered_data)
            error_msg = f"{sub_context}: Invalid data for SubTask model: {e}"
            logger.error(f"{sub_context}: Pydantic validation error: {e}")
            all_validation_errors.append(error_msg)
            return None, all_validation_errors
        except Exception as e:  # Catch any other unexpected errors
            error_msg = f"{sub_context}: Unexpected error processing subtask: {e}\n{traceback.format_exc()}"
            logger.error(
                f"{sub_context}: Unexpected processing error: {e}", exc_info=True
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

        logger.debug(f"Subtasks processed: {subtasks}")
        logger.debug(f"Subtask validation errors: {subtask_validation_errors}")

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

    async def _create_task_with_structured_output(
        self, objective: str, max_retries: int = 3
    ) -> Task:
        """
        Create a task plan using structured output from the LLM.

        This method attempts to have the LLM generate a complete task plan
        as a JSON object that conforms to the schema of the `CreateTaskTool`.
        It includes retry logic: if the LLM's output is not valid JSON or
        fails plan validation (`_validate_structured_output_plan`), it re-prompts
        the LLM with error feedback.

        Args:
            objective: The user's objective for the task plan.
            max_retries: The maximum number of attempts to get a valid plan.

        Returns:
            A `Task` object representing the validated plan.

        Raises:
            ValueError: If a valid task plan cannot be generated after all retries,
                        or if other critical errors occur.
        """
        create_task_tool = CreateTaskTool()
        response_schema: dict = create_task_tool.input_schema
        base_prompt: str = self._render_prompt(
            f"{PLAN_CREATION_TEMPLATE}\n{DEFAULT_AGENT_TASK_TEMPLATE}"
        )

        prompt: str = base_prompt  # Initial prompt
        current_retry: int = 0
        last_error: Optional[Exception] = None

        while current_retry < max_retries:
            attempt = current_retry + 1
            self.display_manager.print(  # Use display manager print
                f"[yellow]Attempt {attempt}/{max_retries} for structured output...[/yellow]"
            )

            # Prepare messages for this attempt
            messages_for_attempt: List[Message] = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=prompt),  # Use potentially updated prompt
            ]

            try:
                # 1. Generate Response
                message = await self.provider.generate_message(
                    messages=messages_for_attempt,
                    model=self.model,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "TaskPlan",
                            "schema": response_schema,
                            "strict": True,
                        },
                    },
                )

                if not isinstance(message.content, str) or not message.content.strip():
                    raise ValueError(
                        "LLM returned empty or non-string content for structured output."
                    )

                # 2. Parse JSON
                try:
                    task_data: dict = json.loads(message.content)
                except json.JSONDecodeError as json_err:
                    error_detail = f"Failed to decode JSON from LLM response: {json_err}.\nRaw Response:\n---\n{message.content}\n---"
                    self.display_manager.print(  # Use display manager print
                        f"[red]JSON Decode Error:[/red]\n{error_detail}"
                    )
                    # Raise a ValueError containing the raw response for retry feedback
                    raise ValueError(error_detail) from json_err

                print(task_data)

                # 3. Validate Plan (Subtasks & Dependencies)
                validated_task, validation_errors = (
                    await self._validate_structured_output_plan(task_data, objective)
                )

                if validation_errors:
                    error_feedback = "\n".join(validation_errors)
                    self.display_manager.print(  # Use display manager print
                        f"[red]Validation Errors (Structured):[/red]\n{error_feedback}"
                    )
                    # Raise ValueError with validation errors for retry feedback
                    raise ValueError(
                        f"Validation failed for structured output plan:\n{error_feedback}"
                    )

                # 4. Success
                if validated_task:
                    self.display_manager.print(  # Use display manager print
                        "[green]Structured output plan created and validated successfully.[/green]"
                    )
                    return validated_task  # Success!
                else:
                    # Should not happen if validation_errors was empty, but handle defensively
                    raise ValueError(
                        "Plan validation reported no errors, but no task object was returned."
                    )

            except Exception as e:
                last_error = e
                current_retry += 1  # Increment retry count
                if current_retry < max_retries:
                    # Prepare prompt for the next retry, including the error message
                    prompt = f"{base_prompt}\n\nPREVIOUS ATTEMPT FAILED WITH ERROR (Attempt {attempt}/{max_retries}):\n{str(e)}\n\nPlease fix the errors and regenerate the full plan according to the schema."
                    self.display_manager.print(  # Use display manager print
                        f"[yellow]Retrying structured output ({current_retry + 1}/{max_retries})...[/yellow]"
                    )
                    await asyncio.sleep(1)  # Optional delay before retry
                else:
                    # Raise the last error encountered after exhausting retries
                    raise ValueError(
                        f"Failed structured output after {max_retries} attempts."
                    ) from last_error

        # Should ideally not be reached if max_retries > 0
        raise ValueError(
            f"Failed to create task with structured output after {max_retries} attempts (unexpected exit). Last error: {str(last_error)}"
        ) from last_error

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
        logger.debug(
            f"Starting generation with retry, max_retries={max_retries}, tools={[t.name for t in tools]}"
        )
        current_retry: int = 0
        last_message: Optional[Message] = None

        while current_retry < max_retries:
            attempt = current_retry + 1
            logger.debug(f"Generation attempt {attempt}/{max_retries}")

            # Generate response using current history
            logger.debug(f"Calling LLM with {len(history)} messages in history")
            message = await self.provider.generate_message(
                messages=history,
                model=self.model,
                tools=tools,
            )
            history.append(
                message
            )  # Add assistant's response to history *before* processing
            last_message = message
            logger.debug(
                f"LLM response received, has_tool_calls={bool(message.tool_calls)}"
            )

            if not message.tool_calls:
                # LLM didn't use the expected tool
                logger.warning(f"LLM did not use required tools on attempt {attempt}")
                if tools and current_retry < max_retries - 1:
                    current_retry += 1
                    tool_names = ", ".join([t.name for t in tools])
                    retry_prompt = f"Please use one of the available tools ({tool_names}) to define the task based on the previous analysis and requirements."
                    history.append(Message(role="user", content=retry_prompt))
                    logger.debug(
                        f"Added retry prompt for tool usage, attempt {current_retry + 1}"
                    )
                    self.display_manager.print(  # Use display manager print
                        f"[yellow]Retry {attempt}/{max_retries}: Asking LLM to use required tool(s).[/yellow]"
                    )
                    continue  # Go to next iteration
                else:
                    # Max retries reached without tool use
                    logger.error(
                        f"Max retries reached without tool usage after {max_retries} attempts"
                    )
                    self.display_manager.print(  # Use display manager print
                        f"[red]Failed after {max_retries} retries: LLM did not use the required tool(s). Last message: {self._format_message_content(message)}[/red]"
                    )
                    return None, last_message

            # Tool call exists, try to process it
            try:
                logger.debug(f"Processing {len(message.tool_calls)} tool call(s)")
                # Process the tool call(s). This adds 'tool' role messages to history
                # and raises ValueError on validation failure.
                task = await self._process_tool_calls(message, history)
                # If _process_tool_calls returns without error, success!
                logger.info(
                    f"Tool calls processed successfully, created task with {len(task.subtasks)} subtasks"
                )
                self.display_manager.print(  # Use display manager print
                    "[green]Tool call processed successfully.[/green]"
                )
                return (
                    task,
                    last_message,
                )  # Return created task and the assistant message
            except ValueError as e:  # Catch validation errors from _process_tool_calls
                logger.warning(f"Tool call validation failed on attempt {attempt}: {e}")
                self.display_manager.print(  # Use display manager print
                    f"[yellow]Validation Error (Retry {attempt}/{max_retries}):[/yellow]\n{str(e)}"
                )
                if current_retry < max_retries - 1:
                    current_retry += 1
                    # Add a user message asking LLM to fix errors. The 'tool' message with error details
                    # should already be in history from _process_tool_calls failing.
                    retry_prompt = f"The previous attempt failed validation. Please review the errors detailed in the tool response and call the tool again correctly:\n{str(e)}"
                    history.append(Message(role="user", content=retry_prompt))
                    logger.debug(
                        f"Added validation error retry prompt, attempt {current_retry + 1}"
                    )
                    self.display_manager.print(  # Use display manager print
                        f"[yellow]Retry {attempt + 1}/{max_retries}: Asking LLM to fix validation errors.[/yellow]"
                    )
                    # Optional: Add a small delay before retrying
                    # await asyncio.sleep(1)
                    continue  # Go to next iteration
                else:
                    # Max retries reached after validation errors
                    logger.error(
                        f"Max retries reached due to persistent validation errors after {max_retries} attempts"
                    )
                    self.display_manager.print(  # Use display manager print
                        f"[red]Failed after {max_retries} retries due to persistent validation errors.[/red]"
                    )
                    return (
                        None,
                        last_message,
                    )  # Return no task and the last assistant message

        # Should only be reached if max_retries is 0 or loop finishes unexpectedly
        logger.error("Generation with retry exited unexpectedly")
        return None, last_message

    async def save_task_plan(self) -> None:
        """
        Save the current task plan to `tasks.yaml` in the workspace directory.

        The plan is serialized to YAML format. Errors during saving are
        printed to the display manager.
        """
        logger.debug(f"Attempting to save task plan to {self.tasks_file_path}")
        if self.task_plan:
            task_dict: dict = self.task_plan.model_dump(
                exclude_none=True
            )  # Use exclude_none
            logger.debug(
                f"Task plan serialized to dict with {len(task_dict.get('tasks', []))} tasks"
            )
            try:
                with open(self.tasks_file_path, "w") as f:
                    yaml.dump(
                        task_dict,
                        f,
                        indent=2,
                        sort_keys=False,
                        default_flow_style=False,
                    )  # Improve formatting
                logger.info(f"Task plan saved successfully to {self.tasks_file_path}")
                self.display_manager.print(  # Use display manager print
                    f"[cyan]Task plan saved to {self.tasks_file_path}[/cyan]"
                )
            except IOError as e:
                logger.error(f"IO error saving task plan: {e}")
                self.display_manager.print(  # Use display manager print
                    f"[red]Error saving task plan to {self.tasks_file_path}: {e}[/red]"
                )
            except (
                Exception
            ) as e:  # Catch other potential errors (e.g., yaml serialization)
                logger.error(f"Unexpected error saving task plan: {e}", exc_info=True)
                self.display_manager.print(  # Use display manager print
                    f"[red]Unexpected error saving task plan: {e}[/red]"
                )
        else:
            logger.warning("Attempted to save task plan but no plan exists")

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

        This is used in prompts to inform the LLM about files that are
        available at the very beginning of the task execution.

        Returns:
            A string listing the input files, or "No input files available"
            if the `input_files` list is empty.
        """
        if not self.input_files:
            return "No input files available"

        input_files_info = "Input files:\n"
        for file_path in self.input_files:
            input_files_info += f"- {file_path}\n"
        return input_files_info.strip()  # Remove trailing newline

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

    def _generate_batch_subtask_instructions(
        self, base_content: str, batch_info: dict
    ) -> dict:
        """
        Generate instructions for a batch processing subtask.

        Args:
            base_content: The base subtask content
            batch_info: Dictionary with batch details (start_idx, end_idx, items)

        Returns:
            Modified subtask data for batch processing
        """
        batch_content = f"""
{base_content}

BATCH PROCESSING INSTRUCTIONS:
1. Process items {batch_info['start_idx']} to {batch_info['end_idx']} from the input
2. Write results progressively to the output file using WriteFileTool in append mode
3. For each item processed:
   - Write the result as a JSON object on a single line (JSONL format)
   - Include item index/id for tracking
   - Flush the file after each write to prevent data loss
4. If an item fails, log the error but continue with remaining items
5. At the end, write a summary line with batch statistics
6. Use minimal memory - process one item at a time
7. Do NOT keep all results in memory
"""

        return {
            "content": batch_content,
            "batch_processing": {
                "enabled": True,
                "batch_size": batch_info["end_idx"] - batch_info["start_idx"] + 1,
                "start_index": batch_info["start_idx"],
                "end_index": batch_info["end_idx"],
                "total_items": batch_info.get("total_items", 0),
            },
        }

    def _generate_aggregation_subtask(
        self, batch_outputs: List[str], final_output: dict
    ) -> dict:
        """
        Generate the aggregation subtask for combining batch results.

        Args:
            batch_outputs: List of output files from batch subtasks
            final_output: Final output requirements

        Returns:
            Subtask data for aggregation
        """
        return {
            "content": f"""
Aggregate results from all batch processing subtasks into the final output.

AGGREGATION INSTRUCTIONS:
1. Read each batch result file line by line (JSONL format)
2. Process results progressively to minimize memory usage:
   - Use ReadFileTool to read one line at a time
   - Extract relevant data from each line
   - Update aggregated statistics/results
3. Combine results according to the output schema
4. Generate final summary and metadata
5. Handle any batch processing errors gracefully
6. Ensure the final output matches the required schema and format
7. Use file pointers {"path": "filename"} when passing data between tools
""",
            "input_files": batch_outputs,
            "output_file": final_output["file"],
            "output_type": final_output["type"],
            "output_schema": final_output["schema"],
            "max_iterations": 5,  # Aggregation typically needs fewer iterations
            "is_intermediate_result": False,
            "batch_processing": {
                "enabled": False  # Aggregation is not batch processing
            },
        }

    def _prepare_list_processing_plan(
        self, objective: str, item_count: int, batch_size: int = 15
    ) -> dict:
        """
        Prepare a plan structure for list processing with batching.

        Args:
            objective: The processing objective
            item_count: Estimated number of items
            batch_size: Items per batch (default 15)

        Returns:
            Dictionary with plan structure hints
        """
        num_batches = (item_count + batch_size - 1) // batch_size

        return {
            "processing_type": "batch",
            "estimated_items": item_count,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "parallel_capable": True,
            "requires_aggregation": True,
            "context_optimization": {
                "use_streaming": True,
                "use_file_pointers": True,
                "progressive_writes": True,
                "minimal_memory": True,
            },
        }

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
