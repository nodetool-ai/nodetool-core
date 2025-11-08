"""
🧠 SubTask Execution Context: Orchestrating Focused Task Execution

This module provides the `SubTaskContext` class, the dedicated engine for executing
a single, isolated subtask within a larger agentic workflow. It manages the state,
communication, and tool usage necessary to fulfill the subtask's objective, ensuring
each step operates independently but contributes to the overall task goal.

Core Components:
---------------
*   `SubTaskContext`: The central class managing the lifecycle of a subtask.
    It maintains an isolated message history, manages available tools, monitors
    resource limits (tokens, iterations), and drives the interaction with the
    chosen language model (LLM).
*   `FinishSubTaskTool` / `FinishTaskTool`: Special tools injected into the context.
    The LLM *must* call one of these tools (`finish_subtask` for regular subtasks,
    `finish_task` for the final aggregation task) to signify completion and provide
    the final result as an object. The result is stored directly in the processing
    context and made available to downstream tasks.
*   Helper Functions (`json_schema_for_output_type`, etc.):
    Utilities for determining output types, generating appropriate JSON schemas for
    tools, and handling object operations.

Execution Algorithm:
--------------------
1.  **Initialization**: A `SubTaskContext` is created for a specific `SubTask`,
    equipped with the necessary `ProcessingContext`, `ChatProvider`, tools (including
    special `finish_subtask` or `finish_task` tools), model, and resource limits.
    A system prompt tailored to the subtask (execution vs. final aggregation) is
    generated. The subtask's internal message history is initialized with this
    system prompt.

2.  **LLM-Driven Execution Loop**: The context enters an iterative loop driven by
    interactions with the LLM. The loop continues as long as the subtask is not
    completed and the number of iterations is less than `max_iterations`:
    a.  **Iteration & Limit Checks (Pre-LLM Call)**:
        - If `max_iterations` is reached, the context proceeds to step 2.f (Forced Completion).
        - Token count of the history is checked. If it exceeds `max_token_limit`
          and not already in conclusion stage, the context transitions to
          `in_conclusion_stage` (see step 2.e).
    b.  **Prepare Prompt & LLM Call**: The overall task description, current subtask
        instructions, input task results (directly injected), and the current message
        history form the prompt. The LLM processes this history and generates a response.
        This response can be text content or a request to use one or more available tools
        (LLM Tool Calls). The LLM's text response (if any) is yielded externally as a
        `Chunk` and added to the internal history as an assistant message.
    c.  **Handling LLM Tool Calls**: If the LLM response includes tool calls:
        i.   For each `ToolCall` requested by the LLM:
             - A user-friendly message describing the tool call is generated
               (via `_generate_tool_call_message` which typically uses the tool's
               `user_message` method).
             - The `ToolCall` object, now including this descriptive message,
               is yielded externally from the `execute` method.
             - If the tool call is for `finish_subtask` or `finish_task`, a
               `TaskUpdateEvent.SUBTASK_COMPLETED` update is also yielded.
        ii.  Internally, after being yielded, these LLM-generated tool calls are
             processed by `_handle_tool_call`. This method orchestrates:
             - **Execution**: Invokes the tool's logic (`_process_tool_execution`).
             - **Binary Artifact Handling**: Saves base64 encoded binary data
               (images, audio) from the result to workspace files (in an 'artifacts'
               folder) and updates the result to point to these files
               (`_handle_binary_artifact`).
             - **Special Side-Effects**: Manages tool-specific actions
               (`_process_special_tool_side_effects`). For `browser` navigation,
               it logs sources. For `finish_subtask`/`finish_task`, it marks
               the subtask complete and stores the result object directly in the
               processing context.
             - **Serialization**: Converts the processed tool result to a JSON
               string (`_serialize_tool_result_for_history`).
        iii. The serialized JSON string (tool output) is then added to the subtask's
             internal message history as a `Message` with role 'tool'.
    d.  **Continuation**: The loop continues to the next LLM interaction unless a
        `finish_...` tool completed the subtask or `max_iterations` was reached.
    e.  **Conclusion Stage**: If the token limit is exceeded, the context enters a
        "conclusion stage". In this stage, available tools are restricted to only
        the `finish_...` tool, and the LLM is prompted to synthesize results and
        conclude the subtask by providing an object result.
    f.  **Forced Completion (Max Iterations)**: If `max_iterations` is reached
        before `finish_...` is called, `_handle_max_iterations_reached` is invoked.
        This forces completion by:
        - Requesting a final structured output from the LLM, conforming to the
          finish tool's schema (which mandates an object result)
          (`request_structured_output`).
        - Storing the result object directly in the processing context.
        - Marking the subtask as complete.
        - Creating an assistant message with the `ToolCall` for record and adding
          it to history.
        - Yielding this record `ToolCall` externally.
        The subtask loop then terminates.

3.  **Object Storage**: When a `finish_...` tool is successfully processed (via
    `_process_special_tool_side_effects` during `_handle_tool_call`) or when forced
    completion occurs (`_handle_max_iterations_reached`), the result object is stored
    directly in the processing context using the task/subtask ID as the key. This
    makes the result available to downstream tasks via direct context access.

4.  **Completion**: The subtask is marked as `completed` (either by a `finish_...`
    tool call or by reaching max iterations). Status updates reflecting completion
    or failure are yielded. The `execute` loop terminates.

Key Data Structures:
--------------------
*   `Task`: Represents the overall goal.
*   `SubTask`: Defines a single step, including its objective (`content`), input task IDs,
    and expected output schema.
*   `Message`: Represents a single turn in the conversation history (system, user,
    assistant, tool).
*   `ToolCall`: Represents the LLM's request to use a tool, including its arguments.
    It also includes a user-facing message string generated before execution, and
    eventually the tool's result (when part of an assistant message in history or
    processed internally).
*   `ProcessingContext`: Holds shared workflow state and stores task results as objects.
*   `Tool`: Interface for tools the LLM can use.

High-Level Execution Flow Diagram:
---------------------------------
```ascii
+-----------------------+
| Init SubTaskContext   |
| (History, Tools,      |
|  System Prompt)       |
+-----------+-----------+
            |
            | (Subtask active & within iters)
            V
+-----------+-------------+<-----------------------------------------------------+ (Loop if active)
| LLM Loop                |
| (Check completion/      |
|  max_iters)             |
+-----------+-------------+
            |                                +-----------------------------------+
            +--(Max iters?)----------------->| `Handle Max Iters`                |
            | Yes                            |  - Req. structured output         |
            V No                             |  - Store result object            |
+--------------------------+                 |  - Mark complete                  |
|   Check Tokens           |                 |  - Log ToolCall to history        |
| (May enter Conclusion    |                 +-----------+-----------------------+
|  Stage if limit reached) |                             |
+--------------------------+                             V
            |                                +-----------+-----------------------+
            V                                | Yield ToolCall (Task ends)        |
+--------------------------+                 +-----------------------------------+
| Execute LLM              |
| (Gen Msg/ToolCalls)      |
+--------------------------+
            |
            | (Tool Calls?)
            No +-----------------------------+ Yes
               V                             V (For each ToolCall)
+--------------------------+                +------------------------------------+
| Yield Text Chunk         |                | `Gen. ToolCall Msg`                |
| Add Asst Msg to History  |                | Yield ToolCall (w/ user msg)       |
+--------------------------+                | (Yield COMPLETED if finish_*)      |
            | (Loop)                        +------------------------------------+
            |                                      |
            |                                      V (Internal processing)
            |                               +----------------------------------------+
            |                               | `_handle_tool_call` (Process ToolCall) |
            |                               |   - Tool Exec                          |
            |                               |   - Compress?                          |
            |                               |   - Save Binaries                      |
            |                               |   - Side Effects (finish_*, nav)       |
            |                               |     - finish_*: Store object directly  |
            |                               |   - Serialize Result (JSON)            |
            |                               +----------------------------------------+
            |                                            |
            |                                            V
            |      +---------------------------------------+
            |      | Add Tool Result to History            |
            +------+ (role='tool')                         | (Loop)
                   +---------------------------------------+
```
"""

import asyncio
import base64
import binascii
import datetime
import logging
import re
from nodetool.providers import BaseProvider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, SubTask, Task, ToolCall
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole

import tiktoken

from nodetool.chat.token_counter import (
    count_message_tokens,
    count_messages_tokens,
)


import json
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    List,
    Sequence,
    Union,
    Dict,
    Optional,
)
from nodetool.config.logging_config import get_logger

from jinja2 import Environment, BaseLoader
from jsonschema import ValidationError, validate as jsonschema_validate

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


DEFAULT_MAX_TOKEN_LIMIT: int = 4096
DEFAULT_MAX_ITERATIONS: int = 10
MESSAGE_COMPRESSION_THRESHOLD: int = 4096

DEFAULT_EXECUTION_SYSTEM_PROMPT: str = """
# Role
You are executing a single subtask within a larger plan. Your job is to complete this subtask end-to-end.

# Subtask Objective
{{ subtask_content }}

# Operating Mode
Persistence and completion:
- Keep going until the subtask is completed; do not hand back early.
- If something is ambiguous, make the most reasonable assumption, proceed, and document the assumption in `metadata.notes`.
- Prefer concrete tool calls and actions over clarifying questions.
- Stay focused: avoid unnecessary exploration or tangential work.

Agentic eagerness control:
- Maximum non-finish tool calls: {{ max_tool_calls }}
- Be efficient and selective with tool usage.
- Complete work directly when possible rather than adding subtasks for trivial steps.

Output style:
- Be concise and minimize token usage.
- Use structured, deterministic outputs over prose.
- Provide brief reasoning explanations, not extensive chain-of-thought.

# Tool Usage Guidelines

## Communication Pattern (Tool Preambles)
Before making tool calls, provide clear progress updates:
1. First assistant message: Restate the subtask objective in one sentence, then list a short numbered plan (1-3 steps).
2. Before each tool call: Emit a one-sentence message describing what you're doing and why.
3. After tool results: Provide a brief update only if the result changes your plan.

## Dynamic Task Management Tools
You have access to `add_subtask` and `list_subtasks` for expanding the task plan when necessary.

**When to use `add_subtask`:**
- You discover substantial, distinct work beyond the current subtask scope (e.g., separate categories requiring focused investigation, new dependencies, or parallel workstreams).
- The additional work would benefit from isolation with its own context and result tracking.
- Examples: Discovering multiple product categories each needing research, finding distinct technical issues requiring separate analysis.

**When NOT to use `add_subtask`:**
- Trivial steps you can complete directly.
- Sequential operations within the current scope.
- Simple follow-up actions.

**How to use `add_subtask`:**
```
{
  "name": "add_subtask",
  "args": {
    "content": "Clear instructions describing what the subtask should accomplish",
    "input_tasks": ["list", "of", "subtask_ids", "that", "must", "complete", "first"],
    "max_tool_calls": 15
  }
}
```

**Use `list_subtasks`:**
- To understand the broader context and see what work has been completed.
- To avoid duplicating work or creating redundant subtasks.

# Execution Protocol
1. **Use provided context:** Upstream task results are already present in context. Do not re-request them.
2. **Execute the work:** Perform the required steps to produce a result conforming to this subtask's schema.
3. **Consider dynamic expansion:** If you discover substantial additional work beyond current scope, use `add_subtask` to create a focused task for it.
4. **Finish properly:** When ready, call `finish_subtask` exactly once with:
   - `result`: The final structured object conforming to the output schema

# Stop Conditions
- Stop immediately after calling `finish_subtask` successfully.
- If you reach tool-call limits or enter conclusion stage, prioritize synthesizing available information and finishing.
- Do not continue working after successful task completion.

# Output Requirements
- Do not reveal internal chain-of-thought or reasoning traces.
- Output only necessary tool calls and required fields.
- Prefer structured, deterministic outputs following the schema.
- Keep all responses concise and token-efficient.
"""

DEFAULT_FINISH_TASK_SYSTEM_PROMPT: str = """
# Role
You are completing the final aggregation task, synthesizing results from prior subtasks into a single deliverable.

# Operating Mode
Persistence and completion:
- Keep going until the final result is produced; do not hand back early.
- Focus on synthesis and aggregation, not additional research.

Agentic eagerness control:
- Maximum non-finish tool calls: {{ max_tool_calls }}
- Be highly selective with tool usage during aggregation.
- Prioritize using existing results over gathering new information.

Output style:
- Be concise and token-efficient.
- Use structured, deterministic outputs.
- Provide brief reasoning, not extensive explanations.

# Tool Usage Guidelines

## Communication Pattern (Tool Preambles)
1. First assistant message: Restate the overall objective in one sentence, then outline a short aggregation plan (1-3 steps).
2. Before each tool call: Provide a one-sentence rationale of what you're doing and why.

## Dynamic Task Management Tools
You have access to `add_subtask` and `list_subtasks` if critical information is missing.

**When to use `add_subtask` during aggregation:**
- Critical information is missing that prevents producing a complete final result.
- Additional analysis or data gathering is essential before synthesis.
- Use sparingly - prefer working with existing results.

**Use `list_subtasks`:**
- To review what work has been completed and identify any gaps.
- To understand the full context before aggregating.

**How to use `add_subtask`:**
```
{
  "name": "add_subtask",
  "args": {
    "content": "Specific description of critical missing information to obtain",
    "input_tasks": ["existing", "subtask", "dependencies"],
    "max_tool_calls": 10
  }
}
```

# Aggregation Protocol
1. **Use provided results:** Upstream subtask results are already in context. Do not re-request them.
2. **Review completeness:** Use `list_subtasks` if needed to understand what work has been done.
3. **Identify gaps:** If critical information is missing and essential for the final result, use `add_subtask` to obtain it.
4. **Extract and synthesize:** Gather key information from completed subtask results.
5. **Produce final output:** Generate the complete deliverable matching the task schema.
6. **Finish properly:** Call `finish_task` exactly once with:
   - `result`: The complete final deliverable conforming to the output schema

# Stop Conditions
- Stop immediately after calling `finish_task` successfully.
- Do not continue working after successful completion.

# Output Requirements
- Do not reveal internal reasoning traces or chain-of-thought.
- Output only necessary tool calls and required fields.
- Prefer structured, deterministic outputs following the schema.
- Keep all responses concise and token-efficient.
"""


def _remove_think_tags(text_content: Optional[str]) -> Optional[str]:
    if text_content is None:
        return None
    # Use regex to remove <think>...</think> blocks, including newlines within them.
    # re.DOTALL makes . match newlines.
    # We also strip leading/trailing whitespace from the result.
    return re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()


def _validate_and_sanitize_schema(
    schema: Any, default_description: str = "Result object"
) -> Dict[str, Any]:
    """
    Validates and sanitizes a JSON schema to ensure it's compatible with OpenAI function calling.

    Args:
        schema: The schema to validate and sanitize
        default_description: Default description if none provided

    Returns:
        A valid JSON schema dict
    """
    if schema is None:
        raise ValueError("Schema is None")

    if isinstance(schema, str):
        schema = json.loads(schema)

    if not isinstance(schema, dict):
        raise ValueError(f"Schema is not a dict (type: {type(schema)})")

    # Make a deep copy to avoid modifying the original
    try:
        result_schema = json.loads(json.dumps(schema))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Schema contains non-serializable data: {e}")

    # Ensure required fields
    if "type" not in result_schema:
        result_schema["type"] = "object"

    if "description" not in result_schema:
        result_schema["description"] = default_description

    # Validate common schema issues
    disallowed_extension_keys = {
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
    }

    def _should_default_additional_properties(obj: Dict[str, Any]) -> bool:
        """Determine if we should set additionalProperties to False for this node."""
        if "additionalProperties" in obj:
            return False

        if any(key in obj for key in disallowed_extension_keys):
            return False

        schema_type = obj.get("type")
        if isinstance(schema_type, list):
            # Only treat as plain object if the type list exclusively contains "object"
            if len(schema_type) != 1 or schema_type[0] != "object":
                return False
        elif schema_type is not None and schema_type != "object":
            return False

        # Treat as object if explicitly typed or if properties imply it
        if schema_type == "object" or (
            schema_type is None and obj.get("properties") is not None
        ):
            return True

        return False

    def _clean_schema_recursive(obj: Any) -> Any:
        """Recursively clean schema objects to ensure compatibility."""
        if isinstance(obj, dict):
            cleaned = {key: _clean_schema_recursive(value) for key, value in obj.items()}
            if _should_default_additional_properties(cleaned):
                cleaned["additionalProperties"] = False
            return cleaned
        if isinstance(obj, list):
            return [_clean_schema_recursive(item) for item in obj]
        return obj

    result_schema = _clean_schema_recursive(result_schema)

    # Final validation by attempting to serialize
    try:
        json.dumps(result_schema)
    except (TypeError, ValueError) as e:
        # Note: We don't have access to display_manager here, so we'll keep this as a simple warning
        import warnings

        warnings.warn(
            f"Schema serialization failed after cleaning: {e}, using default object schema"
        )
        return {"type": "object", "description": default_description}

    return result_schema


class FinishTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a task as done and saves its results
    """

    name: str = "finish_task"
    description: str = """
    Finish a task by saving its final result as an object. Provide the result directly
    in the 'result' parameter - it will be stored as an object and made available to
    downstream processes. 
    """
    input_schema: Dict[str, Any]  # Defined in __init__

    def __init__(
        self,
        output_schema: Any,
    ):
        super().__init__()  # Call parent constructor

        # Validate and prepare the result schema
        result_schema = _validate_and_sanitize_schema(output_schema, "The task result")
        self.result_schema = result_schema

        # Final input schema for the tool
        self.input_schema = {
            "type": "object",
            "properties": {
                "result": result_schema,
            },
            "required": ["result"],
            "additionalProperties": False,
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]):
        return params.get("result", None)


class FinishSubTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a subtask as done and saves its results
    """

    name: str = "finish_subtask"
    description: str = """
    Finish a subtask by saving its final result as an object. Provide the result directly
    in the 'result' parameter - it will be stored as an object and made available to
    downstream tasks.
    """
    input_schema: Dict[str, Any]  # Defined in __init__

    def __init__(
        self,
        output_schema: Any,
    ):
        super().__init__()  # Call parent constructor

        # Validate and prepare the result schema
        result_schema = _validate_and_sanitize_schema(
            output_schema, "The subtask result"
        )
        self.result_schema = result_schema

        # Final input schema for the tool
        self.input_schema = {
            "type": "object",
            "properties": {
                "result": result_schema,
            },
            "required": ["result"],
            "additionalProperties": False,
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]):
        return params.get("result", None)


class SubTaskContext:
    """
    🧠 The Task-Specific Brain - Isolated execution environment for a single subtask

    This class maintains a completely isolated context for each subtask, with its own:
    - Message history: Tracks all interactions in this specific subtask
    - System prompt: Automatically selected based on subtask type (reasoning vs. standard)
    - Tools: Available for information gathering and task completion
    - Token tracking: Monitors context size with automatic summarization when needed

    Each subtask operates like a dedicated worker with exactly the right skills and
    information for that specific job, without interference from other tasks.

    Key Features:
    - Token limit monitoring with automatic context summarization when exceeding thresholds
    - Two-stage execution model: tool calling stage → conclusion stage
    - Safety limits: iteration tracking, max tool calls, and max token controls
    - Explicit reasoning capabilities for "thinking" subtasks
    - Progress reporting throughout execution
    """

    task: Task
    subtask: SubTask
    processing_context: ProcessingContext
    model: str
    provider: BaseProvider
    max_token_limit: int
    use_finish_task: bool
    jinja_env: Environment
    system_prompt: str
    finish_tool: Union[FinishTaskTool, FinishSubTaskTool]
    tools: Sequence[Tool]
    history: List[Message]
    iterations: int
    max_iterations: int
    sources: List[str]
    progress: List[Any]
    encoding: tiktoken.Encoding
    _chunk_buffer: str
    _is_buffering_chunks: bool
    in_conclusion_stage: bool
    _output_result: Any
    tool_calls_made: int
    max_tool_calls: int
    input_tokens_total: int
    output_tokens_total: int

    def __init__(
        self,
        task: Task,
        subtask: SubTask,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        model: str,
        provider: BaseProvider,
        system_prompt: Optional[str] = None,
        use_finish_task: bool = False,
        max_token_limit: int | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        display_manager: Optional[AgentConsole] = None,
    ):
        """
        Initialize a subtask execution context.

        Args:
            task (Task): The task to execute
            subtask (SubTask): The subtask to execute
            processing_context (ProcessingContext): The processing context
            system_prompt (str): The system prompt for this subtask
            tools (List[Tool]): Tools available to this subtask
            model (str): The model to use for this subtask
            provider (ChatProvider): The provider to use for this subtask
            use_finish_task (bool): Whether to use the finish_task tool
            max_token_limit (int): Maximum token limit before summarization
            max_iterations (int): Maximum iterations for the subtask
            display_manager (Optional[AgentConsole]): Console for beautiful terminal output
        """
        self.task = task
        self.subtask = subtask
        self.processing_context = processing_context
        self.model = model
        self.provider = provider
        self.max_token_limit = max_token_limit or provider.get_context_length(model)
        self.use_finish_task = use_finish_task
        self.message_compression_threshold = max(
            self.max_token_limit // 4,
            MESSAGE_COMPRESSION_THRESHOLD,
        )
        self._output_result = None  # Added
        self.display_manager = display_manager

        # Initialize tool call tracking
        self.tool_calls_made = 0
        self.max_tool_calls = subtask.max_tool_calls

        # Initialize token usage tracking
        self.input_tokens_total = 0
        self.output_tokens_total = 0

        # Note: Initialization debug messages will be logged after setting current subtask in execute()
        self._init_debug_messages = [
            f"Initializing SubTaskContext for subtask: {subtask.id}",
            f"Task: {task.title}",
            f"Subtask content: {subtask.content}",
            f"Subtask output_schema: {subtask.output_schema}",
            f"Subtask output_schema type: {type(subtask.output_schema)}",
            f"Model: {model}, Provider: {provider.__class__.__name__}",
            f"Max token limit: {self.max_token_limit}",
            f"Max iterations: {max_iterations}",
            f"Use finish task: {use_finish_task}",
            f"Available tools: {[tool.name for tool in tools]}",
        ]

        # --- Prepare prompt templates ---
        self.jinja_env = Environment(loader=BaseLoader())

        if use_finish_task:
            base_system_prompt = system_prompt or DEFAULT_FINISH_TASK_SYSTEM_PROMPT
            prompt_context = {
                "max_tool_calls": self.max_tool_calls,
            }
            self.finish_tool = FinishTaskTool(self.subtask.output_schema)
        else:  # Standard execution subtask
            base_system_prompt = system_prompt or DEFAULT_EXECUTION_SYSTEM_PROMPT
            prompt_context = {
                "subtask_content": self.subtask.content,
                "max_tool_calls": self.max_tool_calls,
            }  # Provide subtask content context
            self.finish_tool = FinishSubTaskTool(self.subtask.output_schema)

        self.system_prompt = self._render_prompt(base_system_prompt, prompt_context)

        self.tools = list(tools) + [
            self.finish_tool,
        ]
        self.system_prompt = (
            self.system_prompt
            + "\n\nToday's date is "
            + datetime.datetime.now().strftime("%Y-%m-%d")
        )

        # Initialize isolated message history for this subtask
        self.history = [Message(role="system", content=self.system_prompt)]

        # Track iterations for this subtask
        self.iterations = 0
        if max_iterations < 3:
            raise ValueError("max_iterations must be at least 3")
        self.max_iterations = max_iterations

        # Track sources for data lineage
        self.sources = []

        # Track progress for this subtask
        self.progress = []

        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Add a buffer for aggregating chunks
        self._chunk_buffer = ""
        self._is_buffering_chunks = False

        # Flag to track which stage we're in - normal execution flow for all tasks
        self.in_conclusion_stage = False

    def _render_prompt(self, template_string: str, context: dict) -> str:
        """Renders a prompt template using Jinja2."""
        template = self.jinja_env.from_string(template_string)
        return template.render(context)

    def _count_tokens(self, messages: List[Message]) -> int:
        """
        Count the number of tokens in the message history.

        Args:
            messages: The messages to count tokens for

        Returns:
            int: The approximate token count
        """
        return count_messages_tokens(messages, encoding=self.encoding)

    def _count_single_message_tokens(self, msg: Message) -> int:
        """
        Count the number of tokens in a single message.

        Args:
            msg: The message to count tokens for.

        Returns:
            int: The approximate token count for the single message.
        """
        return count_message_tokens(msg, encoding=self.encoding)

    def _normalize_tool_result(self, value: Any) -> Any:
        """
        Convert tool results into JSON-serializable primitives recursively.
        """
        if hasattr(value, "model_dump"):
            return self._normalize_tool_result(value.model_dump())
        if isinstance(value, dict):
            return {key: self._normalize_tool_result(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._normalize_tool_result(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def get_result(self) -> Any | None:
        """
        Returns the stored result object.
        """
        return self._output_result

    async def execute(
        self,
    ) -> AsyncGenerator[Union[Chunk, ToolCall, TaskUpdate], None]:
        """
        Runs a single subtask to completion using the LLM-driven execution loop.

        Yields:
            Union[Chunk, ToolCall, TaskUpdate]: Live updates during task execution
        """
        # Record the start time of the subtask
        current_time = int(time.time())
        self.subtask.start_time = current_time

        # Ensure subtask logs are initialized
        if not hasattr(self.subtask, "logs") or self.subtask.logs is None:
            self.subtask.logs = []

        # Set the current subtask for the display manager
        if self.display_manager:
            self.display_manager.set_current_subtask(self.subtask)

        # Log initialization messages now that current subtask is set
        for msg in self._init_debug_messages:
            log.debug(msg)

        # Display beautiful subtask start panel
        if self.display_manager:
            self.display_manager.display_subtask_start(self.subtask)

        # --- LLM-based Execution Logic ---
        prompt_parts = [
            f"**Overall Task:**\nTitle: {self.task.title}\nDescription: {self.task.description}\n",
            f"**Current Subtask Instructions:**\n{self.subtask.content}\n",  # Treat content as instructions
        ]

        if self.subtask.input_tasks:
            input_tasks_str = ", ".join(self.subtask.input_tasks)
            prompt_parts.append(
                f"**Input Tasks for this Subtask:**\n{input_tasks_str}\n"
            )

            # Fetch and inject input results directly into the prompt
            input_results = []
            for input_task_id in self.subtask.input_tasks:
                try:
                    result = self.processing_context.get(input_task_id)
                    if result is not None:
                        input_results.append(
                            f"**Result from Task {input_task_id}:**\n{json.dumps(result, indent=2, ensure_ascii=False)}\n"
                        )
                    else:
                        input_results.append(
                            f"**Result from Task {input_task_id}:** No result available\n"
                        )
                except Exception as e:
                    if self.display_manager:
                        self.display_manager.warning(
                            f"Failed to fetch result for task {input_task_id}: {e}"
                        )
                    input_results.append(
                        f"**Result from Task {input_task_id}:** Error fetching result: {e}\n"
                    )

            if input_results:
                prompt_parts.append(
                    "**Input Data from Upstream Tasks:**\n" + "\n".join(input_results)
                )

        prompt_parts.append(
            "Please perform the subtask based on the provided context, instructions, and upstream task results."
        )
        task_prompt = "\n".join(prompt_parts)

        # Add the task prompt to this subtask's history
        self.history.append(Message(role="user", content=task_prompt))

        log.debug(f"Task prompt added to history: {task_prompt[:200]}...")

        # Yield task update for subtask start
        yield TaskUpdate(
            task=self.task,
            subtask=self.subtask,
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

        # Display task update event
        if self.display_manager:
            self.display_manager.display_task_update(
                "SUBTASK_STARTED", self.subtask.content
            )

        # Continue executing until the task is completed or max iterations reached
        while not self.subtask.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            log.debug(f"Starting iteration {self.iterations}/{self.max_iterations}")

            # Calculate total token count AFTER potential compression
            token_count = self._count_tokens(self.history)

            # Display beautiful iteration status
            # self.display_manager.display_iteration_status(
            #     self.iterations, self.max_iterations, token_count, self.max_token_limit
            # )

            # Check if we need to transition to conclusion stage
            if (token_count > self.max_token_limit) and not self.in_conclusion_stage:
                # Log and display token warning
                log.warning(f"Token usage: {token_count}/{self.max_token_limit}")
                await self._transition_to_conclusion_stage()
                # Yield the event after transitioning
                yield TaskUpdate(
                    task=self.task,
                    subtask=self.subtask,
                    event=TaskUpdateEvent.ENTERED_CONCLUSION_STAGE,
                )
                if self.display_manager:
                    self.display_manager.display_task_update("ENTERED_CONCLUSION_STAGE")

            # Process current iteration
            message = await self._process_iteration()

            if message.tool_calls:
                if self.display_manager:
                    self.display_manager.debug_subtask_only(
                        f"LLM returned {len(message.tool_calls)} tool calls"
                    )
                message.tool_calls = self._filter_tool_calls_for_current_stage(
                    message.tool_calls
                )

            # Add assistant message to history after any tool-call adjustments so the
            # stored conversation precisely matches what we'll execute/respond to.
            self.history.append(message)

            if message.tool_calls:
                # Separate finish tools from other tools - finish tools are always allowed
                finish_tools = [
                    tc
                    for tc in message.tool_calls
                    if tc.name in ("finish_subtask", "finish_task")
                ]
                other_tools = [
                    tc
                    for tc in message.tool_calls
                    if tc.name not in ("finish_subtask", "finish_task")
                ]

                # Check if we would exceed the tool call limit for non-finish tools
                if self.tool_calls_made + len(other_tools) > self.max_tool_calls:
                    remaining_calls = self.max_tool_calls - self.tool_calls_made
                    if remaining_calls <= 0:
                        # No more non-finish tool calls allowed
                        if finish_tools:
                            # Allow only finish tools
                            message.tool_calls = finish_tools
                            log.warning(
                                f"Tool call limit ({self.max_tool_calls}) reached. Only allowing finish tools."
                            )
                        else:
                            # Force completion if no finish tools available
                            log.warning(
                                f"Tool call limit ({self.max_tool_calls}) reached. Forcing completion."
                            )
                            tool_call = await self._handle_max_tool_calls_reached()
                            yield tool_call
                            yield TaskUpdate(
                                task=self.task,
                                subtask=self.subtask,
                                event=TaskUpdateEvent.MAX_TOOL_CALLS_REACHED,
                            )
                            log.warning(
                                f"Tool calls: {self.tool_calls_made}/{self.max_tool_calls}"
                            )
                            # Remove the assistant message with unprocessed tool calls
                            # since we won't be responding to them.
                            if self.history and self.history[-1] is message:
                                self.history.pop()
                            break
                    else:
                        # Allow remaining non-finish calls plus all finish calls
                        allowed_other_tools = other_tools[:remaining_calls]
                        message.tool_calls = finish_tools + allowed_other_tools
                        log.warning(
                            f"Tool call limit approaching. Processing {len(finish_tools)} finish tools and {len(allowed_other_tools)} of {len(other_tools)} other tool calls."
                        )

                if message.content:
                    yield Chunk(content=str(message.content))
                for tool_call in message.tool_calls:
                    # Log tool execution only to subtask (not phase)
                    if tool_call.name not in ("finish_subtask", "finish_task"):
                        log.debug(f"Executing tool: {tool_call.name}")
                        self.tool_calls_made += 1
                    tool_message = self._generate_tool_call_message(tool_call)
                    yield ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        subtask_id=self.subtask.id,
                        message=tool_message,
                    )
                    if (
                        tool_call.name == "finish_subtask"
                        or tool_call.name == "finish_task"
                    ):
                        log.debug(f"Subtask completed via {tool_call.name}")
                        yield TaskUpdate(
                            task=self.task,
                            subtask=self.subtask,
                            event=TaskUpdateEvent.SUBTASK_COMPLETED,
                        )
                        log.debug(f"Subtask completed: {self.subtask.content}")

                await self._process_tool_call_results(message.tool_calls)
            elif message.content:
                # Handle potential text chunk yields if provider supports streaming text
                yield Chunk(content=str(message.content))

        # If we've reached the last iteration and haven't completed yet, generate summary
        if self.iterations >= self.max_iterations and not self.subtask.completed:
            tool_call = await self._handle_max_iterations_reached()
            yield tool_call
            yield TaskUpdate(
                task=self.task,
                subtask=self.subtask,
                event=TaskUpdateEvent.MAX_ITERATIONS_REACHED,
            )
            if self.display_manager:
                self.display_manager.display_task_update(
                    "MAX_ITERATIONS_REACHED",
                    f"Iterations: {self.iterations}/{self.max_iterations}",
                )
            yield tool_call

        # Display completion event
        if self.display_manager:
            self.display_manager.display_completion_event(
                self.subtask, self.subtask.completed, self._output_result
            )
            log.debug(f"Total iterations: {self.iterations}")
            log.debug(f"Total messages in history: {len(self.history)}")

        # Useful debug printing
        # for m in self.history:
        #     print("-" * 100)
        #     print(m.role, m.content)
        #     print("-" * 100)

    async def _transition_to_conclusion_stage(self) -> None:
        """
        Transition from tool calling stage to conclusion stage.

        This method:
        1. Sets the conclusion stage flag
        2. Adds a clear transition message to the conversation history
        3. Logs the transition to the console
        4. Restricts available tools to only finish_subtask
        """
        self.in_conclusion_stage = True
        transition_message = f"""
        SYSTEM: The conversation history is approaching the token limit ({self.max_token_limit} tokens).
        ENTERING CONCLUSION STAGE: You MUST now synthesize all gathered information and finalize the subtask.
        Your ONLY available tool is '{self.finish_tool.name}'. Use it to provide the final result.
        Do not request any other tools. Focus on generating the complete output based on the work done so far.
        """
        # Check if the message already exists to prevent duplicates if called multiple times
        if not any(
            m.role == "system" and "ENTERING CONCLUSION STAGE" in str(m.content)
            for m in self.history
        ):
            self.history.append(Message(role="system", content=transition_message))
            if self.display_manager:
                self.display_manager.display_conclusion_stage()

    async def _process_iteration(
        self,
    ) -> Message:
        """
        Process a single iteration of the task.
        """
        tools_for_iteration = (
            [self.finish_tool]  # Only allow finish tool in conclusion stage
            if self.in_conclusion_stage
            else self.tools  # Allow all tools otherwise
        )

        # Create a dictionary to track unique tools by name
        unique_tools = {tool.name: tool for tool in tools_for_iteration}
        final_tools = list(unique_tools.values())

        try:
            log.debug(f"Calling LLM with {len(self.history)} messages in history")
            # Count input tokens from current history prior to generation
            try:
                input_tokens_now = self._count_tokens(self.history)
                self.input_tokens_total += input_tokens_now
                log.debug(
                    f"Input tokens this call: {input_tokens_now} (cumulative: {self.input_tokens_total})"
                )
            except Exception as e:
                log.warning(f"Failed to count input tokens: {e}")
            message = await self.provider.generate_message(
                messages=self.history,
                model=self.model,
                tools=final_tools,
            )
            log.debug(
                f"LLM response received - content length: {len(str(message.content)) if message.content else 0}"
            )
            # Count output tokens from returned assistant message
            try:
                output_tokens_now = self._count_single_message_tokens(message)
                self.output_tokens_total += output_tokens_now
                log.debug(
                    f"Output tokens this call: {output_tokens_now} (cumulative: {self.output_tokens_total})"
                )
            except Exception as e:
                log.warning(f"Failed to count output tokens: {e}")
            if message.tool_calls:
                log.debug(
                    f"LLM requested tool calls: {[tc.name for tc in message.tool_calls]}"
                )
        except Exception as e:
            log.error(f"Error generating message: {e}", exc_info=True)
            raise e

        # Clean assistant message content
        if isinstance(message.content, str):
            message.content = _remove_think_tags(message.content)
        elif isinstance(message.content, list):
            for part_dict in message.content:  # Iterate directly over parts
                if isinstance(part_dict, dict) and part_dict.get("type") == "text":
                    text_val = part_dict.get("text")
                    if isinstance(text_val, str):
                        cleaned_text = _remove_think_tags(text_val)
                        part_dict["text"] = cleaned_text
                    elif text_val is None:
                        cleaned_text = _remove_think_tags(None)  # Explicitly pass None
                        part_dict["text"] = cleaned_text  # Assigns None back

        return message

    def _filter_tool_calls_for_current_stage(
        self, tool_calls: Optional[list[ToolCall]]
    ) -> list[ToolCall]:
        """Filter tool calls based on whether we're in the conclusion stage."""
        if not tool_calls:
            return []

        if not self.in_conclusion_stage:
            return list(tool_calls)

        allowed_calls: list[ToolCall] = []
        for tc in tool_calls:
            if tc.name == self.finish_tool.name:
                allowed_calls.append(tc)
            else:
                log.warning(
                    f"LLM attempted to call disallowed tool '{tc.name}' in conclusion stage. Ignoring."
                )

        if not allowed_calls:
            log.warning(
                "LLM did not call the required finish tool in conclusion stage."
            )

        return allowed_calls

    async def _process_tool_call_results(
        self, valid_tool_calls: list[ToolCall]
    ) -> None:
        """Execute tool calls and add their results to history."""
        if not valid_tool_calls:
            return

        log.debug(f"Processing {len(valid_tool_calls)} valid tool calls")

        tool_results = await asyncio.gather(
            *[self._handle_tool_call(tool_call) for tool_call in valid_tool_calls],
            return_exceptions=True,
        )

        valid_tool_messages: list[Message] = []
        for tool_call, result in zip(valid_tool_calls, tool_results):
            if isinstance(result, Exception):
                log.error(
                    f"Tool call {tool_call.id} ({tool_call.name}) failed with exception: {result}"
                )
                error_message = Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error executing tool: {str(result)}",
                )
                valid_tool_messages.append(error_message)
            elif isinstance(result, Message):
                valid_tool_messages.append(result)
            else:
                log.error(
                    f"Tool call {tool_call.id} ({tool_call.name}) returned unexpected type: {type(result)}"
                )
                error_message = Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Unexpected tool result type: {type(result)}",
                )
                valid_tool_messages.append(error_message)

        self.history.extend(valid_tool_messages)
        log.debug(f"Added {len(valid_tool_messages)} tool results to history")

    def _generate_tool_call_message(self, tool_call: ToolCall) -> str:
        """
        Generate a message object from a tool call.
        """
        for tool in self.tools:
            if tool.name == tool_call.name:
                return tool.user_message(tool_call.args)

        raise ValueError(f"Tool '{tool_call.name}' not found in available tools.")

    def _validate_finish_tool_result(
        self, tool_result: Any, tool_call: ToolCall
    ) -> tuple[bool, Optional[str], Optional[Dict[str, Any]], Any]:
        """
        Validate finish tool results against the declared schema.
        """
        tool_instance: Optional[Tool] = None
        for tool in self.tools:
            if tool.name == tool_call.name:
                tool_instance = tool
                break

        schema: Optional[Dict[str, Any]] = None
        if tool_instance is not None:
            schema = getattr(tool_instance, "result_schema", None)
            if schema is None:
                schema = (
                    getattr(tool_instance, "input_schema", {})
                    .get("properties", {})
                    .get("result")
                )

        normalized_result = self._normalize_tool_result(tool_result)

        if not schema:
            log.debug(
                "No result schema available for finish tool %s; skipping validation",
                tool_call.name,
            )
            return True, None, None, normalized_result

        try:
            jsonschema_validate(normalized_result, schema)
            return True, None, schema, normalized_result
        except ValidationError as exc:
            error_detail = exc.message
            json_path = getattr(exc, "json_path", None)
            if json_path:
                error_detail = f"{json_path}: {exc.message}"
            return False, error_detail, schema, normalized_result
        except Exception as exc:  # pragma: no cover - unexpected errors
            return False, str(exc), schema, normalized_result

    async def _handle_tool_call(self, tool_call: ToolCall) -> Message:
        """
        Handle a tool call by executing it, processing its result, and returning a message for history.
        This involves execution, potential compression, artifact handling, special side-effects, and serialization.

        Args:
            tool_call (ToolCall): The tool call from the assistant message.

        Returns:
            Message: A message object with role 'tool' containing the processed and serialized result.
        """
        if self.display_manager:
            self.display_manager.debug_subtask_only(
                f"Handling tool call: {tool_call.name} (ID: {tool_call.id})"
            )

        # 1. Execute the tool
        tool_result = await self._process_tool_execution(tool_call)

        if self.display_manager:
            self.display_manager.debug_subtask_only(
                f"Tool {tool_call.name} execution completed"
            )

        is_finish_tool = tool_call.name in ("finish_task", "finish_subtask")
        is_valid_finish = True
        finish_error_detail: Optional[str] = None
        expected_schema: Optional[Dict[str, Any]] = None
        normalized_result: Any = None

        if is_finish_tool:
            (
                is_valid_finish,
                finish_error_detail,
                expected_schema,
                normalized_result,
            ) = self._validate_finish_tool_result(tool_result, tool_call)

            if not is_valid_finish:
                if self.display_manager:
                    self.display_manager.warning(
                        f"{tool_call.name} result failed validation: {finish_error_detail}"
                    )
                log.warning(
                    "Finish tool %s result failed schema validation: %s",
                    tool_call.name,
                    finish_error_detail,
                )
                error_payload: Dict[str, Any] = {
                    "error": "finish_tool_validation_failed",
                    "detail": finish_error_detail
                    or "Result did not match the expected schema.",
                    "resolution": "Adjust the result to match the expected schema and call the finish tool again.",
                }
                if expected_schema is not None:
                    error_payload["expected_schema"] = expected_schema
                if normalized_result is not None:
                    error_payload["submitted_result"] = normalized_result
                tool_result = error_payload

        # 3. Handle binary artifacts (images, audio)
        if isinstance(tool_result, dict) and (not is_finish_tool or is_valid_finish):
            if "image" in tool_result:
                tool_result = self._handle_binary_artifact(
                    tool_result, tool_call.name, "image"
                )
            elif "audio" in tool_result:
                tool_result = self._handle_binary_artifact(
                    tool_result, tool_call.name, "audio"
                )

        # 4. Process special tool side-effects (e.g., finish_task, browser navigation)
        if not is_finish_tool or is_valid_finish:
            self._process_special_tool_side_effects(tool_result, tool_call)
        else:
            log.debug(
                "Skipping special side effects for %s due to validation failure",
                tool_call.name,
            )

        # Log tool result only to subtask tree (not phase)
        if tool_call.name not in ("finish_subtask", "finish_task"):
            if self.display_manager:
                self.display_manager.info_subtask_only(
                    f"Tool result received from {tool_call.name}"
                )

        # 5. Serialize the final processed result for history
        content_str = self._serialize_tool_result_for_history(
            tool_result, tool_call.name
        )

        # Return the tool result as a message to be added to history
        return Message(
            role="tool",
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content_str,
        )

    async def _process_tool_execution(self, tool_call: ToolCall) -> Any:
        """Executes the specified tool and returns its raw result."""
        for tool in self.tools:
            if tool.name == tool_call.name:
                return await tool.process(self.processing_context, tool_call.args)
        raise ValueError(f"Tool '{tool_call.name}' not found in available tools.")

    def _handle_binary_artifact(
        self, tool_result: Dict[str, Any], tool_call_name: str, artifact_type: str
    ) -> Dict[str, Any]:
        """Handles saving binary artifacts (image or audio) and updating the tool result."""
        artifact_key = artifact_type  # "image" or "audio"
        base64_data = tool_result.get(artifact_key)

        if not isinstance(base64_data, str):
            if self.display_manager:
                self.display_manager.warning(
                    f"No valid base64 data found for artifact type '{artifact_type}' in tool '{tool_call_name}' result."
                )
            return tool_result

        # Determine file extension
        file_ext = "png"  # Default for images
        if artifact_type == "audio":
            file_ext = str(tool_result.get("format", "mp3")).strip().lower()

        try:
            # Generate a unique filename
            artifact_filename = f"artifact_{uuid.uuid4().hex[:8]}.{file_ext}"
            artifact_rel_path = artifact_filename  # Save at the root
            artifact_abs_path = self.processing_context.resolve_workspace_path(
                artifact_rel_path
            )

            # Ensure artifacts directory exists (No longer needed for root saving, parent is workspace_dir)
            # os.makedirs(os.path.dirname(artifact_abs_path), exist_ok=True)

            # Decode and write the artifact
            decoded_data = base64.b64decode(base64_data)
            with open(artifact_abs_path, "wb") as artifact_file:
                artifact_file.write(decoded_data)

            if self.display_manager:
                self.display_manager.info_subtask_only(
                    f"Saved base64 {artifact_type} from tool '{tool_call_name}' to {artifact_rel_path}"
                )

            # Update result: add path, remove original base64 data
            tool_result[f"{artifact_type}_path"] = artifact_rel_path
            del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]  # Clean up format key for audio

        except (binascii.Error, ValueError) as e:
            if self.display_manager:
                self.display_manager.error(
                    f"Failed to decode base64 {artifact_type} from tool '{tool_call_name}': {e}"
                )
            tool_result[f"{artifact_type}_path"] = (
                f"Error decoding {artifact_type}: {e}"
            )
            if artifact_key in tool_result:
                del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]
        except Exception as e:
            if self.display_manager:
                self.display_manager.error(
                    f"Failed to save {artifact_type} artifact from tool '{tool_call_name}': {e}"
                )
            tool_result[f"{artifact_type}_path"] = f"Error saving {artifact_type}: {e}"
            if artifact_key in tool_result:
                del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]
        return tool_result

    def _process_special_tool_side_effects(self, tool_result: Any, tool_call: ToolCall):
        """Handles side effects for specific tools, like 'browser' or 'finish_*'."""
        log.debug(f"Processing special side effects for tool: {tool_call.name}")

        if tool_call.name == "browser" and isinstance(tool_call.args, dict):
            action = tool_call.args.get("action", "")
            url = tool_call.args.get("url", "")
            if action == "navigate" and url:
                if url not in self.sources:  # Avoid duplicates
                    self.sources.append(url)
                    log.debug(f"Added browser source: {url}")

        if tool_call.name == "finish_task":
            log.debug("Processing finish_task - marking subtask as completed")
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            # Store the result object directly
            self.processing_context.set(self.task.id, tool_result)
            self.processing_context.set(self.subtask.id, tool_result)
            self._output_result = tool_result  # Store for completion display

        if tool_call.name == "finish_subtask":
            log.debug("Processing finish_subtask - marking subtask as completed")
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            # Store the result object directly
            self.processing_context.set(self.subtask.id, tool_result)
            self._output_result = tool_result  # Store for completion display
            log.debug(f"Subtask {self.subtask.id} completed with result: {tool_result}")

    def _serialize_tool_result_for_history(
        self, tool_result: Any, tool_name: str
    ) -> str:
        """Serializes the tool result to a JSON string for message history."""

        try:
            if tool_result is None:
                return "Tool returned no output."
            normalized = self._normalize_tool_result(tool_result)
            return json.dumps(normalized, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            log.error(
                f"Failed to serialize tool result for '{tool_name}' to JSON: {e}. Result: {tool_result}"
            )
            return json.dumps(
                {
                    "error": f"Failed to serialize tool result: {e}",
                    "result_repr": repr(tool_result),
                }
            )

    async def _handle_max_iterations_reached(self):
        """
        Handle the case where max iterations are reached without completion by prompting
        the LLM to call the finish tool.
        """
        log.warning(
            f"Subtask '{self.subtask.content}' reached max iterations ({self.max_iterations}). Forcing completion."
        )

        log.debug(f"Max iterations reached for subtask {self.subtask.id}")
        log.debug("Prompting LLM to call finish tool")

        # Determine the appropriate finish tool name
        tool_name = "finish_task" if self.use_finish_task else "finish_subtask"

        # Add a system message prompting the LLM to finish
        force_completion_prompt = f"""
SYSTEM: You have reached the maximum allowed iterations ({self.max_iterations}) for this subtask.
You MUST now call the '{tool_name}' tool to complete the task.
Synthesize all the work done so far and provide the best possible result based on the conversation history.
Do not attempt any other tool calls - only call '{tool_name}' with your final result.
"""

        self.history.append(Message(role="system", content=force_completion_prompt))

        # Get the LLM to generate the finish tool call
        try:
            # Count input tokens for finish prompt call
            input_tokens_now = self._count_tokens(self.history)
            self.input_tokens_total += input_tokens_now
            log.debug(
                f"Input tokens (max iterations finish): {input_tokens_now} (cumulative: {self.input_tokens_total})"
            )
        except Exception as e:
            log.warning(f"Failed to count input tokens (max iterations): {e}")

        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=[self.finish_tool],  # Only allow the finish tool
        )

        try:
            output_tokens_now = self._count_single_message_tokens(message)
            self.output_tokens_total += output_tokens_now
            log.debug(
                f"Output tokens (max iterations finish): {output_tokens_now} (cumulative: {self.output_tokens_total})"
            )
        except Exception as e:
            log.warning(f"Failed to count output tokens (max iterations): {e}")

        # Add the assistant message to history
        self.history.append(message)

        # Process the tool calls if any
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Should be the finish tool call
            await self._handle_tool_call(tool_call)
            return tool_call
        else:
            # If LLM didn't call the tool, create a fallback
            log.warning(
                "LLM failed to call finish tool at max iterations, creating fallback"
            )
            fallback_result = {
                "result": {
                    "status": "completed_at_max_iterations_fallback",
                    "message": f"Task completed after reaching maximum iterations ({self.max_iterations}) - LLM failed to generate proper finish call",
                    "iteration_count": self.iterations,
                }
            }

            tool_call = ToolCall(
                id=f"max_iterations_fallback_{tool_name}",
                name=tool_name,
                args=fallback_result,
            )

            await self._handle_tool_call(tool_call)
            return tool_call

    async def _handle_max_tool_calls_reached(self):
        """
        Handle the case where max tool calls are reached without completion by prompting
        the LLM to call the finish tool.
        """
        if self.display_manager:
            self.display_manager.warning(
                f"Subtask '{self.subtask.content}' reached max tool calls ({self.max_tool_calls}). Forcing completion."
            )

        if self.display_manager:
            self.display_manager.debug(
                f"Max tool calls reached for subtask {self.subtask.id}"
            )
            self.display_manager.debug("Prompting LLM to call finish tool")

        # Determine the appropriate finish tool name
        tool_name = "finish_task" if self.use_finish_task else "finish_subtask"

        # Add a system message prompting the LLM to finish
        force_completion_prompt = f"""
SYSTEM: You have reached the maximum allowed tool calls ({self.max_tool_calls}) for this subtask.
You MUST now call the '{tool_name}' tool to complete the task.
Synthesize all the work done so far and provide the best possible result based on the conversation history.
Do not attempt any other tool calls - only call '{tool_name}' with your final result.
"""

        self.history.append(Message(role="system", content=force_completion_prompt))

        # Get the LLM to generate the finish tool call
        try:
            # Count input tokens for finish prompt call
            input_tokens_now = self._count_tokens(self.history)
            self.input_tokens_total += input_tokens_now
            log.debug(
                f"Input tokens (max tool calls finish): {input_tokens_now} (cumulative: {self.input_tokens_total})"
            )
        except Exception as e:
            log.warning(f"Failed to count input tokens (max tool calls): {e}")

        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=[self.finish_tool],  # Only allow the finish tool
        )

        try:
            output_tokens_now = self._count_single_message_tokens(message)
            self.output_tokens_total += output_tokens_now
            log.debug(
                f"Output tokens (max tool calls finish): {output_tokens_now} (cumulative: {self.output_tokens_total})"
            )
        except Exception as e:
            log.warning(f"Failed to count output tokens (max tool calls): {e}")

        # Add the assistant message to history
        self.history.append(message)

        # Process the tool calls if any
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Should be the finish tool call
            await self._handle_tool_call(tool_call)
            return tool_call
        else:
            # If LLM didn't call the tool, create a fallback
            log.warning(
                "LLM failed to call finish tool at max tool calls, creating fallback"
            )
            fallback_result = {
                "result": {
                    "status": "completed_at_max_tool_calls_fallback",
                    "message": f"Task completed after reaching maximum tool calls ({self.max_tool_calls}) - LLM failed to generate proper finish call",
                    "tool_calls_made": self.tool_calls_made,
                }
            }

            tool_call = ToolCall(
                id=f"max_tool_calls_fallback_{tool_name}",
                name=tool_name,
                args=fallback_result,
            )

            await self._handle_tool_call(tool_call)
            return tool_call

    async def _execute_tool(self, tool_call: ToolCall) -> Any:
        """
        Execute a tool call using the available tools.

        Args:
            tool_call (ToolCall): The tool call to execute

        Returns:
            ToolCall: The tool call with the result attached
        """
        for tool in self.tools:
            if tool.name == tool_call.name:
                return await tool.process(self.processing_context, tool_call.args)

        raise ValueError(f"Tool '{tool_call.name}' not found")

    async def _compress_tool_result(
        self, result_content: Any, tool_name: str, tool_args: dict
    ) -> Union[dict, str]:
        """
        Compresses large tool result content using an LLM call.

        Args:
            result_content: The original tool result content (often a dict).
            tool_name: The name of the tool that produced the result.
            tool_args: The arguments passed to the tool.

        Returns:
            The compressed result (potentially a dict summarizing the original,
            or a string summary), or the original content if compression fails.
        """
        try:
            # Serialize the original content for the LLM prompt
            original_content_str = json.dumps(
                result_content, indent=2, ensure_ascii=False
            )
        except Exception as e:
            log.error(f"Failed to serialize result content for compression: {e}")
            return {
                "error": "Failed to serialize content for compression",
                "original_content_preview": repr(result_content)[:500],
            }

        compression_system_prompt = f"""
        # Goal
        Reduce the size of the 'TOOL RESULT TO COMPRESS' by removing duplicate information and summarizing the content.
        Ensure that all information vital to achieving the subtask's objective ('{self.subtask.content}') is retained.

        # Output Format
        - If the input is JSON, the output should ideally be a valid, smaller JSON object preserving the essential structure and data.
        - If the input is not JSON or cannot be effectively summarized as JSON, provide a concise text summary.

        # Context
        - Tool Name: {tool_name}
        - Tool Arguments: {json.dumps(tool_args, ensure_ascii=False)}
        - Overall Task: {self.task.title} - {self.task.description}
        - Current Subtask: {self.subtask.content}

        Instructions:
        - Focus ONLY on summarizing the 'TOOL RESULT TO COMPRESS' provided below.
        - **Crucially, ensure that all information vital to achieving the subtask's objective ('{self.subtask.content}') is retained.**
        - Keep all URLs, file paths, and other references to external sources.
        - Keep all code snippets and other non-text information.
        - Keep all name, entity, and other specific information.
        - Output *only* the compressed summary.
        - KEEP all relevant details of the original content.
        """

        compression_user_prompt = f"TOOL RESULT TO COMPRESS:\n---\n{original_content_str}\n---\nCOMPRESSED SUMMARY:"

        try:
            # Count input tokens for compression call
            try:
                input_tokens_now = self._count_tokens(
                    [
                        Message(role="system", content=compression_system_prompt),
                        Message(role="user", content=compression_user_prompt),
                    ]
                )
                self.input_tokens_total += input_tokens_now
                log.debug(
                    f"Input tokens (compression): {input_tokens_now} (cumulative: {self.input_tokens_total})"
                )
            except Exception as e:
                log.warning(f"Failed to count input tokens (compression): {e}")

            # Use the same provider and model for consistency, but no tools
            compression_response = await self.provider.generate_message(
                messages=[
                    Message(role="system", content=compression_system_prompt),
                    Message(role="user", content=compression_user_prompt),
                ],
                model=self.model,  # Or potentially a cheaper/faster model
                tools=[],
                max_tokens=self.message_compression_threshold,  # Limit the summary size
            )

            try:
                output_tokens_now = self._count_single_message_tokens(
                    compression_response
                )
                self.output_tokens_total += output_tokens_now
                log.debug(
                    f"Output tokens (compression): {output_tokens_now} (cumulative: {self.output_tokens_total})"
                )
            except Exception as e:
                log.warning(f"Failed to count output tokens (compression): {e}")

            # Clean compression_response.content before str() and strip()
            if isinstance(compression_response.content, str):
                compression_response.content = _remove_think_tags(
                    compression_response.content
                )
            elif isinstance(compression_response.content, list):
                for (
                    part_dict
                ) in compression_response.content:  # Iterate directly over parts
                    if isinstance(part_dict, dict) and part_dict.get("type") == "text":
                        text_val = part_dict.get("text")
                        if isinstance(text_val, str):
                            cleaned_text = _remove_think_tags(text_val)
                            part_dict["text"] = cleaned_text
                        elif text_val is None:
                            cleaned_text = _remove_think_tags(
                                None
                            )  # Explicitly pass None
                            part_dict["text"] = cleaned_text  # Assigns None back

            compressed_content_str = str(compression_response.content).strip()

            # Attempt to parse the compressed content as JSON if the original was likely JSON
            # This helps maintain structure if the LLM cooperated.
            if isinstance(result_content, (dict, list)):
                try:
                    parsed_json = json.loads(compressed_content_str)
                    return parsed_json  # Return parsed JSON if successful
                except json.JSONDecodeError:
                    if self.display_manager:
                        self.display_manager.warning(
                            f"Compressed content for '{tool_name}' was not valid JSON, returning as string summary."
                        )
                    # Fall through to return the string if JSON parsing fails

            return compressed_content_str  # Return as string summary

        except Exception as e:
            if self.display_manager:
                self.display_manager.error(
                    f"Error during LLM call for tool result compression ('{tool_name}'): {e}"
                )
            # Return a structured error message instead of the original large content
            return {
                "error": f"Failed to compress tool result via LLM: {e}",
                "compression_failed": True,
                "original_content_preview": original_content_str[:500]
                + "...",  # Include a preview
            }
