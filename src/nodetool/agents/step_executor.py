"""
🧠 Step Execution Context: Orchestrating Focused Task Execution

This module provides the `StepExecutor` class, the dedicated engine for executing
a single, isolated step within a larger agentic workflow. It manages the state,
communication, and tool usage necessary to fulfill the step's objective, ensuring
each step operates independently but contributes to the overall task goal.

Core Components:
---------------
*   `StepExecutor`: The central class managing the lifecycle of a step.
    It maintains an isolated message history, manages available tools, monitors
    resource limits (tokens, iterations), and drives the interaction with the
    chosen language model (LLM).
*   Structured completion responses: Instead of calling finish tools, the LLM ends
    each step by emitting a compact JSON block with `"status": "completed"`
    and a `result` object that matches the declared output schema. The result is
    stored directly in the processing context and made available to downstream tasks.
*   Helper Functions (`json_schema_for_output_type`, etc.):
    Utilities for determining output types, generating appropriate JSON schemas for
    tools, and handling object operations.

Execution Algorithm:
--------------------
1.  **Initialization**: A `StepExecutor` is created for a specific `Step`,
    equipped with the necessary `ProcessingContext`, `ChatProvider`, tools,
    model, and resource limits.
    A system prompt tailored to the step (execution vs. final aggregation) is
    generated. The step's internal message history is initialized with this
    system prompt.

2.  **LLM-Driven Execution Loop**: The context enters an iterative loop driven by
    interactions with the LLM. The loop continues as long as the step is not
    completed:
    a.  **Iteration & Limit Checks (Pre-LLM Call)**:
        - Token count of the history is checked. If it exceeds `max_token_limit`
          and not already in conclusion stage, the context transitions to
          `in_conclusion_stage` (see step 2.e), which disables tool calls and
          asks the LLM to conclude.
    b.  **Prepare Prompt & LLM Call**: The overall task description, current step
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
        ii.  Internally, after being yielded, these LLM-generated tool calls are
             processed by `_handle_tool_call`. This method orchestrates:
             - **Execution**: Invokes the tool's logic (`_process_tool_execution`).
             - **Binary Artifact Handling**: Saves base64 encoded binary data
               (images, audio) from the result to workspace files (in an 'artifacts'
               folder) and updates the result to point to these files
               (`_handle_binary_artifact`).
             - **Special Side-Effects**: Manages tool-specific actions
               (`_process_special_tool_side_effects`). For `browser` navigation,
               it logs sources.
             - **Serialization**: Converts the processed tool result to a JSON
               string (`_serialize_tool_result_for_history`).
        iii. The serialized JSON string (tool output) is then added to the step's
             internal message history as a `Message` with role 'tool'.
    d.  **Continuation**: The loop continues to the next LLM interaction unless the
        step has emitted a completion JSON.
    e.  **Conclusion Stage**: If the token limit is exceeded, the context enters a
        "conclusion stage". Tool calls are disabled and the LLM is instructed to
        synthesize results and conclude the step by providing the completion JSON.

3.  **Object Storage**: When a completion JSON is accepted (or when forced
    completion occurs), the result object is stored directly in the processing
    context using the step ID (and the task ID if this is the final aggregation
    step). This makes the result available to downstream tasks via direct
    context access.

4.  **Completion**: Once a valid completion JSON is captured (or synthesized as a
    fallback), the step is marked `completed`, `StepResult` / `TaskUpdate`
    events are emitted, and the execution loop terminates.

Key Data Structures:
--------------------
*   `Task`: Represents the overall goal.
*   `Step`: Defines a single step, including its objective (`content`), input task IDs,
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
| Init StepExecutor   |
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
import json
import logging
import re
import time
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

import tiktoken
from jinja2 import BaseLoader, Environment
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from nodetool.agents.base_agent import DEFAULT_TOKEN_LIMIT
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.finish_step_tool import FinishStepTool
from nodetool.chat.token_counter import (
    count_message_tokens,
    count_messages_tokens,
)
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Step, Task, ToolCall
from nodetool.providers import BaseProvider
from nodetool.ui.console import AgentConsole
from nodetool.utils.message_parsing import (
    extract_json_from_message,
    remove_think_tags,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, LogUpdate, StepResult, TaskUpdate, TaskUpdateEvent

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


DEFAULT_MAX_TOKEN_LIMIT: int = 4096
MESSAGE_COMPRESSION_THRESHOLD: int = 4096
MAX_TOOL_RESULT_CHARS: int = 20000
VALID_COMPLETION_STATUSES: set[str] = {"completed", "complete", "done", "success"}
JSON_FAILURE_ALERT_THRESHOLD: int = 3
MAX_JSON_PARSE_FAILURES: int = 6

DEFAULT_EXECUTION_SYSTEM_PROMPT: str = """
# Role
You are executing EXACTLY one step within a larger plan. Complete this step end-to-end.

# Objective
{{ step_content }}

# Hard Constraint: No Human Feedback
- Do NOT ask clarifying questions or request user input.
- If something is ambiguous or missing, choose the simplest reasonable assumption and proceed.

# Scope & Discipline
- Do ONLY what is required to satisfy this step objective; avoid tangents and extra work.
- Use upstream step results already present in context; do not ask for them again.
- Never invent fields: the final result must match the schema exactly (no extra keys; include all required keys).

Output style:
- Keep non-tool messages concise (≤2 sentences).
- Do not reveal chain-of-thought or internal reasoning traces.

# Output Schema
- The final `result` object MUST match this schema:
```json
{{ output_schema_json }}
```

# Tool Use
- Use tools only when they materially improve correctness or are required.
- Avoid exploratory or repeated tool calls that are unlikely to change the outcome.
- Progress updates: only when you start a new major phase or your plan changes; ≤1 sentence.

# Completion (Tool Call Only)
- When the step is complete, CALL `finish_step` exactly once with:
  {"result": <result>}
- Do NOT output the final result in assistant text.
- Stop immediately after calling `finish_step`.
"""

DEFAULT_FINISH_TASK_SYSTEM_PROMPT: str = """
# Role
You are completing the final aggregation task, synthesizing results from prior steps into a single deliverable.

# Hard Constraint: No Human Feedback
- Do NOT ask clarifying questions or request user input.
- If something is ambiguous or missing, choose the simplest reasonable assumption and proceed.

# Scope & Discipline
- Focus on synthesis and aggregation only (do not do additional research).
- Use upstream step results already present in context; do not ask for them again.
- Never invent fields: the final result must match the schema exactly (no extra keys; include all required keys).

Output style:
- Keep non-tool messages concise (≤2 sentences).
- Do not reveal chain-of-thought or internal reasoning traces.

# Output Schema
- The final deliverable must match this schema:
```json
{{ output_schema_json }}
```

# Tool Use
- Use tools only when they materially improve correctness or are required.
- Avoid exploratory or repeated tool calls that are unlikely to change the outcome.
- Progress updates: only when you start a new major phase or your plan changes; ≤1 sentence.

# Completion (Tool Call Only)
- When aggregation is complete, CALL `finish_step` exactly once with:
  {"result": <result>}
- Do NOT output the final result in assistant text.
- Stop immediately after calling `finish_step`.
"""

DEFAULT_UNSTRUCTURED_SYSTEM_PROMPT: str = """
# Role
You are executing a task. Your job is to complete it end-to-end.

# Objective
{{ step_content }}

# Hard Constraint: No Human Feedback
- Do NOT ask clarifying questions or request user input.
- If something is ambiguous or missing, choose the simplest reasonable assumption and proceed.

# Operating Mode
- Use tools as needed to achieve the objective.
- When you have the final answer or have completed the task, provide the result as your final response.

# Tool Usage Guidelines
## Communication Pattern (Tool Preambles)
Before making tool calls, provide clear progress updates:
1. First assistant message: Restate the objective in one sentence, then list a short numbered plan (1-3 steps).
2. Before each tool call: Emit a one-sentence message describing what you're doing and why.
3. After tool results: Provide a brief update only if the result changes your plan.
"""


def _validate_and_sanitize_schema(schema: Any, default_description: str = "Result object") -> Dict[str, Any]:
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
        raise ValueError(f"Schema contains non-serializable data: {e}") from e

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
        return schema_type == "object" or (schema_type is None and obj.get("properties") is not None)

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
            f"Schema serialization failed after cleaning: {e}, using default object schema",
            stacklevel=2,
        )
        return {"type": "object", "description": default_description}

    return result_schema


def _remove_think_tags(text: str) -> str:
    """Remove think tags from the text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class StepExecutor:
    """
    🧠 The Task-Specific Brain - Isolated execution environment for a single step

    This class maintains a completely isolated context for each step, with its own:
    - Message history: Tracks all interactions in this specific step
    - System prompt: Automatically selected based on step type (reasoning vs. standard)
    - Tools: Available for information gathering and task completion
    - Token tracking: Monitors context size with automatic summarization when needed

    Each step operates like a dedicated worker with exactly the right skills and
    information for that specific job, without interference from other tasks.

    Key Features:
    - Token limit monitoring with automatic context summarization when exceeding thresholds
    - Two-stage execution model: tool calling stage → conclusion stage
    - Safety limits: iteration tracking and strict token budget controls
    - Explicit reasoning capabilities for "thinking" steps
    - Progress reporting throughout execution
    """

    task: Task
    step: Step
    processing_context: ProcessingContext
    model: str
    provider: BaseProvider
    max_token_limit: int
    use_finish_task: bool
    jinja_env: Environment
    system_prompt: str
    tools: Sequence[Tool]
    history: List[Message]
    iterations: int
    sources: List[str]
    progress: List[Any]
    encoding: tiktoken.Encoding
    _chunk_buffer: str
    _is_buffering_chunks: bool
    in_conclusion_stage: bool
    _output_result: Any
    _finish_step_tool: FinishStepTool | None
    input_tokens_total: int
    output_tokens_total: int

    def __init__(
        self,
        task: Task,
        step: Step,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        model: str,
        provider: BaseProvider,
        system_prompt: Optional[str] = None,
        use_finish_task: bool = False,
        max_token_limit: int | None = None,
        max_iterations: int | None = None,
        display_manager: Optional[AgentConsole] = None,
    ):
        """
        Initialize a step execution context.

        Args:
            task (Task): The task to execute
            step (Step): The step to execute
            processing_context (ProcessingContext): The processing context
            system_prompt (str): The system prompt for this step
            tools (List[Tool]): Tools available to this step
            model (str): The model to use for this step
            provider (ChatProvider): The provider to use for this step
            use_finish_task (bool): Whether this step produces the final aggregated task result
            max_token_limit (int): Maximum token limit before summarization
            max_iterations (int): Deprecated; token budget now controls termination
            display_manager (Optional[AgentConsole]): Console for beautiful terminal output
        """
        self.task = task
        self.step = step
        self.processing_context = processing_context
        self.model = model
        self.provider = provider
        self.max_token_limit = max_token_limit or DEFAULT_TOKEN_LIMIT
        self.use_finish_task = use_finish_task
        self.message_compression_threshold = max(
            self.max_token_limit // 4,
            MESSAGE_COMPRESSION_THRESHOLD,
        )
        self._output_result = None
        self.json_parse_failures = 0
        self.generation_failures = 0
        self.display_manager = display_manager
        self.display_manager = display_manager
        self.result_schema = self._load_result_schema()

        # Initialize token usage tracking
        self.input_tokens_total = 0
        self.output_tokens_total = 0

        # Add finish_step tool with this step's output schema for reliable completion
        self._finish_step_tool = None
        if self.result_schema:
            self._finish_step_tool = FinishStepTool(self.result_schema)
            self.tools = [*list(tools), self._finish_step_tool]
        else:
            self.tools = tools
        self._available_tool_names = [tool.name for tool in self.tools]

        # --- Prepare prompt templates ---
        self.jinja_env = Environment(loader=BaseLoader())

        if self.result_schema:
            schema_json = json.dumps(self.result_schema, indent=2, ensure_ascii=False)
            if use_finish_task:
                base_system_prompt = system_prompt or DEFAULT_FINISH_TASK_SYSTEM_PROMPT
                prompt_context = {
                    "output_schema_json": schema_json,
                }
            else:  # Standard execution step
                base_system_prompt = system_prompt or DEFAULT_EXECUTION_SYSTEM_PROMPT
                prompt_context = {
                    "step_content": self.step.instructions,
                    "output_schema_json": schema_json,
                }
        else:
            base_system_prompt = system_prompt or DEFAULT_UNSTRUCTURED_SYSTEM_PROMPT
            prompt_context = {
                "step_content": self.step.instructions,
            }

        self.system_prompt = self._render_prompt(base_system_prompt, prompt_context)
        self.system_prompt = self.system_prompt + "\n\nToday's date is " + datetime.datetime.now().strftime("%Y-%m-%d")

        # Initialize isolated message history for this step
        self.history = [Message(role="system", content=self.system_prompt)]

        # Track iterations for this step
        self.iterations = 0
        if max_iterations is not None:
            log.debug(
                "StepExecutor received deprecated max_iterations=%s; token budget now governs execution.",
                max_iterations,
            )

        # Track sources for data lineage
        self.sources = []

        # Track progress for this step
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

    def _log_initial_state(self) -> None:
        """Emit a single multiline debug entry describing the initial context."""

        summary_lines = [
            "StepExecutor initialized:",
            f"  step_id: {self.step.id} (final={self.use_finish_task})",
            f"  task_title: {self.task.title}",
            f"  instructions: {self.step.instructions}",
            f"  output_schema: {self.step.output_schema}",
            f"  model: {self.model} ({self.provider.__class__.__name__})",
            f"  max_tokens: {self.max_token_limit}",
            f"  tools: {', '.join(self._available_tool_names) if self._available_tool_names else 'none'}",
        ]
        log.debug("\n".join(summary_lines))

    async def _summarize_messages(self, messages: list[Message]) -> str:
        """Summarize older messages into a concise, factual summary."""

        joined = "\n".join(f"{m.role.upper()}: {m.content}" for m in messages if m.content)
        prompt = (
            "Summarize the following conversation concisely while preserving key facts, "
            "decisions, and results:\n\n" + joined
        )
        try:
            msg = await self.provider.generate_message(
                messages=[
                    Message(role="system", content="Summarize previous context."),
                    Message(role="user", content=prompt),
                ],
                model=self.model,
                tools=[],
                max_tokens=512,
            )
            return str(msg.content).strip()
        except Exception as e:  # pragma: no cover - best effort
            log.warning(f"Failed to summarize history: {e}")
            return "Summary unavailable due to compression error."

    async def _trim_history_if_needed(self) -> None:
        """Trim or summarize older messages to stay within token limits."""

        token_count = self._count_tokens(self.history)
        if token_count < self.max_token_limit * 0.9:
            return

        log.debug("Trimming history (tokens=%d/%d)", token_count, self.max_token_limit)

        preserved: list[Message] = []
        for msg in reversed(self.history):
            preserved.insert(0, msg)
            if len(preserved) >= 6:
                break

        earlier_count = len(self.history) - len(preserved)
        earlier_context = self.history[1:earlier_count] if earlier_count > 1 else []  # exclude initial system prompt

        if earlier_context:
            summary = await self._summarize_messages(earlier_context)
            system_prompt = self.history[0] if self.history else None
            self.history = []
            if system_prompt:
                self.history.append(system_prompt)
            self.history.append(
                Message(
                    role="system",
                    content=f"Summary of previous context:\n{summary}",
                )
            )
        else:
            self.history = self.history[:1]

        self.history.extend(preserved)
        current_tokens = self._count_tokens(self.history)
        trimmed_messages = 0
        while current_tokens > self.max_token_limit * 0.85 and len(self.history) > 2:
            removed = self.history.pop(2)
            trimmed_messages += 1
            current_tokens = self._count_tokens(self.history)
            log.debug(
                "Dropped older message (%s) to control history size",
                removed.role,
            )

        if trimmed_messages:
            log.warning(
                "Removed %d additional history entries to stay within context",
                trimmed_messages,
            )

        log.info("History trimmed at iteration %d. Token count reset.", self.iterations)

    def _load_result_schema(self) -> Optional[Dict[str, Any]]:
        """Parse and sanitize the declared output schema for this step."""

        if not self.step.output_schema:
            return None

        default_description = "The task result" if self.use_finish_task else "The step result"
        raw_schema: Any = self.step.output_schema

        try:
            return _validate_and_sanitize_schema(raw_schema, default_description)
        except Exception as exc:  # pragma: no cover - defensive fallback
            log.warning(
                "Failed to parse output_schema for step %s: %s. Using fallback string schema.",
                self.step.id,
                exc,
            )
            return {
                "type": "string",
                "description": default_description,
            }

    def _validate_result_payload(self, result_payload: Any) -> tuple[bool, Optional[str], Any]:
        """Validate the provided result payload against the declared schema."""

        normalized_result = self._normalize_tool_result(result_payload)

        try:
            jsonschema_validate(normalized_result, self.result_schema)
            return True, None, normalized_result
        except ValidationError as exc:
            return False, exc.message, normalized_result
        except Exception as exc:  # pragma: no cover - unexpected errors
            return False, str(exc), normalized_result

    def _store_completion_result(self, normalized_result: Any) -> None:
        """Persist the final result and mark the step as completed."""

        self.step.completed = True
        self.step.end_time = int(time.time())
        self.processing_context.store_step_result(self.step.id, normalized_result)
        if self.use_finish_task:
            self.processing_context.store_step_result(self.task.id, normalized_result)
        self._output_result = normalized_result

    def _append_completion_feedback(self, detail: str, submitted_result: Any | None = None) -> None:
        """Append a system message instructing the LLM to complete via finish_step."""

        schema_str = json.dumps(self.result_schema, indent=2, ensure_ascii=False)
        message_lines = [
            "SYSTEM: Step completion must be signaled via the `finish_step` tool.",
            f"Detail: {detail}",
            "Call `finish_step` exactly once with:",
            '{"result": <result>}',
            "Schema for `result`:",
            schema_str,
        ]

        if submitted_result is not None:
            try:
                preview = json.dumps(
                    self._normalize_tool_result(submitted_result),
                    indent=2,
                    ensure_ascii=False,
                )
            except Exception:  # pragma: no cover
                preview = str(submitted_result)
            message_lines.extend(
                [
                    "Previous submission preview:",
                    preview,
                ]
            )

        self.history.append(Message(role="system", content="\n".join(message_lines)))

    def _maybe_finalize_from_message(self, message: Optional[Message]) -> tuple[bool, Optional[Any]]:
        """Attempt to parse and store a completion payload from the assistant message."""

        if not message:
            return False, None

        if self.result_schema is None:
            if not message.tool_calls:
                return True, message.content
            return False, None

        parsed = extract_json_from_message(message)
        if not isinstance(parsed, dict):
            return False, None

        status = parsed.get("status")
        if status is not None and status != "completed":
            return False, None

        if status == "completed" and "result" not in parsed:
            self.history.append(
                Message(
                    role="system",
                    content="Missing 'result' in completion payload. Provide: "
                    '{"status": "completed", "result": <your_result>}.',
                )
            )
            return False, None

        candidate_result = parsed.get("result") if "result" in parsed else parsed
        is_valid, error_detail, normalized_result = self._validate_result_payload(candidate_result)
        if not is_valid or normalized_result is None:
            self.history.append(
                Message(
                    role="system",
                    content=f"Schema validation failed: {error_detail or 'unknown error'}",
                )
            )
            return False, None

        self._store_completion_result(normalized_result)
        return True, normalized_result

    def _emit_completion_events(self, normalized_result: Any):
        """Yield completion updates and a StepResult message."""

        yield TaskUpdate(
            task=self.task,
            step=self.step,
            event=TaskUpdateEvent.STEP_COMPLETED,
        )
        yield StepResult(
            step=self.step,
            result=normalized_result,
            is_task_result=self.use_finish_task,
        )

    def _register_json_failure(self, detail: str) -> None:
        """Track JSON parsing/validation failures and enforce a hard stop."""
        self.json_parse_failures += 1
        log.warning(
            "Subtask %s JSON parse/validation failure %d/%d: %s",
            self.step.id,
            self.json_parse_failures,
            MAX_JSON_PARSE_FAILURES,
            detail,
        )

        if self.json_parse_failures == JSON_FAILURE_ALERT_THRESHOLD:
            # Push a clear system nudge and force conclusion mode
            reminder = (
                "SYSTEM: Do NOT output completion JSON in assistant text. You MUST call "
                "`finish_step` with {'result': <result>} matching the schema, with no extra keys."
            )
            self.history.append(Message(role="system", content=reminder))
            self.in_conclusion_stage = True

        if self.json_parse_failures >= MAX_JSON_PARSE_FAILURES:
            raise ValueError(
                f"Exceeded maximum JSON parse attempts ({MAX_JSON_PARSE_FAILURES}) for step {self.step.id}."
            )

    def get_result(self) -> Any | None:
        """
        Returns the stored result object.
        """
        return self._output_result

    async def execute(
        self,
    ) -> AsyncGenerator[Chunk | ToolCall | TaskUpdate | StepResult, None]:
        """
        Runs a single step to completion using the LLM-driven execution loop.

        Yields:
            Union[Chunk, ToolCall, TaskUpdate, StepResult]: Live updates during task execution
        """
        # Record the start time of the step
        current_time = int(time.time())
        self.step.start_time = current_time

        # Ensure step logs are initialized
        if not hasattr(self.step, "logs") or self.step.logs is None:
            self.step.logs = []

        # Set the current step for the display manager
        if self.display_manager:
            self.display_manager.set_current_step(self.step)

        self._log_initial_state()

        # Display beautiful step start panel
        if self.display_manager:
            self.display_manager.display_step_start(self.step)

        # --- LLM-based Execution Logic ---
        prompt_parts = [
            self.step.instructions,
        ]
        if self.step.depends_on:
            # Fetch and inject input results directly into the prompt
            for input_task_id in self.step.depends_on:
                result = self.processing_context.load_step_result(input_task_id)
                if result is not None:
                    prompt_parts.append(
                        f"**Result from Task {input_task_id}:**\n{json.dumps(result, indent=2, ensure_ascii=False)}\n"
                    )

        prompt_parts.append(
            "Please perform the step based on the provided context, instructions, and upstream task results."
        )
        task_prompt = "\n".join(prompt_parts)

        # Add the task prompt to this step's history
        self.history.append(Message(role="user", content=task_prompt))

        # Yield task update for step start
        yield TaskUpdate(
            task=self.task,
            step=self.step,
            event=TaskUpdateEvent.STEP_STARTED,
        )

        # Display task update event
        if self.display_manager:
            self.display_manager.display_task_update("SUBTASK_STARTED", self.step.instructions)

        # Continue executing until the task is completed (token budget enforces termination)
        while not self.step.completed:
            self.iterations += 1
            token_count = self._count_tokens(self.history)
            log.debug(
                "%s | iteration %d | history=%d msgs | tokens=%d/%d",
                self.step.id,
                self.iterations,
                len(self.history),
                token_count,
                self.max_token_limit,
            )

            # Display beautiful iteration status
            # self.display_manager.display_iteration_status(
            #     self.iterations, token_count, self.max_token_limit
            # )

            # Check if we need to transition to conclusion stage
            if (token_count > self.max_token_limit) and not self.in_conclusion_stage:
                # Log and display token warning
                log.warning(f"Token usage: {token_count}/{self.max_token_limit}")
                await self._transition_to_conclusion_stage()
                # Yield the event after transitioning
                yield TaskUpdate(
                    task=self.task,
                    step=self.step,
                    event=TaskUpdateEvent.ENTERED_CONCLUSION_STAGE,
                )
                if self.display_manager:
                    self.display_manager.display_task_update("ENTERED_CONCLUSION_STAGE")

            # Process current iteration
            yield LogUpdate(
                node_id=self.step.id,
                node_name=f"Step: {self.step.id}",
                content="Generating next steps..." if not self.in_conclusion_stage else "Synthesizing final answer...",
                severity="info",
            )
            message = None
            async for update in self._process_iteration():
                if isinstance(update, Message):
                    message = update
                else:
                    yield update

            if message is None:
                log.error("Iteration failed to produce a message")
                break

            # If there was any final content that wasn't streamed (e.g. buffered), yield it now
            if message.content and not message.tool_calls:
                # This is a fallback in case streaming didn't catch everything or provider buffered
                # Note: if streaming worked, this will redundant but Chunk(done=False) is fast.
                pass

            message.tool_calls = self._filter_tool_calls_for_current_stage(message.tool_calls)

            # Add assistant message to history after any tool-call adjustments so the
            # stored conversation precisely matches what we'll execute/respond to.
            self.history.append(message)

            if message.tool_calls:
                # Check for finish_step tool call - handle specially for completion
                finish_step_call = next((tc for tc in message.tool_calls if tc.name == "finish_step"), None)

                if finish_step_call:
                    # Handle finish_step tool for step completion
                    tool_message = self._generate_tool_call_message(finish_step_call)
                    yield ToolCall(
                        id=finish_step_call.id,
                        name=finish_step_call.name,
                        args=finish_step_call.args,
                        step_id=self.step.id,
                        message=tool_message,
                    )

                    # Extract and validate result from tool call args
                    result_payload = (
                        finish_step_call.args.get("result") if isinstance(finish_step_call.args, dict) else None
                    )
                    if result_payload is not None:
                        is_valid, error_detail, normalized_result = self._validate_result_payload(result_payload)
                        if is_valid and normalized_result is not None:
                            # Add tool result to history
                            self.history.append(
                                Message(
                                    role="tool",
                                    tool_call_id=finish_step_call.id,
                                    name="finish_step",
                                    content='{"status": "completed"}',
                                )
                            )
                            self._store_completion_result(normalized_result)
                            log.debug(
                                f"StepExecutor: {self.step.id} completed via tool. use_finish_task={self.use_finish_task}"
                            )
                            for event in self._emit_completion_events(normalized_result):
                                if isinstance(event, StepResult):
                                    log.debug(
                                        f"StepExecutor: Yielding tool StepResult. is_task_result={event.is_task_result}"
                                    )
                                yield event
                            break
                        else:
                            # Invalid result - add feedback and continue loop
                            log.warning(f"finish_step result validation failed: {error_detail}")
                            self.history.append(
                                Message(
                                    role="tool",
                                    tool_call_id=finish_step_call.id,
                                    name="finish_step",
                                    content=f'{{"error": "Result validation failed: {error_detail}"}}',
                                )
                            )
                            self._append_completion_feedback(
                                error_detail or "Result failed schema validation.", result_payload
                            )
                    else:
                        log.warning("finish_step called without result")
                        self.history.append(
                            Message(
                                role="tool",
                                tool_call_id=finish_step_call.id,
                                name="finish_step",
                                content='{"error": "Missing result in finish_step call"}',
                            )
                        )
                else:
                    # Process non-finish_step tool calls normally
                    for tool_call in message.tool_calls:
                        tool_message = self._generate_tool_call_message(tool_call)
                        yield ToolCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            args=tool_call.args,
                            step_id=self.step.id,
                            message=tool_message,
                        )

                    # Provide immediate feedback that execution has started
                    tool_names_str = ", ".join(tc.name for tc in message.tool_calls)
                    yield LogUpdate(
                        node_id=self.step.id,
                        node_name=f"Step: {self.step.id}",
                        content=f"Executing tools: {tool_names_str}...",
                        severity="info",
                    )

                    await self._process_tool_call_results(message.tool_calls)

                    yield LogUpdate(
                        node_id=self.step.id,
                        node_name=f"Step: {self.step.id}",
                        content=f"Completed tool execution: {tool_names_str}.",
                        severity="info",
                    )
            elif message.content:
                # Text chunks already yielded during _process_iteration
                pass

            completed, normalized_result = self._maybe_finalize_from_message(message)
            if completed and normalized_result is not None:
                self._store_completion_result(normalized_result)
                log.debug(f"StepExecutor: {self.step.id} completed via message. use_finish_task={self.use_finish_task}")
                for event in self._emit_completion_events(normalized_result):
                    if isinstance(event, StepResult):
                        log.debug(f"StepExecutor: Yielding message StepResult. is_task_result={event.is_task_result}")
                    yield event
                break

        # Display completion event
        if self.display_manager:
            self.display_manager.display_completion_event(self.step, self.step.completed, self._output_result)

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
        4. Restricts available tools to finish_step so the LLM can still complete reliably
        """
        self.in_conclusion_stage = True
        if self._finish_step_tool:
            transition_message = f"""
            SYSTEM: The conversation history is approaching the token limit ({self.max_token_limit} tokens).
            ENTERING CONCLUSION STAGE: You MUST now synthesize all gathered information and finalize the step.
            Only the `finish_step` tool is available. Call `finish_step` exactly once with:
            {{"result": <result>}} where <result> matches the declared schema.
            """
        else:
            transition_message = f"""
            SYSTEM: The conversation history is approaching the token limit ({self.max_token_limit} tokens).
            ENTERING CONCLUSION STAGE: You MUST now synthesize all gathered information and finalize the step.
            Tools are not available. Provide the final answer concisely.
            """
        # Check if the message already exists to prevent duplicates if called multiple times
        if not any(m.role == "system" and "ENTERING CONCLUSION STAGE" in str(m.content) for m in self.history):
            self.history.append(Message(role="system", content=transition_message))
            if self.display_manager:
                self.display_manager.display_conclusion_stage()

    async def _process_iteration(
        self,
    ) -> AsyncGenerator[Chunk | Message, None]:
        """
        Process a single iteration of the task.
        """
        await self._trim_history_if_needed()
        if self.in_conclusion_stage:
            tools_for_iteration = [self._finish_step_tool] if self._finish_step_tool else []
        else:
            tools_for_iteration = list(self.tools)

        # Create a dictionary to track unique tools by name
        unique_tools = {tool.name: tool for tool in tools_for_iteration}
        final_tools = list(unique_tools.values())

        try:
            input_tokens_now = self._count_tokens(self.history)
            self.input_tokens_total += input_tokens_now
        except Exception as e:
            log.warning(f"Failed to count input tokens: {e}")

        message = None
        try:
            content = ""
            tool_calls = []
            async for chunk in self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=final_tools,
            ):
                if isinstance(chunk, Chunk):
                    if chunk.content:
                        content += chunk.content
                    yield chunk
                elif isinstance(chunk, ToolCall):
                    tool_calls.append(chunk)

            message = Message(
                role="assistant",
                content=content if content else None,
                tool_calls=tool_calls if tool_calls else None,
            )

        except Exception as e:
            self.generation_failures += 1
            log.error(f"Failed to generate message: {e}")
            if isinstance(e, IndexError) or self.generation_failures >= 3:
                raise
            message = Message(
                role="assistant",
                content=f"Error generating message: {e}",
            )

        if message.tool_calls:
            for tool_call in message.tool_calls:
                log.debug(f"{self.step.id} | {tool_call.name}: {tool_call.args}")

        # Clean message content from think tags to save tokens for local models
        if isinstance(message.content, str):
            message.content = remove_think_tags(message.content)
        elif isinstance(message.content, list):
            for part_dict in message.content:  # Iterate directly over parts
                if isinstance(part_dict, dict) and part_dict.get("type") == "text":
                    text_val = part_dict.get("text")
                    if isinstance(text_val, str):
                        cleaned_text = remove_think_tags(text_val)
                        part_dict["text"] = cleaned_text
                    elif text_val is None:
                        cleaned_text = remove_think_tags(None)  # Explicitly pass None
                        part_dict["text"] = cleaned_text  # Assigns None back

        try:
            self.output_tokens_total += self._count_single_message_tokens(message)
        except Exception as e:
            log.warning(f"Failed to count output tokens: {e}")

        yield message

    def _filter_tool_calls_for_current_stage(self, tool_calls: Optional[list[ToolCall]]) -> list[ToolCall]:
        """Filter tool calls based on whether we're in the conclusion stage."""
        if not tool_calls:
            return []

        if not self.in_conclusion_stage:
            return list(tool_calls)

        allowed = [tool_call for tool_call in tool_calls if tool_call.name == "finish_step"]
        ignored = [tool_call for tool_call in tool_calls if tool_call.name != "finish_step"]
        if ignored:
            log.warning(
                "LLM attempted to call non-finish_step tools during conclusion stage; ignoring %d tool call(s).",
                len(ignored),
            )
        return allowed

    async def _process_tool_call_results(self, valid_tool_calls: list[ToolCall]) -> None:
        """Execute tool calls and add their results to history."""
        if not valid_tool_calls:
            return

        tool_results = await asyncio.gather(
            *[self._handle_tool_call(tool_call) for tool_call in valid_tool_calls],
            return_exceptions=True,
        )

        valid_tool_messages: list[Message] = []
        for tool_call, result in zip(valid_tool_calls, tool_results, strict=True):
            if isinstance(result, Exception):
                log.error(f"Tool call {tool_call.id} ({tool_call.name}) failed with exception: {result}")
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
                log.error(f"Tool call {tool_call.id} ({tool_call.name}) returned unexpected type: {type(result)}")
                error_message = Message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Unexpected tool result type: {type(result)}",
                )
                valid_tool_messages.append(error_message)

        self.history.extend(valid_tool_messages)

    def _generate_tool_call_message(self, tool_call: ToolCall) -> str:
        """
        Generate a message object from a tool call.
        """
        for tool in self.tools:
            if tool.name == tool_call.name:
                return tool.user_message(tool_call.args)

        raise ValueError(f"Tool '{tool_call.name}' not found in available tools.")

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
            self.display_manager.debug_step_only(f"Handling tool call: {tool_call.name} (ID: {tool_call.id})")

        # 1. Execute the tool
        tool_result = await self._process_tool_execution(tool_call)

        if self.display_manager:
            self.display_manager.debug_step_only(f"Tool {tool_call.name} execution completed")

        # 3. Handle binary artifacts (images, audio)
        if isinstance(tool_result, dict):
            if "image" in tool_result:
                tool_result = self._handle_binary_artifact(tool_result, tool_call.name, "image")
            elif "audio" in tool_result:
                tool_result = self._handle_binary_artifact(tool_result, tool_call.name, "audio")

        # 4. Process special tool side-effects (e.g., browser navigation)
        self._process_special_tool_side_effects(tool_result, tool_call)

        # Log tool result only to step tree (not phase)
        if self.display_manager:
            self.display_manager.info_step_only(f"Tool result received from {tool_call.name}")

        # 5. Serialize the final processed result for history
        content_str = self._serialize_tool_result_for_history(tool_result, tool_call.name)

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
            artifact_abs_path = self.processing_context.resolve_workspace_path(artifact_rel_path)

            # Ensure artifacts directory exists (No longer needed for root saving, parent is workspace_dir)
            # os.makedirs(os.path.dirname(artifact_abs_path), exist_ok=True)

            # Decode and write the artifact
            decoded_data = base64.b64decode(base64_data)
            with open(artifact_abs_path, "wb") as artifact_file:
                artifact_file.write(decoded_data)

            if self.display_manager:
                self.display_manager.info_step_only(
                    f"Saved base64 {artifact_type} from tool '{tool_call_name}' to {artifact_rel_path}"
                )

            # Update result: add path, remove original base64 data
            tool_result[f"{artifact_type}_path"] = artifact_rel_path
            del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]  # Clean up format key for audio

        except (binascii.Error, ValueError) as e:
            if self.display_manager:
                self.display_manager.error(f"Failed to decode base64 {artifact_type} from tool '{tool_call_name}': {e}")
            tool_result[f"{artifact_type}_path"] = f"Error decoding {artifact_type}: {e}"
            tool_result.pop(artifact_key, None)
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]
        except Exception as e:
            if self.display_manager:
                self.display_manager.error(f"Failed to save {artifact_type} artifact from tool '{tool_call_name}': {e}")
            tool_result[f"{artifact_type}_path"] = f"Error saving {artifact_type}: {e}"
            tool_result.pop(artifact_key, None)
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]
        return tool_result

    def _process_special_tool_side_effects(self, tool_result: Any, tool_call: ToolCall):
        """Handles side effects for specific tools, like 'browser' or 'finish_*'."""

        if tool_call.name == "browser" and isinstance(tool_call.args, dict):
            url = tool_call.args.get("url", "")
            if url and url not in self.sources:  # Avoid duplicates
                self.sources.append(url)

    def _serialize_tool_result_for_history(self, tool_result: Any, tool_name: str) -> str:
        """Serializes the tool result to a JSON string for message history."""

        try:
            if tool_result is None:
                return "Tool returned no output."
            normalized = self._normalize_tool_result(tool_result)
            serialized = json.dumps(normalized, ensure_ascii=False)
            if len(serialized) > MAX_TOOL_RESULT_CHARS:
                log.warning(
                    "Truncating tool result (%d chars) for history entry",
                    len(serialized),
                )
                serialized = serialized[:MAX_TOOL_RESULT_CHARS] + "... [truncated to maintain context size]"
            return serialized
        except (TypeError, ValueError) as e:
            log.error(f"Failed to serialize tool result for '{tool_name}' to JSON: {e}. Result: {tool_result}")
            return json.dumps(
                {
                    "error": f"Failed to serialize tool result: {e}",
                    "result_repr": repr(tool_result),
                }
            )

    async def _request_completion_response(self, system_prompt: str) -> Optional[Any]:
        """Ask the LLM to provide the final completion JSON and return the parsed result."""

        self.history.append(Message(role="system", content=system_prompt))

        try:
            input_tokens_now = self._count_tokens(self.history)
            self.input_tokens_total += input_tokens_now
            log.debug(f"Input tokens (forced completion): {input_tokens_now} (cumulative: {self.input_tokens_total})")
        except Exception as exc:  # pragma: no cover
            log.warning(f"Failed to count input tokens (forced completion): {exc}")

        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=[],
        )

        try:
            output_tokens_now = self._count_single_message_tokens(message)
            self.output_tokens_total += output_tokens_now
            log.debug(
                f"Output tokens (forced completion): {output_tokens_now} (cumulative: {self.output_tokens_total})"
            )
        except Exception as exc:  # pragma: no cover
            log.warning(f"Failed to count output tokens (forced completion): {exc}")

        self.history.append(message)
        completed, normalized_result = self._maybe_finalize_from_message(message)
        if completed:
            return normalized_result
        return None

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

    async def _compress_tool_result(self, result_content: Any, tool_name: str, tool_args: dict) -> dict | str:
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
            original_content_str = json.dumps(result_content, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error(f"Failed to serialize result content for compression: {e}")
            return {
                "error": "Failed to serialize content for compression",
                "original_content_preview": repr(result_content)[:500],
            }

        compression_system_prompt = f"""
        # Goal
        Reduce the size of the 'TOOL RESULT TO COMPRESS' by removing duplicate information and summarizing the content.
        Ensure that all information vital to achieving the step's objective ('{self.step.instructions}') is retained.

        # Output Format
        - If the input is JSON, the output should ideally be a valid, smaller JSON object preserving the essential structure and data.
        - If the input is not JSON or cannot be effectively summarized as JSON, provide a concise text summary.

        # Context
        - Tool Name: {tool_name}
        - Tool Arguments: {json.dumps(tool_args, ensure_ascii=False)}
        - Overall Task: {self.task.title} - {self.task.description}
        - Current Subtask: {self.step.instructions}

        Instructions:
        - Focus ONLY on summarizing the 'TOOL RESULT TO COMPRESS' provided below.
        - **Crucially, ensure that all information vital to achieving the step's objective ('{self.step.instructions}') is retained.**
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
                log.debug(f"Input tokens (compression): {input_tokens_now} (cumulative: {self.input_tokens_total})")
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
                output_tokens_now = self._count_single_message_tokens(compression_response)
                self.output_tokens_total += output_tokens_now
                log.debug(f"Output tokens (compression): {output_tokens_now} (cumulative: {self.output_tokens_total})")
            except Exception as e:
                log.warning(f"Failed to count output tokens (compression): {e}")

            # Clean compression_response.content before str() and strip()
            if isinstance(compression_response.content, str):
                compression_response.content = _remove_think_tags(compression_response.content)
            elif isinstance(compression_response.content, list):
                for part_dict in compression_response.content:  # Iterate directly over parts
                    if isinstance(part_dict, dict) and part_dict.get("type") == "text":
                        text_val = part_dict.get("text")
                        if isinstance(text_val, str):
                            cleaned_text = _remove_think_tags(text_val)
                            part_dict["text"] = cleaned_text
                        elif text_val is None:
                            cleaned_text = _remove_think_tags(None)  # Explicitly pass None
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
                self.display_manager.error(f"Error during LLM call for tool result compression ('{tool_name}'): {e}")
            # Return a structured error message instead of the original large content
            return {
                "error": f"Failed to compress tool result via LLM: {e}",
                "compression_failed": True,
                "original_content_preview": original_content_str[:500] + "...",  # Include a preview
            }
