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
import mimetypes
import re
from nodetool.chat.providers import ChatProvider
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, SubTask, Task, ToolCall, LogEntry
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole

import tiktoken
import yaml


import json
import os
import time
import uuid
import warnings
from typing import (
    Any,
    AsyncGenerator,
    List,
    Sequence,
    Union,
    Dict,
    Optional,
    cast,
)

from jinja2 import Environment, BaseLoader


DEFAULT_MAX_TOKEN_LIMIT: int = 4096
DEFAULT_MAX_ITERATIONS: int = 10
MESSAGE_COMPRESSION_THRESHOLD: int = 4096

DEFAULT_EXECUTION_SYSTEM_PROMPT: str = """
You are executing a single subtask from a larger task plan.
YOUR GOAL IS TO PRODUCE THE INTENDED RESULT FOR THIS SUBTASK.

EXECUTION PROTOCOL:
1. Focus exclusively on the current subtask objective: {{ subtask_content }}.
2. Use the provided input data from upstream tasks efficiently:
    - The results from upstream tasks are already provided in your context.
    - Process the provided data directly rather than requesting additional information.
    - Use tools that can work with the data structure directly.
3. Perform the required steps to generate the result.
4. Ensure the final result matches the expected structure defined in the subtask schema.
5. **Tool Call Limit**: You have a maximum of {{ max_tool_calls }} tool calls for this subtask. Use them wisely and efficiently.
6. **Crucially**: Call `finish_subtask` ONCE at the end with the final result.
    - Provide the result directly as an object in the `result` parameter.
    - The result will be stored as an object and made available to downstream tasks.
    - Always include relevant `metadata` (title, description, sources). Sources should cite original inputs and any external sources used.
7. Do NOT call `finish_subtask` multiple times. Structure your work to produce the final output, then call `finish_subtask`.
8. Reasoning privacy: Do not reveal chain-of-thought. Only output tool calls and required fields.
9. Efficiency: Keep text minimal; prefer structured outputs and tool calls.
"""

DEFAULT_FINISH_TASK_SYSTEM_PROMPT: str = """
You are completing the final task and aggregating results from previous subtasks.
The goal is to combine the information from upstream task results into a single, final result according to the overall task objective.

FINISH_TASK PROTOCOL:
1. Use the provided results from previous subtasks that are already included in your context.
2. Analyze the provided results efficiently - extract only the key information needed for aggregation.
3. Synthesize and aggregate the information to create the final task result.
4. Ensure the final result conforms to the required structure defined in the task schema.
5. **Tool Call Limit**: You have a maximum of {{ max_tool_calls }} tool calls for this task. Use them wisely and efficiently.
6. Call `finish_task` ONCE with the complete, aggregated `result` and relevant `metadata` (title, description, sources - citing original sources where possible).
7. Reasoning privacy: Do not reveal chain-of-thought. Only output tool calls and required fields.
8. Efficiency: Keep text minimal; prefer structured outputs and tool calls.
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
    def _clean_schema_recursive(obj: Any) -> Any:
        """Recursively clean schema objects to ensure compatibility."""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                if key == "additionalProperties":
                    cleaned[key] = False
                else:
                    cleaned[key] = _clean_schema_recursive(value)
            return cleaned
        elif isinstance(obj, list):
            return [_clean_schema_recursive(item) for item in obj]
        else:
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
    downstream processes. Include relevant 'metadata' with title, description, and sources.
    This will hold the final aggregated result of all subtasks.
    """
    input_schema: Dict[str, Any]  # Defined in __init__

    def __init__(
        self,
        output_schema: Any,
    ):
        super().__init__()  # Call parent constructor

        # Validate and prepare the result schema
        result_schema = _validate_and_sanitize_schema(output_schema, "The task result")

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
    downstream tasks. Include relevant 'metadata' with title, description, and sources.
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
    provider: ChatProvider
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

    def __init__(
        self,
        task: Task,
        subtask: SubTask,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        model: str,
        provider: ChatProvider,
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
        self.display_manager = display_manager or AgentConsole(verbose=True)

        # Initialize tool call tracking
        self.tool_calls_made = 0
        self.max_tool_calls = subtask.max_tool_calls

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

        self.tools: Sequence[Tool] = list(tools) + [
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
        token_count = 0
        for msg in messages:
            token_count += self._count_single_message_tokens(msg)

        self.display_manager.debug(
            f"Token count for {len(messages)} messages: {token_count}"
        )

        return token_count

    def _count_single_message_tokens(self, msg: Message) -> int:
        """
        Count the number of tokens in a single message.

        Args:
            msg: The message to count tokens for.

        Returns:
            int: The approximate token count for the single message.
        """
        token_count = 0
        # Count tokens in the message content
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                token_count += len(self.encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                # For multimodal content, just count the text parts
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        token_count += len(self.encoding.encode(part.get("text", "")))

        # Count tokens in tool calls if present
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                # Count function name
                token_count += len(self.encoding.encode(tool_call.name))
                # Count arguments
                if isinstance(tool_call.args, dict):
                    token_count += len(self.encoding.encode(json.dumps(tool_call.args)))
                else:
                    token_count += len(self.encoding.encode(str(tool_call.args)))

        # Count tokens in tool results if present (role="tool")
        # Note: Tool results content is often JSON string, handled by the content check above.
        # If tool results were stored differently, add logic here.
        # Example: if msg.role == "tool" and msg.result: token_count += ...

        return token_count

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
        self.display_manager.set_current_subtask(self.subtask)

        # Log initialization messages now that current subtask is set
        for msg in self._init_debug_messages:
            self.display_manager.debug(msg)

        # Display beautiful subtask start panel
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

        self.display_manager.debug(
            f"Task prompt added to history: {task_prompt[:200]}..."
        )

        # Yield task update for subtask start
        yield TaskUpdate(
            task=self.task,
            subtask=self.subtask,
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

        # Display task update event
        self.display_manager.display_task_update(
            "SUBTASK_STARTED", self.subtask.content
        )

        # Continue executing until the task is completed or max iterations reached
        while not self.subtask.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            self.display_manager.debug(
                f"Starting iteration {self.iterations}/{self.max_iterations}"
            )

            # Calculate total token count AFTER potential compression
            token_count = self._count_tokens(self.history)

            # Display beautiful iteration status
            # self.display_manager.display_iteration_status(
            #     self.iterations, self.max_iterations, token_count, self.max_token_limit
            # )

            # Check if we need to transition to conclusion stage
            if (token_count > self.max_token_limit) and not self.in_conclusion_stage:
                # Log and display token warning
                self.display_manager.warning(
                    f"Token usage: {token_count}/{self.max_token_limit}"
                )
                self.display_manager.display_token_warning(
                    token_count, self.max_token_limit
                )
                await self._transition_to_conclusion_stage()
                # Yield the event after transitioning
                yield TaskUpdate(
                    task=self.task,
                    subtask=self.subtask,
                    event=TaskUpdateEvent.ENTERED_CONCLUSION_STAGE,
                )
                self.display_manager.display_task_update("ENTERED_CONCLUSION_STAGE")

            # Process current iteration
            message = await self._process_iteration()
            if message.tool_calls:
                self.display_manager.debug_subtask_only(
                    f"LLM returned {len(message.tool_calls)} tool calls"
                )

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
                            self.display_manager.warning(
                                f"Tool call limit ({self.max_tool_calls}) reached. Only allowing finish tools."
                            )
                        else:
                            # Force completion if no finish tools available
                            self.display_manager.warning(
                                f"Tool call limit ({self.max_tool_calls}) reached. Forcing completion."
                            )
                            tool_call = await self._handle_max_tool_calls_reached()
                            yield tool_call
                            yield TaskUpdate(
                                task=self.task,
                                subtask=self.subtask,
                                event=TaskUpdateEvent.MAX_TOOL_CALLS_REACHED,
                            )
                            self.display_manager.display_task_update(
                                "MAX_TOOL_CALLS_REACHED",
                                f"Tool calls: {self.tool_calls_made}/{self.max_tool_calls}",
                            )
                            break
                    else:
                        # Allow remaining non-finish calls plus all finish calls
                        allowed_other_tools = other_tools[:remaining_calls]
                        message.tool_calls = finish_tools + allowed_other_tools
                        self.display_manager.warning(
                            f"Tool call limit approaching. Processing {len(finish_tools)} finish tools and {len(allowed_other_tools)} of {len(other_tools)} other tool calls."
                        )

                if message.content:
                    yield Chunk(content=str(message.content))
                for tool_call in message.tool_calls:
                    # Log tool execution only to subtask (not phase)
                    if tool_call.name not in ("finish_subtask", "finish_task"):
                        self.display_manager.info_subtask_only(
                            f"Executing tool: {tool_call.name}"
                        )
                        self.tool_calls_made += 1
                    message = self._generate_tool_call_message(tool_call)
                    yield ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        subtask_id=self.subtask.id,
                        message=message,
                    )
                    if (
                        tool_call.name == "finish_subtask"
                        or tool_call.name == "finish_task"
                    ):
                        self.display_manager.debug(
                            f"Subtask completed via {tool_call.name}"
                        )
                        yield TaskUpdate(
                            task=self.task,
                            subtask=self.subtask,
                            event=TaskUpdateEvent.SUBTASK_COMPLETED,
                        )
                        self.display_manager.display_task_update(
                            "SUBTASK_COMPLETED", self.subtask.content
                        )
            # Handle potential text chunk yields if provider supports streaming text
            elif message.content:
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
            self.display_manager.display_task_update(
                "MAX_ITERATIONS_REACHED",
                f"Iterations: {self.iterations}/{self.max_iterations}",
            )
            yield tool_call

        # Display completion event
        self.display_manager.display_completion_event(
            self.subtask, self.subtask.completed, self._output_result
        )
        self.display_manager.debug(f"Total iterations: {self.iterations}")
        self.display_manager.debug(f"Total messages in history: {len(self.history)}")

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
            # Display beautiful conclusion stage transition
            self.display_manager.display_conclusion_stage()

    async def _process_iteration(
        self,
    ) -> Message:
        """
        Process a single iteration of the task.
        """
        self.display_manager.debug("Processing iteration")

        tools_for_iteration = (
            [self.finish_tool]  # Only allow finish tool in conclusion stage
            if self.in_conclusion_stage
            else self.tools  # Allow all tools otherwise
        )

        self.display_manager.debug(f"Conclusion stage: {self.in_conclusion_stage}")
        self.display_manager.debug(
            f"Tools available: {[t.name for t in tools_for_iteration]}"
        )

        # Create a dictionary to track unique tools by name
        unique_tools = {tool.name: tool for tool in tools_for_iteration}
        final_tools = list(unique_tools.values())

        try:
            self.display_manager.debug(
                f"Calling LLM with {len(self.history)} messages in history"
            )
            message = await self.provider.generate_message(
                messages=self.history,
                model=self.model,
                tools=final_tools,
            )
            self.display_manager.debug(
                f"LLM response received - content length: {len(str(message.content)) if message.content else 0}"
            )
            if message.tool_calls:
                self.display_manager.debug(
                    f"LLM requested tool calls: {[tc.name for tc in message.tool_calls]}"
                )
        except Exception as e:
            self.display_manager.error(f"Error generating message: {e}", exc_info=True)
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

        # Add the message to history
        self.history.append(message)

        if message.tool_calls:
            # Check if finish tool was called in conclusion stage, otherwise filter disallowed tools
            valid_tool_calls = []
            if self.in_conclusion_stage:
                for tc in message.tool_calls:
                    if tc.name == self.finish_tool.name:
                        valid_tool_calls.append(tc)
                    else:
                        self.display_manager.warning(
                            f"LLM attempted to call disallowed tool '{tc.name}' in conclusion stage. Ignoring."
                        )
            else:
                valid_tool_calls = (
                    message.tool_calls
                )  # Allow all tools if not in conclusion stage

            if valid_tool_calls:
                self.display_manager.debug(
                    f"Processing {len(valid_tool_calls)} valid tool calls"
                )

                # Standard parallel processing for tool calls
                tool_results = await asyncio.gather(
                    *[
                        self._handle_tool_call(tool_call)
                        for tool_call in valid_tool_calls
                    ]
                )
                print("************************************************")
                print(tool_results)
                print("************************************************")
                self.history.extend(tool_results)
                self.display_manager.debug(
                    f"Added {len(tool_results)} tool results to history"
                )
            elif self.in_conclusion_stage and not valid_tool_calls:
                # If in conclusion stage and LLM didn't call finish_tool, add a nudge?
                # Or handle it in the max_iterations logic? For now, let loop continue.
                self.display_manager.warning(
                    "LLM did not call the required finish tool in conclusion stage."
                )

        return message

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
        self.display_manager.debug_subtask_only(
            f"Handling tool call: {tool_call.name} (ID: {tool_call.id})"
        )

        # 1. Execute the tool
        tool_result = await self._process_tool_execution(tool_call)

        self.display_manager.debug_subtask_only(
            f"Tool {tool_call.name} execution completed"
        )

        # 3. Handle binary artifacts (images, audio)
        if isinstance(tool_result, dict):
            if "image" in tool_result:
                tool_result = self._handle_binary_artifact(
                    tool_result, tool_call.name, "image"
                )
            elif "audio" in tool_result:
                tool_result = self._handle_binary_artifact(
                    tool_result, tool_call.name, "audio"
                )

        # 4. Process special tool side-effects (e.g., finish_task, browser navigation)
        self._process_special_tool_side_effects(tool_result, tool_call)

        # Log tool result only to subtask tree (not phase)
        if tool_call.name not in ("finish_subtask", "finish_task"):
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

            self.display_manager.info_subtask_only(
                f"Saved base64 {artifact_type} from tool '{tool_call_name}' to {artifact_rel_path}"
            )

            # Update result: add path, remove original base64 data
            tool_result[f"{artifact_type}_path"] = artifact_rel_path
            del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]  # Clean up format key for audio

        except (binascii.Error, ValueError) as e:
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
        self.display_manager.debug(
            f"Processing special side effects for tool: {tool_call.name}"
        )

        if tool_call.name == "browser" and isinstance(tool_call.args, dict):
            action = tool_call.args.get("action", "")
            url = tool_call.args.get("url", "")
            if action == "navigate" and url:
                if url not in self.sources:  # Avoid duplicates
                    self.sources.append(url)
                    self.display_manager.debug_subtask_only(
                        f"Added browser source: {url}"
                    )

        if tool_call.name == "finish_task":
            self.display_manager.debug(
                "Processing finish_task - marking subtask as completed"
            )
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            # Store the result object directly
            self.processing_context.set(self.task.id, tool_result)
            self.processing_context.set(self.subtask.id, tool_result)
            self._output_result = tool_result  # Store for completion display

        if tool_call.name == "finish_subtask":
            self.display_manager.debug(
                "Processing finish_subtask - marking subtask as completed"
            )
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            # Store the result object directly
            self.processing_context.set(self.subtask.id, tool_result)
            self._output_result = tool_result  # Store for completion display
            self.display_manager.info_subtask_only(
                f"Subtask {self.subtask.id} completed with result: {tool_result}"
            )

    def _serialize_tool_result_for_history(
        self, tool_result: Any, tool_name: str
    ) -> str:
        """Serializes the tool result to a JSON string for message history."""
        try:
            if tool_result is None:
                return "Tool returned no output."
            return json.dumps(tool_result, ensure_ascii=False)
        except TypeError as e:
            self.display_manager.error(
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
        self.display_manager.warning(
            f"Subtask '{self.subtask.content}' reached max iterations ({self.max_iterations}). Forcing completion."
        )

        self.display_manager.debug(
            f"Max iterations reached for subtask {self.subtask.id}"
        )
        self.display_manager.debug("Prompting LLM to call finish tool")

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
        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=[self.finish_tool],  # Only allow the finish tool
        )

        # Add the assistant message to history
        self.history.append(message)

        # Process the tool calls if any
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Should be the finish tool call
            await self._handle_tool_call(tool_call)
            return tool_call
        else:
            # If LLM didn't call the tool, create a fallback
            self.display_manager.warning(
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
        self.display_manager.warning(
            f"Subtask '{self.subtask.content}' reached max tool calls ({self.max_tool_calls}). Forcing completion."
        )

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
        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=[self.finish_tool],  # Only allow the finish tool
        )

        # Add the assistant message to history
        self.history.append(message)

        # Process the tool calls if any
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Should be the finish tool call
            await self._handle_tool_call(tool_call)
            return tool_call
        else:
            # If LLM didn't call the tool, create a fallback
            self.display_manager.warning(
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
            self.display_manager.error(
                f"Failed to serialize result content for compression: {e}"
            )
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
                    self.display_manager.warning(
                        f"Compressed content for '{tool_name}' was not valid JSON, returning as string summary."
                    )
                    # Fall through to return the string if JSON parsing fails

            return compressed_content_str  # Return as string summary

        except Exception as e:
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
