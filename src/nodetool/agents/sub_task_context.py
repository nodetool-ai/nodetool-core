"""
🧠 SubTask Execution Context: Orchestrating Focused Task Execution

This module provides the `SubTaskContext` class, the dedicated engine for executing
a single, isolated subtask within a larger Nodetool workflow. It manages the state,
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
    the final result as a file pointer `{"path": "..."}`. The referenced file must be
    at the workspace root.
*   Helper Functions (`is_binary_output_type`, `json_schema_for_output_type`, etc.):
    Utilities for determining output types, generating appropriate JSON schemas for
    tools, and handling file operations.

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
        instructions, input files, and the current message history form the prompt.
        The LLM processes this history and generates a response. This response can
        be text content or a request to use one or more available tools (LLM Tool Calls).
        The LLM's text response (if any) is yielded externally as a `Chunk` and added
        to the internal history as an assistant message.
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
             - **Result Compression**: Compresses large tool results via LLM-based
               summarization if they exceed a token threshold
               (`_maybe_compress_tool_result` calling `_compress_tool_result`).
             - **Binary Artifact Handling**: Saves base64 encoded binary data
               (images, audio) from the result to workspace files (in an 'artifacts'
               folder) and updates the result to point to these files
               (`_handle_binary_artifact`).
             - **Special Side-Effects**: Manages tool-specific actions
               (`_process_special_tool_side_effects`). For `browser` navigation,
               it logs sources. For `finish_subtask`/`finish_task`, it marks
               the subtask complete and triggers result saving (step 3), expecting
               the result to be a file pointer.
             - **Serialization**: Converts the processed tool result to a JSON
               string (`_serialize_tool_result_for_history`).
        iii. The serialized JSON string (tool output) is then added to the subtask's
             internal message history as a `Message` with role 'tool'.
    d.  **Continuation**: The loop continues to the next LLM interaction unless a
        `finish_...` tool completed the subtask or `max_iterations` was reached.
    e.  **Conclusion Stage**: If the token limit is exceeded, the context enters a
        "conclusion stage". In this stage, available tools are restricted to only
        the `finish_...` tool, and the LLM is prompted to synthesize results and
        conclude the subtask by providing a file pointer result.
    f.  **Forced Completion (Max Iterations)**: If `max_iterations` is reached
        before `finish_...` is called, `_handle_max_iterations_reached` is invoked.
        This forces completion by:
        - Requesting a final structured output from the LLM, conforming to the
          finish tool's schema (which mandates a file pointer for `result`)
          (`request_structured_output`).
        - Triggering result saving with this structured output (step 3).
        - Marking the subtask as complete.
        - Creating an assistant message with the `ToolCall` for record and adding
          it to history.
        - Yielding this record `ToolCall` externally.
        The subtask loop then terminates.

3.  **Output Saving (`_save_to_output_file`)**: This method is called when a `finish_...`
    tool is successfully processed (via `_process_special_tool_side_effects` during
    `_handle_tool_call`) or when forced completion occurs (`_handle_max_iterations_reached`).
    It handles saving the final subtask result.
    - The result (from the `finish_...` tool's arguments) MUST be a file pointer
      object `{"path": "..."}`. The referenced workspace file/directory is copied
      to the `output_file` location.
    - Handles cases where `output_file` itself is a directory (source copied into it).

4.  **Completion**: The subtask is marked as `completed` (either by a `finish_...`
    tool call or by reaching max iterations). Status updates reflecting completion
    or failure are yielded. The `execute` loop terminates.

Key Data Structures:
--------------------
*   `Task`: Represents the overall goal.
*   `SubTask`: Defines a single step, including its objective (`content`), input/output
    files, expected `output_type`, `output_schema`.
*   `Message`: Represents a single turn in the conversation history (system, user,
    assistant, tool).
*   `ToolCall`: Represents the LLM's request to use a tool, including its arguments.
    It also includes a user-facing message string generated before execution, and
    eventually the tool's result (when part of an assistant message in history or
    processed internally).
*   `ProcessingContext`: Holds shared workflow state like the workspace directory.
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
            V No                             |  - Save output                    |
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
            |                               |     - finish_*: Save, mark complete    |
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
from nodetool.metadata.types import Message, MessageFile, SubTask, Task, ToolCall
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.workflows.processing_context import ProcessingContext

import tiktoken
import yaml


import json
import os
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
    Tuple,
    cast,
)
import logging
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.workspace_tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
import shutil

from jinja2 import Environment, BaseLoader

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKEN_LIMIT: int = 4096
DEFAULT_MAX_ITERATIONS: int = 10
MESSAGE_COMPRESSION_THRESHOLD: int = 4096

DEFAULT_EXECUTION_SYSTEM_PROMPT: str = """
You are a executing a single subtask from a larger task plan.
YOUR GOAL IS TO PRODUCE THE INTENDED `output_file` FOR THIS SUBTASK.

EXECUTION PROTOCOL:
1. Focus exclusively on the current subtask objective: {{ subtask_content }}.
2. Use the provided input files efficiently:
    - Read only the necessary parts of input files using available tools if possible.
    - Avoid loading entire large files into your working context unless absolutely required.
    - Prefer tools that process data directly (e.g., summarizing, extracting) over just reading content.
3. Perform the required steps to generate the result.
4. Ensure the final result matches the expected `output_type` and `output_schema`.
5. **Crucially**: Call `finish_subtask` ONCE at the end with the final result.
    - If the result is content (text, JSON, etc.), provide it directly in the `result` parameter.
    - If a tool directly generated the final `output_file` (e.g., download_file to `downloaded.pdf` at the root), provide `result` as `{"path": "downloaded.pdf"}`. The path MUST be to a file at the workspace root.
    - Always include relevant `metadata` (title, description, sources). Sources should cite original inputs, not just intermediate files.
6. Do NOT call `finish_subtask` multiple times. Structure your work to produce the final output, then call `finish_subtask`.
"""

DEFAULT_FINISH_TASK_SYSTEM_PROMPT: str = """
You are completing the final task and aggregating results from previous subtasks.
The goal is to combine the information from the input files (outputs of previous subtasks) into a single, final result according to the overall task objective.

FINISH_TASK PROTOCOL:
1. Analyze the provided input files (likely outputs from previous subtasks).
2. Use the `read_file` to read the content of necessary input files. Read efficiently - extract only the key information needed for aggregation if files are large.
3. Synthesize and aggregate the information to create the final task result.
4. Ensure the final result conforms to the required output schema: {{ output_schema }}.
5. Call `finish_task` ONCE with the complete, aggregated `result` and relevant `metadata` (title, description, sources - citing original sources where possible).
"""

# Define common binary types/extensions
_common_binary_types: set[str] = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "mp3",
    "wav",
    "ogg",
    "flac",
    "aac",
    "mpeg",
    "mpg",
    "mp4",
    "avi",
    "mov",
    "wmv",
    "mkv",
    "flv",
    "zip",
    "tar",
    "gz",
    "rar",
    "7z",
    "exe",
    "dll",
    "so",
    "dylib",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",  # Often treated as binary blobs
    "bin",
    "dat",
    "pickle",
    "joblib",  # Generic binary data
    "onnx",
    "pt",
    "pb",  # ML models
}

# Define known text types handled explicitly by json_schema_for_output_type or common usage
_known_text_types: set[str] = {
    "string",
    "markdown",
    "json",
    "yaml",
    "csv",
    "html",
    "python",
    "javascript",
    "typescript",
    "shell",
    "sql",
    "css",
    "svg",
    "text",
    "txt",
    "log",
    "xml",
    "rst",  # Added more text types
}


def _remove_think_tags(text_content: Optional[str]) -> Optional[str]:
    if text_content is None:
        return None
    # Use regex to remove <think>...</think> blocks, including newlines within them.
    # re.DOTALL makes . match newlines.
    # We also strip leading/trailing whitespace from the result.
    return re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()


def is_binary_output_type(output_type: str) -> bool:
    """
    Determines if an output type likely corresponds to a binary format.
    Checks common types, known text types, and guesses mime type based on potential extension.

    Args:
        output_type: The output type string (e.g., 'pdf', 'json', 'png').

    Returns:
        True if the type is likely binary, False otherwise.
    """
    if not output_type or not isinstance(output_type, str):
        return False  # Treat invalid input as non-binary

    output_type_lower = output_type.lower().strip()

    if output_type_lower in _common_binary_types:
        return True
    if output_type_lower in _known_text_types:
        return False

    # Try guessing mime type assuming output_type might be an extension
    potential_extension = f".{output_type_lower}"
    mime_type, _ = mimetypes.guess_type(f"dummy{potential_extension}")

    if mime_type:
        # Check if the guessed mime type starts with 'text/' or is a known text-based application type
        if mime_type.startswith("text/"):
            return False
        if mime_type in {
            "application/json",
            "application/xml",
            "application/javascript",
            "application/typescript",
        }:
            return False
        # Otherwise, assume binary if mime type is known but not text
        return True
    else:
        # Fallback: If it's not explicitly known as text or binary, and mime guess failed,
        # we conservatively assume it might be binary to enforce file pointer usage for safety.
        # This avoids potential issues with large or complex unknown types being passed as content.
        logger.warning(
            f"Unknown output_type '{output_type}'. Assuming binary and requiring file pointer."
        )
        return True


def mime_type_from_path(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

    # Define allowed mime types
    allowed_mime_types = [
        "application/pdf",
        "audio/mpeg",
        "audio/mp3",
        "audio/wav",
        "image/png",
        "image/jpeg",
        "image/webp",
        "text/plain",
        "video/mov",
        "video/mpeg",
        "video/mp4",
        "video/mpg",
        "video/avi",
        "video/wmv",
        "video/mpegps",
        "video/flv",
    ]

    # Check if the mime type is allowed, otherwise return a default safe type
    if mime_type not in allowed_mime_types:
        # Return text/plain as a safe default
        logger.warning(
            f"Disallowed MIME type '{mime_type}' for path '{path}', defaulting to 'text/plain'."
        )
        return "text/plain"

    return mime_type


METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "Metadata for the result",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the result",
        },
        "locale": {
            "type": "string",
            "description": "The locale of the result",
        },
        "url": {
            "type": "string",
            "description": "The URL of the result",
        },
        "site_name": {
            "type": "string",
            "description": "The site name of the result",
        },
        "tags": {
            "type": "array",
            "description": "Tags for the result",
            "items": {"type": "string"},
        },
        "description": {
            "type": "string",
            "description": "The description of the result",
        },
        "sources": {
            "type": "array",
            "description": "The sources of the result, either http://example.com, file://path/to/file.txt or search://query",
            "items": {
                "type": "string",
            },
        },
    },
    "additionalProperties": False,
    "required": [
        "title",
        "description",
        "sources",
        "locale",
        "url",
        "site_name",
        "tags",
    ],
}


def json_schema_for_output_type(output_type: str) -> Dict[str, Any]:
    if output_type == "string":
        return {"type": "string"}
    elif output_type == "markdown":
        return {
            "type": "string",
            "description": "Markdown formatted text",
            "contentMediaType": "text/markdown",
        }
    elif output_type == "json":
        return {"type": "object"}
    elif output_type == "yaml":
        return {"type": "object"}
    elif output_type == "csv":
        return {"type": "array", "items": {"type": "string"}}
    elif output_type == "html":
        return {
            "type": "string",
            "description": "HTML markup",
            "contentMediaType": "text/html",
        }
    elif output_type == "python":
        return {
            "type": "string",
            "description": "Python source code",
            "contentMediaType": "text/x-python",
        }
    elif output_type == "javascript":
        return {
            "type": "string",
            "description": "JavaScript source code",
            "contentMediaType": "application/javascript",
        }
    elif output_type == "typescript":
        return {
            "type": "string",
            "description": "TypeScript source code",
            "contentMediaType": "application/typescript",
        }
    elif output_type == "shell":
        return {
            "type": "string",
            "description": "Shell script",
            "contentMediaType": "text/x-shellscript",
        }
    elif output_type == "sql":
        return {
            "type": "string",
            "description": "SQL query or script",
            "contentMediaType": "text/x-sql",
        }
    elif output_type == "css":
        return {
            "type": "string",
            "description": "CSS stylesheet",
            "contentMediaType": "text/css",
        }
    elif output_type == "svg":
        return {
            "type": "string",
            "description": "SVG image markup",
            "contentMediaType": "image/svg+xml",
        }
    else:
        return {"type": "string"}


# Define the schema for the file pointer object
FILE_POINTER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": "An object indicating the result is a file path within the workspace.",
    "properties": {
        "path": {
            "type": "string",
            "description": "The path to the result file relative to the workspace root (e.g., 'result_data.json'). Must be at the top level, not in a subdirectory.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


class FinishTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a task as done and saves its results
    """

    name: str = "finish_task"
    description: str = """
    Finish a task by saving its final result. Provide a file pointer
    object {"path": "final_file.ext"} in 'result'. The path must be to a file at the workspace root.
    Include 'metadata'. This will hold the final result of all subtasks.
    The 'result' parameter MUST ALWAYS be a file pointer object like {"path": "your_final_output_file.ext"}.
    """
    input_schema: Dict[str, Any]  # Defined in __init__

    def __init__(
        self,
        output_type: str,
        output_schema: Any,
    ):
        super().__init__()  # Call parent constructor
        self.output_type: str = output_type  # Store output_type

        # Always use FILE_POINTER_SCHEMA regardless of output type
        content_schema = FILE_POINTER_SCHEMA

        # Final input schema for the tool
        self.input_schema = {
            "type": "object",
            "properties": {
                "result": content_schema,
                "metadata": METADATA_SCHEMA,
            },
            "required": ["result", "metadata"],
            "additionalProperties": False,
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Validation is handled by the JSON schema.
        # We just pass the params through, as the saving logic is in SubTaskContext._save_to_output_file
        return params


class FinishSubTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a subtask as done and saves its results

    This tool serves as the formal completion mechanism for subtasks by:
    1. Requiring the final output to be provided in `result` as an object `{"path": "..."}`
       pointing to a file in the workspace at the root level.
    2. The referenced file is copied to the designated output location (`output_file` defined
       in the subtask, which also must be a root-level file).
    3. Preserving structured metadata about the results.
    4. Marking the subtask as completed.

    **IMPORTANT**:
    - The `result` parameter MUST ALWAYS be a file pointer object `{"path": "path/to/workspace_root_file.ext"}`.
    - Always include the `metadata`.

    Example usage with file path:
    finish_subtask(result={"path": "final_report.pdf"}, metadata={...})

    Example usage with directory path (where "analysis_results/" is a directory at the workspace root created by a previous step):
    finish_subtask(result={"path": "analysis_results/"}, metadata={...})
    """

    name: str = "finish_subtask"
    description: str = """
    Finish a subtask by saving its final result. Provide a file pointer
    object {"path": "generated_file.ext"} in 'result'. The path must point to a file at the workspace root.
    Include 'metadata'. The result will be saved to the output_path defined in the subtask (must also be a root-level file).
    The 'result' parameter MUST ALWAYS be a file pointer object like {"path": "your_subtask_output_file.ext"}.
    """
    input_schema: Dict[str, Any]  # Defined in __init__

    def __init__(
        self,
        output_type: str,
        output_schema: Any,
    ):
        super().__init__()  # Call parent constructor
        self.output_type: str = output_type  # Store output_type

        # Always use FILE_POINTER_SCHEMA regardless of output type
        content_schema = FILE_POINTER_SCHEMA

        # Final input schema for the tool
        self.input_schema = {
            "type": "object",
            "properties": {
                "result": content_schema,
                "metadata": METADATA_SCHEMA,
            },
            "required": ["result", "metadata"],
            "additionalProperties": False,
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Validation is largely handled by the JSON schema.
        # We just pass the params through.
        return params


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
    workspace_dir: str
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
        """
        self.task = task
        self.subtask = subtask
        self.processing_context = processing_context
        self.model = model
        self.provider = provider
        self.workspace_dir = processing_context.workspace_dir
        self.max_token_limit = max_token_limit or provider.get_max_token_limit(model)
        self.use_finish_task = use_finish_task
        self.message_compression_threshold = max(
            self.max_token_limit // self.subtask.max_iterations,
            MESSAGE_COMPRESSION_THRESHOLD,
        )

        # --- Prepare prompt templates ---
        self.jinja_env = Environment(loader=BaseLoader())

        if use_finish_task:
            base_system_prompt = system_prompt or DEFAULT_FINISH_TASK_SYSTEM_PROMPT
            prompt_context = {
                "output_schema": json.dumps(
                    self.subtask.output_schema, indent=2
                )  # Provide schema context
            }
            finish_tool_output_schema = self.subtask.output_schema
            finish_tool_class = FinishTaskTool
        else:  # Standard execution subtask
            base_system_prompt = system_prompt or DEFAULT_EXECUTION_SYSTEM_PROMPT
            prompt_context = {
                "subtask_content": self.subtask.content,
                # Execution tasks might still refer to an expected output format/schema
                "output_schema": json.dumps(
                    self.subtask.output_schema or {"type": "string"}, indent=2
                ),
            }  # Provide subtask content context
            finish_tool_output_schema = self.subtask.output_schema
            finish_tool_class = FinishSubTaskTool

        self.system_prompt = self._render_prompt(base_system_prompt, prompt_context)

        # Initialize finish tool based on context (class determined above)
        self.finish_tool = finish_tool_class(
            self.subtask.output_type,
            finish_tool_output_schema,  # Use the determined schema
        )

        self.tools: Sequence[Tool] = list(tools) + [
            self.finish_tool,
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
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

    def _is_file_pointer(self, data: Any) -> bool:
        """Checks if the provided data matches the file pointer structure."""
        return isinstance(data, dict) and isinstance(data.get("path"), str)

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        """Detect if an exception is related to context window limits."""
        msg = str(error).lower()
        keywords = [
            "context length",
            "context_length",
            "context window",
            "maximum context",
            "token limit",
            "too many tokens",
            "prompt is too long",
            "max_tokens",
        ]
        return any(kw in msg for kw in keywords)

    def _find_unique_summary_path(self, base_dir: str, base_name: str, ext: str) -> str:
        """Finds a unique path for a summary file, avoiding collisions."""
        summary_path = os.path.join(base_dir, f"{base_name}{ext}")
        counter = 1
        while os.path.exists(summary_path):
            summary_path = os.path.join(base_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        return summary_path

    def _write_content_to_file(
        self, file_path: str, content: Any, metadata: dict, file_ext: str
    ):
        """Encapsulates the logic for writing different content types to a file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                if file_ext == ".md":
                    if metadata:
                        f.write("---\n")
                        try:
                            yaml.dump(
                                metadata,
                                f,
                                default_flow_style=False,
                                allow_unicode=True,
                            )
                        except yaml.YAMLError as ye:
                            logger.warning(f"Metadata YAML dump failed: {ye}")
                        if metadata:  # Re-check if dump was successful
                            f.write("---\n\n")
                    f.write(str(content))
                elif file_ext in (".yaml", ".yml"):
                    output_data = {}
                    if isinstance(content, str):
                        try:
                            parsed = yaml.safe_load(content)
                            if isinstance(parsed, dict):
                                parsed["metadata"] = metadata
                                output_data = parsed
                            else:  # Handle non-dict YAML content string
                                output_data = {"result": parsed, "metadata": metadata}
                        except yaml.YAMLError:
                            output_data = {"result": content, "metadata": metadata}
                    elif isinstance(content, dict):
                        content["metadata"] = metadata
                        output_data = content
                    else:
                        output_data = {"result": content, "metadata": metadata}
                    yaml.dump(
                        output_data,
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                elif file_ext == ".json":
                    output_data = {}
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                parsed["metadata"] = metadata
                                output_data = parsed
                            else:  # Handle non-dict JSON content string
                                output_data = {"result": parsed, "metadata": metadata}
                        except json.JSONDecodeError:
                            output_data = {"result": content, "metadata": metadata}
                    elif isinstance(content, dict):
                        content["metadata"] = metadata
                        output_data = content
                    else:
                        output_data = {"result": content, "metadata": metadata}
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                elif file_ext == ".jsonl":
                    # Assume content is list/tuple of dicts, or a single dict
                    if isinstance(content, (list, tuple)):
                        for item in content:
                            # Optionally add metadata to each line? For now, maybe not.
                            json.dump(item, f, ensure_ascii=False)
                            f.write("\n")
                    elif isinstance(content, dict):
                        content["metadata"] = (
                            metadata  # Add metadata to the single object
                        )
                        json.dump(content, f, ensure_ascii=False)
                        f.write("\n")
                    else:  # Fallback for non-iterable/non-dict content
                        json.dump(
                            {"result": content, "metadata": metadata},
                            f,
                            ensure_ascii=False,
                        )
                        f.write("\n")
                elif file_ext == ".csv":
                    import csv

                    if metadata:
                        f.write("# -- Metadata --\n")
                        [f.write(f"# {k}: {v}\n") for k, v in metadata.items()]
                        f.write("# -- End Metadata --\n")
                    # Assume content is list of lists, list of strings, or just a string
                    if isinstance(content, (list, tuple)):
                        if all(isinstance(row, (list, tuple)) for row in content):
                            csv.writer(f).writerows(content)
                        elif all(isinstance(item, str) for item in content):
                            # Treat list of strings as single-column CSV
                            csv.writer(f).writerows([[item] for item in content])
                        else:
                            # Fallback for mixed/unsupported list types
                            f.write(str(content))
                    else:
                        f.write(str(content))
                else:  # Default: treat as plain text, embed metadata in comment
                    if metadata:
                        f.write("/* -- Metadata --\n")
                        try:
                            yaml.dump(
                                metadata,
                                f,
                                default_flow_style=False,
                                allow_unicode=True,
                            )
                        except yaml.YAMLError as ye:
                            logger.warning(f"Metadata YAML dump failed: {ye}")
                        f.write("-- End Metadata -- */\n\n")
                    f.write(str(content))
        except Exception as e:
            logger.error(f"Error writing content to {file_path}: {e}")
            # Attempt to write error info even if content write failed
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "error": f"Failed to write result content: {e}",
                            "metadata": metadata,  # Include original metadata attempt
                        },
                        f,
                        indent=2,
                    )
            except Exception as write_err:
                logger.error(
                    f"Failed even to write error to file {file_path}: {write_err}"
                )

    def _save_to_output_file(self, finish_params: dict) -> None:
        """
        Save the result of a subtask to its designated output file.
        Handles results that are direct content or a file/directory pointer object.
        If the output path is a directory, content is saved to a summary file inside it.
        """
        metadata = finish_params.get("metadata", {})
        result_value = finish_params.get("result")
        output_path_rel = self.subtask.output_file
        output_path_abs = self.processing_context.resolve_workspace_path(
            output_path_rel
        )

        # Add tracked sources to metadata
        if self.sources:
            if "sources" in metadata and isinstance(metadata["sources"], list):
                metadata["sources"].extend(
                    s for s in self.sources if s not in metadata["sources"]
                )
            else:
                metadata["sources"] = list(self.sources)

        # --- Handle File Pointer Case ---
        if self._is_file_pointer(result_value):
            assert isinstance(result_value, dict)
            source_path_rel_raw = result_value["path"]
            source_path_rel = os.path.normpath(source_path_rel_raw)
            source_path_abs = self.processing_context.resolve_workspace_path(
                source_path_rel
            )

            if source_path_abs != output_path_abs:
                shutil.copy2(source_path_abs, output_path_abs)
                logger.info(
                    f"Successfully copied file '{source_path_abs}' to '{output_path_abs}'."
                )

        # --- Handle Direct Content Case ---
        elif result_value is not None:
            result_content = result_value
            if not os.path.exists(output_path_abs):
                file_ext = os.path.splitext(output_path_rel)[1].lower()
                self._write_content_to_file(
                    output_path_abs, result_content, metadata, file_ext
                )

        # --- Handle Null Result Case ---
        else:  # result_value is None
            error_msg = f"finish_... tool called with null 'result' for target file '{output_path_rel}'"
            logger.error(error_msg)
            error_filename = (
                f"error_null_{os.path.basename(output_path_rel)}.json"
                if os.path.basename(output_path_rel)
                else f"error_null_{self.subtask.id}.json"
            )
            error_file_path_abs = self.processing_context.resolve_workspace_path(
                error_filename
            )
            try:
                with open(error_file_path_abs, "w", encoding="utf-8") as f:
                    json.dump({"error": error_msg, "metadata": metadata}, f, indent=2)
            except Exception as write_err:
                logger.error(
                    f"Failed to write error file to {error_file_path_abs}: {write_err}"
                )

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

        # --- LLM-based Execution Logic ---
        prompt_parts = [
            f"**Overall Task:**\nTitle: {self.task.title}\nDescription: {self.task.description}\n",
            f"**Current Subtask Instructions:**\n{self.subtask.content}\n",  # Treat content as instructions
        ]

        if self.subtask.input_files:
            # Document that input files can be directories
            input_files_str = "\n".join(
                [
                    f"- {f} (must be at root unless it is an initial task input file. Initial inputs can be files or directories in subdirs.)"
                    for f in self.subtask.input_files
                ]
            )
            prompt_parts.append(
                f"**Input Files for this Subtask:**\n{input_files_str}\n"
            )

        # Mention output path can be a directory
        prompt_parts.append(
            f"**Expected Output File (must be at workspace root):**\n{self.subtask.output_file}\n"
        )

        prompt_parts.append(
            "Please perform the subtask based on the provided context, instructions, inputs, and expected output file."
        )
        task_prompt = "\n".join(prompt_parts)

        # Add the task prompt to this subtask's history
        self.history.append(Message(role="user", content=task_prompt))

        # Yield task update for subtask start
        yield TaskUpdate(
            task=self.task,
            subtask=self.subtask,
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

        # Continue executing until the task is completed or max iterations reached
        while not self.subtask.completed and self.iterations < self.max_iterations:
            self.iterations += 1

            # Calculate total token count AFTER potential compression
            token_count = self._count_tokens(self.history)

            # Check if we need to transition to conclusion stage
            if (token_count > self.max_token_limit) and not self.in_conclusion_stage:
                await self._transition_to_conclusion_stage()
                # Yield the event after transitioning
                yield TaskUpdate(
                    task=self.task,
                    subtask=self.subtask,
                    event=TaskUpdateEvent.ENTERED_CONCLUSION_STAGE,
                )

            # Process current iteration
            message = await self._process_iteration()
            if message.tool_calls:
                if message.content:
                    yield Chunk(content=str(message.content))
                for tool_call in message.tool_calls:
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
                        yield TaskUpdate(
                            task=self.task,
                            subtask=self.subtask,
                            event=TaskUpdateEvent.SUBTASK_COMPLETED,
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
            yield tool_call

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
            logger.warning(
                f"Token limit approaching. Transitioning subtask '{self.subtask.content}' to conclusion stage."
            )

    async def _optimize_context_window(self) -> None:
        """
        Optimize the context window by cleaning up the message history using several strategies:
        1. Remove "thinking" sections in messages
        2. Summarize long tool results
        3. Remove/summarize less relevant history
        4. Keep only the most recent interactions
        5. Ensure system and task messages are preserved
        """
        logger.warning(
            f"Optimizing context window for subtask '{self.subtask.content}'"
        )

        # Make sure we have at least 3 messages (system, user, and 1+ exchanges)
        if len(self.history) < 3:
            logger.warning("History too short to optimize, nothing to do")
            return

        # Step 1: Never touch the system prompt (index 0) and the task prompt (index 1)
        preserved_messages = self.history[:2]  # System and task prompt
        working_history = self.history[2:]

        # Step 2: Calculate token counts for each message to identify largest ones
        message_sizes = [
            (i + 2, msg, self._count_single_message_tokens(msg))
            for i, msg in enumerate(working_history)
        ]

        # Step 3: Find the largest tool responses (typically file reads, web fetches)
        large_tool_responses = []
        for idx, msg, tokens in message_sizes:
            if msg.role == "tool" and tokens > MESSAGE_COMPRESSION_THRESHOLD:
                large_tool_responses.append((idx, msg, tokens))

        # Step 4: Compress large tool responses
        for idx, msg, tokens in large_tool_responses:
            try:
                # Check if it's JSON content that we can parse and compress
                if msg.content and isinstance(msg.content, str):
                    try:
                        content_data = json.loads(msg.content)
                        # If it's a read_file result, extract the essential metadata
                        if msg.name == "read_file" and isinstance(content_data, dict):
                            path = content_data.get("path", "unknown")
                            # Keep line info and token info if available
                            line_info = content_data.get("line_info", {})
                            token_info = content_data.get("token_info", {})
                            content_size = len(content_data.get("content", ""))
                            # Create a compact summary instead of full content
                            summary = {
                                "success": content_data.get("success", False),
                                "path": path,
                                "action": "read_file",
                                "content_summary": f"[File content ({content_size} chars, {token_info.get('count', 0)} tokens) removed to save context space]"
                                f" File had {line_info.get('total_lines', 0)} lines.",
                            }
                            # Replace the content with our summary
                            self.history[idx].content = json.dumps(summary)

                        # Summarize other large tool results (like web fetches)
                        elif msg.name in [
                            "browser",
                            "search",
                            "web_fetch",
                        ] and isinstance(content_data, dict):
                            # Create a compact summary for web results
                            summary = {
                                "action": msg.name,
                                "content_summary": f"[Web content ({tokens} tokens) removed to save context space]",
                            }
                            if "url" in content_data:
                                summary["url"] = content_data.get("url")
                            if "title" in content_data:
                                summary["title"] = content_data.get("title")
                            self.history[idx].content = json.dumps(summary)
                    except json.JSONDecodeError:
                        # Not valid JSON, use a simple text truncation approach
                        if len(msg.content) > 500:
                            self.history[idx].content = (
                                f"{msg.content[:500]}... [Content truncated to save context space]"
                            )
            except Exception as e:
                logger.error(f"Error optimizing message {idx}: {e}")

        # Step 5: If we're still over limit, keep only the most recent interactions
        # First count tokens again after the optimizations above
        current_token_count = self._count_tokens(self.history)
        if (
            current_token_count > self.max_token_limit * 0.85
        ):  # If still using >85% of limit
            # Keep system message, task message, and the most recent exchanges
            # The most important part is the system message and recent context
            essential_count = len(preserved_messages)  # System + task messages
            recent_count = min(
                4, len(working_history)
            )  # Keep at least 4 recent messages if available
            oldest_to_remove = len(working_history) - recent_count

            if oldest_to_remove > 0:
                # Keep only the most recent exchanges
                self.history = preserved_messages + working_history[-recent_count:]
                # Insert a marker message to show history was truncated
                self.history.insert(
                    2,
                    Message(
                        role="system",
                        content=f"[Context window optimized: {oldest_to_remove} older messages removed to stay within token limits]",
                    ),
                )
                logger.info(
                    f"Removed {oldest_to_remove} older messages to optimize context window"
                )

        # Log the final optimization results
        original_tokens = current_token_count
        optimized_tokens = self._count_tokens(self.history)
        reduction = original_tokens - optimized_tokens
        logger.info(
            f"Context window optimization complete: {reduction} tokens removed ({int(reduction/original_tokens*100)}% reduction)"
        )

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

        # Ensure ReadFileTool is always available if not already included
        if not any(isinstance(t, ReadFileTool) for t in tools_for_iteration):
            tools_for_iteration = list(tools_for_iteration) + [ReadFileTool()]

        # Create a dictionary to track unique tools by name
        unique_tools = {tool.name: tool for tool in tools_for_iteration}
        final_tools = list(unique_tools.values())

        try:
            message = await self.provider.generate_message(
                messages=self.history,
                model=self.model,
                tools=final_tools,
            )
        except Exception as e:
            if SubTaskContext._is_context_length_error(e):
                await self._optimize_context_window()
                message = await self.provider.generate_message(
                    messages=self.history,
                    model=self.model,
                    tools=final_tools,
                )
            else:
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

        # Check if the message contains output files and use them as subtask output
        if hasattr(message, "output_files") and message.output_files:
            # Use the first output file as the subtask output
            output_file = self.processing_context.resolve_workspace_path(
                self.subtask.output_file
            )
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Write the file content to the output file
            with open(output_file, "wb") as f:
                f.write(message.output_files[0].content)

            # Mark subtask as completed
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())

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
                        logger.warning(
                            f"LLM attempted to call disallowed tool '{tc.name}' in conclusion stage. Ignoring."
                        )
            else:
                valid_tool_calls = (
                    message.tool_calls
                )  # Allow all tools if not in conclusion stage

            if valid_tool_calls:
                tool_results = await asyncio.gather(
                    *[
                        self._handle_tool_call(tool_call)
                        for tool_call in valid_tool_calls
                    ]
                )
                self.history.extend(tool_results)
            elif self.in_conclusion_stage and not valid_tool_calls:
                # If in conclusion stage and LLM didn't call finish_tool, add a nudge?
                # Or handle it in the max_iterations logic? For now, let loop continue.
                logger.warning(
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
        # 1. Execute the tool
        raw_tool_result = await self._process_tool_execution(tool_call)

        # 2. Conditionally compress the tool result
        # processed_tool_result = await self._maybe_compress_tool_result(
        #     raw_tool_result, tool_call
        # )
        processed_tool_result = raw_tool_result

        # 3. Handle binary artifacts (images, audio)
        if isinstance(processed_tool_result, dict):
            if "image" in processed_tool_result:
                processed_tool_result = self._handle_binary_artifact(
                    processed_tool_result, tool_call.name, "image"
                )
            elif "audio" in processed_tool_result:
                processed_tool_result = self._handle_binary_artifact(
                    processed_tool_result, tool_call.name, "audio"
                )

        # 4. Process special tool side-effects (e.g., finish_task, browser navigation)
        self._process_special_tool_side_effects(processed_tool_result, tool_call)

        # 5. Serialize the final processed result for history
        content_str = self._serialize_tool_result_for_history(
            processed_tool_result, tool_call.name
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

    async def _maybe_compress_tool_result(
        self, tool_result: Any, tool_call: ToolCall
    ) -> Any:
        """Compresses the tool result if it's too large."""
        # Skip compression for finish tools as their structure is required for saving
        if tool_call.name in ("finish_task", "finish_subtask"):
            return tool_result

        try:
            # Serialize the result to check its size
            result_str_for_size_check = json.dumps(tool_result, ensure_ascii=False)
            result_token_count = len(self.encoding.encode(result_str_for_size_check))

            if result_token_count > self.message_compression_threshold:
                logger.info(
                    f"Tool result for '{tool_call.name}' ({result_token_count} tokens) "
                    f"exceeds threshold ({self.message_compression_threshold}). Compressing..."
                )
                compressed_result = await self._compress_tool_result(
                    tool_result, tool_call.name, tool_call.args
                )
                new_token_count = len(
                    self.encoding.encode(
                        json.dumps(compressed_result, ensure_ascii=False)
                    )
                )
                logger.info(
                    f"Compressed tool result for '{tool_call.name}' to {new_token_count} tokens."
                )
                return compressed_result
            return tool_result
        except TypeError as e:
            logger.warning(
                f"Could not serialize tool result for '{tool_call.name}' to check size: {e}. Skipping compression check."
            )
            return tool_result
        except Exception as e:
            logger.error(
                f"Error during tool result compression check for '{tool_call.name}': {e}"
            )
            return tool_result  # Return original result if compression check fails

    def _handle_binary_artifact(
        self, tool_result: Dict[str, Any], tool_call_name: str, artifact_type: str
    ) -> Dict[str, Any]:
        """Handles saving binary artifacts (image or audio) and updating the tool result."""
        artifact_key = artifact_type  # "image" or "audio"
        base64_data = tool_result.get(artifact_key)

        if not isinstance(base64_data, str):
            logger.warning(
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

            print(
                f"Saved base64 {artifact_type} from tool '{tool_call_name}' to {artifact_rel_path}"
            )

            # Update result: add path, remove original base64 data
            tool_result[f"{artifact_type}_path"] = artifact_rel_path
            del tool_result[artifact_key]
            if artifact_type == "audio" and "format" in tool_result:
                del tool_result["format"]  # Clean up format key for audio

        except (binascii.Error, ValueError) as e:
            logger.error(
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
            logger.error(
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
        if tool_call.name == "browser" and isinstance(tool_call.args, dict):
            action = tool_call.args.get("action", "")
            url = tool_call.args.get("url", "")
            if action == "navigate" and url:
                if url not in self.sources:  # Avoid duplicates
                    self.sources.append(url)

        if tool_call.name == "finish_task":
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            self._save_to_output_file(cast(dict, tool_result))

        if tool_call.name == "finish_subtask":
            self.subtask.completed = True
            self.subtask.end_time = int(time.time())
            self._save_to_output_file(cast(dict, tool_result))

    def _serialize_tool_result_for_history(
        self, tool_result: Any, tool_name: str
    ) -> str:
        """Serializes the tool result to a JSON string for message history."""
        try:
            return json.dumps(tool_result, ensure_ascii=False)
        except TypeError as e:
            logger.error(
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
        the LLM to explicitly call the finish tool.
        """
        logger.warning(
            f"Subtask '{self.subtask.content}' reached max iterations ({self.max_iterations}). Forcing completion."
        )
        # --- Determine schema and tool name dynamically ---
        tool_name = "finish_task" if self.use_finish_task else "finish_subtask"
        # The finish_tool already has the correct schema based on __init__
        final_call_schema = self.finish_tool.input_schema
        # --- End schema/tool determination ---

        # Request the final output using the determined schema
        structured_result = await self.request_structured_output(
            final_call_schema, tool_name
        )

        # Save the structured result (which includes result and metadata keys)
        # _save_to_output_file expects the dict passed to finish_subtask/finish_task
        self._save_to_output_file(structured_result)
        self.subtask.completed = True
        self.subtask.end_time = int(time.time())

        # Create the tool call based on the structured data received
        # Note: The 'result' field *within* structured_result holds the actual subtask/task result
        tool_call = ToolCall(
            id=f"max_iterations_{tool_name}",
            name=tool_name,
            args=structured_result,
        )

        # Add the tool call to history for completeness
        self.history.append(Message(role="assistant", tool_calls=[tool_call]))

        return tool_call

    async def request_structured_output(self, schema: dict, schema_name: str) -> dict:
        """
        Request a final structured output from the LLM when max iterations are reached,
        using response_format parameter with JSON schema to ensure valid structure.

        Args:
            schema: The JSON schema the output must conform to.
            schema_name: A descriptive name for the root of the schema.

        Returns:
            A dictionary conforming to the schema, or an error dict.
        """
        logger.info(
            f"Requesting structured output conforming to schema '{schema_name}' due to max iterations."
        )
        # Create a JSON-specific system prompt
        json_system_prompt = f"""
        You MUST provide the final output for the subtask in JSON format, strictly matching the '{schema_name}' tool's input schema.
        You have reached the maximum iterations allowed ({self.max_iterations}). Synthesize all previous work from the conversation history into a single, valid JSON response.
        Ensure the JSON includes all required fields specified in the schema, particularly 'result' and 'metadata'.
        For the 'result' field, you must provide the actual computed data that fulfills the subtask's objective and conforms to its defined schema. Do NOT provide the schema definition itself as the value for the 'result' field.
        If you cannot determine appropriate values for the 'result' or 'metadata' fields from the history, use sensible defaults or indicate the missing information clearly within the data structure (e.g., in the description field of metadata, or as null/empty values within the 'result' data if its schema allows).
        Do NOT include any explanatory text outside the JSON object. Your entire response must be the JSON object itself.
        """

        # Create a focused user prompt
        json_user_prompt = f"""
        The subtask has reached the maximum allowed iterations ({self.max_iterations}).

        Based on all previous information in the conversation history, generate the most complete and accurate JSON output possible
        conforming EXACTLY to the '{schema_name}' schema.
        Ensure all required fields (like 'result' and 'metadata') are present. Provide the best possible result based on the available context.
        YOUR RESPONSE MUST BE ONLY THE JSON OBJECT.
        """

        # Create a minimal history with just the system prompt and request
        # Include previous history to provide context for the final generation
        json_history = [
            Message(role="system", content=json_system_prompt),
            *self.history[1:],  # Include previous interactions for context
            Message(role="user", content=json_user_prompt),
        ]

        # --- Modify schema before sending ---
        modified_schema = SubTaskContext._enforce_strict_object_schema(schema)
        # --- End schema modification ---

        # Get response with response_format set to JSON schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": modified_schema,
                "strict": True,
            },
        }

        try:
            message = await self.provider.generate_message(
                messages=json_history,
                model=self.model,
                tools=[],  # No tools allowed for this specific call
                response_format=response_format,
            )

            # Parse the JSON content
            content = str(message.content)
            # Add message to history for debugging/audit trail
            self.history.append(message)

            try:
                parsed_json = json.loads(content)
                # Basic validation: check if required keys are present (adjust as needed)
                if "result" in parsed_json and "metadata" in parsed_json:
                    return parsed_json
                else:
                    print(
                        f"Warning: LLM JSON output missing required keys ('result', 'metadata'). Raw: {content}"
                    )
                    # Attempt to salvage if possible, otherwise return error
                    return {
                        "error": "Generated JSON missing required keys",
                        "result": parsed_json.get("result", "Missing result key"),
                        "metadata": parsed_json.get(
                            "metadata",
                            {
                                "title": "Error",
                                "description": "Missing metadata",
                                "sources": [],
                            },
                        ),
                    }

            except json.JSONDecodeError as e:
                print(
                    f"Error: Failed to decode JSON from LLM after max iterations. Error: {e}. Raw: {content}"
                )
                return {
                    "error": f"Failed to generate valid JSON: {e}",
                    # Include raw content in 'result' field within the error structure
                    "result": f"Invalid JSON received: {content}",
                    "metadata": {
                        "title": "JSON Error",
                        "description": "Failed to parse LLM JSON output",
                        "sources": [],
                    },
                }

        except Exception as e:
            # Catch potential API errors or other issues during generation
            print(f"Error: Exception during structured output generation: {e}")
            # Add an error message to history
            error_msg = Message(
                role="assistant", content=f"Error generating structured output: {e}"
            )
            self.history.append(error_msg)
            return {
                "error": f"Failed to generate structured output: {e}",
                "result": "Error during generation",
                "metadata": {
                    "title": "Generation Error",
                    "description": str(e),
                    "sources": [],
                },
            }

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
            original_token_count = len(self.encoding.encode(original_content_str))
        except Exception as e:
            logger.error(f"Failed to serialize result content for compression: {e}")
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
                    logger.warning(
                        f"Compressed content for '{tool_name}' was not valid JSON, returning as string summary."
                    )
                    # Fall through to return the string if JSON parsing fails

            return compressed_content_str  # Return as string summary

        except Exception as e:
            logger.error(
                f"Error during LLM call for tool result compression ('{tool_name}'): {e}"
            )
            # Return a structured error message instead of the original large content
            return {
                "error": f"Failed to compress tool result via LLM: {e}",
                "compression_failed": True,
                "original_content_preview": original_content_str[:500]
                + "...",  # Include a preview
            }

    # --- Helper Function for Schema Modification ---
    @staticmethod
    def _enforce_strict_object_schema(schema: Any) -> Any:
        """Recursively adds 'additionalProperties': False and sets all properties as required for object schemas."""
        if isinstance(schema, dict):
            if schema.get("type") == "object" and "properties" in schema:
                schema["additionalProperties"] = False
                # Set all properties as required
                schema["required"] = list(schema["properties"].keys())
                # Recursively apply to properties
                for key, prop_schema in schema["properties"].items():
                    schema["properties"][key] = (
                        SubTaskContext._enforce_strict_object_schema(prop_schema)
                    )
            elif schema.get("type") == "array" and "items" in schema:
                # Recursively apply to array items schema
                schema["items"] = SubTaskContext._enforce_strict_object_schema(
                    schema["items"]
                )
            # Recursively apply to other nested structures like allOf, anyOf, etc.
            for key in ["allOf", "anyOf", "oneOf", "not"]:
                if key in schema and isinstance(schema[key], list):
                    schema[key] = [
                        SubTaskContext._enforce_strict_object_schema(sub_schema)
                        for sub_schema in schema[key]
                    ]
            if "definitions" in schema and isinstance(schema["definitions"], dict):
                for key, def_schema in schema["definitions"].items():
                    schema["definitions"][key] = (
                        SubTaskContext._enforce_strict_object_schema(def_schema)
                    )

        elif isinstance(schema, list):
            # Handle cases where the schema itself might be a list (e.g., type: [string, null])
            return [
                SubTaskContext._enforce_strict_object_schema(item) for item in schema
            ]

        return schema

    # --- End Helper Function ---
