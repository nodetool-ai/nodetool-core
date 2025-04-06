import asyncio
import datetime
import mimetypes
from nodetool.chat.providers import ChatProvider
from nodetool.chat.tools import Tool
from nodetool.chat.tools.base import resolve_workspace_path
from nodetool.metadata.types import Message, MessageFile, SubTask, Task, ToolCall
from nodetool.workflows.types import Chunk, TaskUpdate, TaskUpdateEvent
from nodetool.workflows.processing_context import ProcessingContext

import tiktoken
import yaml
from pydantic import BaseModel


import json
import os
import time
from typing import Any, AsyncGenerator, List, Sequence, Union

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools.workspace import ReadWorkspaceFileTool
from enum import Enum

DEFAULT_EXECUTION_SYSTEM_PROMPT = """
You are a executing a single subtask from a larger task plan.
PROVIDE THE RESULT IN THE FINISH_SUBTASK CALL TO SAVE THE RESULT TO THE OUTPUT FILE.
YOU CAN ONLY USE TOOLS TO PRODUCE THE OUTPUT FILE.

EXECUTION PROTOCOL:
1. Focus exclusively on the current subtask
2. ALWAYS call finish_subtask with the result
3. Results MUST be a JSON object
4. Pick the output type defined in the subtask
5. Include metadata with these fields: title, description, source, images, videos, audio, documents
6. Call `finish_subtask` with a result object, output_file, and metadata
7. When downloading files, use the output_file parameter to save the file to the workspace
"""

DEFAULT_FINISH_TASK_SYSTEM_PROMPT = """
You are a completing a task and aggregating the results.
Accept the results of input files and aggregate them into a single result.
PROVIDE THE RESULT IN THE FINISH_TASK CALL.

FINISH_TASK PROTOCOL:
1. ALWAYS call finish_task with the result
2. Results MUST be a JSON object
3. Results should be in the format defined in the output_schema
4. Pick the output type defined in the subtask
5. Include metadata with these fields: title, description, source, images, videos, audio, documents
6. Provide the final task result in the result field
7. Use the ReadWorkspaceFileTool to read the contents
"""


def mime_type_from_path(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


METADATA_SCHEMA = {
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
        "images": {
            "type": "array",
            "description": "The URLs of the images in the result",
            "items": {
                "type": "string",
            },
        },
        "videos": {
            "type": "array",
            "description": "The URLs of the videos in the result",
            "items": {
                "type": "string",
            },
        },
        "audio": {
            "type": "array",
            "description": "The URLs of the audio in the result",
            "items": {
                "type": "string",
            },
        },
        "documents": {
            "type": "array",
            "description": "The URLs of the documents in the result",
            "items": {
                "type": "string",
            },
        },
        "sources": {
            "type": "array",
            "description": "The sources of the result, either http://example.com, file://path/to/file.txt or search://query",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["title", "description", "sources"],
}


def json_schema_for_output_type(output_type: str) -> dict:
    if output_type == "string":
        return {"type": "string"}
    elif output_type == "html":
        return {
            "type": "string",
        }
    elif output_type == "markdown":
        return {
            "type": "string",
        }
    elif output_type == "json":
        return {"type": "object"}
    elif output_type == "yaml":
        return {"type": "object"}
    elif output_type == "csv":
        return {"type": "array", "items": {"type": "string"}}
    elif output_type == "html":
        return {"type": "string"}
    elif output_type == "xml":
        return {"type": "string"}
    elif output_type == "jsonl":
        return {"type": "array", "items": {"type": "string"}}
    elif output_type in ["python", "py"]:
        return {
            "type": "string",
            "description": "Python source code",
            "contentMediaType": "text/x-python",
        }
    elif output_type in ["javascript", "js", "typescript", "ts"]:
        return {
            "type": "string",
            "description": "JavaScript/TypeScript source code",
            "contentMediaType": "application/javascript",
        }
    elif output_type == "java":
        return {
            "type": "string",
            "description": "Java source code",
            "contentMediaType": "text/x-java",
        }
    elif output_type in ["cpp", "c++"]:
        return {
            "type": "string",
            "description": "C++ source code",
            "contentMediaType": "text/x-c++src",
        }
    elif output_type == "go":
        return {
            "type": "string",
            "description": "Go source code",
            "contentMediaType": "text/x-go",
        }
    elif output_type == "rust":
        return {
            "type": "string",
            "description": "Rust source code",
            "contentMediaType": "text/x-rust",
        }
    # Add common development formats
    elif output_type == "diff":
        return {
            "type": "string",
            "description": "Unified diff format",
            "contentMediaType": "text/x-diff",
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
    elif output_type == "dockerfile":
        return {
            "type": "string",
            "description": "Dockerfile",
            "contentMediaType": "text/x-dockerfile",
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


class FinishTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a task as done and saves its results
    """

    name = "finish_task"
    description = """
    Finish a task by saving its final result to a file in the workspace.
    This will hold the final result of all subtasks.
    """

    def __init__(
        self,
        workspace_dir: str,
        output_type: str,
        output_schema: Any,
    ):
        self.workspace_dir = workspace_dir
        if output_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "result": json_schema_for_output_type(output_type),
                    "metadata": METADATA_SCHEMA,
                },
                "required": ["result", "metadata"],
            }
        else:
            try:
                self.input_schema = {
                    "type": "object",
                    "properties": {
                        "result": output_schema,
                        "metadata": METADATA_SCHEMA,
                    },
                    "required": ["result", "metadata"],
                }
            except Exception as e:
                print(f"Error parsing output schema: {e}")
                self.input_schema = json_schema_for_output_type(output_type)


class FinishSubTaskTool(Tool):
    """
    🏁 Task Completion Tool - Marks a subtask as done and saves its results

    This tool serves as the formal completion mechanism for subtasks by:
    1. Saving the final output to the designated workspace location
    2. Preserving structured metadata about the results
    3. Marking the subtask as completed

    This tool is EXCLUSIVELY AVAILABLE during the Conclusion Stage of complex subtasks,
    forcing the agent to synthesize findings before completion. For simple 'tool_call'
    subtasks, it is called immediately after the information-gathering tool.

    Usage pattern:
    - First call information tools to gather data (Tool Calling Stage)
    - Then call finish_subtask to formalize and save results (Conclusion Stage)

    Example result format:
    {
        "content": {
            "analysis": "The data shows a clear trend of...",
            "recommendations": ["First, consider...", "Second, implement..."]
        },
        "metadata": {
            "title": "Market Analysis Results",
            "description": "Comprehensive analysis of market trends",
            "timestamp": "2023-06-15T10:30:00Z",
            "sources": [
                "http://example.com",
                "file://path/to/file.txt",
                "search://market trends",
            ],
            "images": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
            ],
            "videos": [
                "https://example.com/video1.mp4",
                "https://example.com/video2.mp4",
            ],
            "audio": [
                "https://example.com/audio1.mp3",
                "https://example.com/audio2.mp3",
            ],
            "documents": [
                "https://example.com/document1.pdf",
                "https://example.com/document2.pdf",
            ],
        },
    }
    """

    name = "finish_subtask"
    description = """
    Finish a subtask by saving its final result to a file in the workspace.
    The result will be saved to the output_file path, defined in the subtask.
    """

    def __init__(
        self,
        workspace_dir: str,
        output_type: str,
        output_schema: Any,
    ):
        self.workspace_dir = workspace_dir
        if output_schema is None:
            self.input_schema = {
                "type": "object",
                "properties": {
                    "result": json_schema_for_output_type(output_type),
                    "metadata": METADATA_SCHEMA,
                },
                "required": ["result", "metadata"],
            }
        else:
            try:
                self.input_schema = {
                    "type": "object",
                    "properties": {
                        "result": output_schema,
                        "metadata": METADATA_SCHEMA,
                    },
                    "required": ["result", "metadata"],
                }
            except Exception as e:
                print(f"Error parsing output schema: {e}")
                self.input_schema = json_schema_for_output_type(output_type)

    async def process(self, context: ProcessingContext, params: dict):
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

    def __init__(
        self,
        task: Task,
        subtask: SubTask,
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        model: str,
        provider: ChatProvider,
        system_prompt: str | None = None,
        use_finish_task: bool = False,
        max_token_limit: int = -1,
        max_iterations: int = 5,
        enable_tracing: bool = True,
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
            enable_tracing (bool): Whether to enable LLM message tracing
        """
        self.task = task
        self.subtask = subtask
        self.processing_context = processing_context
        self.model = model
        self.provider = provider
        self.workspace_dir = processing_context.workspace_dir
        self.max_token_limit = max_token_limit
        self.enable_tracing = enable_tracing
        if use_finish_task:
            self.system_prompt = system_prompt or DEFAULT_FINISH_TASK_SYSTEM_PROMPT
            self.tools: Sequence[Tool] = [
                FinishTaskTool(
                    self.workspace_dir,
                    self.subtask.output_type,
                    self.subtask.output_schema,
                ),
                ReadWorkspaceFileTool(self.workspace_dir),
            ]
        else:
            self.system_prompt = system_prompt or DEFAULT_EXECUTION_SYSTEM_PROMPT
            self.tools: Sequence[Tool] = list(tools) + [
                FinishSubTaskTool(
                    self.workspace_dir,
                    self.subtask.output_type,
                    self.subtask.output_schema,
                ),
                ReadWorkspaceFileTool(self.workspace_dir),
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

        # Track tool calls for this subtask
        self.tool_call_count = 0

        # Track sources for data lineage
        self.sources = []

        # Track progress for this subtask
        self.progress = []

        # Flag to track which stage we're in - normal execution flow for all tasks
        self.in_conclusion_stage = False

        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Add a list to track yielded events to avoid duplicates
        self.yielded_events = []

        # Setup tracing directory and file
        if self.enable_tracing:
            self.traces_dir = os.path.join(self.workspace_dir, "traces")
            os.makedirs(self.traces_dir, exist_ok=True)

            # Create a unique trace file name based on task and subtask IDs
            sanitized_subtask_name = "".join(
                c if c.isalnum() else "_" for c in self.subtask.content[:40]
            )
            self.trace_file_path = os.path.join(
                self.traces_dir,
                f"trace_{sanitized_subtask_name}.jsonl",
            )

            # Initialize trace with basic metadata
            self._log_trace_event(
                "trace_initialized",
                {
                    "subtask_content": self.subtask.content,
                    "model": self.model,
                    "max_iterations": self.max_iterations,
                    "output_file": self.subtask.output_file,
                },
            )

            # Log the system prompt
            self._log_trace_event(
                "message",
                {
                    "direction": "system",
                    "content": self.system_prompt,
                    "role": "system",
                },
            )

        # Add a buffer for aggregating chunks for trace logging
        self._chunk_buffer = ""
        self._is_buffering_chunks = False

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
            # Count tokens in the message content
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, str):
                    token_count += len(self.encoding.encode(msg.content))
                elif isinstance(msg.content, list):
                    # For multi-modal content, just count the text parts
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            token_count += len(
                                self.encoding.encode(part.get("text", ""))
                            )

            # Count tokens in tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # Count function name
                    token_count += len(self.encoding.encode(tool_call.name))
                    # Count arguments
                    if isinstance(tool_call.args, dict):
                        token_count += len(
                            self.encoding.encode(json.dumps(tool_call.args))
                        )
                    else:
                        token_count += len(self.encoding.encode(str(tool_call.args)))

        return token_count

    def _save_to_output_file(self, output: Any) -> None:
        """
        Save the result of a tool call to the output file.
        Includes metadata in appropriate formats based on file type.
        """
        # Extract metadata from the result if it's a dictionary
        if isinstance(output, dict) and "result" in output:
            metadata = output.pop("metadata", {})
            output = output.get("result", output)
        else:
            metadata = {}

        # Add tracked sources to metadata
        if self.sources and isinstance(output, dict):
            # If sources already exist in metadata, extend them
            if "sources" in metadata and isinstance(metadata["sources"], list):
                metadata["sources"].extend(self.sources)
            else:
                metadata["sources"] = self.sources

        # Get file extension
        file_ext = os.path.splitext(self.subtask.output_file)[1].lower()

        output_file = resolve_workspace_path(
            self.workspace_dir, self.subtask.output_file
        )

        # If file already exists, assume it was created by a tool and skip saving
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists, skipping save")
            return

        print(f"Saving result to {output_file}")

        # Create parent directory for output file if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w") as f:
            if file_ext == ".md":
                # For Markdown files, add metadata as YAML frontmatter
                if metadata:
                    f.write("---\n")
                    yaml.dump(metadata, f)
                    f.write("---\n\n")
                f.write(str(output))

            elif file_ext == ".yaml" or file_ext == ".yml":
                if isinstance(output, str):
                    f.write(output)
                else:
                    if isinstance(output, dict):
                        output["metadata"] = metadata
                    yaml.dump(output, f)

            elif file_ext == ".json":
                if isinstance(output, str):
                    f.write(output)
                else:
                    if isinstance(output, dict):
                        output["metadata"] = metadata
                    json.dump(output, f, indent=2)

            elif file_ext == ".jsonl":
                # For JSONL, write each line as a separate JSON object
                if isinstance(output, (list, tuple)):
                    for item in output:
                        json.dump(item, f)
                        f.write("\n")
                else:
                    json.dump(output, f)
                    f.write("\n")

            elif file_ext == ".csv":
                import csv

                # Handle CSV output based on data type
                if isinstance(output, (list, tuple)):
                    writer = csv.writer(f)
                    for row in output:
                        writer.writerow(row)  # Write multiple rows
                else:
                    f.write(str(output))

            else:
                # For all other formats, just write the result as a string
                f.write(str(output))

    def _log_trace_event(self, event_type: str, data: dict) -> None:
        """
        Log an event to the trace file.

        Args:
            event_type (str): Type of event (message, tool_call, etc.)
            data (dict): Event data to log
        """
        if not self.enable_tracing:
            return

        trace_entry = {"timestamp": time.time(), "event": event_type, "data": data}

        with open(self.trace_file_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

    async def execute(
        self,
    ) -> AsyncGenerator[Union[Chunk, ToolCall, TaskUpdate], None]:
        """
        ⚙️ Task Executor - Runs a single subtask to completion using appropriate strategy

        This execution path handles:
        1. Single tool execution tasks: Direct tool execution and result storage
        2. Complex tasks requiring multiple steps:
           a. Tool Calling Stage: Multiple iterations of information gathering using any tools
           b. Conclusion Stage: Final synthesis with restricted access (only finish_subtask tool)

        The method automatically tracks iterations, manages token limits, and enforces
        transitions between stages based on limits or progress.

        Yields:
            Union[Chunk, ToolCall, TaskUpdate]: Live updates during task execution
        """
        # Record the start time of the subtask
        current_time = int(time.time())
        self.subtask.start_time = current_time

        # For tasks with both tool_name and input_files, or regular tasks
        task_prompt = (
            self.task.title
            + "\n\n"
            + self.task.description
            + "\n\n"
            + self.subtask.content
        )

        input_files = []
        if self.subtask.input_files and self.provider.has_code_interpreter:
            for input_file in self.subtask.input_files:
                path = resolve_workspace_path(self.workspace_dir, input_file)
                with open(path, "rb") as f:
                    content = f.read()
                input_files.append(
                    MessageFile(
                        content=content,
                        mime_type=mime_type_from_path(path),
                    )
                )

        # Add the task prompt to this subtask's history
        self.history.append(
            Message(role="user", content=task_prompt, input_files=input_files)
        )

        # Log the task prompt in the trace
        if self.enable_tracing:
            self._log_trace_event(
                "message",
                {"direction": "outgoing", "content": task_prompt, "role": "user"},
            )

        # Yield task update for subtask start
        yield TaskUpdate(
            task=self.task,
            subtask=self.subtask,
            event=TaskUpdateEvent.SUBTASK_STARTED,
        )

        # Continue executing until the task is completed or max iterations reached
        while not self.subtask.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            token_count = self._count_tokens(self.history)

            # Check if we need to transition to conclusion stage
            if (token_count > self.max_token_limit) and not self.in_conclusion_stage:
                await self._transition_to_conclusion_stage()
                # Yield task update for transition to conclusion stage
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
                    yield ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                    )
                    if tool_call.name == "finish_subtask":
                        yield TaskUpdate(
                            task=self.task,
                            subtask=self.subtask,
                            event=TaskUpdateEvent.SUBTASK_COMPLETED,
                        )

        # If we've reached the last iteration and haven't completed yet, generate summary
        if self.iterations >= self.max_iterations and not self.subtask.completed:
            await self._handle_max_iterations_reached()
            # Yield task update for max iterations reached
            yield TaskUpdate(
                task=self.task,
                subtask=self.subtask,
                event=TaskUpdateEvent.MAX_ITERATIONS_REACHED,
            )

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
        ENTER CONCLUSION STAGE NOW: Synthesize all the information gathered and finish the this subtask.
        """
        self.history.append(Message(role="user", content=transition_message))

        # Log the transition in trace
        if self.enable_tracing:
            self._log_trace_event(
                "stage_transition",
                {
                    "from_stage": "tool_calling",
                    "to_stage": "conclusion",
                    "message": transition_message,
                    "iterations_completed": self.iterations,
                },
            )

    async def _process_iteration(
        self,
    ) -> Message:
        """
        Process a single iteration of the task.
        """

        tools = (
            self.tools
            if not self.in_conclusion_stage
            else [
                t
                for t in self.tools
                if t.name == "finish_subtask" or t.name == "finish_task"
            ]
        )
        # Only read files in the workspace directory
        tools = list(tools) + [ReadWorkspaceFileTool(self.workspace_dir)]

        # Create a dictionary to track unique tools by name
        unique_tools = {tool.name: tool for tool in tools}
        tools = list(unique_tools.values())

        message = await self.provider.generate_message(
            messages=self.history,
            model=self.model,
            tools=tools,
            use_code_interpreter=self.subtask.use_code_interpreter,
        )

        # Check if the message contains output files and use them as subtask output
        if hasattr(message, "output_files") and message.output_files:
            # Log that we're using output files from message response
            if self.enable_tracing:
                self._log_trace_event(
                    "using_message_output_files",
                    {
                        "num_files": len(message.output_files),
                        "mime_types": [file.mime_type for file in message.output_files],
                    },
                )

            # Use the first output file as the subtask output
            output_file = resolve_workspace_path(
                self.workspace_dir, self.subtask.output_file
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

        # Log the message in trace
        if self.enable_tracing and isinstance(message.content, str):
            self._log_trace_event(
                "message",
                {
                    "direction": "incoming",
                    "content": message.content,
                    "role": "assistant",
                },
            )

        if message.tool_calls:
            messages = await asyncio.gather(
                *[self._handle_tool_call(tool_call) for tool_call in message.tool_calls]
            )
            self.history.extend(messages)

        return message

    async def _handle_tool_call(self, tool_call: ToolCall) -> Message:
        """
        Handle a tool call.

        Args:
            chunk (ToolCall): The tool call to handle
        """
        # Log tool call in trace
        if self.enable_tracing:
            self._log_trace_event(
                "tool_call",
                {
                    "direction": "incoming",
                    "tool_name": tool_call.name,
                    "tool_args": tool_call.args,
                    "tool_id": tool_call.id,
                },
            )

        # Increment tool call counter
        self.tool_call_count += 1

        args_json = json.dumps(tool_call.args)[:100]
        print(f"Executing tool: {tool_call.name} with {args_json}")

        tool_result = await self._execute_tool(tool_call)

        # Track sources for data lineage
        if tool_call.name == "google_search" and isinstance(tool_call.args, dict):
            search_query = tool_call.args.get("query", "")
            if search_query:
                self.sources.append(f"search://{search_query}")

        elif tool_call.name == "browser" and isinstance(tool_call.args, dict):
            action = tool_call.args.get("action", "")
            url = tool_call.args.get("url", "")
            if action == "navigate" and url:
                self.sources.append(url)

        elif tool_call.name == "read_workspace_file" and isinstance(
            tool_call.args, dict
        ):
            file_path = tool_call.args.get("path", "")
            if file_path:
                self.sources.append(f"file://{file_path}")

        # Log tool result in trace
        if self.enable_tracing:
            self._log_trace_event(
                "tool_result",
                {
                    "direction": "outgoing",
                    "tool_name": tool_call.name,
                    "tool_id": tool_call.id,
                    "result": tool_result.result,
                },
            )

        # Handle finish_subtask tool specially
        if tool_call.name == "finish_task":
            self.subtask.completed = True
            self._save_to_output_file(tool_result.result)
            self.subtask.end_time = int(time.time())
            if self.enable_tracing:
                self._log_trace_event(
                    "task_completed",
                    {
                        "output_file": self.subtask.output_file,
                    },
                )

        if tool_call.name == "finish_subtask":
            self.subtask.completed = True
            self._save_to_output_file(tool_result.result)
            # Record the end time when the subtask is completed
            self.subtask.end_time = int(time.time())

            if self.enable_tracing:
                self._log_trace_event(
                    "subtask_completed",
                    {
                        "output_file": self.subtask.output_file,
                        "duration_seconds": self.subtask.end_time
                        - self.subtask.start_time,
                    },
                )

        # Add the tool result to history
        return Message(
            role="tool",
            tool_call_id=tool_result.id,
            name=tool_call.name,
            content=json.dumps(tool_result.result),
        )

    async def _handle_max_iterations_reached(self):
        """
        Handle the case where max iterations are reached without completion.

        Returns:
            ToolCall: A tool call representing the summary result
        """
        default_result = await self.request_final_output()
        self._save_to_output_file(default_result)
        self.subtask.completed = True
        self.subtask.end_time = int(time.time())

    async def _execute_tool(self, tool_call: ToolCall) -> ToolCall:
        """
        Execute a tool call using the available tools.

        Args:
            tool_call (ToolCall): The tool call to execute

        Returns:
            ToolCall: The tool call with the result attached
        """
        for tool in self.tools:
            if tool.name == tool_call.name:
                try:
                    result = await tool.process(self.processing_context, tool_call.args)

                    return ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        result=result,
                    )
                except Exception as e:
                    return ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        args=tool_call.args,
                        result={"error": str(e)},
                    )

        # Tool not found
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result={"error": f"Tool '{tool_call.name}' not found"},
        )

    async def request_final_output(self) -> str:
        """
        Request a final summary from the LLM when max iterations are reached.
        This is used when the subtask has gone through both the Tool Calling Stage
        and the Conclusion Stage but still failed to complete properly.
        """
        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with provide the final output for the subtask.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        The subtask has reached the maximum allowed iterations ({self.max_iterations}).
        Provide the final output for the subtask.
        The output should be in the in the format: {self.subtask.output_type}
        """

        # Create a minimal history with just the system prompt and summary request
        summary_history = [
            Message(role="system", content=summary_system_prompt),
            *self.history[1:],
            Message(role="user", content=summary_user_prompt),
        ]

        # Get response without tools
        message = await self.provider.generate_message(
            messages=summary_history,
            model=self.model,
            tools=[],
        )

        return str(message.content)
