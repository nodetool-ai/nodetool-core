from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.providers.ollama import OllamaProvider
from nodetool.chat.tools import Tool
from nodetool.chat.tools.workspace import WorkspaceBaseTool
from nodetool.metadata.types import FunctionModel, Message, SubTask, Task, ToolCall
from nodetool.workflows.processing_context import ProcessingContext

import tiktoken
import yaml


import json
import os
import time
from typing import Any, AsyncGenerator, List, Union

from nodetool.workflows.processing_context import ProcessingContext


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
        system_prompt: str,
        tools: List[Tool],
        model: str,
        provider: ChatProvider,
        workspace_dir: str,
        print_usage: bool = True,
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
            workspace_dir (str): The workspace directory
            print_usage (bool): Whether to print token usage
            max_token_limit (int): Maximum token limit before summarization
            max_iterations (int): Maximum iterations for the subtask
            enable_tracing (bool): Whether to enable LLM message tracing
        """
        self.task = task
        self.subtask = subtask
        self.processing_context = processing_context
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
        self.provider = provider
        self.workspace_dir = workspace_dir
        self.max_token_limit = max_token_limit
        self.enable_tracing = enable_tracing

        # Initialize isolated message history for this subtask
        self.history = [Message(role="system", content=self.system_prompt)]

        # Track iterations for this subtask
        self.iterations = 0
        if max_iterations < 3:
            raise ValueError("max_iterations must be at least 3")
        self.max_iterations = max_iterations
        self.max_tool_calling_iterations = max_iterations - 2

        # Track tool calls for this subtask
        self.tool_call_count = 0
        # Default max tool calls if not specified in subtask
        self.max_tool_calls = getattr(subtask, "max_tool_calls", float("inf"))

        # Track progress for this subtask
        self.progress = []

        # Flag to track which stage we're in - normal execution flow for all tasks
        self.in_conclusion_stage = False

        self.print_usage = print_usage
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Ensure the output file path is already in the /workspace format
        if not self.subtask.output_file.startswith("/workspace/"):
            self.subtask.output_file = os.path.join(
                "/workspace", self.subtask.output_file.lstrip("/")
            )

        # For the actual file system operations, strip the /workspace prefix
        self.output_file_path = os.path.join(
            self.workspace_dir, os.path.relpath(self.subtask.output_file, "/workspace")
        )

        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)

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
                    "max_tool_calls": self.max_tool_calls,
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
        Includes metadata in the appropriate format based on file type.
        """
        # Extract metadata from the result if it's a dictionary
        if isinstance(output, dict):
            metadata = output.pop("metadata", {})
            result = output.get("result", output)
        else:
            metadata = {}
            result = output

        print(f"Saving result to {self.output_file_path}")
        is_markdown = self.output_file_path.endswith(".md")
        is_json = self.output_file_path.endswith(".json")

        with open(self.output_file_path, "w") as f:
            if is_markdown:
                # For Markdown files, add metadata as YAML frontmatter
                if metadata:
                    f.write("---\n")
                    yaml.dump(metadata, f)
                    f.write("---\n\n")

                f.write(str(result))
            elif is_json:
                output = {"data": result, "metadata": metadata}
                if metadata:
                    output["metadata"] = metadata
                json.dump(output, f, indent=2)
            else:
                # For string results being written to non-markdown files
                f.write(str(result))

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
        task_prompt: str,
    ) -> AsyncGenerator[Union[Chunk, ToolCall], None]:
        """
        ⚙️ Task Executor - Runs a single subtask to completion using appropriate strategy

        This execution path handles complex tasks requiring multiple steps:
        1. Tool Calling Stage: Multiple iterations of information gathering using any tools
        2. Conclusion Stage: Final synthesis with restricted access (only finish_subtask tool)

        The method automatically tracks iterations, manages token limits, and enforces
        transitions between stages based on limits or progress.

        Args:
            task_prompt (str): The task prompt with specific instructions

        Yields:
            Union[Chunk, ToolCall]: Live updates during task execution
        """
        # Record the start time of the subtask
        current_time = int(time.time())
        self.subtask.start_time = current_time

        # Create the enhanced prompt with two-stage explanation
        enhanced_prompt = self._create_enhanced_prompt(task_prompt)

        # Add the task prompt to this subtask's history
        self.history.append(Message(role="user", content=enhanced_prompt))

        # Log the task prompt in the trace
        if self.enable_tracing:
            self._log_trace_event(
                "message",
                {"direction": "outgoing", "content": enhanced_prompt, "role": "user"},
            )

        # Signal that we're executing this subtask
        print(f"Executing subtask: {self.subtask.content}")

        # Continue executing until the task is completed or max iterations reached
        while not self.subtask.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            token_count = self._count_tokens(self.history)

            # Check if we need to transition to conclusion stage
            if (
                self.iterations > self.max_tool_calling_iterations
                and not self.in_conclusion_stage
            ):
                await self._transition_to_conclusion_stage()

            # Check if token count exceeds limit and summarize if needed
            if token_count > self.max_token_limit:
                await self._handle_token_limit_exceeded(token_count)
                token_count = self._count_tokens(self.history)

            # Process current iteration
            async for chunk in self._process_iteration():
                yield chunk

        # If we've reached the last iteration and haven't completed yet, generate summary
        if self.iterations >= self.max_iterations and not self.subtask.completed:
            await self._handle_max_iterations_reached()

        # print(self.provider.usage)

    def _create_enhanced_prompt(self, task_prompt: str) -> str:
        """
        Create an enhanced prompt with two-stage execution explanation.

        Appends instructions about the two-stage execution model to the original task prompt,
        helping the LLM understand the constraints and expectations of each stage.

        Args:
            task_prompt (str): The original task prompt

        Returns:
            str: The enhanced prompt with two-stage execution instructions
        """
        return (
            task_prompt
            + """
        
        IMPORTANT: This task will be executed in TWO STAGES:
        
        STAGE 1: TOOL CALLING STAGE
        - You may use any available tools to gather information and make progress
        - This stage is limited to a maximum of """
            + str(self.max_tool_calling_iterations)
            + """ iterations
        - Be efficient with your tool calls and focus on making meaningful progress
        - After """
            + str(self.max_tool_calling_iterations)
            + """ iterations, you will automatically transition to Stage 2
        
        STAGE 2: CONCLUSION STAGE
        - You will ONLY have access to the finish_subtask tool
        - You must synthesize your findings and complete the task
        - No further information gathering will be possible
        
        Your goal is to complete the task in as few iterations as possible while producing high-quality results.
        """
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

        # Add transition message to history
        transition_message = """
        STAGE 1 (TOOL CALLING) COMPLETE ⚠️
        
        You have reached the maximum number of iterations for the tool calling stage.
        
        ENTERING STAGE 2: CONCLUSION STAGE
        
        In this stage, you ONLY have access to the finish_subtask tool.
        You must now synthesize all the information you've gathered and complete the task.
        
        Please summarize what you've learned, draw conclusions, and use the finish_subtask
        tool to save your final result.
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

    async def _handle_token_limit_exceeded(self, token_count: int) -> None:
        """
        Handle the case where token count exceeds the limit.

        This method:
        1. Logs the token overflow situation
        2. Triggers context summarization to reduce token count
        3. Logs the new token count after summarization

        Args:
            token_count (int): The current token count that exceeded the limit
        """
        print(
            f"Token count ({token_count}) exceeds limit ({self.max_token_limit}). Summarizing context..."
        )
        await self._summarize_context()
        print(
            f"After summarization: {self._count_tokens(self.history)} tokens in context"
        )

    async def _process_iteration(self) -> AsyncGenerator[Union[Chunk, ToolCall], None]:
        """
        Process a single iteration of the task.

        An iteration consists of:
        1. Selecting appropriate tools based on current stage
        2. Generating a response from the LLM
        3. Processing each chunk or tool call from the response
        4. Updating conversation history with results

        Yields:
            Union[Chunk, ToolCall]: Live updates during iteration processing
        """
        # Get the appropriate tools based on the stage
        current_tools = (
            [tool for tool in self.tools if tool.name == "finish_subtask"]
            if self.in_conclusion_stage
            else self.tools
        )

        # Generate response
        generator = self.provider.generate_messages(
            messages=self.history,
            model=self.model,
            tools=current_tools,
        )

        async for chunk in generator:  # type: ignore
            yield chunk

            if isinstance(chunk, Chunk):
                await self._handle_chunk(chunk)
            elif isinstance(chunk, ToolCall):
                await self._handle_tool_call(chunk)

    async def _handle_chunk(self, chunk: Chunk) -> None:
        """
        Handle a response chunk.

        Args:
            chunk (Chunk): The chunk to handle
        """
        # Update history with assistant message
        if (
            len(self.history) > 0
            and self.history[-1].role == "assistant"
            and isinstance(self.history[-1].content, str)
        ):
            # Update existing assistant message
            self.history[-1].content += chunk.content

            # Log the chunk in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "chunk",
                    {
                        "direction": "incoming",
                        "content": chunk.content,
                        "role": "assistant",
                        "is_partial": True,
                    },
                )
        else:
            # Add new assistant message
            self.history.append(Message(role="assistant", content=chunk.content))

            # Log the chunk in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "chunk",
                    {
                        "direction": "incoming",
                        "content": chunk.content,
                        "role": "assistant",
                        "is_partial": False,
                    },
                )

    async def _handle_tool_call(self, chunk: ToolCall) -> None:
        """
        Handle a tool call.

        Args:
            chunk (ToolCall): The tool call to handle
        """
        # Add tool call to history
        self.history.append(
            Message(
                role="assistant",
                tool_calls=[chunk],
            )
        )

        # Log tool call in trace
        if self.enable_tracing:
            self._log_trace_event(
                "tool_call",
                {
                    "direction": "incoming",
                    "tool_name": chunk.name,
                    "tool_args": chunk.args,
                    "tool_id": chunk.id,
                },
            )

        # Increment tool call counter
        self.tool_call_count += 1

        print(f"Executing tool: {chunk.name}")
        tool_result = await self._execute_tool(chunk)

        # Log tool result in trace
        if self.enable_tracing:
            self._log_trace_event(
                "tool_result",
                {
                    "direction": "outgoing",
                    "tool_name": chunk.name,
                    "tool_id": chunk.id,
                    "result": tool_result.result,
                },
            )

        # Handle finish_subtask tool specially
        if chunk.name == "finish_subtask":
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
        self.history.append(
            Message(
                role="tool",
                tool_call_id=tool_result.id,
                name=chunk.name,
                content=json.dumps(tool_result.result),
            )
        )

        # Check if we've reached the tool call limit and force conclusion stage if needed
        if (
            not self.in_conclusion_stage
            and self.tool_call_count >= self.max_tool_calls
            and chunk.name != "finish_subtask"
        ):
            await self._force_conclusion_stage_tool_limit()

    async def _force_conclusion_stage_tool_limit(self) -> None:
        """
        Force transition to conclusion stage due to reaching max tool calls.
        """
        self.in_conclusion_stage = True

        # Add transition message to history
        transition_message = f"""
        MAXIMUM TOOL CALLS REACHED ⚠️
        
        You have reached the maximum number of allowed tool calls ({self.max_tool_calls}).
        
        ENTERING STAGE 2: CONCLUSION STAGE
        
        In this stage, you ONLY have access to the finish_subtask tool.
        You must now synthesize all the information you've gathered and complete the task.
        
        Please summarize what you've learned, draw conclusions, and use the finish_subtask
        tool to save your final result.
        """
        self.history.append(Message(role="user", content=transition_message))

        print(
            f"  Reached maximum tool calls ({self.max_tool_calls}). Transitioning to conclusion stage."
        )

    async def _handle_max_iterations_reached(self):
        """
        Handle the case where max iterations are reached without completion.

        Returns:
            ToolCall: A tool call representing the summary result
        """
        default_result = await self.request_summary()
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

    async def request_summary(self) -> str:
        """
        Request a final summary from the LLM when max iterations are reached.
        This is used when the subtask has gone through both the Tool Calling Stage
        and the Conclusion Stage but still failed to complete properly.
        """
        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with providing a concise summary of work completed so far.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        The subtask has reached the maximum allowed iterations ({self.max_iterations}).
        Summarize the work completed so far, in detail.
        """

        # Create a minimal history with just the system prompt and summary request
        summary_history = [
            Message(role="system", content=summary_system_prompt),
            *self.history[1:],
            Message(role="user", content=summary_user_prompt),
        ]

        # Get response without tools
        generator = self.provider.generate_messages(
            messages=summary_history,
            model=self.model,
            tools=[],  # No tools allowed for summary
        )

        summary_content = ""
        async for chunk in generator:  # type: ignore
            if isinstance(chunk, Chunk):
                summary_content += chunk.content

        return summary_content

    async def _summarize_context(self) -> None:
        """
        Summarize the conversation history to reduce token count.

        This token management method:
        1. Preserves the system prompt (always kept intact)
        2. Preserves the current execution stage information
        3. Uses the LLM to summarize conversation to ~50% of original length
        4. Replaces detailed history with the compressed summary
        5. Maintains all critical information needed to continue execution

        This helps prevent context overflow while preserving execution continuity.
        """
        # Keep the system prompt
        system_prompt = self.history[0]

        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with creating a detailed summary of the conversation so far.
        Do not compress too aggressively - aim for about 30% reduction, not more.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        Please summarize the conversation history for the subtask so far.
        
        Include:
        1. The original task/objective in full detail
        2. All key information discovered
        3. Any important context or details that would be needed to continue the task
        """

        # Create a minimal history with just the system prompt and summary request
        summary_history = [
            Message(role="system", content=summary_system_prompt),
            Message(
                role="user",
                content=summary_user_prompt
                + "\n\nHere's the conversation to summarize:\n"
                + "\n".join(
                    [
                        f"{msg.role}: {msg.content}"
                        for msg in self.history[1:]
                        if hasattr(msg, "content") and msg.content
                    ]
                ),
            ),
        ]

        # Get response without tools
        generator = self.provider.generate_messages(
            messages=summary_history,
            model=self.model,
            tools=[],  # No tools allowed for summary
        )

        summary_content = ""
        async for chunk in generator:  # type: ignore
            if isinstance(chunk, Chunk):
                summary_content += chunk.content

        # Replace history with system prompt and summary
        self.history = [
            system_prompt,
            Message(
                role="user",
                content=f"CONVERSATION HISTORY (SUMMARIZED TO ~50% LENGTH):\n{summary_content}\n\nPlease continue with the task based on this detailed summary.",
            ),
        ]


class FinishSubTaskTool(WorkspaceBaseTool):
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
            "source": "calculation",
            "timestamp": "2023-06-15T10:30:00Z"
        }
    }
    """

    name = "finish_subtask"
    description = """
    Finish a subtask by saving its final result to a file in the workspace.
    This tool is the ONLY tool available during the Conclusion Stage.
    
    Use this when you have gathered sufficient information in the Tool Calling Stage
    and are ready to synthesize your findings into a final result.
    
    The result will be saved to the output_file path, defined in the subtask.
    """
    input_schema = {
        "type": "object",
        "properties": {
            "result": {
                "oneOf": [
                    {
                        "type": "object",
                        "description": "The final result of the subtask as a structured object",
                    },
                    {
                        "type": "string",
                        "description": "The final result of the subtask as a simple string",
                    },
                ],
                "description": "The final result of the subtask (can be an object or string)",
            },
            "metadata": {
                "type": "object",
                "description": "Metadata for the result",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the result",
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
                    "source": {
                        "type": "string",
                        "description": "The source of the result",
                        "enum": ["search", "website", "file", "reasoning", "other"],
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL of the data source",
                    },
                },
                "required": ["title", "description", "source", "url"],
            },
        },
        "required": ["result", "metadata"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        """
        Save the subtask result to a file and mark the subtask as finished.

        Args:
            context (ProcessingContext): The processing context
            params (dict): Parameters containing the resultand metadata

        Returns:
            dict: Response containing the file path where the result was stored
        """
        result = params.get("result", {})
        metadata = params.get("metadata", {})

        return {
            "result": result,
            "metadata": metadata,
        }
