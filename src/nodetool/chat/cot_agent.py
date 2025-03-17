"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A TaskPlanner class that creates a task list with dependencies
2. A TaskExecutor class that executes tasks in the correct order
3. A CoTAgent class that combines planning and execution (legacy)
4. Integration with the existing provider and tool system
5. Support for streaming results during reasoning

Features:
- Three-phase approach: planning, dependency identification, and execution
- Step-by-step reasoning with tool use capabilities
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Streaming results during the reasoning process
- Configurable reasoning steps and prompt templates
- Task dependency management for proper execution order
"""

import datetime
import json
import os
import shutil
import subprocess
import platform
import tiktoken
from typing import AsyncGenerator, List, Optional, Union, Dict, Any, Set, Tuple

from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    Message,
    ToolCall,
    FunctionModel,
    Task,
    SubTask,
    TaskPlan,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.tools.workspace import WorkspaceBaseTool


class CreateTaskPlanTool(Tool):
    """
    A tool that creates a task plan.
    """

    name = "create_task_plan"
    description = "Create a task plan for the given objective"
    input_schema = TaskPlan.model_json_schema()

    _task_plan: TaskPlan | None = None

    async def process(self, context: ProcessingContext, params: dict) -> str:
        """
        Create a task plan for the given objective.
        """
        self._task_plan = TaskPlan(**params)
        return "Task plan created successfully"

    def get_task_plan(self) -> TaskPlan:
        """
        Get the task plan.
        """
        if self._task_plan is None:
            raise ValueError("Task plan not created")
        return self._task_plan


class FinishSubTaskTool(WorkspaceBaseTool):
    """
    A tool that finishes a subtask.
    """

    name = "finish_subtask"
    description = """
    Finish a subtask with its final result. 
    Use this when you have completed all necessary work for a subtask.
    Provide the full result of the subtask as the argument to the tool.
    The result will be stored and retrieved for subsequent tasks.
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
            },
        },
        "required": ["result", "metadata"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        """
        Mark a subtask as finished with its final result.

        Args:
            context (ProcessingContext): The processing context
            params (dict): Parameters containing the result and optional file_path

        Returns:
            dict: Response containing either the result or the file_path where result was stored
        """
        return {
            "result": params.get("result"),
            "metadata": params.get("metadata"),
        }


# Add these constants at the top of the file, after the imports
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are a strategic planning assistant that creates precise, efficient task plans.

PLANNING INSTRUCTIONS:
1. Create MINIMAL plans with only essential tasks - fewer tasks are better.
2. Scale plan complexity to match objective complexity - simple objectives need 1-3 tasks.
3. Document your reasoning in the "thoughts" field with clear analysis.
4. Merge related actions into single tasks - avoid fragmentation.
5. Order subtasks logically - they execute sequentially in the order provided.
6. Use descriptive IDs for each subtask (e.g., "research_topic", "analyze_data").
7. Define dependencies using exact subtask IDs.
8. Set "output_type": "text" for results that has prose, formatting, or other non-structured data.
9. Set "output_type": "object" for results that has structured data.
10. Make sure to save artifacts in the workspace directory.
11. A good artifact is Markdown file for sub reports, or a CSV file for data analysis.
12. Set "thinking" for subtasks that require reasoning.
13. Set "thinking": false to focus on executing tools.
14. Eliminate redundant steps - optimize for efficiency.
15. Use absolute paths from workspace root for all file operations.

Use the CreateTaskPlanTool to submit your final task plan.

TASK PLAN FORMAT:
```json
{
    "type": "task_plan",
    "title": "Concise Objective Title",
    "thoughts": "Your strategic analysis and reasoning for this plan structure",
    "tasks": [
        {
            "type": "task",
            "title": "Task Title",
            "subtasks": [
                {
                    "type": "subtask",
                    "id": "descriptive_id",
                    "content": "Specific action to perform",
                    "thinking": false,
                    "output_type": "text",
                    "dependencies": ["prior_subtask_id"]
                }
            ]
        }
    ]
}
```
"""

DEFAULT_EXECUTION_SYSTEM_PROMPT = """
You are a task execution agent that completes subtasks efficiently.

WORKSPACE CONSTRAINTS:
- All file operations must use the workspace directory as root
- Never attempt to access files outside the workspace directory
- Use absolute paths from workspace root for all file operations
- Make sure to save artifacts in the workspace directory.

EXECUTION CONTEXT:
- You are executing a single subtask from a larger task plan
- You have access to specific tools defined in the subtask
- Results from dependency subtasks are provided when available
- Each subtask must be completed with a call to finish_subtask

EXECUTION PROTOCOL:
1. Focus exclusively on the current subtask - ignore other subtasks
2. Always call finish_subtask when the subtask is complete
3. You can make 0-5 tool calls per subtask

WEB CRAWLING:
- You can use `google_search` tool to find web pages
- You can fetch web pages with `browser_control` tool
- You can fetch multiple pages with `batch_download` tool
- Try to batch as many pages as possible to save time

RESULT REQUIREMENTS:
1. Results can be a string, a JSON object
2. Pick the output type defined in the subtask
3. Include metadata with these fields:
   - title: Descriptive title of the result
   - description: Detailed description of what was accomplished
   - source: Origin of the data (e.g., URL, calculation, file)
4. Call finish_subtask with a result object or string

MULTIMEDIA:
1. Link to images, videos, and other files in the result
2. Use screenshot tool to capture images

CITING:
1. Subtask results include metadata with source information
2. You should cite the source of the data in the result
3. Provide many links in the final report

"""


class WorkspaceManager:
    """
    Manages the workspace for the agent.
    """

    workspace_root = "/tmp/nodetool-workspaces"

    def __init__(self):
        os.makedirs(self.workspace_root, exist_ok=True)
        self.create_new_workspace()

    def create_new_workspace(self):
        """Creates a new workspace with a unique name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = f"workspace_{timestamp}"
        workspace_path = os.path.join(self.workspace_root, workspace_name)
        os.makedirs(workspace_path, exist_ok=True)
        self.current_workspace = workspace_path

    async def execute_command(self, cmd: str) -> str:
        """
        Execute workspace commands in a controlled environment.

        This method parses and executes file system commands within the workspace
        boundary, preventing operations outside the allowed workspace directory.
        All paths are validated to ensure they remain within the workspace.

        Supported commands:
            - pwd/cwd: Show current working directory
            - ls [path]: List directory contents
            - cd [path]: Change directory
            - mkdir [path]: Create directory
            - rm [-r/-rf] [path]: Remove file or directory
            - open [path]: Open file with system default application

        Args:
            cmd (str): The command to execute

        Returns:
            str: The result of the command execution or error message
        """
        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command == "pwd" or command == "cwd":
            return self.current_workspace or ""

        elif command == "ls":
            path = (
                os.path.join(self.current_workspace, *args)
                if args
                else self.current_workspace
            )
            try:
                items = os.listdir(path)
                return "\n".join(items)
            except Exception as e:
                return f"Error: {str(e)}"

        elif command == "cd":
            if not args:
                return "Error: Missing directory argument"
            new_path = os.path.join(self.current_workspace, args[0])
            if not os.path.exists(new_path):
                return f"Error: Directory {args[0]} does not exist"
            if not new_path.startswith(self.workspace_root):
                return "Error: Cannot navigate outside workspace"
            self.current_workspace = new_path
            return f"Changed directory to {new_path}"

        elif command == "mkdir":
            if not args:
                return "Error: Missing directory name"
            try:
                os.makedirs(
                    os.path.join(self.current_workspace, args[0]), exist_ok=True
                )
                return f"Created directory {args[0]}"
            except Exception as e:
                return f"Error creating directory: {str(e)}"

        elif command == "rm":
            if not args:
                return "Error: Missing path argument"
            path = os.path.join(self.current_workspace, args[0])
            if not path.startswith(self.workspace_root):
                return "Error: Cannot remove files outside workspace"
            try:
                if os.path.isdir(path):
                    if "-r" in args or "-rf" in args:
                        shutil.rmtree(path)
                    else:
                        os.rmdir(path)
                else:
                    os.remove(path)
                return f"Removed {args[0]}"
            except Exception as e:
                return f"Error removing {args[0]}: {str(e)}"

        elif command == "open":
            if not args:
                return "Error: Missing file argument"
            path = os.path.join(self.current_workspace, args[0])
            if not os.path.exists(path):
                return f"Error: File {args[0]} does not exist"
            try:
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", path])
                elif platform.system() == "Windows":  # Windows
                    os.startfile(path)  # type: ignore
                else:  # linux variants
                    subprocess.run(["xdg-open", path])
                return f"Opened {args[0]}"
            except Exception as e:
                return f"Error opening file: {str(e)}"

        return f"Unknown command: {command}"


class TaskPlanner:
    """
    Creates a task plan with dependencies using a language model.

    This class handles the planning phase of the Chain of Thought process,
    breaking down a problem into tasks and subtasks, and identifying
    dependencies between them.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        objective: str,
        system_prompt: str | None = None,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            objective (str): The objective to solve
            system_prompt (str, optional): Custom system prompt
            user_prompt (str, optional): Custom user prompt
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.task_plan = None
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_PLANNING_SYSTEM_PROMPT
        )
        self.user_prompt = """
        Create an efficient task plan for the following objective:

        {objective}

        Analyze requirements, think carefully about the problem, and then output the tasks and subtasks in the required JSON format.
        """

        self.user_prompt = f"Objective to solve: {objective}\n\n{self.user_prompt}"
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding

    async def create_plan(self) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        Create a task plan for the given problem using the LLM.

        This method:
        1. Sends a system prompt and user prompt to the LLM
        2. Uses CreateTaskPlanTool to capture a structured task plan
        3. Yields chunks and tool calls during plan generation
        4. Populates self.task_plan with the final plan structure

        The method relies on the LLM to properly invoke the CreateTaskPlanTool
        with valid TaskPlan data. If the LLM fails to do so, task_plan may remain None.

        Yields:
            Union[Message, Chunk]: Generation chunks during plan creation

        Raises:
            ValueError: If the plan couldn't be created after generation completes
        """
        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.user_prompt),
        ]

        # Generate plan
        task_plan_tool = CreateTaskPlanTool("/tmp")
        chunks = []
        generator = self.provider.generate_messages(
            messages=history,
            model=self.model,
            tools=[task_plan_tool],
            thinking=True,
        )

        async for chunk in generator:  # type: ignore
            chunks.append(chunk)
            if isinstance(chunk, Chunk):
                yield chunk
            elif isinstance(chunk, ToolCall):
                print(chunk)
                self.task_plan = TaskPlan(**chunk.args)
                break


class SubTaskContext:
    """
    Encapsulates the execution context for a single subtask.

    This class maintains an isolated conversation history for each subtask,
    ensuring that subtasks don't share message history and can be executed
    independently while still being able to access results from dependencies.
    """

    def __init__(
        self,
        task: Task,
        subtask: SubTask,
        system_prompt: str,
        tools: List[Tool],
        model: FunctionModel,
        provider: ChatProvider,
        workspace_dir: str,
        result_store: Dict[str, Any],
        print_usage: bool = True,
        max_token_limit: int = 20000,
    ):
        """
        Initialize a subtask execution context.

        Args:
            task (Task): The task to execute
            subtask (SubTask): The subtask to execute
            system_prompt (str): The system prompt for this subtask
            tools (List[Tool]): Tools available to this subtask
            model (FunctionModel): The model to use for this subtask
            provider (ChatProvider): The provider to use for this subtask
            workspace_dir (str): The workspace directory
            result_store (Dict[str, Any]): Global dictionary to store results
            print_usage (bool): Whether to print token usage
            max_token_limit (int): Maximum token limit before summarization
        """
        self.task = task
        self.subtask = subtask
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
        self.provider = provider
        self.workspace_dir = workspace_dir
        self.result_store = result_store  # Reference to global result store
        self.max_token_limit = max_token_limit

        # Initialize isolated message history for this subtask
        self.history = [Message(role="system", content=system_prompt)]

        # Track iterations for this subtask
        self.iterations = 0
        self.max_iterations = 10

        # Track progress for this subtask
        self.progress = []

        # Flag to track if subtask is finished
        self.completed = False

        self.print_usage = print_usage
        self.encoding = tiktoken.get_encoding("cl100k_base")

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

    async def execute(
        self, task_prompt: str, dependency_results: Dict[str, Any] = {}
    ) -> AsyncGenerator[Union[Chunk, ToolCall], None]:
        """
        Execute this subtask with its isolated context.

        This method:
        1. Creates an enhanced prompt with dependency results
        2. Manages an isolated conversation with the LLM for this subtask
        3. Handles tool calls, including the finish_subtask tool
        4. Stores results in the global result_store
        5. Enforces maximum iterations to prevent infinite loops
        6. Auto-completes the subtask if max iterations are reached
        7. Summarizes context if token count exceeds the limit

        The method maintains contextual isolation to prevent context overflow
        across different subtasks, while allowing access to dependency results.

        Args:
            task_prompt (str): The base prompt for this subtask
            dependency_results (Dict[str, Any], optional): Results from dependencies

        Yields:
            Union[Chunk, ToolCall]: Chunks of text or tool calls during execution

        Warning:
            If max_iterations is reached without the subtask being completed via
            the finish_subtask tool, the method will auto-complete the subtask
            with a summary of progress.
        """
        # Create the task prompt with dependency context
        enhanced_prompt = task_prompt

        # Include dependency results if available
        if dependency_results:
            enhanced_prompt += f"\n\nDependencies: {json.dumps(dependency_results)}"

        # Add the task prompt to this subtask's history
        self.history.append(Message(role="user", content=enhanced_prompt))

        # Signal that we're executing this subtask
        print(f"Executing task: {self.task.title} - {self.subtask.content}")

        # Continue executing until the task is completed or max iterations reached
        while not self.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            token_count = self._count_tokens(self.history)
            print(
                f"  Iteration {self.iterations}/{self.max_iterations} for subtask {self.subtask.id}"
            )
            print(
                f"\n[Debug: {token_count} tokens in context for subtask {self.subtask.id}]\n"
            )
            print(self.provider.usage)

            # Check if token count exceeds limit and summarize if needed
            if token_count > self.max_token_limit:
                print(
                    f"Token count ({token_count}) exceeds limit ({self.max_token_limit}). Summarizing context..."
                )
                await self._summarize_context()
                token_count = self._count_tokens(self.history)
                print(f"After summarization: {token_count} tokens in context")

            # Get response for this subtask using its isolated history
            generator = self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=self.tools,
            )

            async for chunk in generator:  # type: ignore
                yield chunk

                if isinstance(chunk, Chunk):
                    # Update history with assistant message
                    if (
                        len(self.history) > 0
                        and self.history[-1].role == "assistant"
                        and isinstance(self.history[-1].content, str)
                    ):
                        # Update existing assistant message
                        self.history[-1].content += chunk.content
                    else:
                        # Add new assistant message
                        self.history.append(
                            Message(role="assistant", content=chunk.content)
                        )

                elif isinstance(chunk, ToolCall):
                    # Add tool call to history
                    self.history.append(
                        Message(
                            role="assistant",
                            tool_calls=[chunk],
                        )
                    )

                    # Execute the tool call
                    tool_result = await self._execute_tool(chunk)

                    print(f"Tool call: {chunk.name}")
                    print(f"Tool result: {tool_result.result}")

                    # Handle finish_subtask tool specially
                    if chunk.name == "finish_subtask":
                        # Get the result from the tool call
                        result = chunk.args.get("result", {})

                        # Store in the global result store
                        self.result_store[self.subtask.id] = result

                        # Mark subtask as finished
                        self.completed = True

                        # Add completion notice to progress
                        print(f"Subtask {self.subtask.id} completed.")

                    # Add the tool result to history
                    self.history.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            name=chunk.name,
                            content=json.dumps(tool_result.result),
                        )
                    )

            # If we've reached the last iteration and haven't completed yet, generate summary
            if self.iterations >= self.max_iterations and not self.completed:
                default_result = await self.request_summary()

                # Store in the global result store
                self.result_store[self.subtask.id] = default_result
                self.completed = True

                yield ToolCall(
                    id=f"{self.subtask.id}_max_iterations_reached",
                    name="finish_subtask",
                    args=default_result,
                )

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
                from nodetool.workflows.processing_context import ProcessingContext

                context = ProcessingContext(user_id="cot_agent", auth_token="")
                result = await tool.process(context, tool_call.args)
                return ToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    args=tool_call.args,
                    result=result,
                )

        # Tool not found
        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result={"error": f"Tool '{tool_call.name}' not found"},
        )

    async def request_summary(self) -> dict:
        """
        Request a final summary from the LLM when max iterations are reached.
        This uses a modified system prompt to specifically ask for a summary
        without tool calls, while including the full subtask history.
        """
        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with providing a concise summary of work completed so far.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        The subtask '{self.subtask.id}' has reached the maximum allowed iterations ({self.max_iterations}).
        Please provide a brief summary of:
        1. What has been accomplished
        2. What remains to be done
        3. Any blockers or issues encountered
        
        Respond with a clear, concise summary only.
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

        # Create a structured result with the summary
        summary_result = {
            "status": "max_iterations_reached",
            "message": f"Reached maximum iterations ({self.max_iterations}) for subtask {self.subtask.id}",
            "summary": summary_content,
        }

        return summary_result

    async def _summarize_context(self) -> None:
        """
        Summarize the conversation history to reduce token count.

        This method:
        1. Preserves the system prompt
        2. Summarizes the conversation history to approximately half its original length
        3. Replaces the history with the system prompt and summary
        """
        # Keep the system prompt
        system_prompt = self.history[0]

        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with creating a detailed summary of the conversation so far.
        Maintain approximately 50% of the original content length.
        Include all important information, decisions made, and current state.
        Your summary will replace the detailed conversation history to reduce token usage.
        Do not compress too aggressively - aim for about 50% reduction, not more.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        Please summarize the conversation history for subtask '{self.subtask.id}' so far.
        
        IMPORTANT: Create a summary that is approximately 50% of the original length.
        
        Include:
        1. The original task/objective in full detail
        2. All key information discovered
        3. All actions taken and their results
        4. Current state and what needs to be done next
        5. Any important context or details that would be needed to continue the task
        
        Do not compress too aggressively. Maintain approximately 50% of the original content.
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


class TaskExecutor:
    """
    Executes tasks from a TaskPlan using isolated execution contexts.

    This class handles the execution phase of the Chain of Thought process,
    executing tasks in the correct order based on their dependencies. Each subtask
    has its own isolated execution context to prevent context window overflow.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        workspace_dir: str,
        tools: List[Tool],
        task_plan: TaskPlan,
        system_prompt: str | None = None,
        max_steps: int = 50,
        max_subtask_iterations: int = 10,
        max_token_limit: int = 8000,  # Default token limit
    ):
        """
        Initialize the IsolatedTaskExecutor.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            workspace_dir (str): Directory for workspace files
            tools (List[Tool]): List of tools available for task execution
            task_plan (TaskPlan): The task list to execute
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum execution steps to prevent infinite loops
            max_subtask_iterations (int, optional): Maximum iterations allowed per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
        """
        self.provider = provider
        self.model = model
        self.workspace_dir = workspace_dir
        self.tools = tools
        self.task_plan = task_plan
        self.max_token_limit = max_token_limit

        # Prepare system prompt
        prefix = f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        Your workspace directory is: {self.workspace_dir}
        """
        self.system_prompt = prefix + (
            system_prompt if system_prompt else DEFAULT_EXECUTION_SYSTEM_PROMPT
        )

        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.result_store = {}

    async def execute_tasks(
        self,
        print_usage: bool = True,
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Execute the tasks in the provided task plan sequentially.

        Args:
            print_usage (bool, optional): Whether to print provider usage statistics

        Yields:
            Union[Message, Chunk, ToolCall]: Objects representing parts of the execution process
        """
        steps_taken = 0

        # Continue until all tasks are complete or we reach max steps
        while not self._all_tasks_complete() and steps_taken < self.max_steps:
            steps_taken += 1

            # Find all executable tasks
            executable_tasks = self._get_all_executable_tasks()

            if not executable_tasks:
                # If no tasks are executable but we're not done, there might be a dependency issue
                if not self._all_tasks_complete():
                    yield Chunk(
                        content="\nNo executable tasks but not all complete. Possible dependency cycle.\n",
                        done=False,
                    )
                break

            # Execute the first available executable task (sequential execution)
            task, subtask = executable_tasks[0]

            # Create subtask context
            context = SubTaskContext(
                task=task,
                subtask=subtask,
                system_prompt=self.system_prompt,
                tools=self.tools,
                model=self.model,
                provider=self.provider,
                workspace_dir=self.workspace_dir,
                result_store=self.result_store,
                print_usage=print_usage,
                max_token_limit=self.max_token_limit,  # Pass the token limit
            )

            # Prepare dependency results
            dependency_results = {}
            for dep_id in subtask.dependencies:
                if dep_id in self.result_store:
                    dependency_results[dep_id] = self.result_store[dep_id]

            # Create the task prompt
            task_prompt = self._create_task_prompt(task, subtask)

            # Start the subtask execution
            print(f"Executing subtask {subtask.id}: {subtask.content}")

            # Execute the subtask and forward messages directly
            async for message in context.execute(task_prompt, dependency_results):
                yield message

            subtask.completed = True

        # Print workspace files when agent completes all tasks
        if os.path.exists(self.workspace_dir):
            yield Chunk(
                content=f"\n\nWorkspace files in {self.workspace_dir}:\n",
                done=False,
            )
            for root, dirs, files in os.walk(self.workspace_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.workspace_dir)
                    yield Chunk(content=f"- {relative_path}\n", done=False)

    def _get_all_executable_tasks(self) -> List[Tuple[Task, SubTask]]:
        """
        Get all executable tasks from the task list, respecting dependencies.

        A subtask is considered executable when:
        1. It has not been completed yet
        2. All its dependencies (if any) have been completed

        This method efficiently checks each subtask by:
        1. Verifying its completion status
        2. Looking up all dependencies in the task plan
        3. Creating a list of tasks ready for execution

        Note that this implementation enforces sequential execution order
        based on defined dependencies, not parallel execution.

        Returns:
            List[Tuple[Task, SubTask]]: All executable tasks and subtasks

        Raises:
            ValueError: If a dependent task is referenced but not found in the plan
        """
        executable_tasks = []

        for task in self.task_plan.tasks:
            for subtask in task.subtasks:
                if not subtask.completed and subtask.id:
                    # Check if all dependencies are completed
                    all_dependencies_met = True
                    for dep_id in subtask.dependencies:
                        # Find the dependent task and check if it's completed
                        _, dependent_task = self.task_plan.find_task_by_id(dep_id)
                        if not dependent_task:
                            raise ValueError(f"Dependent task {dep_id} not found")
                        if not dependent_task.completed:
                            all_dependencies_met = False
                            break

                    if all_dependencies_met:
                        executable_tasks.append((task, subtask))

        return executable_tasks

    def _create_task_prompt(self, task: Task, subtask: SubTask) -> str:
        """
        Create a specific prompt for this task.

        Args:
            task: The high-level task
            subtask: The specific subtask to execute
            iteration: The current iteration count

        Returns:
            str: The task-specific prompt
        """
        prompt = f"""
        Execute this sub task to accomplish the high level task:
        Task Title: {task.title}
        Task Description: {task.description if task.description else task.title}
        Subtask ID: {subtask.id}
        Subtask Content: {subtask.content}
        Use Thinking: {subtask.thinking}
        Dependencies: {subtask.dependencies}

        When you have completed all necessary work for this subtask, call the finish_subtask tool with the final result.
        """

        return prompt

    def _all_tasks_complete(self) -> bool:
        """
        Check if all tasks are marked as complete.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        for task in self.task_plan.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    return False
        return True

    def get_results(self) -> Dict[str, Any]:
        """
        Get all subtask results from the global result store.

        Returns:
            Dict[str, Any]: Dictionary of results indexed by subtask ID
        """
        return self.result_store


class CoTAgent:
    """
    Agent that implements Chain of Thought (CoT) reasoning with language models.

    The CoTAgent class orchestrates a step-by-step reasoning process using language models
    to solve complex problems. It manages the conversational context, tool calling, and
    the overall reasoning flow, breaking problems down into logical steps.

    This class now uses TaskPlanner and IsolatedTaskExecutor internally to separate the
    planning and execution phases, making them available as standalone components.
    Each subtask has its own isolated execution context to prevent context window overflow.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        objective: str,
        workspace_dir: str,
        tools: List[Tool],
        max_steps: int = 30,
        max_subtask_iterations: int = 5,
        max_token_limit: int = 8000,  # Default token limit
    ):
        """
        Initializes the CoT agent.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            objective (str): The objective to solve
            workspace_dir (str): Directory for workspace files
            tools (List[Tool]): List of tools available for task execution
            max_steps (int, optional): Maximum reasoning steps to prevent infinite loops. Defaults to 30
            max_subtask_iterations (int, optional): Maximum iterations allowed per subtask. Defaults to 5
            max_token_limit (int, optional): Maximum token limit before summarization. Defaults to 8000
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_token_limit = max_token_limit
        self.workspace_dir = workspace_dir

        # Add FinishSubTaskTool to tools
        finish_subtask_tool = FinishSubTaskTool(workspace_dir)
        self.tools = (
            tools + [finish_subtask_tool] if finish_subtask_tool not in tools else tools
        )

        self.chat_history: List[Message] = (
            []
        )  # Store all chat interactions for reference
        # Create planner and executor components
        self.planner = TaskPlanner(provider, model, self.objective)

    async def solve_problem(
        self, print_usage: bool = False
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Solves the given problem using a two-phase approach: planning, then execution.

        This is the main method that coordinates the entire problem-solving process:
        1. Planning Phase: Uses TaskPlanner to break down the objective into tasks
        2. Execution Phase: Uses TaskExecutor to execute each task in the correct order

        The method yields intermediate results during both phases, allowing
        for streaming output to the user.

        Args:
            print_usage (bool, optional): Whether to print token usage statistics. Defaults to False.

        Yields:
            Union[Message, Chunk, ToolCall]: Intermediate results during planning and execution

        Raises:
            ValueError: If the planner fails to create a valid task plan
        """
        # Phase 1: Planning
        # The planner is already initialized in __init__ or set externally
        async for item in self.planner.create_plan():
            yield item

        # Get the task plan from the planner
        task_plan = self.planner.task_plan
        if not task_plan:
            raise ValueError("Failed to create a valid task plan")

        # Phase 2: Execution
        # Create a task executor with the same model as the agent
        executor = TaskExecutor(
            provider=self.provider,
            model=self.model,  # Use the executor model
            workspace_dir=self.workspace_dir,
            tools=self.tools,
            task_plan=task_plan,
            max_steps=self.max_steps,
            max_subtask_iterations=self.max_subtask_iterations,
            max_token_limit=self.max_token_limit,
        )

        # Execute the tasks and yield results
        async for item in executor.execute_tasks(print_usage=print_usage):
            yield item

        # Store the results for later retrieval
        self.results = executor.get_results()

    def get_results(self) -> Dict[str, Any]:
        """
        Get all subtask results from the global result store.

        Returns:
            Dict[str, Any]: Dictionary of results indexed by subtask ID
        """
        if hasattr(self, "results"):
            return self.results
        return {}
