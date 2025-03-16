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
import re
import shutil
import subprocess
import platform
import tiktoken
from typing import AsyncGenerator, List, Optional, Union, Dict, Any

from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.tools.assets import (
    ListAssetsDirectoryTool,
    ReadAssetTool,
    SaveAssetTool,
)
from nodetool.metadata.types import (
    Message,
    ToolCall,
    FunctionModel,
    Task,
    SubTask,
    TaskList,
)

# Add these constants at the top of the file, after the imports
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are a strategic planning assistant that creates clear, organized plans.
PLANNING INSTRUCTIONS:
1. Create MINIMAL plans - use only as many tasks as absolutely necessary.
2. Match plan complexity to objective complexity - simple objectives should have few tasks.
3. Combine related steps into single tasks whenever possible.
4. Each top-level task should have 1-3 subtasks for most objectives.
5. Only create separate subtasks when there's a clear dependency or tool change.
6. The sub tasks will get executed in the execution phase, in the order they are written.
7. Each sub task should get a unique id which can be used to reference the task.
8. Each sub task should reference the tool it will use.
9. Dependencies are other task IDs that must be completed before the subtask can be executed.
10. Keep the plan as streamlined as possible while still achieving the objective.
11. When working with files, always use the workspace directory as the root.

Write JSON with the following format:
```json
{
    "type": "task_list",
    "title": "Write a Simple Note",
    "tasks": [
        {
            "type": "task",
            "title": "Create and Write Note",
            "subtasks": [
                {
                    "type": "subtask",
                    "id": "note1",
                    "content": "Create a new note file 'quick_note.txt' with initial content",
                    "tool": "create_workspace_file",
                    "dependencies": []
            }
        ]
        }
    ]
}
```
"""

DEFAULT_PLANNING_USER_PROMPT_TEMPLATE = """
Create an efficient plan by:
1. Analyzing the problem requirements
2. Thinking about the problem in-depth
3. Breaking it into logical tasks
4. Breaking each task into subtasks
5. Identifying tools you'll need during execution
6. Considering which subtasks might depend on outputs from other tasks

Output the task list in the specified JSON format. Each task should have:
- A title
- A list of subtasks with:
  - content (string)
  - completed (boolean)
  - subtask_id (string)
  - dependencies (list of strings)
"""

DEFAULT_EXECUTION_SYSTEM_PROMPT = """
You are an agent that executes plans.

All file operations must be performed within this workspace directory.

EXECUTION INSTRUCTIONS:
1. Focus on the task at hand and use the tools provided to you to complete the task.
2. When you need information from a previous tool execution, reference it by its task ID.
3. DO NOT include the full output of tools in your responses to save context space.
4. Respect task dependencies - do not execute a task until all its dependencies have been completed.
5. When working with files:
   - Always use paths relative to the workspace directory
   - Do not attempt to access files outside the workspace
   - Use the workspace tools for file operations
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
        """Execute workspace commands"""
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
        tools: List[Tool],
        objective: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            tools (List[Tool]): List of tools available for task execution
            objective (str): The objective to solve
            system_prompt (str, optional): Custom system prompt
            user_prompt (str, optional): Custom user prompt
        """
        self.provider = provider
        self.model = model
        self.tools = tools
        self.objective = objective
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_PLANNING_SYSTEM_PROMPT
        )
        self.system_prompt += """
        Consider the following tools to be used in sub tasks:
        """
        for tool in self.tools:
            self.system_prompt += f"- {tool.name}\n"
        self.system_prompt += """
        """
        self.user_prompt = (
            user_prompt
            if user_prompt
            else DEFAULT_PLANNING_USER_PROMPT_TEMPLATE.format(objective=objective)
        )
        self.user_prompt = f"Objective to solve: {objective}\n\n{self.user_prompt}"
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding

    async def create_plan(self) -> TaskList:
        """
        Create a task plan for the given problem.

        Returns:
            TaskList: A structured plan with tasks and subtasks
        """
        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.user_prompt),
        ]

        # Get JSON schema from TaskList model
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_list",
                "schema": TaskList.model_json_schema(),
            },
        }

        # Generate plan
        chunks = []
        generator = self.provider.generate_messages(
            messages=history,
            model=self.model,
            tools=[],
            response_format=response_format,
        )

        combined_content = ""
        async for chunk in generator:  # type: ignore
            chunks.append(chunk)
            if isinstance(chunk, Chunk):
                combined_content += chunk.content

        # Parse the JSON response and create TaskList
        try:
            json_pattern = r"```json(.*?)```"
            json_match = re.search(
                json_pattern, combined_content, re.DOTALL | re.MULTILINE
            )
            if json_match:
                combined_content = json_match.group(1).strip()
            tasks_json = json.loads(combined_content)
            return TaskList.model_validate(tasks_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            # Return an empty TaskList if parsing fails
            return TaskList(title=f"Failed plan for: {self.objective}")


class TaskExecutor:
    """
    Executes tasks from a TaskList using a language model and tools.

    This class handles the execution phase of the Chain of Thought process,
    executing tasks in the correct order based on their dependencies.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        workspace_dir: str,
        tools: List[Tool],
        task_list: TaskList,
        system_prompt: str | None = None,
        max_steps: int = 30,
    ):
        """
        Initialize the TaskExecutor.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            workspace_dir (str): Directory for workspace files
            tools (List[Tool]): List of tools available for task execution
            task_list (TaskList): The task list to execute
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum execution steps to prevent infinite loops. Defaults to 30
        """
        self.provider = provider
        self.model = model
        self.workspace_dir = workspace_dir
        self.tools = tools
        self.task_list = task_list
        prefix = f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        Your workspace directory is: {self.workspace_dir}
        """
        self.system_prompt = prefix + (
            system_prompt if system_prompt else DEFAULT_EXECUTION_SYSTEM_PROMPT
        )
        self.max_steps = max_steps
        self.history: List[Message] = []
        self.tool_memory: Dict[str, Any] = {}
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.history = [
            Message(role="system", content=self.system_prompt),
        ]

    def _count_tokens(self, messages: List[Message]) -> int:
        """
        Count the number of tokens in the message history.

        Args:
            messages (List[Message]): The messages to count tokens for

        Returns:
            int: The approximate token count
        """
        # Simple token counting approach
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

    async def execute_tasks(
        self,
        show_thinking: bool = True,
        print_usage: bool = False,
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Execute the tasks in the provided task list.

        Args:
            tasks_list (TaskList): The task list to execute
            problem (str): The original problem statement
            show_thinking (bool, optional): Whether to include thinking steps in the output
            print_usage (bool, optional): Whether to print provider usage statistics

        Yields:
            Union[Message, Chunk, ToolCall]: Objects representing parts of the execution process
        """
        self.tool_memory = {}

        # Reasoning loop for execution with one task at a time
        while not self._all_tasks_complete() and len(self.task_list.tasks) > 0:
            # Find the next executable task
            task, sub_task = self._get_next_executable_task()
            if not task or not sub_task:
                break

            yield Chunk(
                content=f"\nExecuting task: {task.title} - {sub_task.content}\n",
                done=False,
            )

            # Create a specific prompt for this task with references to previous task outputs if needed
            task_prompt = f"""
            Execute this sub task to accomplish the high level task:
            Task Title: {task.title}
            Task Description: {task.description}
            Subtask ID: {sub_task.id}
            Subtask Content: {sub_task.content}
            """

            # Check if subtask references other tasks
            for ref_id in sub_task.dependencies:
                if ref_id in self.tool_memory:
                    task_prompt += f"\n\This task is depending on the output from {ref_id}:\n{json.dumps(self.tool_memory[ref_id])}"

            # Add the task-specific prompt to history
            self.history.append(
                Message(role="user", content=task_prompt, task_id=sub_task.id)
            )

            # Get response for this specific task
            chunks = []
            generator = self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=self.tools,
            )

            if print_usage:
                yield Chunk(
                    content=f"\nProvider usage: {self.provider.usage}\n", done=False
                )

            async for chunk in generator:  # type: ignore
                yield chunk
                chunks.append(chunk)

            # Process the chunks
            await self._process_chunks(chunks, sub_task.id)

            # Mark this specific task as complete
            sub_task.completed = True

            # Prune history to remove messages no longer needed
            self._prune_history()

            # Print token count after each task for debugging
            if show_thinking:
                token_count = self._count_tokens(self.history)
                yield Chunk(
                    content=f"\n[Debug: {token_count} tokens in history after task {sub_task.id}]\n",
                    done=False,
                )

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

    def _get_next_executable_task(self) -> tuple[Optional[Task], Optional[SubTask]]:
        """
        Get the next executable task from the task list, respecting dependencies.

        Returns:
            tuple: The next executable task and subtask, or (None, None) if no tasks are executable
        """
        for task in self.task_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    # Check if all dependencies are completed
                    all_dependencies_met = True
                    for dep_id in subtask.dependencies:
                        # Find the dependent task and check if it's completed
                        task, dependent_task = self.task_list.find_task_by_id(dep_id)
                        if not dependent_task:
                            raise ValueError(f"Dependent task {dep_id} not found")
                        if not dependent_task.completed:
                            all_dependencies_met = False
                            break

                    if all_dependencies_met:
                        return task, subtask

        return None, None

    async def _process_chunks(
        self, chunks: List[Union[Chunk, ToolCall]], task_id: str
    ) -> None:
        """
        Process a list of chunks and tool calls, updating the conversation history.

        Args:
            chunks: List of chunks and tool calls to process
            task_id: The ID of the task the chunks belong to
        """
        for chunk in chunks:
            # Handle chunks and tool calls to update history
            if isinstance(chunk, Chunk):
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
                        Message(
                            role="assistant", content=chunk.content, task_id=task_id
                        )
                    )

            elif isinstance(chunk, ToolCall):
                # Add tool call to history
                self.history.append(
                    Message(
                        role="assistant",
                        tool_calls=[chunk],
                        task_id=task_id,
                    )
                )

                # Execute tool
                tool_result = await self._execute_tool(chunk)

                # Get current task and subtask ID for reference
                task, subtask = self._get_current_task()
                result_id = subtask.id if task and subtask else f"tool_{chunk.id}"

                # Store the result in memory for reference
                self.tool_memory[result_id] = tool_result.result

                # Add the full tool result to history instead of just a summary
                self.history.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_result.id,
                        name=chunk.name,
                        content=json.dumps(tool_result.result),
                        task_id=task_id,
                    )
                )

    def _prune_history(self) -> None:
        """
        Remove messages from history that are no longer needed for dependency resolution.

        This method identifies messages with task_ids that are no longer referenced
        as dependencies in any remaining incomplete tasks and removes them from history.
        """
        # Skip if history is too short
        if len(self.history) <= 2:  # Keep at least system prompt and user query
            return

        # Collect all task IDs that are still needed as dependencies
        needed_task_ids = set()

        # Add all task IDs from incomplete tasks and their dependencies
        for task in self.task_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    # The task itself is needed
                    needed_task_ids.add(subtask.id)
                    # All its dependencies are needed
                    needed_task_ids.update(subtask.dependencies)

        # Always keep system message and the most recent user message
        pruned_history = []
        for msg in self.history:
            # Always keep system messages
            if msg.role == "system":
                pruned_history.append(msg)
                continue

            # Always keep the most recent user message
            if msg.role == "user" and msg == self.history[-1]:
                pruned_history.append(msg)
                continue

            # Keep messages without task_id
            if not hasattr(msg, "task_id") or msg.task_id is None:
                pruned_history.append(msg)
                continue

            # Keep messages with task_id that's still needed
            if msg.task_id in needed_task_ids:
                pruned_history.append(msg)
                continue

        # Update the history
        self.history = pruned_history

    def _get_current_task(self) -> tuple[Optional[Task], Optional[SubTask]]:
        """
        Get the current task being executed.

        Returns:
            tuple: The current task and subtask
        """
        for task in self.task_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    return task, subtask
        return None, None

    def _all_tasks_complete(self) -> bool:
        """
        Check if all tasks are marked as complete.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        for task in self.task_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    return False
        return True

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


class CoTAgent:
    """
    Agent that implements Chain of Thought (CoT) reasoning with language models.

    The CoTAgent class orchestrates a step-by-step reasoning process using language models
    to solve complex problems. It manages the conversational context, tool calling, and
    the overall reasoning flow, breaking problems down into logical steps.

    This class now uses TaskPlanner and TaskExecutor internally to separate the
    planning and execution phases, making them available as standalone components.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        objective: str,
        workspace_dir: str,
        tools: List[Tool],
        max_steps: int = 30,
    ):
        """
        Initializes the CoT agent.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use wCith the provider
            objective (str): The objective to solve
            workspace_dir (str): Directory for workspace files
            max_steps (int, optional): Maximum reasoning steps to prevent infinite loops. Defaults to 10
        """
        self.provider = provider
        self.model = model
        self.objective = objective
        self.max_steps = max_steps
        self.workspace_dir = workspace_dir
        self.tools = tools
        self.chat_history: List[Message] = (
            []
        )  # Store all chat interactions for reference
        # Create planner and executor components
        self.planner = TaskPlanner(provider, model, self.tools, self.objective)

    async def solve_problem(
        self, show_thinking: bool = True, print_usage: bool = False
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Solves the given problem using a two-phase approach: planning, then execution.

        Args:
            problem (str): The problem or question to solve
            show_thinking (bool, optional): Whether to include thinking steps in the output
            print_usage (bool, optional): Whether to print provider usage statistics. Defaults to False

        Yields:
            Union[Message, Chunk, ToolCall]: Objects representing parts of the reasoning process
        """
        # Add to chat history for reference
        self.chat_history.append(Message(role="user", content=self.objective))

        # Run planning phase
        yield Chunk(content="Planning phase started...\n", done=False)
        tasks_list = await self.planner.create_plan()

        # Display the plan
        yield Chunk(content="\nGenerated plan:\n", done=False)
        for task in tasks_list.tasks:
            yield Chunk(content=task.to_markdown() + "\n", done=False)

        # Run execution phase
        yield Chunk(content="\nExecution phase started...\n", done=False)
        self.executor = TaskExecutor(
            self.provider, self.model, self.workspace_dir, self.tools, tasks_list
        )

        async for result in self.executor.execute_tasks(show_thinking, print_usage):
            yield result
