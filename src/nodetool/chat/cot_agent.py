"""
Chain of Thought (CoT) Agent implementation with tool calling capabilities.

This module implements a Chain of Thought reasoning agent that can use large language
models (LLMs) from various providers (OpenAI, Anthropic, Ollama) to solve problems
step by step. The agent can leverage external tools to perform actions like mathematical
calculations, web browsing, file operations, and shell command execution.

The implementation provides:
1. A CoTAgent class that manages the step-by-step reasoning process
2. Integration with the existing provider and tool system
3. Support for streaming results during reasoning

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
from typing import AsyncGenerator, Sequence, List, Optional, Union, Dict, Any

from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.tools.task_management import FinishTaskTool, SubTask, Task, TaskList
from nodetool.metadata.types import Message, ToolCall, FunctionModel
from nodetool.chat.tools import (
    SearchEmailTool,
    GoogleSearchTool,
    AddLabelTool,
    BrowserTool,
    ScreenshotTool,
    SearchFileTool,
    ChromaTextSearchTool,
    ChromaHybridSearchTool,
    ExtractPDFTablesTool,
    ExtractPDFTextTool,
    ConvertPDFToMarkdownTool,
    CreateAppleNoteTool,
    ReadAppleNotesTool,
    SemanticDocSearchTool,
    KeywordDocSearchTool,
    CreateWorkspaceFileTool,
    ReadWorkspaceFileTool,
    UpdateWorkspaceFileTool,
    DeleteWorkspaceFileTool,
    ListWorkspaceContentsTool,
    ExecuteWorkspaceCommandTool,
    AddTaskTool,
    TaskList,
)
from nodetool.chat.tools.development import (
    RunNodeJSTool,
    RunNpmCommandTool,
    RunEslintTool,
    DebugJavaScriptTool,
    RunJestTestTool,
    ValidateJavaScriptTool,
)


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


class CoTAgent:
    """
    Agent that implements Chain of Thought (CoT) reasoning with language models.

    The CoTAgent class orchestrates a step-by-step reasoning process using language models
    to solve complex problems. It manages the conversational context, tool calling, and
    the overall reasoning flow, breaking problems down into logical steps.

    The agent operates in three phases:
    1. Planning Phase: Breaks down the problem into tasks and subtasks
    2. Dependency Phase: Identifies dependencies between tasks and creates a dependency graph
    3. Execution Phase: Executes tasks in the correct order based on dependencies

    This agent can work with different LLM providers (OpenAI, Anthropic, Ollama) and
    can use various tools to augment the language model's capabilities.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        workspace_dir: str,
        max_steps: int = 30,
        prompt_builder=None,
        max_tool_results: int = 5,  # New parameter to control tool result history
    ):
        """
        Initializes the CoT agent.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            workspace_dir (str): Directory for workspace files
            max_steps (int, optional): Maximum reasoning steps to prevent infinite loops. Defaults to 10
            prompt_builder (callable, optional): Custom function to build the initial prompt.
                                               Defaults to None (use internal builders)
            max_tool_results (int, optional): Maximum number of recent tool results to keep. Defaults to 5
        """
        self.provider = provider
        self.model = model
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder
        self.max_tool_results = max_tool_results
        self.history: List[Message] = []
        self.chat_history: List[Message] = (
            []
        )  # Store all chat interactions for reference
        self.tasks_list = TaskList()
        self.workspace_dir = workspace_dir
        # Add short-term memory to store tool results
        self.tool_memory: Dict[str, Any] = {}

        self.tools: List[Tool] = [
            SearchEmailTool(),
            GoogleSearchTool(),
            AddLabelTool(),
            BrowserTool(),
            ScreenshotTool(),
            SearchFileTool(),
            ChromaTextSearchTool(),
            ChromaHybridSearchTool(),
            ExtractPDFTablesTool(),
            ExtractPDFTextTool(),
            ConvertPDFToMarkdownTool(),
            CreateAppleNoteTool(),
            ReadAppleNotesTool(),
            SemanticDocSearchTool(),
            KeywordDocSearchTool(),
            CreateWorkspaceFileTool(workspace_dir),
            ReadWorkspaceFileTool(workspace_dir),
            UpdateWorkspaceFileTool(workspace_dir),
            DeleteWorkspaceFileTool(workspace_dir),
            ListWorkspaceContentsTool(workspace_dir),
            ExecuteWorkspaceCommandTool(workspace_dir),
            RunNodeJSTool(workspace_dir),
            RunNpmCommandTool(workspace_dir),
            RunEslintTool(workspace_dir),
            DebugJavaScriptTool(workspace_dir),
            RunJestTestTool(workspace_dir),
            ValidateJavaScriptTool(workspace_dir),
            AddTaskTool(self.tasks_list),
            FinishTaskTool(self.tasks_list),
        ]

    def _get_planning_system_prompt(self, tools: List[Tool]) -> str:
        """
        Get the system prompt for the planning phase.

        Returns:
            str: The system prompt with instructions for planning
        """
        prompt = f"""
        You are a strategic planning assistant that creates clear, organized plans.
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        Your workspace directory is: {self.workspace_dir}
        """

        prompt += """
        All file operations should be performed within this workspace directory.
        
        PLANNING INSTRUCTIONS:
        1. Analyze the problem to identify key components and challenges
        2. Break down the problem into logical, sequential steps
        3. Top level tasks should have sub tasks.
        4. The sub tasks will get executed in the execution phase, in the order they are written.
        5. The tasks should not include planning, which only happens in the planning phase.
        6. Each sub task should get a unique id which can be used to reference the task in the execution phase.
        7. Each sub task should reference the tool it will use.
        8. Each sub task has an ID, a title, a description, and a list of dependencies.
        9. The dependencies are other task IDs that must be completed before the sub task can be executed.
        10. The dependencies should be direct dependencies only - don't create indirect dependencies.
        11. The sub tasks should be actionable.
        12. When working with files, always use the workspace directory as the root.

        Write JSON with the following format:
        ```json
        {
            "title": "Write an Original Poem",
            "tasks": [
                {
                "title": "Brainstorming Phase",
                "subtasks": [
                    {
                    "id": "br1",
                    "content": "Create a new workspace file 'poem_ideas.txt' to store brainstorming notes",
                    "tool": "create_workspace_file",
                    "dependencies": []
                    },
                    {
                    "id": "br2", 
                    "content": "Write initial theme ideas and potential topics in poem_ideas.txt",
                    "tool": "update_workspace_file",
                    "dependencies": ["br1"]
                    }
                ]
            },
        }
        ```

        Consider the following tools to be used in sub tasks:
        """
        for tool in tools:
            prompt += f"- {tool.name}\n"
        prompt += """
        """

        return prompt

    def _get_execution_system_prompt(self) -> str:
        """
        Get the system prompt for the execution phase.

        Returns:
            str: The system prompt with instructions for execution
        """
        return f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        You are an agent that executes plans.
        
        Your workspace directory is: {self.workspace_dir}
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

    def _get_planning_user_prompt(self, problem: str) -> str:
        """
        Creates a user prompt for the planning phase.

        Args:
            problem (str): The problem to solve

        Returns:
            str: Formatted prompt for planning
        """
        return (
            f"Problem to solve: {problem}\n\n"
            "Create an efficient plan by:\n"
            "1. Analyzing the problem requirements\n"
            "2. Thinking about the problem in-depth\n"
            "3. Breaking it into logical tasks\n"
            "4. Breaking each task into subtasks\n"
            "5. Identifying tools you'll need during execution\n"
            "6. Considering which subtasks might depend on outputs from other tasks\n\n"
            "Output the task list in the specified JSON format. Each task should have:\n"
            "- A title\n"
            "- A list of subtasks with:\n"
            "  - content (string)\n"
            "  - completed (boolean)\n"
            "  - subtask_id (string)\n"
            "  - dependencies (list of strings)\n"
        )

    def _filter_context(self, messages: List[Message]) -> List[Message]:
        """
        Filter conversation history to keep only recent tool results.

        Args:
            messages (List[Message]): The full conversation history

        Returns:
            List[Message]: Filtered conversation history
        """
        if len(messages) <= 2:  # Keep at least system prompt and user query
            return messages

        # Always include system message and the most recent user message
        system_message = next((msg for msg in messages if msg.role == "system"), None)
        user_messages = [msg for msg in messages if msg.role == "user"]
        latest_user_message = user_messages[-1] if user_messages else None

        # Find tool result messages and their corresponding tool calls
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if msg.role == "tool":  # This is a tool result
                tool_result_indices.append(i)

        # Keep only the last N tool results and their tool calls
        recent_tool_indices = set()
        for idx in tool_result_indices[-self.max_tool_results :]:
            recent_tool_indices.add(idx)  # The tool result
            if (
                idx > 0
                and messages[idx - 1].role == "assistant"
                and hasattr(messages[idx - 1], "tool_calls")
            ):
                recent_tool_indices.add(idx - 1)  # The corresponding tool call

        # Build the filtered message list
        filtered_messages = []

        # Add system message if it exists
        if system_message:
            filtered_messages.append(system_message)

        # Add recent assistant responses, tool calls, and tool results
        for i, msg in enumerate(messages):
            # Skip system message (already added)
            if msg.role == "system":
                continue

            # Keep relevant tool calls and results
            if i in recent_tool_indices:
                filtered_messages.append(msg)
            # Keep assistant messages that aren't tool calls
            elif msg.role == "assistant" and not hasattr(msg, "tool_calls"):
                filtered_messages.append(msg)
            # Keep all user messages
            elif msg.role == "user":
                filtered_messages.append(msg)

        # Make sure we include the latest user message if it wasn't already added
        if latest_user_message and latest_user_message not in filtered_messages:
            filtered_messages.append(latest_user_message)

        return filtered_messages

    async def solve_problem(
        self, problem: str, show_thinking: bool = True, print_usage: bool = False
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Solves the given problem using a three-phase approach: planning, then execution.

        Args:
            problem (str): The problem or question to solve
            show_thinking (bool, optional): Whether to include thinking steps in the output
            print_usage (bool, optional): Whether to print provider usage statistics. Defaults to False

        Yields:
            Union[Message, Chunk, ToolCall]: Objects representing parts of the reasoning process
        """
        planning_prompt = (
            self._get_planning_user_prompt(problem)
            if not self.prompt_builder
            else self.prompt_builder(problem)
        )
        planning_system_message = Message(
            role="system", content=self._get_planning_system_prompt(self.tools)
        )
        execution_system_message = Message(
            role="system", content=self._get_execution_system_prompt()
        )

        # Reset history and tool memory
        self.history = [
            planning_system_message,
            Message(role="user", content=planning_prompt),
        ]
        self.tool_memory = {}

        # Add to chat history for reference
        self.chat_history.append(Message(role="user", content=problem))

        # Run planning phase with JSON schema response format
        print("=" * 100)
        print("Planning phase")

        chunks = []
        filtered_history = self._filter_context(self.history)

        # Get JSON schema from TaskList model
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_list",
                "schema": TaskList.model_json_schema(),
            },
        }

        generator = self.provider.generate_messages(
            messages=filtered_history,
            model=self.model,
            tools=[],
            response_format=response_format,  # Add response format parameter
        )

        async for chunk in generator:  # type: ignore
            chunks.append(chunk)

        # Process the generated plan
        combined_content = ""
        for chunk in chunks:
            if isinstance(chunk, Chunk):
                combined_content += chunk.content
                if (
                    len(self.history) > 0
                    and self.history[-1].role == "assistant"
                    and isinstance(self.history[-1].content, str)
                ):
                    self.history[-1].content += chunk.content
                else:
                    self.history.append(
                        Message(role="assistant", content=chunk.content)
                    )

        # Parse the JSON response and update task list
        try:
            json_pattern = r"```json(.*?)```"
            json_match = re.search(
                json_pattern, combined_content, re.DOTALL | re.MULTILINE
            )
            if json_match:
                combined_content = json_match.group(1).strip()
            tasks_json = json.loads(combined_content)
            self.tasks_list = TaskList.model_validate(tasks_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return

        for task in self.tasks_list.tasks:
            print(task.to_markdown())

        # Transition to execution phase with the task list
        self.history = [
            execution_system_message,
            Message(role="user", content=problem),
        ]

        # Reasoning loop for execution with one task at a time
        while not self._all_tasks_complete() and len(self.tasks_list.tasks) > 0:
            # Find the next executable task
            task, sub_task = self._get_next_executable_task()
            if not task or not sub_task:
                break

            print(f"Executing task: {task.title} - {sub_task.content}")

            # Create a specific prompt for this task with references to previous task outputs if needed
            task_prompt = f"""
            Execut this sub task as part of a wider plan:
            Task Title: {task.title}
            Task Description: {task.description}
            Subtask ID: {sub_task.id}
            Subtask Content: {sub_task.content}
            """

            # Check if subtask references other tasks
            for ref_id in sub_task.dependencies:
                if ref_id in self.tool_memory:
                    task_prompt += f"\n\nData from {ref_id}:\n{json.dumps(self.tool_memory[ref_id])}"

            # print(task_prompt)
            # Add the task-specific prompt to history
            self.history.append(Message(role="user", content=task_prompt))

            # Get response for this specific task
            chunks = []
            filtered_history = self._filter_context(self.history)
            generator = self.provider.generate_messages(
                messages=filtered_history,
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
            await self._process_chunks(chunks)

            # Mark this specific task as complete
            sub_task.completed = True

            # for task in self.tasks_list.tasks:
            #     print(task.to_markdown())

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
            Task: The next executable task, or None if no tasks are executable
        """
        for task in self.tasks_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    # Check if all dependencies are completed
                    all_dependencies_met = True
                    for dep_id in subtask.dependencies:
                        # Find the dependent task and check if it's completed
                        task, dependent_task = self.tasks_list.find_task_by_id(dep_id)
                        if not dependent_task:
                            raise ValueError(f"Dependent task {dep_id} not found")
                        if not dependent_task.completed:
                            all_dependencies_met = False
                            break

                    if all_dependencies_met:
                        return task, subtask

        return None, None

    async def _process_chunks(self, chunks: List[Union[Chunk, ToolCall]]) -> None:
        """
        Process a list of chunks and tool calls, updating the conversation history.

        Args:
            chunks: List of chunks and tool calls to process
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
                        Message(role="assistant", content=chunk.content)
                    )

            elif isinstance(chunk, ToolCall):
                # Add tool call to history
                self.history.append(Message(role="assistant", tool_calls=[chunk]))

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
                    )
                )

    def _get_current_task(self) -> tuple[Optional[Task], Optional[SubTask]]:
        """
        Get the current task being executed.

        Returns:
            tuple: The current task and subtask
        """
        for task in self.tasks_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
                    return task, subtask
        return None, None

    def retrieve_from_memory(self, result_id: str) -> Any:
        """
        Retrieve a stored tool result from memory.

        Args:
            result_id: The ID of the stored result

        Returns:
            Any: The stored result, or None if not found
        """
        return self.tool_memory.get(result_id)

    def _all_tasks_complete(self) -> bool:
        """
        Check if all tasks are marked as complete.

        Returns:
            bool: True if all tasks are complete, False otherwise
        """
        for task in self.tasks_list.tasks:
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

    def clear_history(self) -> None:
        """
        Clears the conversation history.

        Returns:
            None
        """
        self.history = []
        return None
