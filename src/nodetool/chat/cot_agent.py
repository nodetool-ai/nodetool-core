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
from typing import AsyncGenerator, Sequence, List, Optional, Union, Dict, Any

from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.tools.task_management import FinishTaskTool, SubTask, Task, TaskList
from nodetool.metadata.types import Message, ToolCall, FunctionModel
from nodetool.chat.tools import (
    SearchEmailTool,
    GoogleSearchTool,
    AddLabelTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
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
            ListDirectoryTool(),
            ReadFileTool(),
            WriteFileTool(),
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
        
        PLANNING INSTRUCTIONS:
        1. Analyze the problem to identify key components and challenges
        2. Break down the problem into logical, sequential steps
        3. The tasks should be actionable.
        4. The tasks should not include planning, which only happens in the planning phase.
        5. For reasearch tasks, create only one summary task at the end.
        6. Each major task should be written as a separate heading in markdown
        7. Each subtask should be written as a list item under the appropriate heading

        Consider the following tools to be used in sub tasks:
        """
        for tool in tools:
            prompt += f"- {tool.name}\n"
            prompt += f"  - Description: {tool.description}\n"
        prompt += """
        """

        return prompt

    def _get_dependency_system_prompt(self) -> str:
        """
        Get the system prompt for the dependency phase.

        Returns:
            str: The system prompt with instructions for identifying dependencies
        """
        return f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        You are an expert at managing task dependencies.
        
        DEPENDENCY PHASE INSTRUCTIONS:
        1. Each task already has a unique ID
        2. Review all tasks and identify which subtasks depend on outputs from other tasks
        3. Mark dependencies using the format "(depends on #id)" and add it to the task title
        4. Ensure the dependency graph is acyclic (no circular dependencies)
        5. Focus on direct dependencies only - don't create indirect dependencies
        6. Output the task list in markdown format
        7. The task list should be formatted as follows:
        # TASK LIST
        ## High level task
        - [ ] #subtask_id Subtask 1 (use tool: tool_name) (depends on #dependency_id)
        - [ ] #subtask_id Subtask 2 (thinking about the problem)
        ## High level task
        - [ ] #subtask_id Subtask 1 (use tool: tool_name) (depends on #dependency_id)
        - [ ] #subtask_id Subtask 2 (use tool: tool_name)
        """

    def _get_execution_system_prompt(self) -> str:
        """
        Get the system prompt for the execution phase.

        Returns:
            str: The system prompt with instructions for execution
        """
        return f"""
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        You are an agent that executes plans.
        Focus on the task at hand and use the tools provided to you to complete the task.
        When you need information from a previous tool execution, reference it by its task ID.
        DO NOT include the full output of tools in your responses to save context space.
        Respect task dependencies - do not execute a task until all its dependencies have been completed.
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
            "7. Output the task list in markdown format\n"
            "The task list should be formatted as follows:\n"
            "# TASK LIST\n"
            "## High level task\n"
            "- [ ] Subtask 1 (use tool: tool_name)\n"
            "- [ ] Subtask 2 (thinking about the problem)\n"
            "## High level task\n"
            "- [ ] Subtask 1 (use tool: tool_name)\n"
            "- [ ] Subtask 2 (use tool: tool_name)\n"
        )

    def _get_dependency_user_prompt(self) -> str:
        """
        Creates a user prompt for the dependency phase.

        Returns:
            str: Formatted prompt for dependency identification
        """
        # Convert tasks to markdown to include in the prompt
        task_markdown = self.tasks_list.to_markdown()

        return (
            "Review the task list and identify dependencies between tasks.\n\n"
            "TASK LIST:\n"
            f"{task_markdown}\n\n"
            "For each task that depends on the output of another task, add 'depends on task <id>' to the task title.\n"
            "Be specific about which task ID it depends on. If no dependencies, leave it as is.\n"
            "Return the complete task list with dependencies marked.\n"
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
        self, problem: str, show_thinking: bool = True
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Solves the given problem using a three-phase approach: planning, then execution.

        Args:
            problem (str): The problem or question to solve
            show_thinking (bool, optional): Whether to include thinking steps in the output

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
        dependency_system_message = Message(
            role="system", content=self._get_dependency_system_prompt()
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

        # Run planning phase - single iteration only, no tools
        print("=" * 100)
        print("Planning phase")

        chunks = []
        filtered_history = self._filter_context(self.history)
        generator = self.provider.generate_messages(
            messages=filtered_history, model=self.model, tools=[]
        )

        async for chunk in generator:  # type: ignore
            yield chunk
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

        # Extract tasks from the planning response and add them
        tasks = combined_content.split("# TASK LIST")[1]
        self.tasks_list.from_markdown(tasks)

        # Second phase: Dependency identification
        # print("=" * 100)
        # print("Dependency phase")

        # # Display current task list before dependency phase
        # for task in self.tasks_list.tasks:
        #     print(task.to_markdown())

        # # Set up dependency phase conversation
        # dependency_prompt = self._get_dependency_user_prompt()
        # self.history = [
        #     dependency_system_message,
        #     Message(role="user", content=dependency_prompt),
        # ]

        # # Run dependency phase
        # chunks = []
        # filtered_history = self._filter_context(self.history)
        # generator = self.provider.generate_messages(
        #     messages=filtered_history, model=self.model, tools=[]
        # )

        # async for chunk in generator:  # type: ignore
        #     yield chunk
        #     chunks.append(chunk)

        # # Process the dependency phase output
        # combined_content = ""
        # for chunk in chunks:
        #     if isinstance(chunk, Chunk):
        #         combined_content += chunk.content
        #         if (
        #             len(self.history) > 0
        #             and self.history[-1].role == "assistant"
        #             and isinstance(self.history[-1].content, str)
        #         ):
        #             self.history[-1].content += chunk.content
        #         else:
        #             self.history.append(
        #                 Message(role="assistant", content=chunk.content)
        #             )

        # # Extract updated task list with dependencies
        # if "# TASK LIST" in combined_content:
        #     tasks = combined_content.split("# TASK LIST")[1]
        # else:
        #     tasks = combined_content

        # print(tasks)

        # # Update the task list with the new dependencies
        # self.tasks_list.from_markdown(tasks)

        # # Display updated task list with dependencies
        # print("=" * 100)
        # print("Execution phase")
        # for task in self.tasks_list.tasks:
        #     print(task.to_markdown())

        # Third phase: Execution

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
            Execute this specific task: {task.title} - {sub_task.content}
            Subtask ID: {sub_task.subtask_id}
            """

            # Check if subtask references other tasks
            references = self._extract_task_references(sub_task.content)
            for ref_id in references:
                if ref_id in self.tool_memory:
                    task_prompt += f"\n\nData from {ref_id}:\n{json.dumps(self.tool_memory[ref_id])}"

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
                        dependent_task = self.tasks_list.find_task_by_id(dep_id)
                        if not dependent_task or not dependent_task.is_completed():
                            all_dependencies_met = False
                            break

                    if all_dependencies_met:
                        return task, subtask

        return None, None

    def _get_next_incomplete_task(self) -> tuple[Optional[Task], Optional[SubTask]]:
        """
        Get the next incomplete task from the task list.

        Returns:
            Task: The next incomplete task, or None if all tasks are complete
        """
        for task in self.tasks_list.tasks:
            for subtask in task.subtasks:
                if not subtask.completed:
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
                result_id = (
                    subtask.subtask_id if task and subtask else f"tool_{chunk.id}"
                )

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
        tasks = self.tasks_list.get_all_tasks()
        if not tasks:
            return True  # No tasks means all tasks are complete

        return all(task.get("completed", False) for task in tasks)

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

    def _extract_task_references(self, content: str) -> List[str]:
        """
        Extract references to other task outputs from the task content.

        Args:
            content: The task content to analyze

        Returns:
            List[str]: List of task reference IDs
        """
        import re

        # Look for dependency formats like "depends on #1", "depends on task_id", or "depends on #1, #2, #4"
        task_ids = re.findall(r"depends on (?:#?([a-zA-Z0-9_]+))", content)
        # Also match comma-separated lists of dependencies
        comma_separated_ids = re.findall(
            r"depends on #[a-zA-Z0-9_]+(?:, #([a-zA-Z0-9_]+))+", content
        )
        if comma_separated_ids:
            task_ids.extend([id for sublist in comma_separated_ids for id in sublist])

        return task_ids
