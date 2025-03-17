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

from typing import AsyncGenerator, List, Union, Dict, Any


from nodetool.chat.sub_task_context import FinishSubTaskTool
from nodetool.chat.task_executor import DEFAULT_EXECUTION_SYSTEM_PROMPT, TaskExecutor
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    Message,
    ToolCall,
    FunctionModel,
    Task,
    TaskPlan,
)


# Modified DEFAULT_EXECUTION_SYSTEM_PROMPT with emphasis on /workspace prefix
# Modified RETRIEVAL_SYSTEM_PROMPT with stronger emphasis on parallelization and chain-of-thought reasoning
RETRIEVAL_SYSTEM_PROMPT = """
You are a specialized retrieval agent focused on gathering information with minimal steps.

RETRIEVAL INSTRUCTIONS:
1. Your sole purpose is to gather information using retrieval tools
2. Complete each retrieval task in as few steps as possible (ideally 1-2 tool calls per subtask)
3. Use search, browser, and API tools to collect focused, specific data
4. FOCUS ON ONE SPECIFIC PIECE OF INFORMATION PER SUBTASK
5. DO NOT ATTEMPT COMPREHENSIVE RETRIEVALS - each subtask should gather one specific fact or dataset
6. Store all retrieved information in the /workspace directory
7. Format and organize information for later processing
8. Do not attempt to analyze or summarize - just retrieve and store
9. EXECUTE ONE FOCUSED RETRIEVAL ACTION - each subtask should have exactly one search concept
10. NEVER CHAIN MULTIPLE SEARCHES - your subtask should perform only one search query

WORKSPACE CONSTRAINTS:
- All file operations must use the /workspace directory as root
- Never attempt to access files outside the /workspace directory
- All file paths must start with /workspace/ for proper access
- Make sure to save artifacts in the /workspace directory

WEB CRAWLING EFFICIENCY:
- Use `google_search` to find relevant pages for your SPECIFIC search topic only
- ALWAYS PREFER `web_fetch` over `browser_control` - it's significantly faster
- Only use `browser_control` when `web_fetch` fails or for dynamic content that requires interaction
- Use `batch_download` to fetch MULTIPLE pages in a single operation
- Avoid sequential browsing - retrieve everything in minimal batched calls
- Take screenshots only of critical visual content
- ONE SEARCH TOPIC PER SUBTASK - do not try to be comprehensive

DATA STORAGE:
- Save raw content in appropriate formats (html, txt, csv, json)
- Use descriptive filenames that indicate specific content source
- Create focused output files with clear naming conventions
- Include metadata with each saved file (source URL, timestamp)

REASONING APPROACH:
1. First, identify exactly what information you need to collect
2. Think through the most direct way to retrieve that information
3. Choose the most appropriate tool for the specific data needed
4. Execute your search with specific, targeted parameters
5. Validate that the retrieved data matches what was requested
6. Store the data in a clean, organized format

RESULT REQUIREMENTS:
1. Save all retrieved information as files in the /workspace directory
2. Use descriptive filenames that indicate specific content
3. Include metadata with sources for all retrieved information
4. Call finish_subtask with a manifest of all retrieved information
"""

# Modified SUMMARIZATION_SYSTEM_PROMPT with emphasis on focused summarization and chain-of-thought
SUMMARIZATION_SYSTEM_PROMPT = """
You are a specialized summarization agent focused on synthesizing information efficiently.

SUMMARIZATION INSTRUCTIONS:
1. Your purpose is to create clear, focused summaries from retrieved information
2. Complete each summarization task in as few steps as possible (ideally 1-2 tool calls)
3. FOCUS ON SPECIFIC ASPECTS: Each subtask should summarize one specific aspect or topic
4. DO NOT CREATE COMPREHENSIVE SUMMARIES - focus on your assigned specific area only
5. Read only the documents relevant to your specific summarization subtask
6. Create focused, narrow summaries with proper citations
7. Ensure summaries are accurate and maintain nuance from source material

WORKSPACE CONSTRAINTS:
- All file operations must use the /workspace directory as root
- Never attempt to access files outside the /workspace directory
- All file paths must start with /workspace/ for proper access
- Make sure to save artifacts in the /workspace directory

EFFICIENCY REQUIREMENTS:
- Read only files relevant to your specific subtask
- Analyze all content before starting to write summaries
- Avoid multiple passes over the same content
- Create focused summaries in a single writing phase

REASONING APPROACH:
1. First, scan all input files to understand the available information
2. Identify the key points relevant to your specific summary focus
3. Organize these points in a logical hierarchy or structure
4. Draft the summary, ensuring you maintain accuracy while being concise
5. Include proper citations to source materials
6. Review for completeness and accuracy before finalizing

SUMMARY CREATION:
- Create focused summary documents in markdown format
- Include proper citations to source documents
- Organize summaries with clear headings and structure
- Stay focused on your specific assigned topic - don't try to be comprehensive
- Present balanced view of any contradicting information

RESULT REQUIREMENTS:
1. Create a focused summary file for your specific assigned topic
2. Include proper citations to all source documents
3. Organize summaries with clear structure
4. Call finish_subtask with the final summary path and content
"""


class Agent:
    """
    🤖 Agent Base Class - Foundation for specialized agents

    This class provides the core functionality for all specialized agents,
    establishing a common interface and execution flow that can be customized
    by specific agent types (retrieval, summarization, etc.).

    Think of it as a base employee template that defines standard duties,
    while specialized roles add their unique skills and responsibilities.

    Features:
    - Task execution capabilities
    - Tool integration
    - Result tracking
    - System prompt customization
    """

    def __init__(
        self,
        name: str,
        objective: str,
        description: str,
        provider: ChatProvider,
        model: FunctionModel,
        workspace_dir: str,
        tools: List[Tool],
        system_prompt: str | None = None,
        max_steps: int = 50,
        max_subtask_iterations: int = 5,
        max_token_limit: int = 20000,
    ):
        """
        Initialize the base agent.

        Args:
            name (str): The name of the agent
            objective (str): The objective of the agent
            description (str): The description of the agent
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            workspace_dir (str): Directory for workspace files
            tools (List[Tool]): List of tools available for this agent
            system_prompt (str, optional): Custom system prompt
            max_steps (int, optional): Maximum reasoning steps
            max_subtask_iterations (int, optional): Maximum iterations per subtask
            max_token_limit (int, optional): Maximum token limit before summarization
        """
        self.name = name
        self.objective = objective
        self.description = description
        self.provider = provider
        self.model = model
        self.workspace_dir = workspace_dir
        self.max_steps = max_steps
        self.max_subtask_iterations = max_subtask_iterations
        self.max_token_limit = max_token_limit

        # Add FinishSubTaskTool to tools
        finish_subtask_tool = FinishSubTaskTool(workspace_dir)
        self.tools = tools + [finish_subtask_tool]

        # Set system prompt
        self.system_prompt = (
            DEFAULT_EXECUTION_SYSTEM_PROMPT if system_prompt is None else system_prompt
        )

        # Results storage
        self.results = {}

    async def execute_task(
        self,
        task_plan: TaskPlan,
        task: Task,
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Execute a task with all its subtasks.

        Args:
            task (Task): The task to execute

        Yields:
            Union[Message, Chunk, ToolCall]: Execution progress
        """
        yield Chunk(
            content=f"\nExecuting task '{task.title}' with {self.name} agent\n",
            done=False,
        )

        # Create task executor
        executor = TaskExecutor(
            provider=self.provider,
            model=self.model,
            workspace_dir=self.workspace_dir,
            tools=self.tools,
            task_plan=task_plan,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            max_subtask_iterations=self.max_subtask_iterations,
            max_token_limit=self.max_token_limit,
        )

        # Execute all subtasks within this task and yield results
        async for item in executor.execute_tasks():
            yield item

        # Store results
        self.results.update(executor.get_results())

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results produced by this agent.

        Returns:
            Dict[str, Any]: Results indexed by subtask ID
        """
        return self.results
