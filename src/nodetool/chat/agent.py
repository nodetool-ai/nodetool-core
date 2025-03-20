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

from typing import AsyncGenerator, List, Sequence, Union, Dict, Any


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
from nodetool.workflows.processing_context import ProcessingContext


# Modified DEFAULT_EXECUTION_SYSTEM_PROMPT with emphasis on /workspace prefix
# Modified RETRIEVAL_SYSTEM_PROMPT with stronger emphasis on parallelization and chain-of-thought reasoning
RETRIEVAL_SYSTEM_PROMPT = """
You are a specialized retrieval agent focused on gathering information with minimal steps.

RETRIEVAL INSTRUCTIONS:
1. Your sole purpose is to gather information using retrieval tools
2. Use search, browser, and API tools to collect focused, specific data
3. FOCUS ON ONE SPECIFIC PIECE OF INFORMATION PER SUBTASK
4. Store all retrieved information in the /workspace directory
5. Format and organize information for later processing

TOOLS:
google_search - Execute precise Google searches with advanced parameters:
   - Use site: filters (e.g., "site:example.com") to narrow results to specific websites
   - Use filetype: filters (e.g., "filetype:pdf") to find specific document types
   - Use exact phrase matching with quotes for precise searches
   - Leverage advanced search parameters like intitle:, inurl:, and intext:
   - Filter by time_period: "past_24h", "past_week", "past_month", "past_year"
   - Specify country and language codes for localized results
   - Control pagination with start parameter
browser_control - Control a web browser to navigate and interact with web pages
   - Use the browser_control tool to navigate, click, and interact with web pages
   - The browser can access the internet and retrieve information from the web
   - You can browse Reddit, Google, Facebook, Instagram, LinkedIn, X, and more
   - Always use google_search to find the URL of the information you need
   
WORKSPACE CONSTRAINTS:
- All generated files must use the /workspace directory as root
- All file paths must start with /workspace/ for proper access
- Make sure to save artifacts in the /workspace directory
- Use descriptive filenames that indicate specific content
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

REASONING APPROACH:
1. First, scan all input files to understand the available information
2. Identify the key points relevant to your specific summary focus
3. Organize these points in a logical hierarchy or structure
4. Include proper citations to source materials

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
        model: str,
        workspace_dir: str,
        tools: Sequence[Tool],
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
        self.tools = list(tools) + [finish_subtask_tool]

        # Set system prompt
        self.system_prompt = (
            DEFAULT_EXECUTION_SYSTEM_PROMPT if system_prompt is None else system_prompt
        )
        self.results = []

    async def execute_task(
        self,
        task: Task,
        processing_context: ProcessingContext,
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
            processing_context=processing_context,
            tools=self.tools,
            task=task,
            system_prompt=self.system_prompt,
            max_steps=self.max_steps,
            max_subtask_iterations=self.max_subtask_iterations,
            max_token_limit=self.max_token_limit,
        )

        # Execute all subtasks within this task and yield results
        async for item in executor.execute_tasks():
            yield item

        # Store results
        self.results.extend(executor.get_results())

    def get_results(self) -> List[Any]:
        """
        Get the results produced by this agent.

        Returns:
            List[Any]: Results
        """
        return self.results
