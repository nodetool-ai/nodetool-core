from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.providers.ollama import OllamaProvider
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    Message,
    SubTask,
    Task,
    TaskPlan,
    ToolCall,
)


import tiktoken


import json
import os
from pathlib import Path
from typing import AsyncGenerator, List, Union

from nodetool.workflows.processing_context import ProcessingContext


class CreateTaskPlanTool(Tool):
    """
    ✏️ Blueprint Creator - Tool for generating structured task plans

    This tool allows an agent to formalize its planning process by creating
    a structured TaskPlan with tasks, subtasks, and their dependencies.
    It's like an architect creating a blueprint before construction begins.

    The created plan becomes the foundation for all subsequent execution,
    defining what work needs to be done and in what order.
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


# Add ADVANCED_PLANNING_SYSTEM_PROMPT for more sophisticated planning
DEFAULT_PLANNING_SYSTEM_PROMPT = """
You are a sophisticated task planning agent that creates optimized, executable plans.

IMPORTANT: USE the create_task_plan tool to create the task plan.

STRATEGIC PLANNING APPROACH:
1. GOAL DECOMPOSITION: Break the main objective into clear sub-goals
2. TASK IDENTIFICATION: For each sub-goal, identify specific tasks needed
3. DEPENDENCY MAPPING: Create a directed acyclic graph (DAG) of task dependencies
4. PARALLEL OPTIMIZATION: Maximize concurrent execution opportunities
5. CRITICAL PATH ANALYSIS: Identify and optimize the longest dependency chain
6. RISK ASSESSMENT: Anticipate potential failure points and create contingencies

TASK DESIGN PRINCIPLES:
- ATOMIC TASKS: Each subtask should do exactly one thing well
- MINIMAL COUPLING: Reduce dependencies between tasks where possible
- MAP-REDUCE PATTERN: Distribute independent data gathering, then consolidate
- APPROPRIATE GRANULARITY: Not too large (sequential) or too small (overhead)
- SELF-CONTAINED: Each task should have everything it needs to execute, or read it from the workspace
- VERIFIABLE OUTPUTS: Task completion should produce a concrete artifact

FILE MANAGEMENT:
- Use the output_file field to specify the path to the file where the task result will be saved
- Use the file_dependencies field to specify the paths to the files that the task depends on
- The workspace directory is /workspace
- DO NOT USE FILES FROM THE RESEARCH PLANNING PHASE IN THE TASK PLAN

SUBTASK TYPES - CHOOSE THE MOST EFFICIENT TYPE FOR EACH SUBTASK:
- multi_step: Use for subtasks requiring multiple tool calls but minimal reasoning
  * Example: "Search for X, then summarize the results"
  * Set max_tool_calls to the number of tool calls needed

SUBTASK MODEL:
- Use the model field to specify the model to use for the subtask
- If the model is not specified, the default model will be used
- Select the most efficient model for the subtask
- Use only models for the provider you are using
- For OpenAI models, o1 and o3-mini are recommended for reasoning tasks
- For OpenAI models, gpt-4o is recommended for multi-step tasks
- For OpenAI models, gpt-4o-mini is recommended for long context windows
- For Anthropic models, claude-3-7-sonnet-20250219 is recommended for reasoning tasks
- For Anthropic models, claude-3-5-haiku-20240307 is recommended for long context windows
- For Ollama models, llama3.1 and llama3.1:8b are recommended for reasoning tasks
- For Ollama models, llama3.1:8b is recommended for multi-step tasks

REASONING
- Use for complex subtasks requiring detailed chain-of-thought reasoning
- Example: "Analyze data and explain the implications"
- Set max_tool_calls to the number of tool calls needed
- Reserve for tasks requiring significant analysis or creativity

IMPLEMENTATION DETAILS:
- Use precise, descriptive task IDs that indicate purpose
- Define the max_tool_calls field to specify the maximum number of tool calls for the subtask
- Specify exact file paths for all dependencies and outputs
- Use the output_type field to specify the type of output the subtask will return
- Tag tasks with appropriate agent name based on required capability
- Include thought process explaining your planning decisions
- ALWAYS set task_type to one of: "multi_step", or "reasoning"
- Set max_tool_calls to the number of tool calls needed

TASK PLAN FORMAT:
```json
{
    "type": "task_plan",
    "title": "Concise Objective Title",
    "tasks": [
        {
            "type": "task",
            "title": "Task Title",
            "agent_name": "Agent Name",
            "subtasks": [
                {
                    "type": "subtask",
                    "id": "descriptive_id",
                    "content": "Specific action to perform",
                    "task_type": "multi_step|reasoning",
                    "max_tool_calls": 2,
                    "output_type": "md|json|txt",
                    "output_file": "/workspace/subtask_output.{output_type}",
                    "file_dependencies": ["/workspace/dependency.md"],
                }
            ]
        }
    ]
}
```
"""


class CreateTaskTool(Tool):
    """
    ✏️ Task Creator - Tool for generating a single high-level task for an agent

    This tool allows the planner to create one task at a time, with each task
    explicitly assigned to a specific agent. Previous tasks are considered when
    creating new tasks to ensure proper dependencies.
    """

    name = "create_task"
    description = "Create a single task for a specific agent"

    # Define the input schema for a single task
    input_schema = {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "const": "subtask"},
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "task_type": {
                            "type": "string",
                            "enum": ["multi_step", "reasoning"],
                            "description": "The type of subtask: multi_step (multiple tools), or reasoning (complex thinking)",
                        },
                        "max_tool_calls": {"type": "number"},
                        "output_type": {"type": "string"},
                        "output_file": {"type": "string"},
                        "file_dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["type", "id", "content", "task_type"],
                },
            },
        },
        "required": ["subtasks"],
    }

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
        self._tasks = []

    async def process(self, context: ProcessingContext, params: dict) -> str:
        """
        Create a single task for a specific agent.
        """
        # Add the task type
        params["type"] = "task"

        # Add the task to our collection
        self._tasks.append(params)

        return f"Task '{params['title']}' created successfully for agent '{params['agent_name']}'"

    def get_tasks(self) -> List[dict]:
        """
        Get all the tasks created so far.
        """
        return self._tasks


class TaskPlanner:
    """
    🧩 The Master Planner - Breaks complex problems into executable chunks

    This strategic component divides large objectives into smaller, manageable tasks
    with dependencies between them. It's like a project manager breaking down a large
    project into sprints and tickets, identifying which tasks depend on others.

    The planner can also conduct research before planning to gather relevant information,
    ensuring the plan is well-informed and realistic. Plans are saved to enable
    persistence across sessions.

    Features:
    - Research capabilities to gather information before planning
    - Dependency tracking between subtasks
    - Parallel execution optimization
    - Plan persistence through JSON storage
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        tools: List[Tool],
        agents: List[Agent],
        task_models: list[str] = [],
        system_prompt: str | None = None,
        max_research_iterations: int = 3,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (str): The model to use with the provider
            task_models (list[str]): The models to use for the tasks
            objective (str): The objective to solve
            workspace_dir (str): The workspace directory path
            tools (List[Tool]): Tools available for research during planning
            agents (List[Agent]): Agents available for planning
            system_prompt (str, optional): Custom system prompt
            max_research_iterations (int, optional): Maximum number of research iterations
        """
        self.provider = provider
        self.model = model
        self.task_models = task_models
        self.objective = objective
        self.workspace_dir = workspace_dir
        self.task_plan = None
        self.agents = agents
        self.system_prompt = (
            system_prompt if system_prompt else DEFAULT_PLANNING_SYSTEM_PROMPT
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Default encoding

        # Check if tasks.json exists in the workspace directory
        self.tasks_file_path = Path(workspace_dir) / "tasks.json"

        # Store tools for research
        self.tools = tools or []
        self.max_research_iterations = max_research_iterations

        if not self.tools:
            raise ValueError("No tools provided to TaskPlanner")

        if not self.agents:
            raise ValueError("No agents provided to TaskPlanner")

        # Create a directory to store research findings
        self.research_dir = os.path.join(workspace_dir, "research")
        os.makedirs(self.research_dir, exist_ok=True)

        # Store research findings
        self.research_findings = []

    async def _load_existing_plan(self):
        """
        Try to load an existing task plan from the workspace.

        Returns:
            bool: True if plan was loaded successfully, False otherwise
        """
        if self.tasks_file_path.exists():
            try:
                with open(self.tasks_file_path, "r") as f:
                    task_plan_data = json.load(f)
                    self.task_plan = TaskPlan(**task_plan_data)
                    return True
            except Exception as e:
                return False
        return False

    async def _prepare_research(self) -> tuple[str, str, List[Message]]:
        """
        Prepare the system prompt, research prompt, and initial history for research.

        Returns:
            tuple: (system_prompt, research_prompt, initial_history)
        """
        research_system_prompt = """
        You are a research assistant tasked with gathering information to help plan a complex task.
        Your goal is to use search and browsing tools to collect relevant information about the objective.
        
        RESEARCH STRATEGY:
        1. First, identify what information you need to create an effective plan
        2. Use search tools to find relevant sources and information
        3. Use browser tools to access and extract key information from websites
        4. Focus on gathering practical, actionable information
        5. Store findings in an organized format for later use
        
        Be concise, focused, and thorough in your research. Only search for information directly
        relevant to planning the task. Don't go down rabbit holes.
        """

        research_prompt = f"""
        I need you to research information that will help me plan how to accomplish this objective:
        
        OBJECTIVE: {self.objective}
        
        What information should we gather to create an effective plan? Use the available search and browser
        tools to conduct focused research on this objective. Limit yourself to {self.max_research_iterations}
        search or browser actions.
        
        After each search or browsing action, summarize what you've learned and how it informs the planning.
        """

        # Initialize research history
        research_history = [
            Message(role="system", content=research_system_prompt),
            Message(role="user", content=research_prompt),
        ]

        return research_system_prompt, research_prompt, research_history

    async def _process_tool_call(
        self, chunk: ToolCall, research_history: List[Message]
    ) -> AsyncGenerator[Chunk, None]:
        """
        Process a tool call during research and update research findings.

        Args:
            chunk: The tool call to process
            research_history: The research conversation history to update

        Yields:
            Chunk: Progress updates
        """
        # Add tool call to history
        research_history.append(
            Message(
                role="assistant",
                tool_calls=[chunk],
            )
        )

        # Execute the tool call
        from nodetool.workflows.processing_context import ProcessingContext

        context = ProcessingContext(user_id="planner_research", auth_token="")

        # Find the tool
        for tool in self.tools:
            if tool.name == chunk.name:
                tool_result = await tool.process(context, chunk.args)
                yield Chunk(content=f"\nUsed tool: {chunk.name}\n", done=False)

                # Add the tool result to history
                research_history.append(
                    Message(
                        role="tool",
                        tool_call_id=chunk.id,
                        name=chunk.name,
                        content=json.dumps(tool_result),
                    )
                )
                self.research_findings.append(tool_result)

                # Save to file
                ext = "json" if isinstance(tool_result, dict) else "md"
                finding_file = os.path.join(
                    self.research_dir,
                    f"{chunk.name}.{ext}",
                )
                with open(finding_file, "w") as f:
                    if isinstance(tool_result, dict):
                        json.dump(tool_result, f, indent=2)
                    else:
                        f.write(str(tool_result))

                break

    async def _conduct_research_iteration(
        self, iteration: int, research_history: List[Message]
    ) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        Conduct a single research iteration.

        Args:
            iteration: The current iteration number
            research_history: The research conversation history

        Yields:
            Union[Message, Chunk]: Progress updates
        """
        yield Chunk(
            content=f"\nResearch iteration {iteration}/{self.max_research_iterations}...\n",
            done=False,
        )

        # Generate research step using available tools
        generator = self.provider.generate_messages(
            messages=research_history,
            model=self.model,
            tools=self.tools,
            thinking=True,
        )

        content = ""
        tool_used = False

        async for chunk in generator:  # type: ignore
            if isinstance(chunk, Chunk):
                yield chunk
                content += chunk.content

                # Update research history with assistant message
                if (
                    len(research_history) > 0
                    and research_history[-1].role == "assistant"
                    and isinstance(research_history[-1].content, str)
                ):
                    research_history[-1].content += chunk.content
                else:
                    research_history.append(
                        Message(role="assistant", content=chunk.content)
                    )

            elif isinstance(chunk, ToolCall):
                tool_used = True
                async for tool_chunk in self._process_tool_call(
                    chunk, research_history
                ):
                    yield tool_chunk

        # Add a reflection step after each research iteration
        if tool_used:
            reflection_prompt = """
            Based on the research findings so far, reflect on:
            1. What key information have we discovered?
            2. What questions still need to be answered?
            3. How does this information help with planning the objective?
            
            Be brief and focus on implications for creating an effective task plan.
            """
            research_history.append(Message(role="user", content=reflection_prompt))
        else:
            # If no tool was used, we might be stuck or have enough information
            research_history.append(
                Message(
                    role="user",
                    content="You didn't use any research tools in the last iteration. "
                    "Do you have sufficient information to create a plan now? "
                    "If not, please use the available search or browser tools "
                    "to gather more information.",
                )
            )

    async def _summarize_research(
        self, research_history: List[Message]
    ) -> AsyncGenerator[Chunk, None]:
        """
        Generate a summary of all research findings.

        Args:
            research_history: The research conversation history

        Yields:
            Chunk: Progress updates
        """
        summary_prompt = f"""
        Please provide a concise summary of all the research findings related to our objective:
        
        OBJECTIVE: {self.objective}
        
        Synthesize the key information discovered during research into an organized summary
        that will inform the planning process. Focus on actionable insights and important
        constraints or requirements discovered.
        """

        research_history.append(Message(role="user", content=summary_prompt))

        generator = self.provider.generate_messages(
            messages=research_history,
            model=self.model,
            tools=[],  # No tools for summary
        )

        summary_content = ""
        async for chunk in generator:  # type: ignore
            if isinstance(chunk, Chunk):
                yield chunk
                summary_content += chunk.content

        # Save research summary
        summary_file = os.path.join(self.research_dir, "research_summary.md")
        with open(summary_file, "w") as f:
            f.write(summary_content)

        yield Chunk(
            content=f"\nResearch completed. Findings saved to {summary_file}\n",
            done=False,
        )

    async def research_objective(self) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        📚 Knowledge Hunter - Gathers info before creating a plan

        This method uses search and browsing tools to conduct research on the objective,
        allowing the planning process to be well-informed. It's like doing homework
        before drawing up a project plan.

        The research findings are saved to the workspace for reference and are
        incorporated into the planning process to create a more effective task plan.

        Yields:
            Union[Message, Chunk]: Live updates during the research process
        """
        if not self.tools:
            yield Chunk(
                content="No research tools available for planning phase.\n", done=False
            )
            return

        yield Chunk(content=f"Researching objective: {self.objective}\n", done=False)

        # Prepare research prompts and history
        _, _, research_history = await self._prepare_research()

        # Conduct research iterations
        iterations = 0

        while iterations < self.max_research_iterations:
            iterations += 1
            async for chunk in self._conduct_research_iteration(
                iterations, research_history
            ):
                yield chunk

        # Generate research summary
        async for chunk in self._summarize_research(research_history):
            yield chunk

    async def _create_task_with_retries(
        self,
        agent: Agent,
        all_tasks: List[Task],
        task_tool: CreateTaskTool,
        max_retries: int = 3,
    ) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        Create a task for a specific agent with multiple retry attempts if needed.

        Args:
            agent: The agent to create a task for
            all_tasks: List of all tasks created so far
            task_tool: The tool to use for creating tasks
            max_retries: Maximum number of retry attempts (default: 3)

        Yields:
            Union[Message, Chunk]: Progress updates
        """
        # Build a prompt that includes previously created tasks
        tasks_created_so_far = []
        if all_tasks:
            tasks_created_so_far = [t.model_dump() for t in all_tasks]

        research_summary = "\n\n".join(str(x) for x in self.research_findings)

        agent_task_prompt = f"""
        Overall Objective: {self.objective}
        Create a list of subtasks for the agent: {agent.name}
        Agent description: {agent.description}
        Agent objective: {agent.objective}
        Current provider: {self.provider.__class__.__name__}
        {f"Available models: {self.task_models}" if self.task_models else ""}
        
        Research Findings:
        {research_summary}
        
        Previously created tasks:
        {tasks_created_so_far if all_tasks else "No tasks created yet."}
        
        Think carefully about:
        1. What this specific agent is best suited to work on
        2. How this task relates to any previously created tasks
        3. What dependencies exist between this task and previous tasks
        4. How to structure subtasks to make them clear and executable
        5. How to structure the task to make it clear and executable
        6. Which subtask type is most appropriate for each subtask:
           - Use "multi_step" when multiple tools are needed but with minimal reasoning
           - Use "reasoning" only for complex tasks requiring deep analysis
        7. Use reasoning models for reasoning tasks
        8. Use smaller models for summarization tasks

        Create subtasks that are clear and executable.
        Focus on making the task leverage the agent's specific capabilities.
        """

        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=agent_task_prompt),
        ]

        retries = 0
        while retries <= max_retries:
            if retries > 0:
                yield Chunk(
                    content=f"\nRetry attempt {retries}/{max_retries} for agent '{agent.name}'...\n",
                    done=False,
                )

            # Generate the task for this agent
            generator = self.provider.generate_messages(
                messages=history,
                model=self.model,
                tools=[task_tool],
                thinking=True,
            )

            content = ""
            async for chunk in generator:  # type: ignore
                if isinstance(chunk, Chunk):
                    yield chunk
                    content += chunk.content
                elif isinstance(chunk, ToolCall) and chunk.name == task_tool.name:
                    try:
                        # Add the task to our collection
                        all_tasks.append(
                            Task(
                                title=agent.objective,
                                agent_name=agent.name,
                                subtasks=[
                                    SubTask(**subtask)
                                    for subtask in chunk.args["subtasks"]
                                ],
                            )
                        )
                        yield Chunk(
                            content=f"\nTask created for agent '{agent.name}'"
                            + (f" after {retries} retries" if retries > 0 else ""),
                            done=False,
                        )
                        return
                    except Exception as e:
                        # Add the error to history and continue to the next retry
                        error_message = f"Error creating task: {str(e)}\nPlease fix the task structure and try again."
                        history.append(Message(role="user", content=error_message))

                        yield Chunk(
                            content=f"\nError creating task for agent '{agent.name}': {str(e)}\n",
                            done=False,
                        )

                        if retries >= max_retries:
                            raise ValueError(
                                f"Failed to create valid task after {max_retries} retries: {str(e)}"
                            )
                        retries += 1
                        break  # Break out of the generator loop to retry

            # If we didn't encounter a ToolCall, increment retry counter
            else:
                if retries >= max_retries:
                    raise ValueError(
                        f"Failed to create valid task: No tool call generated after {max_retries} retries"
                    )
                retries += 1
                history.append(
                    Message(
                        role="user",
                        content="You need to use the create_task tool to generate a task for this agent. Please try again.",
                    )
                )

    async def _create_tasks_for_agents(
        self,
    ) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        Create tasks for each agent in sequence.

        Yields:
            Union[Message, Chunk]: Progress updates
        """
        # Create a tool instance for creating individual tasks
        task_tool = CreateTaskTool(self.workspace_dir)

        # Initialize an empty list to store the tasks
        all_tasks = []

        # For each agent, create a task
        for i, agent in enumerate(self.agents):
            yield Chunk(
                content=f"\nCreating task {i+1}/{len(self.agents)} for agent '{agent.name}'...\n",
                done=False,
            )

            async for chunk in self._create_task_with_retries(
                agent, all_tasks, task_tool
            ):
                yield chunk

        # Now create the full task plan with all tasks
        self.task_plan = TaskPlan(
            title=self.objective,
            tasks=all_tasks,
        )

        # Save the task plan to tasks.json in the workspace
        await self.save_task_plan()
        yield Chunk(
            content=f"\nSaved task plan to {self.tasks_file_path}\n",
            done=False,
        )

    async def create_plan(self) -> AsyncGenerator[Union[Message, Chunk], None]:
        """
        🏗️ Blueprint Designer - Creates or loads a task execution plan

        This method strategically creates one high-level task per agent,
        ensuring proper task assignment and dependencies. If a plan already exists
        in the workspace, it will load that instead of creating a new one.

        The planning process considers research findings if available and focuses on
        maximizing parallelization while respecting necessary dependencies.

        Yields:
            Union[Message, Chunk]: Live updates during the planning process
        """
        # Check if tasks.json exists
        if await self._load_existing_plan():
            yield Chunk(
                content=f"Loaded existing task plan from {self.tasks_file_path}\n",
                done=False,
            )
            return

        # Research the objective if we have tools available
        yield Chunk(content="\nPhase 1a: Researching the objective...\n", done=False)
        async for chunk in self.research_objective():
            yield chunk

        # Prepare for creating tasks one at a time
        yield Chunk(
            content="\nPhase 1b: Creating tasks for each agent...\n", done=False
        )

        # Create tasks for each agent
        async for chunk in self._create_tasks_for_agents():
            yield chunk

    async def save_task_plan(self) -> None:
        """
        Save the current task plan to tasks.json in the workspace directory.
        """
        if self.task_plan:
            # Use pydantic's model_dump_json method for serialization
            with open(self.tasks_file_path, "w") as f:
                f.write(self.task_plan.model_dump_json(indent=2))
