from nodetool.chat.agent import Agent
from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.providers.ollama import OllamaProvider
from nodetool.chat.tools import Tool
from nodetool.metadata.types import (
    FunctionModel,
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

STRATEGIC PLANNING APPROACH:
1. GOAL DECOMPOSITION: Break the main objective into clear sub-goals
2. TASK IDENTIFICATION: For each sub-goal, identify specific tasks needed
3. DEPENDENCY MAPPING: Create a directed acyclic graph (DAG) of task dependencies
4. PARALLEL OPTIMIZATION: Maximize concurrent execution opportunities
5. CRITICAL PATH ANALYSIS: Identify and optimize the longest dependency chain
6. RISK ASSESSMENT: Anticipate potential failure points and create contingencies

EFFICIENCY REQUIREMENTS:
1. BE CONCISE - Create a plan that is easy to understand and execute
2. MINMIZE NUMBER OF TASKS - Create as few tasks as possible
3. AVOID OVERTHINKING - Create a good plan immediately rather than iterating to a perfect one
4. MINIMIZE RESEARCH - Focus only on critical information gaps
5. PRIORITIZE SPEED - Choose the quickest way to execute the plan
6. USE DEFAULTS - When details are unclear, use reasonable assumptions rather than asking questions

TASK DESIGN PRINCIPLES:
- ATOMIC TASKS: Each subtask should do exactly one thing well
- CLEAR INTERFACES: Define precise input/output specifications for each task
- MINIMAL COUPLING: Reduce dependencies between tasks where possible
- APPROPRIATE GRANULARITY: Not too large (sequential) or too small (overhead)
- SELF-CONTAINED: Each task should have everything it needs to execute
- VERIFIABLE OUTPUTS: Task completion should produce a concrete artifact

ADVANCED PARALLELIZATION STRATEGIES:
- MAP-REDUCE PATTERN: Distribute independent data gathering, then consolidate
- PIPELINE PARALLELISM: Begin processing early results while gathering continues
- SPECULATIVE EXECUTION: Start tasks that are likely needed before confirmation
- DYNAMIC REASSIGNMENT: Adjust task allocation based on completion patterns
- CONDITIONAL BRANCHES: Create alternate paths based on intermediate findings

IMPLEMENTATION DETAILS:
- Use precise, descriptive task IDs that indicate purpose
- Define the max_tool_calls field to specify the maximum number of tool calls for the subtask
- Specify exact file paths for all dependencies and outputs
- Use the output_type field to specify the type of output the subtask will return
- Tag tasks with appropriate agent name based on required capability
- Include thought process explaining your planning decisions

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
                    "thinking": true|false,
                    "max_tool_calls": 2,
                    "output_type": "text|object",
                    "output_file": "/workspace/subtask_output.md",
                    "file_dependencies": ["/workspace/dependency.md"],
                }
            ]
        }
    ]
}
```

Use the CreateTaskPlanTool to submit your final task plan.
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
                        "thinking": {"type": "boolean"},
                        "max_tool_calls": {"type": "number"},
                        "output_type": {"type": "string"},
                        "output_file": {"type": "string"},
                        "file_dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["type", "id", "content"],
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
        model: FunctionModel,
        objective: str,
        workspace_dir: str,
        system_prompt: str | None = None,
        tools: List[Tool] = [],
        agents: List[Agent] = [],
        max_research_iterations: int = 5,
    ):
        """
        Initialize the TaskPlanner.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            objective (str): The objective to solve
            workspace_dir (str): The workspace directory path
            system_prompt (str, optional): Custom system prompt
            tools (List[Tool], optional): Tools available for research during planning
            max_research_iterations (int, optional): Maximum number of research iterations
        """
        self.provider = provider
        self.model = model
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

        # Create a directory to store research findings
        self.research_dir = os.path.join(workspace_dir, "research")
        os.makedirs(self.research_dir, exist_ok=True)

        # Store research findings
        self.research_findings = []

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

        # Create research system prompt
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

        # Create research prompt
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

        # Conduct research iterations
        iterations = 0
        research_findings = []

        while iterations < self.max_research_iterations:
            iterations += 1
            yield Chunk(
                content=f"\nResearch iteration {iterations}/{self.max_research_iterations}...\n",
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
                    # Add tool call to history
                    research_history.append(
                        Message(
                            role="assistant",
                            tool_calls=[chunk],
                        )
                    )

                    # Execute the tool call
                    from nodetool.workflows.processing_context import ProcessingContext

                    context = ProcessingContext(
                        user_id="planner_research", auth_token=""
                    )

                    # Find the tool
                    for tool in self.tools:
                        if tool.name == chunk.name:
                            tool_result = await tool.process(context, chunk.args)
                            yield Chunk(
                                content=f"\nUsed tool: {chunk.name}\n", done=False
                            )

                            # Add the tool result to history
                            research_history.append(
                                Message(
                                    role="tool",
                                    tool_call_id=chunk.id,
                                    name=chunk.name,
                                    content=json.dumps(tool_result),
                                )
                            )
                            research_findings.append(tool_result)

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

            # Add a reflection step after each research iteration
            reflection_prompt = f"""
            Based on the research findings so far, reflect on:
            1. What key information have we discovered?
            2. What questions still need to be answered?
            3. How does this information help with planning the objective?
            
            Be brief and focus on implications for creating an effective task plan.
            """

            if tool_used:
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

        # Save the complete research summary
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

        self.research_findings = research_findings
        yield Chunk(
            content=f"\nResearch completed. Findings saved to {summary_file}\n",
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
        if self.tasks_file_path.exists():
            try:
                with open(self.tasks_file_path, "r") as f:
                    task_plan_data = json.load(f)
                    self.task_plan = TaskPlan(**task_plan_data)
                    yield Chunk(
                        content=f"Loaded existing task plan from {self.tasks_file_path}\n",
                        done=False,
                    )
                    return
            except Exception as e:
                yield Chunk(
                    content=f"Error loading task plan from {self.tasks_file_path}: {str(e)}\nCreating new plan...\n",
                    done=False,
                )

        # Research the objective if we have tools available
        yield Chunk(content="\nPhase 1a: Researching the objective...\n", done=False)
        async for chunk in self.research_objective():
            yield chunk

        # Load research findings
        research_summary = "\n\n".join(str(x) for x in self.research_findings)

        # Prepare for creating tasks one at a time
        yield Chunk(
            content="\nPhase 1b: Creating tasks for each agent...\n", done=False
        )

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

            # Build a prompt that includes previously created tasks
            tasks_created_so_far = []
            if all_tasks:
                tasks_created_so_far = [t.model_dump() for t in all_tasks]

            agent_task_prompt = f"""
            Overall Objective: {self.objective}
            
            Create a list of subtasks for the agent: {agent.name}
            
            Agent description: {agent.description}
            
            Agent objective: {agent.objective}
            
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

            Create subtasks that are clear and executable.
            Focus on making the task leverage the agent's specific capabilities.
            """

            history = [
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=agent_task_prompt),
            ]

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
                            content=f"\nTask created for agent '{agent.name}'",
                            done=False,
                        )
                        break
                    except Exception as e:
                        # Add the error to history and ask LLM to fix it
                        error_message = f"Error creating task: {str(e)}\nPlease fix the task structure and try again."
                        history.append(Message(role="user", content=error_message))

                        # Generate a corrected task
                        retry_generator = self.provider.generate_messages(
                            messages=history,
                            model=self.model,
                            tools=[task_tool],
                            thinking=True,
                        )

                        # Process the retry response
                        async for retry_chunk in retry_generator:  # type: ignore
                            if isinstance(retry_chunk, Chunk):
                                yield retry_chunk
                                content += retry_chunk.content
                            elif (
                                isinstance(retry_chunk, ToolCall)
                                and retry_chunk.name == task_tool.name
                            ):
                                try:
                                    all_tasks.append(
                                        Task(
                                            title=agent.objective,
                                            agent_name=agent.name,
                                            subtasks=[
                                                SubTask(**subtask)
                                                for subtask in retry_chunk.args[
                                                    "subtasks"
                                                ]
                                            ],
                                        )
                                    )
                                    yield Chunk(
                                        content=f"\nTask created for agent '{agent.name}' after retry",
                                        done=False,
                                    )
                                    break
                                except Exception as retry_error:
                                    # If still invalid, raise an error
                                    raise ValueError(
                                        f"Failed to create valid task after retry: {str(retry_error)}"
                                    )

                        # Break out of the original loop if we got a valid task from retry
                        break

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

    async def save_task_plan(self) -> None:
        """
        Save the current task plan to tasks.json in the workspace directory.
        """
        if self.task_plan:
            # Use pydantic's model_dump_json method for serialization
            with open(self.tasks_file_path, "w") as f:
                f.write(self.task_plan.model_dump_json(indent=2))
