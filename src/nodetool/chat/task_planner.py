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
from typing import AsyncGenerator, List, Sequence, Union

from nodetool.workflows.processing_context import ProcessingContext
import time


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
- Use read_workspace_file tool to read the dependencies, set max_tool_calls high enough to read the dependencies
- The workspace directory is /workspace
- DO NOT USE FILES FROM THE RESEARCH PLANNING PHASE IN THE TASK PLAN

IMPLEMENTATION DETAILS:
- Use content to describe the subtask
- Define the max_tool_calls field to specify the maximum number of tool calls for the subtask
- Specify exact file paths for all dependencies and outputs

TASK FORMAT:
```json
    {
        "type": "task",
        "title": "Task Title",
        "agent_name": "Agent Name",
        "subtasks": [
            {
                "type": "subtask",
                "content": "Analyze the market trends",
                "output_type": "md",
                "output_file": "/workspace/analyze_market_trends.md",
            },
            {
                "type": "subtask",
                "content": "Analyze the competitors",
                "file_dependencies": ["/workspace/analyze_market_trends.md"],
                "output_type": "md",
            },
            {
                "type": "subtask",
                "content": "Summarize the findings",
                "file_dependencies": ["/workspace/analyze_market_trends.md", "/workspace/analyze_competitors.md"],
                "output_type": "md",
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

        print("==" * 100)
        print(params)
        print("==" * 100)

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
    - Detailed LLM trace logging for debugging and analysis
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: str,
        objective: str,
        workspace_dir: str,
        tools: Sequence[Tool],
        agents: Sequence[Agent],
        task_models: Sequence[str] = [],
        system_prompt: str | None = None,
        max_research_iterations: int = 3,
        enable_tracing: bool = True,
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
            enable_tracing (bool, optional): Whether to enable LLM trace logging
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
        self.enable_tracing = enable_tracing

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

        # Setup tracing directory and file
        if self.enable_tracing:
            self.traces_dir = os.path.join(self.workspace_dir, "traces")
            os.makedirs(self.traces_dir, exist_ok=True)

            # Create a unique trace file name for this planning session
            sanitized_objective = "".join(
                c if c.isalnum() else "_" for c in self.objective[:40]
            )
            self.trace_file_path = os.path.join(
                self.traces_dir,
                f"trace_planner_{sanitized_objective}.jsonl",
            )

            # Initialize trace with basic metadata
            self._log_trace_event(
                "planner_initialized",
                {
                    "objective": self.objective,
                    "model": self.model,
                    "max_research_iterations": self.max_research_iterations,
                    "num_agents": len(self.agents),
                    "agent_names": [agent.name for agent in self.agents],
                },
            )

    def _log_trace_event(self, event_type: str, data: dict) -> None:
        """
        Log an event to the trace file.

        Args:
            event_type (str): Type of event (message, tool_call, research, etc.)
            data (dict): Event data to log
        """
        if not self.enable_tracing:
            return

        trace_entry = {"timestamp": time.time(), "event": event_type, "data": data}

        with open(self.trace_file_path, "a") as f:
            f.write(json.dumps(trace_entry) + "\n")

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

        # Log tool call in trace
        if self.enable_tracing:
            self._log_trace_event(
                "tool_call",
                {
                    "direction": "outgoing",
                    "tool_name": chunk.name,
                    "tool_args": chunk.args,
                    "tool_id": chunk.id,
                },
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

                # Log tool result in trace
                if self.enable_tracing:
                    self._log_trace_event(
                        "tool_result",
                        {
                            "direction": "incoming",
                            "tool_name": chunk.name,
                            "tool_id": chunk.id,
                            "result_summary": (
                                str(tool_result)[:200] + "..."
                                if len(str(tool_result)) > 200
                                else str(tool_result)
                            ),
                        },
                    )

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

        # Log research iteration start in trace
        if self.enable_tracing:
            self._log_trace_event(
                "research_iteration_start",
                {
                    "iteration": iteration,
                    "max_iterations": self.max_research_iterations,
                },
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
        accumulated_content = ""  # Track accumulated content from chunks

        async for chunk in generator:  # type: ignore
            if isinstance(chunk, Chunk):
                yield chunk
                content += chunk.content
                accumulated_content += chunk.content  # Accumulate content

                # Only log when the chunk is done or periodically for large responses
                if chunk.done:
                    # Log accumulated content in trace
                    if self.enable_tracing and accumulated_content.strip():
                        self._log_trace_event(
                            "message",
                            {
                                "direction": "incoming",
                                "content": accumulated_content,
                                "role": "assistant",
                                "is_partial": False,
                            },
                        )
                        accumulated_content = ""  # Reset after logging

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
                # If we have accumulated content, log it before processing the tool call
                if self.enable_tracing and accumulated_content.strip():
                    self._log_trace_event(
                        "message",
                        {
                            "direction": "incoming",
                            "content": accumulated_content,
                            "role": "assistant",
                            "is_partial": True,  # This was cut off by a tool call
                        },
                    )
                    accumulated_content = ""  # Reset after logging

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

            # Log reflection prompt in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "message",
                    {
                        "direction": "outgoing",
                        "content": reflection_prompt,
                        "role": "user",
                        "type": "reflection",
                    },
                )
        else:
            # If no tool was used, we might be stuck or have enough information
            prompt = (
                "You didn't use any research tools in the last iteration. "
                "Do you have sufficient information to create a plan now? "
                "If not, please use the available search or browser tools "
                "to gather more information."
            )
            research_history.append(
                Message(
                    role="user",
                    content=prompt,
                )
            )

            # Log no-tool-use prompt in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "message",
                    {
                        "direction": "outgoing",
                        "content": prompt,
                        "role": "user",
                        "type": "no_tool_used_prompt",
                    },
                )

        # Log research iteration end in trace
        if self.enable_tracing:
            self._log_trace_event(
                "research_iteration_end",
                {
                    "iteration": iteration,
                    "tool_used": tool_used,
                },
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

        # Log summary prompt in trace
        if self.enable_tracing:
            self._log_trace_event(
                "research_summary_request",
                {
                    "prompt": summary_prompt,
                },
            )

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

                # Log summary chunk in trace
                if self.enable_tracing and chunk.content.strip():
                    self._log_trace_event(
                        "research_summary_chunk",
                        {
                            "content": chunk.content,
                            "is_partial": not chunk.done,
                        },
                    )

        # Save research summary
        summary_file = os.path.join(self.research_dir, "research_summary.md")
        with open(summary_file, "w") as f:
            f.write(summary_content)

        # Log completed summary in trace
        if self.enable_tracing:
            self._log_trace_event(
                "research_summary_complete",
                {
                    "summary_file": summary_file,
                    "summary_length": len(summary_content),
                },
            )

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

    async def _validate_subtasks(self, subtasks):
        """
        Validate subtasks to ensure they have non-empty content and output_file fields.

        Args:
            subtasks: List of subtasks to validate

        Returns:
            List[str]: List of validation errors, empty if validation passes
        """
        validation_errors = []
        for i, subtask in enumerate(subtasks):
            if not subtask.get("content") or subtask.get("content").strip() == "":
                validation_errors.append(f"Subtask {i+1} has empty content field")
            if (
                not subtask.get("output_file")
                or subtask.get("output_file").strip() == ""
            ):
                validation_errors.append(f"Subtask {i+1} has empty output_file field")
        return validation_errors

    async def _build_agent_task_prompt(
        self, agent: Agent, all_tasks: List[Task]
    ) -> str:
        """
        Build a prompt for creating tasks for a specific agent.

        Args:
            agent: The agent to create a task for
            all_tasks: List of all tasks created so far

        Returns:
            str: The formatted prompt
        """
        # Convert existing tasks to dict format for the prompt
        tasks_created_so_far = []
        if all_tasks:
            tasks_created_so_far = [t.model_dump() for t in all_tasks]

        # Compile research summary
        research_summary = "\n\n".join(str(x) for x in self.research_findings)

        # Build the prompt
        return f"""
        Overall Objective: {self.objective}
        Create a list of subtasks for the agent: {agent.name}
        Agent description: {agent.description}
        Agent objective: {agent.objective}
        Current provider: {self.provider.__class__.__name__}
        {f"Available models: {self.task_models}" if self.task_models else ""}
        
        Research Findings:
        {research_summary}
        
        Previously created tasks:
        {json.dumps(tasks_created_so_far) if all_tasks else "No tasks created yet."}
        
        Think carefully about:
        1. What this specific agent is best suited to work on
        2. How this task relates to any previously created tasks
        3. What dependencies exist between this task and previous tasks
        4. How to structure subtasks to make them clear and executable
        5. How to structure the task to make it clear and executable

        Create subtasks that are clear and executable.
        """

    async def _process_json_blocks(
        self, content: str, agent: Agent, all_tasks: List[Task]
    ) -> tuple[bool, List[str]]:
        """
        Process JSON code blocks in the content to find task definitions.

        Args:
            content: The content to search for JSON blocks
            agent: The agent to create a task for
            all_tasks: List of all tasks to update if a valid task is found

        Returns:
            tuple: (success_flag, error_messages)
        """
        import re

        # Look for ```json ... ``` patterns
        json_blocks = re.findall(r"```json\s*([\s\S]*?)```", content)

        if not json_blocks:
            return False, []

        for json_str in json_blocks:
            try:
                # Try to parse the JSON
                task_data = json.loads(json_str)

                # Check if it has the required structure (subtasks)
                if "subtasks" in task_data:
                    # Add missing fields for the task
                    task_data["title"] = agent.objective
                    task_data["agent_name"] = agent.name

                    # Validate subtasks
                    validation_errors = await self._validate_subtasks(
                        task_data.get("subtasks", [])
                    )

                    if validation_errors:
                        # Continue to the next JSON block if this one has validation errors
                        continue

                    # If validation passes, add the task to our collection
                    all_tasks.append(
                        Task(
                            title=agent.objective,
                            agent_name=agent.name,
                            subtasks=task_data["subtasks"],
                        )
                    )

                    # Log successful task creation in trace
                    if self.enable_tracing:
                        self._log_trace_event(
                            "task_creation_success_from_json",
                            {
                                "agent_name": agent.name,
                                "subtasks_count": len(task_data["subtasks"]),
                                "from_json_block": True,
                            },
                        )

                    return True, []

            except json.JSONDecodeError:
                # If JSON is invalid, continue to the next block
                continue
            except Exception as e:
                # Log other errors
                if self.enable_tracing:
                    self._log_trace_event(
                        "json_block_error",
                        {
                            "agent_name": agent.name,
                            "error": str(e),
                            "json_block": (
                                json_str[:200] + "..."
                                if len(json_str) > 200
                                else json_str
                            ),
                        },
                    )

        return False, []

    async def _process_task_tool_call(
        self,
        tool_call: ToolCall,
        agent: Agent,
        all_tasks: List[Task],
        retries: int,
        max_retries: int,
    ) -> tuple[bool, List[str]]:
        """
        Process a tool call to create a task.

        Args:
            tool_call: The tool call to process
            agent: The agent to create a task for
            all_tasks: List of all tasks to update if a valid task is found
            retries: Current retry count
            max_retries: Maximum number of retries

        Returns:
            tuple: (success_flag, error_messages)
        """
        try:
            # Validate subtasks to ensure content and output_file are not empty
            validation_errors = await self._validate_subtasks(
                tool_call.args.get("subtasks", [])
            )

            if validation_errors:
                # Return validation errors
                error_message = "Validation errors in subtasks:\n- " + "\n- ".join(
                    validation_errors
                )
                error_message += "\n\nPlease fix these issues and ensure all subtasks have non-empty content and output_file fields."

                # Log validation error in trace
                if self.enable_tracing:
                    self._log_trace_event(
                        "task_validation_error",
                        {
                            "agent_name": agent.name,
                            "errors": validation_errors,
                            "retry_attempt": retries,
                        },
                    )

                return False, [error_message]

            # If validation passes, add the task to our collection
            all_tasks.append(
                Task(
                    title=agent.objective,
                    agent_name=agent.name,
                    subtasks=tool_call.args["subtasks"],
                )
            )

            # Log successful task creation in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "task_creation_success",
                    {
                        "agent_name": agent.name,
                        "subtasks_count": len(tool_call.args["subtasks"]),
                        "subtask_ids": [
                            subtask.get("id", f"subtask_{i}")
                            for i, subtask in enumerate(tool_call.args["subtasks"])
                        ],
                    },
                )

            return True, []

        except Exception as e:
            # Log error in trace
            if self.enable_tracing:
                self._log_trace_event(
                    "task_creation_error",
                    {
                        "agent_name": agent.name,
                        "error": str(e),
                        "retry_attempt": retries,
                    },
                )

            error_message = f"Error creating task: {str(e)}\nPlease fix the task structure and try again."

            if retries >= max_retries:
                raise ValueError(
                    f"Failed to create valid task after {max_retries} retries: {str(e)}"
                )

            return False, [error_message]

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
        # Build the initial prompt
        agent_task_prompt = await self._build_agent_task_prompt(agent, all_tasks)

        # Initialize conversation history
        history = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=agent_task_prompt),
        ]

        # Log task creation start in trace
        if self.enable_tracing:
            self._log_trace_event(
                "task_creation_start",
                {
                    "agent_name": agent.name,
                    "agent_objective": agent.objective,
                    "existing_tasks_count": len(all_tasks),
                },
            )

        retries = 0
        while retries <= max_retries:
            if retries > 0:
                yield Chunk(
                    content=f"\nRetry attempt {retries}/{max_retries} for agent '{agent.name}'...\n",
                    done=False,
                )

                # Log retry attempt in trace
                if self.enable_tracing:
                    self._log_trace_event(
                        "task_creation_retry",
                        {
                            "agent_name": agent.name,
                            "retry_attempt": retries,
                            "max_retries": max_retries,
                        },
                    )

            # Generate the task for this agent
            generator = self.provider.generate_messages(
                messages=history,
                model=self.model,
                tools=[task_tool],
                thinking=True,
            )

            content = ""
            tool_call_received = False

            async for chunk in generator:  # type: ignore
                if isinstance(chunk, Chunk):
                    yield chunk
                    content += chunk.content

                    # Log chunk in trace
                    if self.enable_tracing and chunk.content.strip():
                        self._log_trace_event(
                            "task_creation_message",
                            {
                                "direction": "incoming",
                                "content": chunk.content,
                                "role": "assistant",
                                "is_partial": not chunk.done,
                                "agent_name": agent.name,
                            },
                        )

                    # Check for JSON codeblocks in the content if the chunk is done
                    if chunk.done and not tool_call_received:
                        success, errors = await self._process_json_blocks(
                            content, agent, all_tasks
                        )
                        if success:
                            yield Chunk(
                                content=f"\nTask created from JSON block for agent '{agent.name}'\n",
                                done=False,
                            )
                            return

                elif isinstance(chunk, ToolCall) and chunk.name == task_tool.name:
                    tool_call_received = True
                    success, errors = await self._process_task_tool_call(
                        chunk, agent, all_tasks, retries, max_retries
                    )

                    if success:
                        return

                    if errors:
                        for error in errors:
                            history.append(Message(role="user", content=error))
                            yield Chunk(
                                content=f"\nValidation failed for agent '{agent.name}': {error}\n",
                                done=False,
                            )

                        retries += 1
                        break  # Break out of the generator loop to retry

            # If we didn't encounter a ToolCall or valid JSON block, increment retry counter
            if not tool_call_received:
                if retries >= max_retries:
                    raise ValueError(
                        f"Failed to create valid task: No tool call or valid JSON generated after {max_retries} retries"
                    )
                retries += 1
                error_msg = "You need to use the create_task tool or provide a valid JSON structure for this agent's task. Please try again."
                history.append(
                    Message(
                        role="user",
                        content=error_msg,
                    )
                )

                # Log no tool call in trace
                if self.enable_tracing:
                    self._log_trace_event(
                        "task_creation_no_tool_call",
                        {
                            "agent_name": agent.name,
                            "retry_attempt": retries,
                            "prompt": error_msg,
                        },
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
        yield Chunk(content="\Researching the objective...\n", done=False)
        async for chunk in self.research_objective():
            yield chunk

        # Prepare for creating tasks one at a time
        yield Chunk(content="\nCreating tasks for each agent...\n", done=False)

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
