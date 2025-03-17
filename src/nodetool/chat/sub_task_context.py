from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.tools.workspace import WorkspaceBaseTool
from nodetool.metadata.types import FunctionModel, Message, SubTask, Task, ToolCall


import tiktoken
import yaml


import json
import os
from typing import Any, AsyncGenerator, List, Union

from nodetool.workflows.processing_context import ProcessingContext

# Add a new DETAILED_COT_SYSTEM_PROMPT after the existing system prompts
DETAILED_COT_SYSTEM_PROMPT = """
You are an advanced reasoning agent that uses Chain of Thought (CoT) to solve complex problems methodically.

CHAIN OF THOUGHT PROCESS:
1. UNDERSTAND THE PROBLEM: Break down the problem into clearly defined components
2. PLAN YOUR APPROACH: Outline specific steps needed to solve each component
3. IDENTIFY DEPENDENCIES: Determine what information or results you need before proceeding
4. EXECUTE STEP-BY-STEP: Work through each step sequentially, showing your reasoning
5. VERIFY INTERMEDIATE RESULTS: Validate the output of each step before proceeding
6. RECONSIDER WHEN STUCK: If you encounter an obstacle, backtrack and try another approach
7. SYNTHESIZE FINAL SOLUTION: Combine intermediate results into a comprehensive solution

CRITICAL THINKING GUIDELINES:
- EXPLICIT REASONING: Always explain your thought process for each step
- CONSIDER ALTERNATIVES: Evaluate multiple approaches before choosing one
- TEST ASSUMPTIONS: Verify that your premises are valid before building on them
- ACKNOWLEDGE UNCERTAINTY: When information is incomplete, state assumptions clearly
- STRUCTURED FORMAT: Present your reasoning in a clear, organized manner
- USE TOOLS DELIBERATELY: Choose tools based on their specific capabilities for each step

EFFECTIVE IMPLEMENTATION:
- State each step and show your work for that step
- Label intermediate conclusions clearly
- Use numbered steps for complex calculations or multi-part reasoning
- When using tools, explain why you chose that specific tool
- After receiving tool results, interpret what they mean for your problem

AVOID COMMON PITFALLS:
- DON'T SKIP STEPS: Show each logical connection in your reasoning chain
- DON'T ASSUME RESULTS: Verify that each calculation or conclusion is correct
- DON'T RUSH TO CONCLUSIONS: Take time to evaluate alternative explanations
- DON'T OVERUSE TOOLS: Only call tools when necessary for specific information

RESULTS FORMAT:
1. Present your final answer clearly after showing your complete reasoning
2. Include a brief summary of how you arrived at the solution
3. Note any limitations or assumptions in your approach
4. When appropriate, suggest alternative approaches or next steps
"""


class SubTaskContext:
    """
    🧠 The Task-Specific Brain - Isolated execution environment for a single subtask

    This class maintains a completely isolated context for each subtask, with its own:
    - Message history
    - System prompt (based on subtask type)
    - Tools
    - Token tracking and context management

    It's like giving each subtask its own dedicated worker who has exactly the right
    skills and information for that specific job, without getting distracted by other tasks.

    Features:
    - Token limit monitoring with automatic summarization
    - Two-stage execution: tool calling stage followed by conclusion stage
    - Iteration tracking with max iterations safety
    - Explicit reasoning for "thinking" subtasks
    - Progress reporting during execution
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
        print_usage: bool = True,
        max_token_limit: int = 20000,
        max_iterations: int = 5,
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
            print_usage (bool): Whether to print token usage
            max_token_limit (int): Maximum token limit before summarization
            max_iterations (int): Maximum iterations for the subtask
        """
        self.task = task
        self.subtask = subtask

        # Use the DETAILED_COT_SYSTEM_PROMPT for tasks that require thinking
        if subtask.thinking:
            self.system_prompt = DETAILED_COT_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt

        self.tools = tools
        self.model = model
        self.provider = provider
        self.workspace_dir = workspace_dir
        self.max_token_limit = max_token_limit

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

        # Flag to track if subtask is finished
        self.completed = False
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

    def _save_to_output_file(self, result: dict[str, Any]) -> None:
        """
        Save the result of a tool call to the output file.
        Includes metadata in the appropriate format based on file type.
        """
        # Extract metadata from the result if it's a dictionary
        metadata = result.pop("metadata", {})

        # Get the file extension to determine format
        _, file_ext = os.path.splitext(self.output_file_path)
        is_markdown = file_ext.lower() in [".md", ".markdown"]

        with open(self.output_file_path, "w") as f:
            if is_markdown:
                # For Markdown files, add metadata as YAML frontmatter
                if metadata:
                    f.write("---\n")
                    yaml.dump(metadata, f)
                    f.write("---\n\n")

                    # If result is a dict but we're writing markdown, convert to string
                    content = result.get("content", str(result))
                    f.write(str(content))
                else:
                    f.write(str(result))
            else:
                if "content" in result:
                    # For JSON and other formats
                    if isinstance(result["content"], dict):
                        if metadata:
                            result["metadata"] = metadata
                        json.dump(result, f, indent=2)
                    elif isinstance(result["content"], list):
                        output = {"data": result["content"]}
                        if metadata:
                            output["metadata"] = metadata
                        json.dump(output, f, indent=2)
                else:
                    # For string results being written to non-markdown files
                    f.write(str(result))

    async def execute(
        self,
        task_prompt: str,
    ) -> AsyncGenerator[Union[Chunk, ToolCall], None]:
        """
        ⚙️ Task Executor - Runs a single subtask to completion using a two-stage approach:

        STAGE 1: TOOL CALLING STAGE
        - Allows the agent to use all available tools to complete the task
        - Limited to a maximum number of iterations (max_tool_calling_iterations)
        - Encourages efficient information gathering and task progress
        - SKIPPED for thinking tasks (subtask.thinking = True)

        STAGE 2: CONCLUSION STAGE
        - Only has access to the finish_subtask tool
        - Forces the agent to synthesize findings and complete the task
        - Prevents endless tool calling loops
        - Direct entry point for thinking tasks

        This two-stage approach ensures:
        1. Focused and efficient task execution
        2. Clear transition from exploration to conclusion
        3. Minimal iterations while still accomplishing the task effectively

        Args:
            task_prompt (str): The specific instructions for this subtask

        Yields:
            Union[Chunk, ToolCall]: Live updates during task execution
        """
        # Create the task prompt with file dependency context and two-stage explanation
        if self.subtask.thinking:
            # For thinking tasks, enhance the prompt but don't skip tool calling stage
            enhanced_prompt = (
                task_prompt
                + """
            
            IMPORTANT: This task will be executed in TWO STAGES, with emphasis on detailed reasoning:
            
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
            
            Since this is a thinking task, please use this Chain of Thought approach:
            1. First, understand what needs to be accomplished
            2. Break down the problem into manageable parts
            3. For each part, explain your approach clearly
            4. Show your work for each step before moving to the next
            5. Verify your intermediate results as you go
            6. If you get stuck, try a different approach and explain why
            7. Synthesize your findings into a final solution
            
            Your goal is to complete the task in as few iterations as possible while producing high-quality results.
            """
            )
        else:
            # Normal two-stage process for non-thinking tasks
            enhanced_prompt = (
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

        # Add the task prompt to this subtask's history
        self.history.append(Message(role="user", content=enhanced_prompt))

        # Signal that we're executing this subtask
        print(f"Executing task: {self.task.title} - {self.subtask.content}")

        if self.subtask.thinking:
            print(
                f"  This is a thinking task - processing with Chain of Thought reasoning"
            )

        # Continue executing until the task is completed or max iterations reached
        while not self.completed and self.iterations < self.max_iterations:
            self.iterations += 1
            token_count = self._count_tokens(self.history)

            # Determine which stage we're in
            if (
                self.iterations > self.max_tool_calling_iterations
                and not self.in_conclusion_stage
                and not self.subtask.thinking
            ):
                self.in_conclusion_stage = True
                # Create a list with only the finish_subtask tool
                conclusion_tools = [
                    tool for tool in self.tools if tool.name == "finish_subtask"
                ]

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

                print(
                    f"  Transitioning to conclusion stage for subtask {self.subtask.id}"
                )
                print(
                    f"  Iteration {self.iterations}/{self.max_iterations} (CONCLUSION STAGE)"
                )
            else:
                if self.subtask.thinking:
                    if self.in_conclusion_stage:
                        print(
                            f"  Iteration {self.iterations}/{self.max_iterations} (THINKING - CONCLUSION STAGE)"
                        )
                    else:
                        print(
                            f"  Iteration {self.iterations}/{self.max_iterations} (THINKING - TOOL CALLING STAGE)"
                        )
                elif self.in_conclusion_stage:
                    print(
                        f"  Iteration {self.iterations}/{self.max_iterations} (CONCLUSION STAGE)"
                    )
                else:
                    print(
                        f"  Iteration {self.iterations}/{self.max_iterations} (TOOL CALLING STAGE)"
                    )

            # Check if token count exceeds limit and summarize if needed
            if token_count > self.max_token_limit:
                print(
                    f"Token count ({token_count}) exceeds limit ({self.max_token_limit}). Summarizing context..."
                )
                await self._summarize_context()
                token_count = self._count_tokens(self.history)
                print(f"After summarization: {token_count} tokens in context")

            # Get response for this subtask using its isolated history
            # Use the appropriate tools based on the stage
            current_tools = (
                [tool for tool in self.tools if tool.name == "finish_subtask"]
                if self.in_conclusion_stage
                else self.tools
            )

            generator = self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=current_tools,
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

                    # Increment tool call counter
                    self.tool_call_count += 1
                    print(
                        f"Tool call {self.tool_call_count}/{self.max_tool_calls if self.max_tool_calls != float('inf') else 'unlimited'}: {chunk.name}"
                    )

                    # Execute the tool call
                    tool_result = await self._execute_tool(chunk)

                    # Handle finish_subtask tool specially
                    if chunk.name == "finish_subtask":
                        self.completed = True
                        self._save_to_output_file(tool_result.result)

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

                    # Check if we've reached the tool call limit and force conclusion stage if needed
                    if (
                        not self.in_conclusion_stage
                        and self.tool_call_count >= self.max_tool_calls
                        and chunk.name != "finish_subtask"
                    ):
                        self.in_conclusion_stage = True
                        # Create a list with only the finish_subtask tool
                        conclusion_tools = [
                            tool for tool in self.tools if tool.name == "finish_subtask"
                        ]

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
                        self.history.append(
                            Message(role="user", content=transition_message)
                        )

                        print(
                            f"  Reached maximum tool calls ({self.max_tool_calls}). Transitioning to conclusion stage."
                        )

            # If we've reached the last iteration and haven't completed yet, generate summary
            if self.iterations >= self.max_iterations and not self.completed:
                default_result = await self.request_summary()
                self._save_to_output_file(default_result)
                self.completed = True

                yield ToolCall(
                    id=f"{self.subtask.id}_max_iterations_reached",
                    name="finish_subtask",
                    args={
                        "result": default_result,
                        "output_file": self.subtask.output_file,
                        "metadata": {
                            "title": "Max Iterations Reached",
                            "description": "The subtask reached maximum iterations without completing normally",
                            "status": "timeout",
                        },
                    },
                )
        print(
            f"\n[Debug: {token_count} tokens in context for subtask {self.subtask.id}]\n"
        )
        print(self.provider.usage)

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
        This is used when the subtask has gone through both the Tool Calling Stage
        and the Conclusion Stage but still failed to complete properly.
        """
        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with providing a concise summary of work completed so far.
        
        The subtask has gone through both the Tool Calling Stage and the Conclusion Stage
        but has failed to complete properly. Your job is to summarize what was accomplished
        and what remains to be done.
        """

        # Create a focused user prompt
        summary_user_prompt = f"""
        The subtask '{self.subtask.id}' has reached the maximum allowed iterations ({self.max_iterations})
        after going through both the Tool Calling Stage and the Conclusion Stage.
        
        Please provide a brief summary of:
        1. What has been accomplished during the Tool Calling Stage
        2. What analysis was performed during the Conclusion Stage
        3. What remains to be done
        4. Any blockers or issues encountered
        
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
        2. Preserves which stage the subtask is currently in (Tool Calling or Conclusion)
        3. Summarizes the conversation history to approximately half its original length
        4. Replaces the history with the system prompt and summary
        """
        # Keep the system prompt
        system_prompt = self.history[0]

        # Create a summary-specific system prompt
        summary_system_prompt = """
        You are tasked with creating a detailed summary of the conversation so far.
        Maintain approximately 50% of the original content length.
        Include all important information, decisions made, and current state.
        Your summary will replace the detailed conversation history to reduce token usage.
        
        IMPORTANT: Clearly indicate which stage the task is currently in (Tool Calling Stage
        or Conclusion Stage) and preserve all key information related to the current stage.
        
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
        4. Current stage: {"CONCLUSION STAGE" if self.in_conclusion_stage else "TOOL CALLING STAGE"} 
        5. Current state and what needs to be done next
        6. Any important context or details that would be needed to continue the task
        
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


class FinishSubTaskTool(WorkspaceBaseTool):
    """
    🏁 Task Completion Tool - Marks a subtask as done and saves its results

    This tool is the finish line for subtasks, saving their final output to the workspace
    and marking them as completed. It's the equivalent of signing off on a piece of work
    and filing it away for others to use.

    EXCLUSIVELY AVAILABLE IN CONCLUSION STAGE:
    This tool is the ONLY tool available during the Conclusion Stage of subtask execution.
    It forces the agent to synthesize findings and complete the task without further
    information gathering.

    The result gets saved to the designated output file, making it available for
    dependent subtasks and final reporting. Think of it as publishing a report
    that other team members can now build upon.
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
                    "description": {
                        "type": "string",
                        "description": "The description of the result",
                    },
                    "source": {
                        "type": "string",
                        "description": "The source of the result",
                        "enum": ["url", "file", "calculation", "other"],
                    },
                    "url": {
                        "type": "string",
                        "description": "The URL of the result",
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "The timestamp of the result",
                    },
                },
                "required": ["title", "description", "source", "url", "timestamp"],
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
            "content": result,
            "metadata": metadata,
        }
