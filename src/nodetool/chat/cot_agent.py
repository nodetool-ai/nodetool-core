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
- Step-by-step reasoning with tool use capabilities
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Streaming results during the reasoning process
- Configurable reasoning steps and prompt templates
"""

import datetime
import json
import asyncio
from typing import AsyncGenerator, Sequence, Any, List, Dict, Optional, cast, Union

from nodetool.chat.providers import ChatProvider, Chunk
from nodetool.chat.tools import Tool
from nodetool.metadata.types import Message, ToolCall, FunctionModel


class CoTAgent:
    """
    Agent that implements Chain of Thought (CoT) reasoning with language models.

    The CoTAgent class orchestrates a step-by-step reasoning process using language models
    to solve complex problems. It manages the conversational context, tool calling, and
    the overall reasoning flow, breaking problems down into logical steps before arriving
    at a final answer.

    This agent can work with different LLM providers (OpenAI, Anthropic, Ollama) and
    can use various tools to augment the language model's capabilities.
    """

    def __init__(
        self,
        provider: ChatProvider,
        model: FunctionModel,
        tools: Optional[Sequence[Tool]] = None,
        max_steps: int = 10,
        prompt_builder=None,
    ):
        """
        Initializes the CoT agent.

        Args:
            provider (ChatProvider): An LLM provider instance
            model (FunctionModel): The model to use with the provider
            tools (Optional[Sequence[Tool]], optional): List of Tool instances. Defaults to None (empty list)
            max_steps (int, optional): Maximum reasoning steps to prevent infinite loops. Defaults to 10
            prompt_builder (callable, optional): Custom function to build the initial prompt.
                                               Defaults to the internal _default_prompt_builder
        """
        self.provider = provider
        self.model = model
        self.tools = tools or []
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder or self._default_prompt_builder
        self.system_message = Message(role="system", content=self._get_system_prompt())
        self.history: List[Message] = []
        self.chat_history: List[Message] = (
            []
        )  # Store all chat interactions for reference

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent.

        Returns:
            str: The system prompt with instructions for step-by-step reasoning
        """
        return f"""
        You are a helpful AI assistant that solves problems using step-by-step reasoning.
        Today is {datetime.datetime.now().strftime("%Y-%m-%d")}.
        You first create a clear plan before execution.
        You think step by step through both planning and execution phases.
        You provide a final answer after completing your reasoning.
        Optimize for speed and efficiency.
        """

    def _default_prompt_builder(self, problem: str) -> str:
        """
        Default prompt builder that instructs the model to reason step-by-step.

        Creates a prompt that guides the LLM to break down the problem, show its
        reasoning for each step, consider different approaches, and conclude with
        a final answer.

        Args:
            problem (str): The problem or question to solve.

        Returns:
            str: The formatted prompt with instructions for step-by-step reasoning.
        """
        return (
            f"Problem to solve: {problem}\n\n"
            "Approach:\n"
            "1. Planning Phase:\n"
            "   a. Analyze problem and identify key components\n"
            "   b. Break down the problem into sub-problems\n"
            "   c. Create a specific plan with clear steps for solving each sub-problem\n"
            "   d. Identify potential tools needed for execution\n"
            "2. Execution Phase:\n"
            "   a. Follow your plan step-by-step with clear reasoning\n"
            "   b. Adjust the plan if needed based on new insights\n"
            "   c. Use tools when needed (state which tool and why)\n"
            "3. When complete, provide: Final Answer: [concise solution]\n\n"
            "Begin your planning and reasoning now."
        )

    async def solve_problem(
        self, problem: str, show_thinking: bool = False
    ) -> AsyncGenerator[Union[Message, Chunk, ToolCall], None]:
        """
        Solves the given problem using CoT reasoning and tool calling.

        This method manages the entire reasoning process, including:
        1. Initializing the conversation with the problem statement
        2. Conducting a multi-step reasoning process with the LLM
        3. Using tools when needed to gather information or perform actions
        4. Streaming results during the reasoning process

        Args:
            problem (str): The problem or question to solve
            show_thinking (bool, optional): Whether to include thinking steps in the output.
                                          Defaults to False

        Yields:
            Union[Message, Chunk, ToolCall]: Objects representing parts of the reasoning process
        """
        # Initialize conversation with system prompt and user prompt
        initial_prompt = self.prompt_builder(problem)

        # Reset history and start with system message and user prompt
        self.history = [
            self.system_message,
            Message(role="user", content=initial_prompt),
        ]

        # Add to chat history for reference
        self.chat_history.append(Message(role="user", content=problem))

        # Reasoning loop
        for step in range(self.max_steps):
            if show_thinking:
                yield Chunk(content=f"\nStep {step + 1}:\n", done=False)

            # Get chunks from the async generator
            chunks = []
            async for chunk in self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=self.tools,
            ):  # type: ignore
                # Pass chunks and tool calls directly to the caller
                yield chunk
                chunks.append(chunk)

            # Process accumulated chunks
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

                    # Execute tool and add result to history
                    tool_result = await self._execute_tool(chunk)
                    self.history.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            name=chunk.name,
                            content=json.dumps(
                                tool_result.result,
                                default=lambda x: (
                                    str(x)
                                    if not isinstance(
                                        x,
                                        (dict, list, str, int, float, bool, type(None)),
                                    )
                                    else x
                                ),
                            ),
                        )
                    )

            # Check if the last message contains a "Final Answer" indication
            if self._has_final_answer():
                break

        # Add final request for concise answer if not already provided
        if not self._has_final_answer():
            self.history.append(
                Message(
                    role="user",
                    content="Please provide the most appropriate answer to the problem based on your reasoning steps. Be concise but complete.",
                )
            )

            # Get chunks from the async generator
            chunks = []
            async for chunk in self.provider.generate_messages(
                messages=self.history,
                model=self.model,
                tools=self.tools,
            ):  # type: ignore
                yield chunk
                chunks.append(chunk)

            # Process accumulated chunks
            for chunk in chunks:
                # Update history similar to above
                if isinstance(chunk, Chunk):
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

                elif isinstance(chunk, ToolCall):
                    self.history.append(Message(role="assistant", tool_calls=[chunk]))
                    tool_result = await self._execute_tool(chunk)
                    self.history.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            name=chunk.name,
                            content=json.dumps(
                                tool_result.result,
                                default=lambda x: (
                                    str(x)
                                    if not isinstance(
                                        x,
                                        (dict, list, str, int, float, bool, type(None)),
                                    )
                                    else x
                                ),
                            ),
                        )
                    )

    def _has_final_answer(self) -> bool:
        """
        Check if a final answer has been provided in the conversation.

        Returns:
            bool: True if a final answer was found, False otherwise
        """
        if len(self.history) == 0:
            return False

        last_messages = self.history[-min(3, len(self.history)) :]
        for message in last_messages:
            if message.role == "assistant" and isinstance(message.content, str):
                if "Final Answer:" in message.content:
                    return True
        return False

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
        self.history = [self.system_message]
        return None
