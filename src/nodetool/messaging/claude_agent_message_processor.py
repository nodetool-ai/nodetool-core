"""
Claude Agent SDK message processor module.

This module provides an alternative message processor that uses the Claude Agent SDK
for agent-mode messages, providing access to Claude's agentic capabilities including
tool calling, hooks, and MCP server integration.

See: https://platform.claude.com/docs/en/agent-sdk/overview
"""

import asyncio
import logging
import os
from typing import Any, List
from uuid import uuid4

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    SdkMcpTool,
    create_sdk_mcp_server,
)
from claude_agent_sdk import (
    tool as sdk_tool,
)


from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    Step,
    Task,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    LogUpdate,
    PlanningUpdate,
    StepResult,
    TaskUpdate,
    TaskUpdateEvent,
)

from .message_processor import MessageProcessor

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


class ClaudeAgentMessageProcessor(MessageProcessor):
    """
    Processor for agent mode messages using the Claude Agent SDK.

    This processor uses the Claude Agent SDK (ClaudeSDKClient) to handle
    complex tasks by leveraging Claude's native agentic capabilities,
    including tool calling, MCP server integration, and hooks.

    Key features:
    - Uses Claude's official Agent SDK for improved agent performance
    - Supports MCP (Model Context Protocol) servers for tool integration
    - Provides hooks for tool use events (pre/post)
    - Handles streaming messages and tool calls

    Attributes:
        api_key: Anthropic API key for authentication
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Claude Agent SDK processor.

        Args:
            api_key: Optional Anthropic API key. If not provided,
                     will use ANTHROPIC_API_KEY environment variable.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            log.warning(
                "No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable."
            )

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """
        Process messages using the Claude Agent SDK.

        This method:
        1. Extracts the objective from the user's message
        2. Converts nodetool tools to SDK MCP tools
        3. Creates and runs the Claude Agent SDK client
        4. Streams updates back to the client via send_message

        Args:
            chat_history: The complete chat history
            processing_context: Context for processing including user information
            **kwargs: Additional processor-specific parameters
        """

        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"

        # Extract objective from message content
        objective = self._extract_objective(last_message)

        # Generate a unique execution ID for this agent session
        agent_execution_id = str(uuid4())

        log.info(
            f"Starting Claude Agent SDK execution with objective: {objective[:100]}..."
        )

        # Get selected tools and convert to SDK format
        sdk_tools: list[Any] = []
        selected_tools: list[Tool] = []

        if last_message.tools:
            tool_names = set(last_message.tools)
            selected_tools = await asyncio.gather(
                *[
                    resolve_tool_by_name(name, processing_context.user_id)
                    for name in tool_names
                ]
            )  # type: ignore
            log.debug(f"Selected tools for agent: {[t.name for t in selected_tools]}")

            # Create SDK MCP tools from nodetool tools
            for tool in selected_tools:
                sdk_tools.append(self._create_sdk_tool(tool, processing_context))

        # Include UI proxy tools if client provided a manifest via tool bridge
        try:
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                for (
                    _tool_name,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    try:
                        sdk_tools.append(
                            self._create_ui_proxy_sdk_tool(
                                tool_manifest, processing_context
                            )
                        )
                    except Exception as e:
                        log.warning(f"Failed to register UI tool proxy: {e}")
        except Exception as e:
            log.warning(f"Error while adding UI tool proxies: {e}")

        try:
            # Create an SDK MCP server with our tools
            mcp_server = None
            if sdk_tools:
                mcp_server = create_sdk_mcp_server(
                    name="nodetool",
                    version="1.0.0",
                    tools=sdk_tools,
                )

            # Send planning update - initialization
            await self._send_planning_update(
                PlanningUpdate(
                    phase="initialization",
                    status="InProgress",
                    content="Initializing Claude Agent SDK...",
                    node_id="claude_agent",
                ),
                agent_execution_id,
                last_message,
            )

            # Configure agent options
            options = ClaudeAgentOptions(
                model=last_message.model,
                system_prompt=self._build_system_prompt(objective),
                mcp_servers={"nodetool": mcp_server} if mcp_server else {},
                permission_mode="bypassPermissions",  # Auto-allow tool use
                max_turns=kwargs.get("max_turns", 50),
            )

            # Environment variables for the agent
            env_vars = {"ANTHROPIC_API_KEY": self.api_key} if self.api_key else {}
            options.env = env_vars

            # Create a task to track progress
            task = Task(
                id=str(uuid4()),
                title="Claude Agent Task",
                description=objective,
            )
            step = Step(
                id=str(uuid4()),
                instructions=objective,
            )

            await self._send_task_update(
                TaskUpdate(
                    task=task,
                    step=step,
                    event=TaskUpdateEvent.TASK_CREATED,
                ),
                agent_execution_id,
                last_message,
            )

            # Initialize the SDK client
            client = ClaudeSDKClient(options)

            accumulated_content = ""
            step_count = 0

            try:
                # Add a small delay to allow MCP server initialization
                await asyncio.sleep(0.5)

                # Retry connection logic to handle transient transport errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        await client.connect(prompt=objective)
                        break
                    except Exception as e:
                        if "ProcessTransport" in str(e) and attempt < max_retries - 1:
                            log.warning(
                                f"Transport error during connection (attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                            )
                            await asyncio.sleep(1)
                        else:
                            raise e

                # Stream responses from the client
                async for message in client.receive_messages():
                    if isinstance(message, AssistantMessage):
                        # Process content blocks from assistant
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text = block.text
                                accumulated_content += text
                                await self.send_message(
                                    {
                                        "type": "chunk",
                                        "content": text,
                                        "done": False,
                                    }
                                )

                            elif isinstance(block, ToolUseBlock):
                                step_count += 1
                                tool_call = ToolCall(
                                    id=block.id,
                                    name=block.name,
                                    args=block.input,
                                    step_id=str(step_count),
                                )
                                await self.send_message(
                                    {
                                        "type": "tool_call_update",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_call.name,
                                        "message": f"Calling {tool_call.name}...",
                                        "args": tool_call.args,
                                        "step_id": tool_call.step_id,
                                        "agent_execution_id": agent_execution_id,
                                    }
                                )

                                # Also emit a log update for the tool call
                                await self._send_log_update(
                                    LogUpdate(
                                        node_id="claude_agent",
                                        node_name="Claude Agent",
                                        content=f"Calling tool: {block.name}",
                                        severity="info",
                                    ),
                                    agent_execution_id,
                                    last_message,
                                )

                            elif isinstance(block, ToolResultBlock):
                                # Tool result received
                                await self._send_log_update(
                                    LogUpdate(
                                        node_id="claude_agent",
                                        node_name="Claude Agent",
                                        content=f"Tool result received for: {getattr(block, 'tool_use_id', 'unknown')}",
                                        severity="info",
                                    ),
                                    agent_execution_id,
                                    last_message,
                                )

                            elif isinstance(block, ThinkingBlock):
                                # Claude's thinking/reasoning
                                thinking_text = getattr(block, "thinking", "")
                                if thinking_text:
                                    await self._send_planning_update(
                                        PlanningUpdate(
                                            phase="reasoning",
                                            status="InProgress",
                                            content=thinking_text,
                                            node_id="claude_agent",
                                        ),
                                        agent_execution_id,
                                        last_message,
                                    )

                    elif isinstance(message, ResultMessage):
                        # Final result from the agent
                        result = message.result
                        structured_output = message.structured_output

                        final_result = (
                            structured_output or result or accumulated_content
                        )

                        await self._send_step_result(
                            StepResult(
                                step=step,
                                result=final_result,
                                is_task_result=True,
                            ),
                            agent_execution_id,
                            last_message,
                        )

                        await self._send_task_update(
                            TaskUpdate(
                                task=task,
                                step=step,
                                event=TaskUpdateEvent.TASK_COMPLETED,
                            ),
                            agent_execution_id,
                            last_message,
                        )

                    elif isinstance(message, SystemMessage):
                        # System messages (e.g., status updates)
                        content = getattr(message, "content", [])
                        for block in content:
                            if hasattr(block, "text"):
                                await self._send_log_update(
                                    LogUpdate(
                                        node_id="claude_agent",
                                        node_name="Claude Agent",
                                        content=f"System: {block.text}",
                                        severity="info",
                                    ),
                                    agent_execution_id,
                                    last_message,
                                )

                    elif isinstance(message, UserMessage):
                        # User messages in the conversation
                        pass

            finally:
                await client.disconnect()

            # Send the final assistant message
            final_content = (
                accumulated_content if accumulated_content else "Task completed."
            )

            await self.send_message(
                Message(
                    role="assistant",
                    content=final_content,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})

        except Exception as e:
            log.error(f"Error in Claude Agent SDK execution: {e}", exc_info=True)
            error_msg = f"Claude Agent SDK execution error: {str(e)}"

            await self.send_message(
                {"type": "error", "message": error_msg, "error_type": "agent_error"}
            )

            # Signal completion even on error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            await self.send_message(
                Message(
                    role="assistant",
                    content=error_msg,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

        finally:
            self.is_processing = False

    def _extract_objective(self, message: Message) -> str:
        """Extract objective from message content."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    return content_item.text
        return "Complete the requested task"

    async def _send_planning_update(
        self,
        update: PlanningUpdate,
        agent_execution_id: str,
        last_message: Message,
    ) -> None:
        """Send a planning update message."""
        try:
            content_dict = {
                "type": "planning_update",
                "phase": update.phase,
                "status": update.status,
                "content": update.content,
                "node_id": update.node_id,
            }

            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "planning_update",
                    "agent_execution_id": agent_execution_id,
                    "content": content_dict,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )
            log.debug(
                f"Sent planning_update: phase={update.phase}, status={update.status}"
            )
        except Exception as e:
            log.error(f"Failed to send planning_update message: {e}", exc_info=True)

    async def _send_task_update(
        self,
        update: TaskUpdate,
        agent_execution_id: str,
        last_message: Message,
    ) -> None:
        """Send a task update message."""
        try:
            content_dict = {
                "type": "task_update",
                "event": update.event,
                "task": update.task.model_dump() if update.task else None,
                "step": update.step.model_dump() if update.step else None,
            }

            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "task_update",
                    "agent_execution_id": agent_execution_id,
                    "content": content_dict,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )
            log.debug(f"Sent task_update: event={update.event}")
        except Exception as e:
            log.error(f"Failed to send task_update message: {e}", exc_info=True)

    async def _send_step_result(
        self,
        result: StepResult,
        agent_execution_id: str,
        last_message: Message,
    ) -> None:
        """Send a step result message."""
        try:
            content_dict = {
                "type": "step_result",
                "result": result.result,
                "step": result.step.model_dump() if result.step else None,
                "step_id": result.step.id if result.step else None,
                "error": result.error,
                "is_task_result": result.is_task_result,
            }

            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "step_result",
                    "agent_execution_id": agent_execution_id,
                    "content": content_dict,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )
            log.debug("Sent step_result message")
        except Exception as e:
            log.error(f"Failed to send step_result message: {e}", exc_info=True)

    async def _send_log_update(
        self,
        update: LogUpdate,
        agent_execution_id: str,
        last_message: Message,
    ) -> None:
        """Send a log update message."""
        try:
            log_content = {
                "type": "log_update",
                "node_id": update.node_id,
                "node_name": update.node_name,
                "content": update.content,
                "severity": update.severity,
            }

            await self.send_message(
                {
                    "type": "message",
                    "role": "agent_execution",
                    "execution_event_type": "log_update",
                    "agent_execution_id": agent_execution_id,
                    "content": log_content,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                    "provider": last_message.provider,
                    "model": last_message.model,
                    "agent_mode": True,
                }
            )
        except Exception as e:
            log.error(f"Failed to send log_update message: {e}", exc_info=True)

    def _build_system_prompt(self, objective: str) -> str:
        """Build a system prompt for the Claude Agent SDK."""
        return f"""You are an AI assistant powered by the Claude Agent SDK.
Your objective is to help the user with their request.

Current objective: {objective}

You have access to various tools through the MCP server. Use them as needed to accomplish the task.
Be thorough, accurate, and provide helpful responses.
When using tools, explain what you're doing and why.
If a tool fails, try alternative approaches or ask for clarification."""

    def _create_sdk_tool(self, tool: Tool, context: ProcessingContext) -> Any:
        """
        Create an SDK MCP tool from a nodetool Tool.

        Args:
            tool: The nodetool Tool to convert
            context: Processing context for tool execution

        Returns:
            An SDK MCP tool wrapper
        """

        # Create a wrapper function that calls the nodetool tool
        @sdk_tool(tool.name, tool.description, tool.input_schema)
        async def tool_wrapper(args: dict[str, Any]) -> dict[str, Any]:
            try:
                result = await tool.process(context, args)
                # Convert result to SDK format
                if isinstance(result, dict):
                    content_text = str(result)
                elif isinstance(result, str):
                    content_text = result
                else:
                    content_text = str(result)

                return {
                    "content": [{"type": "text", "text": content_text}],
                }
            except Exception as e:
                log.error(f"Tool {tool.name} execution error: {e}")
                return {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "is_error": True,
                }

        return tool_wrapper

    def _create_ui_proxy_sdk_tool(
        self, tool_manifest: dict, context: ProcessingContext
    ) -> Any:
        """
        Create an SDK MCP tool from a UI tool manifest.

        Args:
            tool_manifest: The tool manifest from the frontend
            context: Processing context for tool execution

        Returns:
            An SDK MCP tool wrapper for UI proxy
        """
        name = tool_manifest["name"]
        description = tool_manifest.get("description", "UI tool")
        input_schema = tool_manifest.get("parameters", {})

        @sdk_tool(name, description, input_schema)
        async def ui_tool_wrapper(args: dict[str, Any]) -> dict[str, Any]:
            if not context.tool_bridge:
                return {
                    "content": [
                        {"type": "text", "text": "Error: Tool bridge not available"}
                    ],
                    "is_error": True,
                }

            tool_call_id = str(uuid4())

            # Forward to frontend
            tool_call_message = {
                "type": "tool_call",
                "tool_call_id": tool_call_id,
                "name": name,
                "args": args,
                "thread_id": getattr(context, "thread_id", ""),
            }

            if hasattr(context, "send_message"):
                await context.send_message(tool_call_message)  # type: ignore

            try:
                payload = await asyncio.wait_for(
                    context.tool_bridge.create_waiter(tool_call_id),
                    timeout=60.0,
                )

                if payload.get("ok"):
                    result = payload.get("result", {})
                    return {
                        "content": [{"type": "text", "text": str(result)}],
                    }
                else:
                    error_msg = payload.get("error", "Unknown error")
                    return {
                        "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                        "is_error": True,
                    }

            except TimeoutError:
                return {
                    "content": [
                        {"type": "text", "text": f"Error: UI tool {name} timed out"}
                    ],
                    "is_error": True,
                }

        return ui_tool_wrapper
