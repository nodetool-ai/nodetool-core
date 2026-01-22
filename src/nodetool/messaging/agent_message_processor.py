"""
Agent message processor module.

This module provides the processor for agent mode messages.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.chat.token_counter import count_json_tokens, get_default_encoding
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    ToolCall,
)
from nodetool.providers.base import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
    LogUpdate,
    PlanningUpdate,
    StepResult,
    TaskUpdate,
)

from .message_processor import MessageProcessor

if TYPE_CHECKING:
    from nodetool.agents.tools.base import Tool

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


def _get_encoding_for_model(model: Optional[str]):
    try:
        import tiktoken  # type: ignore

        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                pass
    except Exception:
        pass

    return get_default_encoding()


def _log_tool_definition_token_breakdown(tools: list, model: Optional[str]) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return

    encoding = _get_encoding_for_model(model)
    per_tool: list[tuple[int, str]] = []
    for tool in tools:
        try:
            per_tool.append((count_json_tokens(tool.tool_param(), encoding=encoding), tool.name))
        except Exception:
            per_tool.append((0, getattr(tool, "name", "unknown")))

    total = sum(tokens for tokens, _ in per_tool)
    per_tool.sort(reverse=True)
    log.debug(
        "Tool definition tokens (model=%s): total=%d tools=%d",
        model,
        total,
        len(per_tool),
    )
    for tokens, name in per_tool[:50]:
        log.debug("  tool=%s tokens=%d", name, tokens)


class AgentMessageProcessor(MessageProcessor):
    """
    Processor for agent mode messages.

    This processor uses the Agent system to handle complex tasks by breaking
    them down into steps and executing them step by step.
    """

    def __init__(self, provider: BaseProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process messages using the Agent system."""
        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"

        # Extract objective from message content
        objective = self._extract_objective(last_message)

        # Generate a unique execution ID for this agent session
        agent_execution_id = str(uuid4())

        # Get selected tools based on message.tools
        selected_tools: list[Tool] = []
        if last_message.tools:
            tool_names = set(last_message.tools)
            resolved_tools = await asyncio.gather(
                *[resolve_tool_by_name(name, processing_context.user_id) for name in tool_names]
            )
            selected_tools = [t for t in resolved_tools if t is not None]
            log.debug(f"Selected tools for agent: {[tool.name for tool in selected_tools]}")

        # Include UI proxy tools if client provided a manifest via tool bridge
        # This mirrors HelpMessageProcessor behavior so the Agent can call frontend tools.
        try:
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                from .help_message_processor import (
                    UIToolProxy,
                )  # local proxy that forwards to frontend

                ui_tools: list[Tool] = []
                for (
                    _,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    try:
                        ui_tools.append(UIToolProxy(tool_manifest))
                    except Exception as e:
                        log.warning(f"Failed to register UI tool proxy: {e}")

                if ui_tools:
                    selected_tools.extend(ui_tools)
                    log.debug(f"Added {len(ui_tools)} UI tool proxies to agent tools")
        except Exception as e:
            log.warning(f"Error while adding UI tool proxies: {e}")

        _log_tool_definition_token_breakdown(selected_tools, last_message.model)

        try:
            from nodetool.agents.agent import Agent

            agent = Agent(
                name="Assistant",
                objective=objective,
                provider=self.provider,
                model=last_message.model,
                tools=selected_tools,
                output_schema={
                    "type": "object",
                    "properties": {
                        "markdown": {
                            "type": "string",
                            "description": "The markdown content of the response",
                        }
                    },
                    "required": ["markdown"],
                },
                verbose=kwargs.get("verbose", False),
                display_manager=kwargs.get("display_manager"),
            )

            async for item in agent.execute(processing_context):
                # Check for cancellation
                if self.is_cancelled():
                    log.info("Agent processing cancelled by user")
                    raise asyncio.CancelledError("Processing cancelled by user")

                log.debug(f"Agent yielded item type: {type(item).__name__}")
                if isinstance(item, Chunk):
                    # Set thread_id if available
                    if last_message.thread_id and not item.thread_id:
                        item.thread_id = last_message.thread_id
                    # Stream chunk to client
                    await self.send_message(
                        {
                            "type": "chunk",
                            "content": item.content,
                            "done": item.done if hasattr(item, "done") else False,
                            "thread_id": item.thread_id,
                        }
                    )
                elif isinstance(item, ToolCall):
                    # Send tool call update to frontend for display
                    try:
                        await self.send_message(
                            {
                                "type": "tool_call_update",
                                "thread_id": last_message.thread_id,
                                "workflow_id": last_message.workflow_id,
                                "tool_call_id": item.id,
                                "name": item.name,
                                "message": f"Calling {item.name}...",
                                "args": item.args,
                                "step_id": item.step_id,
                                "agent_execution_id": agent_execution_id,
                            }
                        )
                        log.debug(f"Sent tool_call_update for {item.name}")
                    except Exception as e:
                        log.error(f"Failed to send tool_call_update: {e}")
                elif isinstance(item, TaskUpdate):
                    # Send task update as agent_execution message (will be saved by base_chat_runner)
                    try:
                        log.info(f"Processing TaskUpdate: event={item.event}")

                        # Prepare content as dict
                        content_dict = {
                            "type": "task_update",
                            "event": item.event,
                            "task": item.task.model_dump() if item.task else None,
                            "step": item.step.model_dump() if item.step else None,
                        }

                        # Send as message via WebSocket (base_chat_runner will save to DB)
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
                        log.info("Sent task_update message")
                    except Exception as e:
                        log.error(f"Failed to send task_update message: {e}", exc_info=True)
                elif isinstance(item, PlanningUpdate):
                    # Send planning update as agent_execution message (will be saved by base_chat_runner)
                    try:
                        content_preview = (
                            str(item.content)[:500] + "..."
                            if item.content and len(str(item.content)) > 500
                            else str(item.content)
                        )
                        log.info(
                            f"Processing PlanningUpdate: phase={item.phase}, status={item.status}, content={content_preview}"
                        )

                        # Prepare content as dict
                        content_dict = {
                            "type": "planning_update",
                            "phase": item.phase,
                            "status": item.status,
                            "content": item.content,
                            "node_id": item.node_id,
                        }

                        # Send as message via WebSocket (base_chat_runner will save to DB)
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
                        log.info("Sent planning_update message")

                        # Also send a persistent LogUpdate for completed phases to ensure they show in chat history
                        if item.status in ["Success", "Failed"]:
                            log_content = {
                                "type": "log_update",
                                "node_id": item.node_id or "agent",
                                "node_name": "Agent",
                                "content": f"{item.phase}: {item.content}",
                                "severity": "error" if item.status == "Failed" else "info",
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
                            log.info(f"Sent persistent log_update for phase {item.phase}")

                    except Exception as e:
                        log.error(
                            f"Failed to send planning_update message: {e}",
                            exc_info=True,
                        )
                elif isinstance(item, LogUpdate):
                    # Handle explicit LogUpdate items from agent
                    try:
                        log_content = {
                            "type": "log_update",
                            "node_id": item.node_id,
                            "node_name": item.node_name,
                            "content": item.content,
                            "severity": item.severity,
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
                elif isinstance(item, StepResult) and not item.is_task_result:
                    # Send step result as agent_execution message (will be saved by base_chat_runner)
                    try:
                        log.info("Processing StepResult")

                        # Prepare content as dict
                        content_dict = {
                            "type": "step_result",
                            "result": item.result,
                            "step": item.step.model_dump() if item.step else None,
                            "step_id": item.step.id if item.step else None,
                            "error": item.error,
                            "is_task_result": item.is_task_result,
                        }

                        # Send as message via WebSocket (base_chat_runner will save to DB)
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
                        log.info("Sent step_result message")
                    except Exception as e:
                        log.error(f"Failed to send step_result message: {e}", exc_info=True)

            # Normalize final agent output to a markdown-friendly string
            content: str
            if isinstance(agent.results, str):
                content = agent.results
            elif isinstance(agent.results, dict):
                markdown_value = agent.results.get("markdown")
                content = markdown_value if isinstance(markdown_value, str) else str(agent.results)
            else:
                content = str(agent.results)

            await self.send_message(
                Message(
                    role="assistant",
                    content=content,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

            # Signal completion
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )

        except Exception as e:
            log.error(f"Error in agent execution: {e}", exc_info=True)
            error_msg = f"Agent execution error: {str(e)}"

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "agent_error",
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )

            # Signal completion even on error
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )

            # Return error message
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
            # Always mark processing as complete
            self.is_processing = False

    def _extract_objective(self, message: Message) -> str:
        """Extract objective from message content."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            # Find the first text content
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    return content_item.text
        return "Complete the requested task"
