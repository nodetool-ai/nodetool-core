"""
Agent message processor module.

This module provides the processor for agent mode messages.
"""

import asyncio
from uuid import uuid4
from nodetool.config.logging_config import get_logger
from typing import List
from nodetool.agents.tools.tool_registry import resolve_tool_by_name

from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    ToolCall,
)
from nodetool.workflows.types import (
    Chunk,
    TaskUpdate,
    PlanningUpdate,
    SubTaskResult,
    TaskUpdateEvent,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool
from nodetool.providers.base import BaseProvider
from .message_processor import MessageProcessor

log = get_logger(__name__)


class AgentMessageProcessor(MessageProcessor):
    """
    Processor for agent mode messages.

    This processor uses the Agent system to handle complex tasks by breaking
    them down into subtasks and executing them step by step.
    """

    def __init__(self, provider: BaseProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
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
            selected_tools = await asyncio.gather(
                *[
                    resolve_tool_by_name(name, processing_context.user_id)
                    for name in tool_names
                ]
            )
            log.debug(
                f"Selected tools for agent: {[tool.name for tool in selected_tools]}"
            )

        # Include UI proxy tools if client provided a manifest via tool bridge
        # This mirrors HelpMessageProcessor behavior so the Agent can call frontend tools.
        try:
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                from .help import UIToolProxy  # local proxy that forwards to frontend

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

        try:
            from nodetool.agents.agent import Agent

            agent = Agent(
                name="Assistant",
                objective=objective,
                provider=self.provider,
                model=last_message.model,
                tools=selected_tools,
                enable_analysis_phase=True,
                enable_data_contracts_phase=True,
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
                verbose=False,  # Disable verbose console output for websocket
            )

            async for item in agent.execute(processing_context):
                log.debug(f"Agent yielded item type: {type(item).__name__}")
                if isinstance(item, Chunk):
                    # Stream chunk to client
                    await self.send_message(
                        {
                            "type": "chunk",
                            "content": item.content,
                            "done": item.done if hasattr(item, "done") else False,
                        }
                    )
                elif isinstance(item, ToolCall):
                    # No need to send tool call update
                    pass
                elif isinstance(item, TaskUpdate):
                    # Send task update as agent_execution message (will be saved by base_chat_runner)
                    try:
                        if item.event == TaskUpdateEvent.SUBTASK_COMPLETED:
                            log.info(f"Processing TaskUpdate: event={item.event}")

                            # Prepare content as dict
                            content_dict = {
                                "type": "task_update",
                                "event": item.event,
                                "task": item.task.model_dump() if item.task else None,
                                "subtask": item.subtask.model_dump()
                                if item.subtask
                                else None,
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
                                }
                            )
                            log.info("Sent task_update message")
                    except Exception as e:
                        log.error(
                            f"Failed to send task_update message: {e}", exc_info=True
                        )
                elif isinstance(item, PlanningUpdate):
                    # Send planning update as agent_execution message (will be saved by base_chat_runner)
                    try:
                        if item.status == "Success":
                            log.info(
                                f"Processing PlanningUpdate: phase={item.phase}, status={item.status}"
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
                                }
                            )
                            log.info("Sent planning_update message")
                    except Exception as e:
                        log.error(
                            f"Failed to send planning_update message: {e}",
                            exc_info=True,
                        )
                elif isinstance(item, SubTaskResult) and not item.is_task_result:
                    # Send subtask result as agent_execution message (will be saved by base_chat_runner)
                    try:
                        log.info("Processing SubTaskResult")

                        # Prepare content as dict
                        content_dict = {
                            "type": "subtask_result",
                            "result": str(item.result),
                            "subtask_id": getattr(item, "subtask_id", None),
                        }

                        # Send as message via WebSocket (base_chat_runner will save to DB)
                        await self.send_message(
                            {
                                "type": "message",
                                "role": "agent_execution",
                                "execution_event_type": "subtask_result",
                                "agent_execution_id": agent_execution_id,
                                "content": content_dict,
                                "thread_id": last_message.thread_id,
                            }
                        )
                        log.info("Sent subtask_result message")
                    except Exception as e:
                        log.error(
                            f"Failed to send subtask_result message: {e}", exc_info=True
                        )

            if isinstance(agent.results, str):
                await self.send_message(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": agent.results,
                        "thread_id": last_message.thread_id,
                    }
                )
            elif isinstance(agent.results, dict):
                await self.send_message(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": agent.results.get("markdown", str(agent.results)),
                        "thread_id": last_message.thread_id,
                    }
                )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})

        except Exception as e:
            log.error(f"Error in agent execution: {e}", exc_info=True)
            error_msg = f"Agent execution error: {str(e)}"

            # Send error message to client
            await self.send_message(
                {"type": "error", "message": error_msg, "error_type": "agent_error"}
            )

            # Signal completion even on error
            await self.send_message({"type": "chunk", "content": "", "done": True})

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
