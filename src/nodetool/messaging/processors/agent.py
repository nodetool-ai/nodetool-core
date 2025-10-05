"""
Agent message processor module.

This module provides the processor for agent mode messages.
"""

import asyncio
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
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool
from nodetool.providers.base import BaseProvider
from .base import MessageProcessor

log = get_logger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)


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
                enable_analysis_phase=False,
                enable_data_contracts_phase=False,
                verbose=False,  # Disable verbose console output for websocket
            )

            accumulated_content = ""

            async for item in agent.execute(processing_context):
                if isinstance(item, Chunk):
                    accumulated_content += item.content
                    # Stream chunk to client
                    await self.send_message(
                        {
                            "type": "chunk",
                            "content": item.content,
                            "done": item.done if hasattr(item, "done") else False,
                        }
                    )
                elif isinstance(item, ToolCall):
                    # Send tool call update
                    await self.send_message(
                        {
                            "type": "tool_call_update",
                            "name": item.name,
                            "args": item.args,
                            "message": f"Calling {item.name}...",
                        }
                    )
                elif isinstance(item, TaskUpdate):
                    # Send task update
                    await self.send_message(
                        {
                            "type": "task_update",
                            "event": item.event,
                            "task": item.task.model_dump() if item.task else None,
                            "subtask": (
                                item.subtask.model_dump() if item.subtask else None
                            ),
                        }
                    )
                elif isinstance(item, PlanningUpdate):
                    # Send planning update
                    await self.send_message(
                        {
                            "type": "planning_update",
                            "phase": item.phase,
                            "status": item.status,
                            "content": item.content,
                            "node_id": item.node_id,
                        }
                    )
                elif isinstance(item, SubTaskResult) and not item.is_task_result:
                    # Send subtask result
                    await self.send_message(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": str(item.result),
                        }
                    )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})

            await self.send_message(
                Message(
                    role="assistant",
                    content=accumulated_content if accumulated_content else None,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=True,
                ).model_dump()
            )

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
