"""
Help message processor module.

This module provides the processor for help mode messages.
"""

import asyncio
import json
from typing import List

import httpx
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.help_tools import (
    SearchExamplesTool,
    SearchNodesTool,
)
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    ToolCall,
)
from nodetool.providers.base import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
)

from .message_processor import MessageProcessor
from .context_packer import create_compact_graph_context

log = get_logger(__name__)

# Safety limit to prevent runaway tool-call loops
MAX_TOOL_ITERATIONS = 25

SYSTEM_PROMPT = """
You are a Nodetool workflow assistant.

## Core Rules
- Do NOT invent node types or property names. Always use search_nodes first.
- When building or editing, use tools then validate with ui_auto_layout.
- Answer in short bullets. No plans or chain-of-thought.

## Tools
- search_nodes(query): find node types and schemas
- search_examples(query): find example workflows

## Data Types
- Primitives: str, int, float, bool, list, dict
- Assets: {"type": "image|audio|video|document", "uri": "..."}

## How Nodetool Works
- Visual editor: build node graphs by connecting outputs to inputs
- Node Menu: press space or double-click canvas to open
- Data Flow: values travel across edges from source to target nodes
- Assets: manage media in Assets panel; drag/drop to create asset nodes
- Models: run locally (GPU/MPS) or via providers (OpenAI, Anthropic, Ollama, etc.)

## Building Workflows
1. Design as Directed Acyclic Graphs (DAGs) - no cycles allowed
2. Connect nodes via edges representing data flow
3. All nodes MUST be valid types - use search_nodes to verify
4. Ensure type compatibility between connected handles
5. Use Loop nodes to iterate over lists/dataframes

## Streaming
- Streaming nodes output a stream of items one at a time
- Output nodes automatically collect streamed items into lists
- Use streaming for iterations and loops
- A producer node generates a stream; connected nodes execute once per item
- This enables efficient processing of large datasets

## Node Metadata Structure
- **properties**: Input fields the node accepts (become targetHandles for edges)
- **outputs**: Output slots the node produces (become sourceHandles for edges)
- **is_dynamic**: Node supports dynamic properties

## I/O Mappings
| Type   | Input Node                   | Output Node                   |
|--------|------------------------------|-------------------------------|
| string | nodetool.input.StringInput   | nodetool.output.StringOutput  |
| int    | nodetool.input.IntegerInput  | nodetool.output.IntegerOutput |
| float  | nodetool.input.FloatInput    | nodetool.output.FloatOutput   |
| bool   | nodetool.input.BooleanInput  | nodetool.output.BooleanOutput |
| image  | nodetool.input.ImageInput    | nodetool.output.ImageOutput   |
| video  | nodetool.input.VideoInput    | nodetool.output.VideoOutput   |
| audio  | nodetool.input.AudioInput    | nodetool.output.AudioOutput   |

## Key Namespaces
- nodetool.input, nodetool.output: workflow I/O
- nodetool.text, nodetool.image, nodetool.audio, nodetool.video: media processing
- nodetool.data: dataframes, lists, dictionaries
- nodetool.agents: AI agent nodes with tool connections
- nodetool.control: Loop, If, Switch for control flow
- nodetool.code: code evaluation

## Connection Conventions
- Input nodes: provide data through "output" handle
- Output nodes: receive data through "value" property
- Most nodes have a single output named "output"
- Edge format: {"source": "node_id", "sourceHandle": "output_name", "target": "...", "targetHandle": "..."}

## Agent Nodes (In Workflows)
- Connect nodes as tools to an Agent node via dynamic outputs
- Tool chains should end with a ToolResult node
- Agent's chunk output streams tokens; text output gives final result

## Using search_nodes Efficiently
- Create Input/Output nodes directly using the mappings above (don't search)
- Search for processing nodes with specific keywords
- Use input_type/output_type filters to narrow results
- Batch related searches: "dataframe group aggregate" instead of separate queries
- Check returned metadata for property names and types

## Official Nodetool Resources
- Website: https://nodetool.ai
- Documentation: https://docs.nodetool.ai
- Community Forum: https://forum.nodetool.ai
- Do NOT invent or hallucinate other domains
"""


class UIToolProxy(Tool):
    """Proxy tool that forwards tool calls to the frontend."""

    def __init__(self, tool_manifest: dict):
        # Configure base Tool fields expected by providers
        self.name = tool_manifest["name"]
        self.description = tool_manifest.get("description", "UI tool")
        # Providers expect JSON schema under input_schema
        self.input_schema = tool_manifest.get("parameters", {})

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        """Forward tool call to frontend and wait for result."""
        if not context.tool_bridge:
            raise ValueError("Tool bridge not available")

        # Generate a unique tool call ID
        import uuid

        tool_call_id = str(uuid.uuid4())

        # Forward to frontend
        tool_call_message = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": self.name,
            "args": params,
            "thread_id": getattr(context, "thread_id", ""),
        }

        await context.send_message(tool_call_message)  # type: ignore

        # Wait for result with timeout
        try:
            payload = await asyncio.wait_for(
                context.tool_bridge.create_waiter(tool_call_id), timeout=60.0
            )

            if payload.get("ok"):
                return payload.get("result", {})
            else:
                error_msg = payload.get("error", "Unknown error")
                raise ValueError(f"Frontend tool execution failed: {error_msg}")

        except TimeoutError as e:
            raise ValueError(
                f"Frontend tool {self.name} timed out after 60 seconds"
            ) from e

    def user_message(self, params: dict) -> str:
        """Generate user-friendly message for tool execution."""
        return f"Executing frontend tool: {self.name}"


class HelpMessageProcessor(MessageProcessor):
    """
    Processor for help mode messages.

    This processor handles help requests using the integrated help system
    with access to help-specific tools and documentation.
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
        """Process help messages with integrated help system."""
        last_message = chat_history[-1]

        try:
            if not last_message.provider:
                raise ValueError("Model provider is not set")

            log.debug(f"Processing help messages with model: {last_message.model}")

            # Create help tools combined with all available tools
            help_tools = [
                SearchNodesTool(),
                SearchExamplesTool(),
            ]
            help_tools_by_name = {t.name: t for t in help_tools}
            if last_message.tools:
                tools = await asyncio.gather(
                    *[
                        resolve_tool_by_name(name, processing_context.user_id)
                        for name in last_message.tools
                    ]
                )
            else:
                tools = []

            # Create UI proxy tools from manifest
            ui_tools = []
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                # Create proxy tools for each UI tool in the manifest
                for (
                    _tool_name,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    ui_tools.append(UIToolProxy(tool_manifest))

            # Create effective messages with help system prompt
            effective_messages = [Message(role="system", content=SYSTEM_PROMPT)]

            # If the latest message includes a workflow graph, include it as context
            # so the provider can ground answers in the user's current workflow.
            try:
                if getattr(last_message, "graph", None):
                    assert last_message.graph
                    # Use compact graph representation to minimize tokens
                    compact_graph = create_compact_graph_context(last_message.graph)
                    # Add workflow context IDs so model doesn't hallucinate them
                    context_info = {
                        "workflow_id": last_message.workflow_id,
                        "thread_id": last_message.thread_id,
                        "graph": compact_graph,
                    }
                    graph_context = Message(
                        role="system",
                        content=(
                            "Current workflow context. Use these exact IDs, do NOT invent workflow or thread IDs.\n"
                            + json.dumps(context_info)
                        ),
                    )
                    effective_messages.append(graph_context)
            except Exception:
                # Best-effort: if serialization fails, continue without graph context
                pass

            # Then append the chat history
            effective_messages.extend(chat_history)

            accumulated_content = ""
            unprocessed_messages = []
            iteration_count = 0

            # Process messages with tool execution
            while True:
                iteration_count += 1
                if iteration_count > MAX_TOOL_ITERATIONS:
                    log.warning(f"Hit MAX_TOOL_ITERATIONS limit ({MAX_TOOL_ITERATIONS})")
                    await self.send_message(
                        {"type": "chunk", "content": "\n\n[Reached tool iteration limit]", "done": False}
                    )
                    break
                # Persist unprocessed messages so the provider sees the full history
                effective_messages.extend(unprocessed_messages)
                messages_to_send = effective_messages
                unprocessed_messages = []
                assert last_message.model, "Model is required"

                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=help_tools + tools + ui_tools,
                    context_window=8192,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        accumulated_content += chunk.content
                        await self.send_message(
                            {"type": "chunk", "content": chunk.content, "done": False}
                        )
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing help tool call: {chunk.name}")

                        # Check if this is a UI tool
                        if (
                            hasattr(processing_context, "ui_tool_names")
                            and chunk.name in processing_context.ui_tool_names
                        ):
                            # Handle UI tool call using provider tool_call id to satisfy OpenAI API
                            tool_call_id = chunk.id
                            assert tool_call_id is not None, "Tool call id is required"
                            tool_call_message = {
                                "type": "tool_call",
                                "tool_call_id": tool_call_id,
                                "name": chunk.name,
                                "args": chunk.args,
                                "thread_id": last_message.thread_id,
                            }

                            await self.send_message(tool_call_message)

                            # Wait for result from frontend
                            try:
                                tool_bridge = getattr(
                                    processing_context, "tool_bridge", None
                                )
                                if tool_bridge is None:
                                    raise ValueError("Tool bridge not available")

                                result = await asyncio.wait_for(
                                    tool_bridge.create_waiter(tool_call_id),
                                    timeout=60.0,
                                )

                            except TimeoutError:
                                result = {
                                    "ok": False,
                                    "error": f"Frontend tool {chunk.name} timed out after 60 seconds",
                                }

                            # Create tool result with the same id
                            tool_result = ToolCall(
                                id=tool_call_id,
                                name=chunk.name,
                                args=chunk.args,
                                result=result,
                            )
                        else:
                            # Handle built-in help tools locally
                            if chunk.name in help_tools_by_name:
                                tool_impl = help_tools_by_name[chunk.name]
                                # Notify client about tool execution
                                await self.send_message(
                                    ToolCallUpdate(
                                        name=chunk.name,
                                        args=chunk.args,
                                        message=tool_impl.user_message(chunk.args),
                                    ).model_dump()
                                )
                                try:
                                    result = await tool_impl.process(
                                        processing_context, chunk.args
                                    )
                                    tool_result = ToolCall(
                                        id=chunk.id,
                                        name=chunk.name,
                                        args=chunk.args,
                                        result=result,
                                    )
                                except (ValueError, TypeError, KeyError) as e:
                                    # Tool execution failed due to invalid parameters
                                    # Return error to model so it can retry with corrected args
                                    log.warning(f"Help tool {chunk.name} failed: {e}. Returning error to model.")
                                    tool_result = ToolCall(
                                        id=chunk.id,
                                        name=chunk.name,
                                        args=chunk.args,
                                        result={
                                            "error": f"Tool execution failed: {str(e)}"
                                        },
                                    )
                            else:
                                # Try to process as regular server tool, with graceful error handling
                                try:
                                    tool_result = await self._run_tool(
                                        processing_context, chunk
                                    )
                                except ValueError as e:
                                    # Tool not found - return error to model instead of crashing
                                    # This helps smaller models that may hallucinate tool names
                                    log.warning(f"Tool not found: {chunk.name}. Returning error to model.")
                                    tool_result = ToolCall(
                                        id=chunk.id,
                                        name=chunk.name,
                                        args=chunk.args,
                                        result={
                                            "error": f"Tool '{chunk.name}' not found. Available tools: search_nodes, search_examples, and UI tools. Use search_nodes to find valid node types."
                                        },
                                    )

                        log.debug(
                            f"Help tool {chunk.name} execution complete, id={tool_result.id}"
                        )

                        # Add tool messages to unprocessed messages
                        assistant_msg = Message(
                            role="assistant",
                            tool_calls=[chunk],
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                            agent_mode=last_message.agent_mode or False,
                            help_mode=True,
                        )
                        unprocessed_messages.append(assistant_msg)
                        await self.send_message(assistant_msg.model_dump())

                        # Convert result to JSON
                        converted_result = self._recursively_model_dump(
                            tool_result.result
                        )
                        tool_result_json = json.dumps(converted_result)
                        tool_msg = Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            content=tool_result_json,
                        )
                        unprocessed_messages.append(tool_msg)
                        await self.send_message(tool_msg.model_dump())

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    break

            # Signal the end of the help stream
            await self.send_message({"type": "chunk", "content": "", "done": True})
            # await self.send_message(
            #     Message(
            #         role="assistant",
            #         content=accumulated_content if accumulated_content else None,
            #         thread_id=last_message.thread_id,
            #         workflow_id=last_message.workflow_id,
            #         provider=last_message.provider,
            #         model=last_message.model,
            #         agent_mode=last_message.agent_mode or False,
            #         help_mode=True,
            #     ).model_dump()
            # )

        except httpx.ConnectError as e:
            # Handle connection errors
            error_msg = self._format_connection_error(e)
            log.error(
                f"httpx.ConnectError in _process_help_messages: {e}", exc_info=True
            )

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                }
            )

            # Signal the end of the help stream with error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            # Return an error message
            await self.send_message(
                Message(
                    role="assistant",
                    content=f"I encountered a connection error while processing the help request: {error_msg}. Please check your network connection and try again.",
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                    help_mode=True,
                ).model_dump()
            )

        finally:
            # Always mark processing as complete
            self.is_processing = False

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
    ) -> ToolCall:
        """Execute a tool call and return the result."""
        from nodetool.agents.tools.tool_registry import resolve_tool_by_name

        tool = await resolve_tool_by_name(tool_call.name, context.user_id)
        log.debug(
            f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}"
        )

        # Send tool call to client
        await self.send_message(
            ToolCallUpdate(
                name=tool_call.name,
                args=tool_call.args,
                message=tool.user_message(tool_call.args),
            ).model_dump()
        )

        result = await tool.process(context, tool_call.args)
        log.debug(f"Tool {tool_call.name} returned: {result}")

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        )

    def _recursively_model_dump(self, obj):
        """Recursively convert BaseModel instances to dictionaries."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._recursively_model_dump(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._recursively_model_dump(item) for item in obj]
        else:
            return obj

    def _format_connection_error(self, e: httpx.ConnectError) -> str:
        """Format connection error message."""
        error_msg = str(e)
        if "nodename nor servname provided" in error_msg:
            return "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
        else:
            return f"Connection error: {error_msg}"
