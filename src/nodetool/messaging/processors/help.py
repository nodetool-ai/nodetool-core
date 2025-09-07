"""
Help message processor module.

This module provides the processor for help mode messages.
"""

import logging
import asyncio
import json
from typing import List
import httpx
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from pydantic import BaseModel

from nodetool.metadata.types import (
    Message,
    ToolCall,
)
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
)
from nodetool.agents.tools.help_tools import (
    SearchNodesTool,
    SearchExamplesTool,
)

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool
from .base import MessageProcessor

log = logging.getLogger(__name__)
# Log level is controlled by env (DEBUG/NODETOOL_LOG_LEVEL)

SYSTEM_PROMPT = """
You are a helpful assistant that provides help for Nodetool.

## Goal
- Provide fast, accurate, actionable help for Nodetool.
- If the user asks about a specific node type, provide a detailed explanation of the node type and how to use it.
- If the user asks to create a workflow, perform tool calls to create the workflow in the UI.
- Default to the shortest useful answer. No plans or repetition.

## Style
- Be direct and specific.
- Ask at most one clarifying question, and only if blocked.
- Do not reveal chain-of-thought; output only answers, tool results, and brief notes.
- Keep examples minimal; include code/JSON only if essential.

## Tool Preamble
- One-line preamble before tool use: say what you're about to do.

## Tools
- search_nodes(query): find node types and schemas.
- search_examples(query): find example workflows.

## Data Types
- str, int, float, bool, list, dict, tuple, union
- asset types: {"type": "image|audio|video|document", "uri": "https://example.com/image.png"}

## Important Node namespaces
- nodetool.agents
- nodetool.audio: audio processing
- nodetool.constants: constant values
- nodetool.image: image processing
- nodetool.input: user input
- nodetool.list: list processing
- nodetool.output: output
- nodetool.dictionary: dictionary processing
- nodetool.generators: generative nodes
- nodetool.data: dataframes, lists, dictionaries
- nodetool.text: text processing
- nodetool.code: code evaluation and execution
- nodetool.control: control flow (Loop, If, Switch)
- nodetool.video: video processing

## How Nodetool Works
- Visual editor: build node graphs; connect outputs to inputs; run with the play button at the bottom.
- Node Menu: press space or double click canvas to open the node menu.
- Nodes: typed inputs/outputs; includes AI/model provider nodes and utilities (conversion, control flow like Loop).
- Data flow: values travel across edges; lists/dataframes iterate via Loop; Preview renders outputs for inspection.
- Assets: manage media in the Assets panel; drag/drop onto canvas to create asset nodes; use as inputs/outputs.
- Models: run locally (GPU/MPS) or via providers (OpenAI, Anthropic, Replicate, Hugging Face, Ollama, ElevenLabs, Google).

## Building workflows
1. **Graph Structure:** Design workflows as Directed Acyclic Graphs (DAGs) with no cycles
2. **Data Flow:** Connect nodes via edges that represent data flow from inputs through processing to outputs
3. **Node Design:** Each node should have a clear, focused purpose
4. **Valid Node Types:** All nodes **must** correspond to available node types. Always use `search_nodes` to discover and verify node types
5. **Type Safety:** Ensure type compatibility throughout the workflow
6. **User-Centric Design:** Create graphs that solve the user's actual problem, not just technical requirements
7. **Reasoning Privacy:** Think step-by-step internally but do not reveal chain-of-thought. Only provide requested, structured outputs or tool calls.
8. **Determinism & Efficiency:** Minimize tokens. Prefer canonical, compact JSON for node specifications. Avoid markdown in structured outputs.

## Streaming
- Streaming nodes: output a stream of items.
- Output nodes automatically collect streamed items into lists.
- Use streaming for iterations and loops
- A producer node can generate a stream of items
- Connected nodes will be executed once for each item and continue to stream

## Agent Nodes
- Connect nodes as tools to an Agent node via dynamic outputs.
- A tool chain should end with a nodetool.workflows.base_node.ToolResult node.
- The dynamic output connecting to the tool node should have the correct type and properly named.
- For example, connect search_google output to search.google.GoogleSearch node's "keyword" property.
- Agent's chunk output streams tokens while the text output streams the final text.

## Node Metadata Structure
Each node type has specific metadata that defines:
- **properties**: Input fields/parameters the node accepts (these become targetHandles for edges)
- **outputs**: Output slots the node produces (these become sourceHandles for edges)
- **is_dynamic**: Boolean flag indicating if the node supports dynamic properties
- **supports_dynamic_outputs**: Boolean flag indicating if the node supports dynamic outputs

## Input and Output Node Mappings
Input nodes: string→StringInput, int→IntegerInput, float→FloatInput, bool→BooleanInput, list[any]→ListInput, image→ImageInput, video→VideoInput, document→DocumentInput, dataframe→DataFrameInput

Output nodes: string→StringOutput, int→IntegerOutput, float→FloatOutput, bool→BooleanOutput, list[any]→ListOutput, image→ImageOutput, video→VideoOutput, document→DocumentOutput, dataframe→DataFrameOutput
## Using `search_nodes` Efficiently

**EFFICIENCY PRIORITY:** Minimize the number of search iterations by being strategic and comprehensive:
- **Plan your searches:** Before starting, identify all the different types of processing you need (e.g., data transformation, aggregation, visualization, text generation)
- **Batch similar searches:** If you need multiple data processing nodes, search for them together with broader queries
- **Use specific, descriptive queries:** Instead of generic terms, use specific keywords that target exactly what you need
- **Target the right namespaces:** Most functionality is in `nodetool.data` (dataframes), `nodetool.text` (text processing), `nodetool.code` (custom code), `lib.*` (visualization/specialized tools)

## Using the `search_nodes` tool:
- Provide a `query` with keywords describing the node's function (e.g., "convert", "summarize", "filter data").
- **Start with targeted searches using `input_type` and `output_type` when you know the data types** - this reduces irrelevant results and speeds up the process
- **Only use broad searches without type parameters if you're unsure about available node types** 
- The available types for `input_type` and `output_type` are: "str", "int", "float", "bool", "list", "dict", "tuple", "union", "enum", "any"
- **Search for multiple related functionalities in a single query** when possible (e.g., "dataframe group aggregate sum" instead of separate searches)
 - Prefer fewer, more capable nodes over long chains of trivial nodes when functionality overlaps.

## Instructions - Node Selection
1. **Create ALL nodes including Input and Output nodes.** For Input and Output nodes, use the exact node types from the system prompt mappings (do NOT search for them). Only search for intermediate processing nodes.
   - For each item in the Input Schema, create a corresponding Input node with a `name` matching the schema's `name`.
   - For each item in the Output Schema, create a corresponding Output node with a `name` matching the schema's `name`.
   
2. **Search for intermediate processing nodes using `search_nodes`**. Be strategic with searches - use specific, targeted queries to find the most appropriate nodes. Prefer fewer, more powerful nodes over many simple ones to improve efficiency.
   - **For dataframe operations**: Search with relevant keywords (e.g., "GroupBy", "Aggregate", "Filter", "Transform", "dataframe"). Many dataframe nodes are in the `nodetool.data` namespace.
   - **For list operations**: Search with `input_type="list"` or `output_type="list"` and relevant keywords.
   - **For text operations**: Search with `input_type="str"` or `output_type="str"` (e.g., "concatenate", "regex", "template").
   - **For agents**: Search "agent". Verify their input/output types by inspecting their metadata from the search results before use.
   
3. **Type conversion patterns** (use keyword-based searches):
   - dataframe → array: Search "dataframe to array" or "to_numpy"
   - dataframe → string: Search "dataframe to string" or "to_csv"
   - array → dataframe: Search "array to dataframe" or "from_array"
   - list → item: Use iterator node
   - item → list: Use collector node

## Configuration Guidelines
- **For nodes found via `search_nodes`**: Check their metadata for required fields and create appropriate property entries.
- **Edge connections**: `{"type": "edge", "source": "source_node_id", "sourceHandle": "output_name"}`

## Important Handle Conventions
- **Most nodes have a single output**: The default output handle is often named "output". Always verify with `search_nodes` if unsure.
- **Input nodes**: Provide data through the `"output"` handle.
- **Output nodes**: Receive data through their `"value"` property.
- **Always check metadata from `search_nodes` results** for exceptions and exact input property names (targetHandles).
"""


class UIToolProxy(Tool):
    """Proxy tool that forwards tool calls to the frontend."""

    def __init__(self, tool_manifest: dict):
        # Configure base Tool fields expected by providers
        self.name = tool_manifest["name"]
        self.description = tool_manifest.get("description", "UI tool")
        # Providers expect JSON schema under input_schema
        self.input_schema = tool_manifest.get("parameters", {})

    async def process(self, context: ProcessingContext, args: dict) -> dict:
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
            "args": args,
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

        except asyncio.TimeoutError:
            raise ValueError(f"Frontend tool {self.name} timed out after 60 seconds")

    def user_message(self, args: dict) -> str:
        """Generate user-friendly message for tool execution."""
        return f"Executing frontend tool: {self.name}"


class HelpMessageProcessor(MessageProcessor):
    """
    Processor for help mode messages.

    This processor handles help requests using the integrated help system
    with access to help-specific tools and documentation.
    """

    def __init__(self, provider: ChatProvider):
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
                    tool_name,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    ui_tools.append(UIToolProxy(tool_manifest))

            # Create effective messages with help system prompt
            effective_messages = [Message(role="system", content=SYSTEM_PROMPT)]

            # If the latest message includes a workflow graph, include it as context
            # so the provider can ground answers in the user's current workflow.
            try:
                if getattr(last_message, "graph", None):
                    graph_dict = (
                        last_message.graph.model_dump()
                        if hasattr(last_message.graph, "model_dump")
                        else last_message.graph
                    )
                    # Keep this as a separate system message to avoid mutating user content
                    graph_context = Message(
                        role="system",
                        content=(
                            "Current workflow graph (JSON). Use this to answer questions "
                            "and suggest precise changes.\n" + json.dumps(graph_dict)
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

            # Process messages with tool execution
            while True:
                # Persist unprocessed messages so the provider sees the full history
                effective_messages.extend(unprocessed_messages)
                messages_to_send = effective_messages
                unprocessed_messages = []
                assert last_message.model, "Model is required"

                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=help_tools + tools + ui_tools,
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

                            except asyncio.TimeoutError:
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
                                result = await tool_impl.process(
                                    processing_context, chunk.args
                                )
                                tool_result = ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result=result,
                                )
                            else:
                                # Process regular server tool call
                                tool_result = await self._run_tool(
                                    processing_context, chunk
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
