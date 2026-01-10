"""
Claude Agent SDK Message Processors
====================================

This module provides message processors that leverage the Claude Agent SDK
for agentic AI capabilities, including autonomous tool use, MCP server
integration, and multi-turn conversations.

See: https://docs.anthropic.com/en/docs/agent-sdk/overview

Architecture Overview
---------------------

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Claude Agent SDK Integration                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Request                                                            │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐     ┌──────────────────────────────────────────────┐   │
│  │ Processor   │────>│            ClaudeSDKClient                    │   │
│  │ .process()  │     │  ┌────────────────────────────────────────┐  │   │
│  └─────────────┘     │  │         MCP Server Registry            │  │   │
│                      │  │  ┌──────────┐  ┌──────────┐  ┌───────┐ │  │   │
│                      │  │  │ nodetool │  │  client  │  │  ...  │ │  │   │
│                      │  │  │ (backend)│  │(frontend)│  │       │ │  │   │
│                      │  │  └──────────┘  └──────────┘  └───────┘ │  │   │
│                      │  └────────────────────────────────────────┘  │   │
│                      └──────────────────────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│                           ┌─────────────────┐                            │
│                           │   Anthropic API │                            │
│                           │   (Claude)      │                            │
│                           └─────────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Tool Execution Flow
-------------------

The processors support two types of tools:

1. **Backend Tools** (nodetool MCP server)
   - Executed directly in Python
   - Examples: SearchNodes, GoogleSearch, Browser

2. **Frontend Tools** (client MCP server)
   - Executed in the browser via WebSocket bridge
   - Examples: ScrollTo, HighlightNode, OpenPanel

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Frontend Tool Execution Flow                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Claude ──> ToolUseBlock ──> Processor Loop                           │
│                                   │                                   │
│                    ┌──────────────┴──────────────┐                    │
│                    │                             │                    │
│                    ▼                             ▼                    │
│            [Backend Tool?]              [Frontend Tool?]              │
│                    │                             │                    │
│                    ▼                             ▼                    │
│           Execute directly            Push ID to Queue                │
│                    │                             │                    │
│                    │              ┌──────────────┴──────────────┐     │
│                    │              │                             │     │
│                    │              ▼                             ▼     │
│                    │      SDK Tool Handler            Message Loop    │
│                    │      (waits on queue)           sends ToolCall   │
│                    │              │                             │     │
│                    │              ▼                             │     │
│                    │      tool_bridge.create_waiter()           │     │
│                    │              │                             │     │
│                    │              │      ┌──────────────────────┘     │
│                    │              │      │                            │
│                    │              │      ▼                            │
│                    │              │   Frontend executes tool          │
│                    │              │      │                            │
│                    │              │      ▼                            │
│                    │              │   WebSocket: tool_result          │
│                    │              │      │                            │
│                    │              ▼      ▼                            │
│                    │      tool_bridge.resolve()                       │
│                    │              │                                   │
│                    │              ▼                                   │
│                    │      Handler returns result                      │
│                    │              │                                   │
│                    └──────────────┴───────────────────────────────>   │
│                                   │                                   │
│                                   ▼                                   │
│                          SDK sends to Claude                          │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

Retry Mechanism
---------------

The SDK can fail with "ProcessTransport is not ready" due to subprocess
race conditions. Both processors implement retry with exponential backoff:

```
Attempt 1 ──> Fail ──> Wait 1s ──> Attempt 2 ──> Fail ──> Wait 2s ──> Attempt 3
```

Classes
-------
- ClaudeAgentMessageProcessor: Full agent mode for complex tasks
- ClaudeAgentHelpMessageProcessor: Help mode for workflow assistance

Helper Functions
----------------
- json_schema_to_simple_types(): Convert JSON Schema to SDK format
- create_sdk_tool_from_nodetool(): Wrap nodetool Tool for SDK
- create_sdk_tool_from_frontend(): Create bridge tool for frontend execution
- create_mcp_server_from_tools(): Build MCP server from tool list
- create_prompt_generator(): Create async generator for streaming input
- is_transport_error(): Check for SDK transport race conditions
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from collections.abc import AsyncIterator
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
from claude_agent_sdk import (
    tool as sdk_tool,
)

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    ToolCall,
)
from nodetool.workflows.processing_context import ProcessingContext

from .help_message_processor import SYSTEM_PROMPT as HELP_SYSTEM_PROMPT
from .message_processor import MessageProcessor

log = get_logger(__name__)
log.setLevel(logging.DEBUG)


def is_transport_error(e: BaseException) -> bool:
    """Check if exception is related to transport race condition, handling ExceptionGroups."""
    error_str = str(e)
    if "ProcessTransport is not ready" in error_str or "CLIConnectionError" in error_str:
        return True

    # Handle ExceptionGroup (Python 3.11+)
    if hasattr(e, "exceptions"):
        for sub_exc in e.exceptions:  # type: ignore
            if is_transport_error(sub_exc):
                return True
    return False


def json_schema_to_simple_types(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Convert JSON Schema to simple type mapping for the SDK @tool decorator.

    The SDK expects: {"param_name": type} or {"param_name": {"type": "string", ...}}
    """
    if not schema:
        return {}

    properties = schema.get("properties", {})
    if not properties:
        # If no properties, assume the schema itself is the type mapping
        return schema

    # Convert JSON Schema properties to simple types
    result = {}
    for name, prop in properties.items():
        json_type = prop.get("type", "string")

        # Map JSON Schema types to Python types
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        result[name] = type_mapping.get(json_type, str)

    return result


def create_sdk_tool_from_nodetool(tool: Tool, context: ProcessingContext) -> Any:
    """
    Create an SDK MCP tool from a nodetool Tool using the @tool decorator.

    Args:
        tool: The nodetool Tool to convert
        context: Processing context for tool execution

    Returns:
        An SDK tool decorated function
    """
    # Convert JSON Schema to simple type mapping for SDK
    simple_schema = json_schema_to_simple_types(tool.input_schema)

    # Define the async handler function
    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await tool.process(context, args)
            # Convert result to SDK format
            if isinstance(result, dict):
                content_text = json.dumps(result)
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

    # Apply the @tool decorator by calling it as a function
    # This is equivalent to: @sdk_tool(name, description, schema)
    decorated_tool = sdk_tool(tool.name, tool.description, simple_schema)(tool_handler)

    return decorated_tool


def create_sdk_tool_from_frontend(
    name: str,
    tool_def: dict,
    tool_bridge: Any,
    id_queue: asyncio.Queue,
) -> Any:
    """
    Create an SDK MCP tool that bridges to a frontend tool.

    Args:
        name: Name of the tool
        tool_def: Tool definition from manifest
        tool_bridge: Bridge for waiting on frontend results
        id_queue: Queue to receive the tool call ID from the message loop
    """
    description = tool_def.get("description", "")
    input_schema = tool_def.get("input_schema", {})
    simple_schema = json_schema_to_simple_types(input_schema)

    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        try:
            # Wait for tool ID pushed by the loop (with timeout)
            # This ensures we have the ID matching the tool call currently being processed by the SDK
            tool_id = await asyncio.wait_for(id_queue.get(), timeout=30.0)

            log.debug(f"Executing frontend tool {name} (id={tool_id}) with args: {args}")

            if not tool_bridge:
                raise ValueError("Tool bridge not available")

            # Create waiter for frontend result
            waiter = tool_bridge.create_waiter(tool_id)

            # Wait for result (long timeout for user interaction)
            # The result payload is what we get from UnifiedWebSocketRunner
            payload = await asyncio.wait_for(waiter, timeout=600.0)

            # Extract result from payload
            # Payload typically: {"type": "tool_result", "result": ..., "status": ...}
            result_data = payload.get("result", payload)

            content_text = json.dumps(result_data) if isinstance(result_data, dict | list) else str(result_data)

            return {
                "content": [{"type": "text", "text": content_text}],
            }
        except TimeoutError:
            log.warning(f"Timeout waiting for frontend tool {name}")
            return {
                "content": [{"type": "text", "text": "Error: Timeout waiting for frontend tool execution"}],
                "is_error": True,
            }
        except Exception as e:
            log.error(f"Frontend tool {name} execution error: {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "is_error": True,
            }

    # Apply the @tool decorator
    return sdk_tool(name, description, simple_schema)(tool_handler)


async def create_mcp_server_from_tools(
    tools: list[Tool], context: ProcessingContext, server_name: str = "nodetool"
) -> Any:
    """
    Create an MCP server from a list of nodetool tools.

    Args:
        tools: List of nodetool Tools to convert
        context: Processing context for tool execution
        server_name: Name for the MCP server

    Returns:
        An MCP server with the converted tools
    """
    sdk_tools = [create_sdk_tool_from_nodetool(tool, context) for tool in tools]

    return create_sdk_mcp_server(name=server_name, version="1.0.0", tools=sdk_tools)


def create_prompt_generator(objective: str) -> AsyncIterator[dict]:
    """Create an async generator for streaming input mode required by MCP servers."""

    async def generator():
        yield {"type": "user", "message": {"role": "user", "content": objective}}

    return generator()


async def query_with_retry(
    objective: str,
    options: ClaudeAgentOptions,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> AsyncIterator[Any]:
    """
    Run SDK query with retry logic for transport race conditions.

    The Claude Agent SDK can fail with "ProcessTransport is not ready for writing"
    due to a race condition with the subprocess startup. This function retries
    the query with exponential backoff.

    Args:
        objective: The prompt/objective for the query
        options: ClaudeAgentOptions for the query
        max_retries: Maximum number of retries (default: 3)
        initial_delay: Initial delay between retries in seconds (default: 1.0)

    Yields:
        Messages from the SDK query
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # Create a fresh prompt generator for each attempt
            prompt_input = create_prompt_generator(objective)

            async for message in query(prompt=prompt_input, options=options):
                yield message

            # If we get here, query completed successfully
            return

        except Exception as e:
            # Check if it's the transport race condition error
            if is_transport_error(e):
                last_error = e
                delay = initial_delay * (2**attempt)
                log.warning(f"SDK transport error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                # For other errors, don't retry
                raise

    # All retries exhausted
    if last_error:
        raise last_error


class ClaudeAgentMessageProcessor(MessageProcessor):
    """
    Agent mode message processor using the Claude Agent SDK.

    This processor enables autonomous task execution by leveraging Claude's
    native agentic capabilities through the official Claude Agent SDK.

    Architecture
    ------------

    ```
    ┌─────────────────────────────────────────────────────────┐
    │              ClaudeAgentMessageProcessor                 │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  process() Entry Point                                   │
    │       │                                                  │
    │       ├── Extract objective from message                 │
    │       ├── Format chat history context                    │
    │       ├── Resolve backend tools (nodetool registry)      │
    │       ├── Resolve frontend tools (client manifest)       │
    │       ├── Create MCP servers for both                    │
    │       │                                                  │
    │       ▼                                                  │
    │  ┌─────────────────────────────────────────────┐         │
    │  │         ClaudeSDKClient Session             │         │
    │  │                                             │         │
    │  │  for attempt in retries:                    │         │
    │  │      async with ClaudeSDKClient as client:  │         │
    │  │          await client.query(prompt)         │         │
    │  │          async for msg in receive():        │         │
    │  │              ├── TextBlock → stream chunk   │         │
    │  │              ├── ToolUseBlock → emit call   │         │
    │  │              └── ResultMessage → complete   │         │
    │  └─────────────────────────────────────────────┘         │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    ```

    Features
    --------
    - **Autonomous Execution**: Claude decides which tools to use
    - **Backend Tools**: Execute Python tools via MCP server
    - **Frontend Tools**: Bridge to browser-based tools via WebSocket
    - **Retry Logic**: Handles SDK transport race conditions
    - **Streaming**: Real-time response chunks to client

    Tool Types Supported
    --------------------
    1. Backend (nodetool server): SearchNodes, Browser, GoogleSearch, etc.
    2. Frontend (client server): ScrollTo, HighlightNode, OpenPanel, etc.

    Example Usage
    -------------
    ```python
    processor = ClaudeAgentMessageProcessor(api_key="...")
    await processor.process(
        chat_history=[user_message],
        processing_context=context,
    )
    # Results streamed via processor.message_queue
    ```

    Attributes
    ----------
    api_key : str | None
        Anthropic API key for authentication
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize the Claude Agent SDK processor.

        Args:
            api_key: Optional Anthropic API key. If not provided,
                     will use ANTHROPIC_API_KEY environment variable.
            base_url: Optional base URL for API requests. If provided,
                      sets ANTHROPIC_BASE_URL for Claude SDK. Used for
                      Anthropic-compatible providers like MiniMax.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            log.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")

    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """
        Process messages using the Claude Agent SDK.

        This method:
        1. Extracts the objective from the user's message
        2. Configures the agent options with MCP tools
        3. Uses the query() function to run the agent
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

        # Build conversation context from history
        history_context = self._format_chat_history(chat_history)
        if history_context:
            objective = f"{objective}\n\n{history_context}"

        log.info(f"Starting Claude Agent SDK execution with objective: {objective[:100]}...")

        # Built-in tools to allow
        allowed_tools = ["WebSearch"]

        # Resolve custom tools from nodetool registry
        mcp_servers = {}
        custom_tools: list[Tool] = []

        if last_message.tools:
            log.debug(f"User selected tools: {last_message.tools}")
            resolved_tools = await asyncio.gather(
                *[resolve_tool_by_name(name, processing_context.user_id) for name in last_message.tools]
            )
            custom_tools = [t for t in resolved_tools if t is not None]

            if custom_tools:
                log.debug(f"Creating MCP server with tools: {[t.name for t in custom_tools]}")
                mcp_server = await create_mcp_server_from_tools(custom_tools, processing_context, "nodetool")
                mcp_servers["nodetool"] = mcp_server

                # Add all tool names to allowed_tools with MCP format
                # Format: mcp__{server_name}__{tool_name}
                for tool in custom_tools:
                    allowed_tools.append(f"mcp__nodetool__{tool.name}")

        # Resolve frontend tools from client manifest/tool bridge
        frontend_tool_queues: dict[str, asyncio.Queue] = {}

        if processing_context.client_tools_manifest:
            frontend_tools = []
            for name, tool_def in processing_context.client_tools_manifest.items():
                q = asyncio.Queue()
                frontend_tool_queues[name] = q
                t = create_sdk_tool_from_frontend(name, tool_def, processing_context.tool_bridge, q)
                frontend_tools.append(t)
                allowed_tools.append(f"mcp__client__{name}")

            if frontend_tools:
                log.debug(f"Creating client MCP server with {len(frontend_tools)} tools")
                client_mcp = create_sdk_mcp_server("client", "1.0.0", frontend_tools)
                mcp_servers["client"] = client_mcp

        try:
            # Set the API key in environment if provided
            if self.api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key

            # Set custom base URL for Anthropic-compatible providers (e.g., MiniMax)
            if self.base_url:
                os.environ["ANTHROPIC_BASE_URL"] = self.base_url

            # Configure agent options
            options = ClaudeAgentOptions(
                model=last_message.model,
                system_prompt=self._build_system_prompt(objective, chat_history),
                allowed_tools=allowed_tools,
                disallowed_tools=["Read", "Edit", "Glob", "Grep", "Bash"],
                mcp_servers=mcp_servers if mcp_servers else {},
                permission_mode="default",
                max_turns=kwargs.get("max_turns", 50),
            )

            accumulated_content = ""
            step_count = 0
            max_retries = 3
            initial_delay = 1.0
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Always use streaming input mode via generator to avoid SDK race conditions
                    prompt_input = create_prompt_generator(objective)

                    async with ClaudeSDKClient(options=options) as client:
                        # Send the query
                        await client.query(prompt=prompt_input)

                        # Process responses
                        async for message in client.receive_response():
                            # Check for cancellation
                            if self.is_cancelled():
                                log.info("Claude agent processing cancelled by user")
                                raise asyncio.CancelledError("Processing cancelled by user")

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
                                                "thread_id": last_message.thread_id,
                                            }
                                        )
                                    elif isinstance(block, ToolUseBlock):
                                        # Tool use block
                                        step_count += 1
                                        tool_name = block.name
                                        tool_id = block.id
                                        tool_input = block.input

                                        log.debug(f"Processing tool call: {tool_name}")

                                        # If it's a frontend tool, queue the ID for the handler
                                        if tool_name in frontend_tool_queues:
                                            frontend_tool_queues[tool_name].put_nowait(tool_id)

                                        tool_call = ToolCall(
                                            id=tool_id,
                                            name=tool_name,
                                            args=tool_input,
                                            step_id=str(step_count),
                                        )
                                        # Emit tool call as a Message object
                                        await self.send_message(
                                            Message(
                                                id=str(uuid4()),
                                                role="assistant",
                                                content=None,
                                                tool_calls=[tool_call],
                                            ).model_dump()
                                        )

                            elif isinstance(message, ResultMessage):
                                log.info(f"Claude agent execution completed. Duration: {message.duration_ms}ms")
                                await self.send_message(
                                    {
                                        "type": "chunk",
                                        "content": "",
                                        "done": True,
                                        "thread_id": last_message.thread_id,
                                    }
                                )

                    # If we complete the client block successfully, return
                    return

                except Exception as e:
                    # Check for cancellation first
                    if isinstance(e, asyncio.CancelledError):
                        raise

                    # Check for transport errors
                    if is_transport_error(e):
                        last_error = e
                        delay = initial_delay * (2**attempt)
                        log.warning(
                            f"SDK transport error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # For other errors, don't retry
                        raise

            # All retries exhausted
            if last_error:
                raise last_error

        except asyncio.CancelledError:
            log.info("Claude agent processing was cancelled")
            await self.send_message({"type": "generation_stopped", "message": "Generation stopped by user"})
            raise

        except Exception as e:
            log.error(f"Error in Claude Agent SDK execution: {e}", exc_info=True)
            error_msg = f"Claude Agent SDK execution error: {str(e)}"

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

    def _format_chat_history(self, chat_history: list[Message]) -> str:
        """Format chat history into a readable context string."""
        if len(chat_history) <= 1:
            return ""

        # Exclude the last message (current objective) and limit history
        history_messages = chat_history[:-1]
        # Limit to last 10 messages to avoid context overflow
        history_messages = history_messages[-10:]

        if not history_messages:
            return ""

        formatted_lines = ["## Previous Conversation"]
        for msg in history_messages:
            role = msg.role.capitalize()
            content = self._extract_objective(msg)
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted_lines.append(f"\n**{role}**: {content}")

        return "\n".join(formatted_lines)

    def _build_system_prompt(self, objective: str, chat_history: list[Message]) -> str:
        """Build a system prompt for the Claude Agent SDK with conversation context."""
        return f"""You are an AI assistant powered by the Claude Agent SDK.
Your objective is to help the user with their request: {objective}

You have access to various tools. Use them as needed to accomplish the task.
Be thorough, accurate, and provide helpful responses.
When using tools, explain what you're doing and why.
If a tool fails, try alternative approaches or ask for clarification."""


class ClaudeAgentHelpMessageProcessor(MessageProcessor):
    """
    Help mode message processor using the Claude Agent SDK.

    This processor provides intelligent workflow assistance by combining
    Claude's agentic capabilities with workflow-specific tools and knowledge.

    Architecture
    ------------

    ```
    ┌─────────────────────────────────────────────────────────┐
    │            ClaudeAgentHelpMessageProcessor               │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  Built-in Help Tools (always available):                 │
    │  ├── SearchNodesTool: Find workflow nodes by query       │
    │  └── SearchExamplesTool: Find example workflows          │
    │                                                          │
    │  + Custom tools from message + Frontend tools            │
    │                                                          │
    │  ┌─────────────────────────────────────────────┐         │
    │  │         Uses HELP_SYSTEM_PROMPT             │         │
    │  │  (workflow-specific instructions)           │         │
    │  └─────────────────────────────────────────────┘         │
    │                                                          │
    │  Same SDK flow as ClaudeAgentMessageProcessor            │
    │  but with help-focused toolset and prompting             │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    ```

    Differences from ClaudeAgentMessageProcessor
    ---------------------------------------------

    | Aspect          | AgentProcessor      | HelpProcessor        |
    |-----------------|---------------------|----------------------|
    | Purpose         | General tasks       | Workflow assistance  |
    | System Prompt   | Generic agent       | HELP_SYSTEM_PROMPT   |
    | Default Tools   | WebSearch           | SearchNodes, Examples|
    | Max Turns       | 50                  | 25                   |

    Features
    --------
    - **Workflow Knowledge**: Access to node and example search
    - **Graph Context**: Understands current workflow structure
    - **Concise Responses**: Optimized prompting for brief answers
    - **Tool Integration**: Same frontend/backend tool bridge

    Example Usage
    -------------
    ```python
    processor = ClaudeAgentHelpMessageProcessor(api_key="...")
    await processor.process(
        chat_history=[help_request],
        processing_context=context,
    )
    # Help response streamed via processor.message_queue
    ```

    Attributes
    ----------
    api_key : str | None
        Anthropic API key for authentication
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize the Claude Agent SDK help processor.

        Args:
            api_key: Optional Anthropic API key. If not provided,
                     will use ANTHROPIC_API_KEY environment variable.
            base_url: Optional base URL for API requests. If provided,
                      sets ANTHROPIC_BASE_URL for Claude SDK. Used for
                      Anthropic-compatible providers like MiniMax.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url
        if not self.api_key:
            log.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")

    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """
        Process help messages using the Claude Agent SDK.

        Args:
            chat_history: The complete chat history
            processing_context: Context for processing including user information
            **kwargs: Additional processor-specific parameters
        """

        last_message = chat_history[-1]
        assert last_message.model, "Model is required for help mode"

        # Extract objective from message content
        objective = self._extract_objective(last_message)

        # Build conversation context from history
        history_context = self._format_chat_history(chat_history)
        if history_context:
            objective = f"{objective}\n\n{history_context}"

        log.info(f"Starting Claude Agent SDK help execution with objective: {objective[:100]}...")

        # Built-in tools for help mode (read-only, no edit/bash)
        allowed_tools = ["WebSearch"]

        # Always include help tools for workflow assistance
        from nodetool.agents.tools.help_tools import SearchExamplesTool, SearchNodesTool

        help_tools: list[Tool] = [
            SearchNodesTool(),
            SearchExamplesTool(),
        ]

        # Resolve additional custom tools from nodetool registry
        custom_tools: list[Tool] = []

        if last_message.tools:
            log.debug(f"User selected tools: {last_message.tools}")
            resolved_tools = await asyncio.gather(
                *[resolve_tool_by_name(name, processing_context.user_id) for name in last_message.tools]
            )
            custom_tools = [t for t in resolved_tools if t is not None]

        # Combine help tools with custom tools
        all_tools = help_tools + custom_tools

        log.debug(f"Creating MCP server with help tools: {[t.name for t in all_tools]}")
        mcp_server = await create_mcp_server_from_tools(all_tools, processing_context, "nodetool")
        mcp_servers = {"nodetool": mcp_server}

        # Add all tool names to allowed_tools with MCP format
        for tool in all_tools:
            allowed_tools.append(f"mcp__nodetool__{tool.name}")

        # Resolve frontend tools from client manifest/tool bridge
        frontend_tool_queues: dict[str, asyncio.Queue] = {}

        if processing_context.client_tools_manifest:
            frontend_tools = []
            for name, tool_def in processing_context.client_tools_manifest.items():
                q = asyncio.Queue()
                frontend_tool_queues[name] = q
                t = create_sdk_tool_from_frontend(name, tool_def, processing_context.tool_bridge, q)
                frontend_tools.append(t)
                allowed_tools.append(f"mcp__client__{name}")

            if frontend_tools:
                log.debug(f"Creating client MCP server with {len(frontend_tools)} tools")
                client_mcp = create_sdk_mcp_server("client", "1.0.0", frontend_tools)
                mcp_servers["client"] = client_mcp

        try:
            # Set the API key in environment if provided
            if self.api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key

            # Set custom base URL for Anthropic-compatible providers (e.g., MiniMax)
            if self.base_url:
                os.environ["ANTHROPIC_BASE_URL"] = self.base_url

            # Configure agent options with help-specific system prompt
            options = ClaudeAgentOptions(
                model=last_message.model,
                system_prompt=HELP_SYSTEM_PROMPT,
                allowed_tools=allowed_tools,
                disallowed_tools=["Read", "Edit", "Glob", "Grep", "Bash"],
                mcp_servers=mcp_servers,
                permission_mode="default",
                max_turns=kwargs.get("max_turns", 25),
            )

            accumulated_content = ""

            max_retries = 3
            initial_delay = 1.0
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Always use streaming input mode via generator to avoid SDK race conditions
                    prompt_input = create_prompt_generator(objective)

                    async with ClaudeSDKClient(options=options) as client:
                        # Send the query
                        await client.query(prompt=prompt_input)

                        # Process responses
                        async for message in client.receive_response():
                            # Check for cancellation
                            if self.is_cancelled():
                                log.info("Claude help processing cancelled by user")
                                raise asyncio.CancelledError("Processing cancelled by user")

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
                                                "thread_id": last_message.thread_id,
                                            }
                                        )
                                    elif isinstance(block, ToolUseBlock):
                                        # Tool use block
                                        tool_name = block.name
                                        tool_id = block.id
                                        tool_input = block.input

                                        log.debug(f"Help mode tool call: {tool_name}")

                                        # If it's a frontend tool, queue the ID for the handler
                                        if tool_name in frontend_tool_queues:
                                            frontend_tool_queues[tool_name].put_nowait(tool_id)

                                        tool_call = ToolCall(
                                            id=tool_id,
                                            name=tool_name,
                                            args=tool_input,
                                        )
                                        # Emit tool call
                                        await self.send_message(
                                            Message(
                                                id=str(uuid4()),
                                                role="assistant",
                                                content=None,
                                                tool_calls=[tool_call],
                                            ).model_dump()
                                        )

                            elif isinstance(message, ResultMessage):
                                log.info(f"Claude help execution completed. Duration: {message.duration_ms}ms")
                                await self.send_message(
                                    {
                                        "type": "chunk",
                                        "content": "",
                                        "done": True,
                                        "thread_id": last_message.thread_id,
                                    }
                                )

                    # If we complete the client block successfully, return
                    return

                except Exception as e:
                    # Check for cancellation first
                    if isinstance(e, asyncio.CancelledError):
                        raise

                    # Check for transport errors
                    if is_transport_error(e):
                        last_error = e
                        delay = initial_delay * (2**attempt)
                        log.warning(
                            f"SDK transport error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # For other errors, don't retry
                        raise

            # All retries exhausted
            if last_error:
                raise last_error

        except asyncio.CancelledError:
            log.info("Claude help processing was cancelled")
            await self.send_message({"type": "generation_stopped", "message": "Generation stopped by user"})
            raise

        except Exception as e:
            log.error(f"Error in Claude Agent SDK help execution: {e}", exc_info=True)
            error_msg = f"Claude Agent SDK help error: {str(e)}"

            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "help_error",
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

            await self.send_message(
                Message(
                    role="assistant",
                    content=error_msg,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    help_mode=True,
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
        return "Help me with my workflow"

    def _format_chat_history(self, chat_history: list[Message]) -> str:
        """Format chat history into a readable context string."""
        if len(chat_history) <= 1:
            return ""

        # Exclude the last message (current objective) and limit history
        history_messages = chat_history[:-1]
        # Limit to last 10 messages to avoid context overflow
        history_messages = history_messages[-10:]

        if not history_messages:
            return ""

        formatted_lines = ["## Previous Conversation"]
        for msg in history_messages:
            role = msg.role.capitalize()
            content = self._extract_objective(msg)
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted_lines.append(f"\n**{role}**: {content}")

        return "\n".join(formatted_lines)
