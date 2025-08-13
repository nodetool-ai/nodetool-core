"""
Message processors for handling different types of chat messages.

This module provides a clean separation of concerns for different message processing
scenarios in the WebSocket chat system. Each processor handles a specific use case
and manages its own message queue for sending responses back to the client.
"""

import logging
import json
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Sequence
from asyncio import Queue
import httpx
from pydantic import BaseModel

from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    MessageImageContent,
    MessageAudioContent,
    MessageVideoContent,
    ImageRef,
    AudioRef,
    VideoRef,
    ToolCall,
)
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
    TaskUpdate,
    PlanningUpdate,
    SubTaskResult,
    OutputUpdate,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool
from nodetool.types.graph import Graph
from nodetool.chat.providers.base import ChatProvider

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

REGULAR_SYSTEM_PROMPT = """
You are a helpful assistant.

# IMAGE TOOLS
When using image tools, you will get an image url as result.
ALWAYS EMBED THE IMAGE AS MARKDOWN IMAGE TAG.
"""


class MessageProcessor(ABC):
    """
    Abstract base class for message processors.

    Each processor handles a specific type of message processing scenario
    and manages its own queue for sending messages back to the client.
    """

    def __init__(self):
        self.message_queue: Queue[Dict[str, Any]] = Queue()
        self.is_processing = True

    @abstractmethod
    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        **kwargs,
    ) -> Message:
        """
        Process messages and return the assistant's response.

        Args:
            chat_history: The complete chat history
            processing_context: Context for processing including user information
            tools: Available tools for the processor to use
            **kwargs: Additional processor-specific parameters

        Returns:
            Message: The assistant's response message
        """
        pass

    async def send_message(self, message: Dict[str, Any]):
        """
        Add a message to the queue for sending to the client.

        Args:
            message: The message dictionary to send
        """
        await self.message_queue.put(message)

    async def get_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the next message from the queue.

        Returns:
            The next message or None if queue is empty
        """
        try:
            return self.message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def has_messages(self) -> bool:
        """Check if there are messages in the queue."""
        return not self.message_queue.empty()


class RegularChatProcessor(MessageProcessor):
    """
    Processor for standard chat messages without workflows or special modes.

    This processor handles regular conversational interactions, including:
    - Text generation with streaming
    - Tool execution
    - Collection context queries
    """

    def __init__(self, provider: ChatProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        collections: Optional[List[str]] = None,
        graph: Optional[Graph] = None,
        **kwargs,
    ):
        """Process regular chat messages with optional collection context."""
        last_message = chat_history[-1]
        content = ""
        unprocessed_messages = []

        # Extract query text for collection search
        query_text = self._extract_query_text(last_message)

        if chat_history[0].role != "system":
            chat_history = [
                Message(role="system", content=REGULAR_SYSTEM_PROMPT)
            ] + chat_history

        # Query collections if specified
        collection_context = ""
        if collections:
            log.debug(f"Querying collections: {collections}")
            collection_context = await self._query_collections(
                collections, query_text, n_results=5
            )
            if collection_context:
                log.debug(
                    f"Retrieved collection context: {len(collection_context)} characters"
                )

        assert last_message.model, "Model is required"

        try:
            # Stream the response chunks
            while True:
                messages_to_send = chat_history + unprocessed_messages

                # Add collection context if available
                if collection_context:
                    messages_to_send = self._add_collection_context(
                        messages_to_send, collection_context
                    )
                    collection_context = ""  # Clear after first use

                unprocessed_messages = []

                log.debug(
                    f"Calling provider.generate_messages with {len(messages_to_send)} messages"
                )
                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=list(tools),
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        content += chunk.content
                        await self.send_message(chunk.model_dump())
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing tool call: {chunk.name}")

                        # Process the tool call
                        tool_result = await self._run_tool(
                            processing_context, chunk, tools, graph
                        )
                        log.debug(
                            f"Tool {chunk.name} execution complete, id={tool_result.id}"
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
                        )
                        unprocessed_messages.append(assistant_msg)

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

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    log.debug("No more unprocessed messages, completing generation")
                    break
                else:
                    log.debug(
                        f"Have {len(unprocessed_messages)} unprocessed messages, continuing loop"
                    )

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                Message(
                    role="assistant",
                    content=content if content else None,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                ).model_dump()
            )

        except httpx.ConnectError as e:
            # Handle connection errors
            error_msg = self._format_connection_error(e)
            log.error(f"httpx.ConnectError in process: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                }
            )

            # Signal completion even on error
            await self.send_message({"type": "chunk", "content": "", "done": True})

            # Return an error message
            await self.send_message(
                Message(
                    role="assistant",
                    content=f"I encountered a connection error: {error_msg}. Please check your network connection and try again.",
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                ).model_dump()
            )

        finally:
            # Always mark processing as complete
            self.is_processing = False

    def _extract_query_text(self, message: Message) -> str:
        """Extract query text from a message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    return content_item.text
        return ""

    async def _query_collections(
        self, collections: List[str], query_text: str, n_results: int
    ) -> str:
        """Query ChromaDB collections and return concatenated results."""
        if not collections or not query_text:
            return ""

        from nodetool.common.chroma_client import get_collection

        all_results = []

        for collection_name in collections:
            collection = get_collection(name=collection_name)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
            )

            if results["documents"] and results["documents"][0]:
                collection_results = f"\n\n### Results from {collection_name}:\n"
                for doc, metadata in zip(
                    results["documents"][0],
                    (
                        results["metadatas"][0]
                        if results["metadatas"]
                        else [{}] * len(results["documents"][0])
                    ),
                ):
                    doc_preview = f"{doc[:200]}..." if len(doc) > 200 else doc
                    collection_results += f"\n- {doc_preview}"
                all_results.append(collection_results)

        return "\n".join(all_results) if all_results else ""

    def _add_collection_context(
        self, messages: List[Message], collection_context: str
    ) -> List[Message]:
        """Add collection context as a system message before the last user message."""
        # Find the last user message index
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                last_user_index = i
                break

        if last_user_index >= 0:
            # Insert collection context before the last user message
            collection_message = Message(
                role="system",
                content=f"Context from knowledge base:\n{collection_context}",
            )
            return (
                messages[:last_user_index]
                + [collection_message]
                + messages[last_user_index:]
            )
        return messages

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
        tools: Sequence[Tool],
        graph: Optional[Graph] = None,
    ) -> ToolCall:
        """Execute a tool call and return the result."""
        from nodetool.agents.tools.base import resolve_tool_by_name

        tool = resolve_tool_by_name(tool_call.name, available_tools=tools)
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
        tools: Sequence[Tool],
        **kwargs,
    ):
        """Process help messages with integrated help system."""
        last_message = chat_history[-1]

        try:
            if not last_message.provider:
                raise ValueError("Model provider is not set")

            log.debug(f"Processing help messages with model: {last_message.model}")

            from nodetool.chat.help import SYSTEM_PROMPT
            from nodetool.agents.tools.help_tools import (
                SearchNodesTool,
                SearchExamplesTool,
            )
            from nodetool.agents.tools import CreateWorkflowTool, EditWorkflowTool

            # Create help tools combined with all available tools
            help_tools = [
                SearchNodesTool(),
                SearchExamplesTool(),
                CreateWorkflowTool(),
                EditWorkflowTool(),
            ]

            # Combine help tools with all other available tools
            all_available_tools = help_tools + list(tools)

            # Create effective messages with help system prompt
            effective_messages = [
                Message(role="system", content=SYSTEM_PROMPT)
            ] + chat_history

            accumulated_content = ""
            unprocessed_messages = []

            # Process messages with tool execution
            while True:
                messages_to_send = effective_messages + unprocessed_messages
                unprocessed_messages = []
                assert last_message.model, "Model is required"

                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=help_tools,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        accumulated_content += chunk.content
                        await self.send_message(
                            {"type": "chunk", "content": chunk.content, "done": False}
                        )
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing help tool call: {chunk.name}")

                        # Process the tool call
                        tool_result = await self._run_tool(
                            processing_context, chunk, all_available_tools
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
            await self.send_message(
                Message(
                    role="assistant",
                    content=accumulated_content if accumulated_content else None,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                    help_mode=True,
                ).model_dump()
            )

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
        tools: Sequence[Tool],
    ) -> ToolCall:
        """Execute a tool call and return the result."""
        from nodetool.agents.tools.base import resolve_tool_by_name

        tool = resolve_tool_by_name(tool_call.name, available_tools=tools)
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


class AgentMessageProcessor(MessageProcessor):
    """
    Processor for agent mode messages.

    This processor uses the Agent system to handle complex tasks by breaking
    them down into subtasks and executing them step by step.
    """

    def __init__(self, provider: ChatProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        **kwargs,
    ):
        """Process messages using the Agent system."""
        last_message = chat_history[-1]
        assert last_message.model, "Model is required for agent mode"

        # Extract objective from message content
        objective = self._extract_objective(last_message)

        # Get selected tools based on message.tools
        selected_tools = []
        if last_message.tools:
            tool_names = set(last_message.tools)
            selected_tools = [tool for tool in list(tools) if tool.name in tool_names]
            log.debug(
                f"Selected tools for agent: {[tool.name for tool in selected_tools]}"
            )

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


class WorkflowMessageProcessor(MessageProcessor):
    """
    Processor for workflow execution messages.

    This processor handles messages that include a workflow_id, executing
    the workflow and streaming results back to the client.
    """

    def __init__(self, user_id: Optional[str]):
        super().__init__()
        self.user_id = user_id

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        **kwargs,
    ):
        """Process messages for workflow execution."""
        import uuid
        from nodetool.workflows.workflow_runner import WorkflowRunner
        from nodetool.workflows.run_job_request import RunJobRequest
        from nodetool.workflows.run_workflow import run_workflow

        job_id = str(uuid.uuid4())
        last_message = chat_history[-1]
        assert last_message.workflow_id is not None, "Workflow ID is required"

        workflow_runner = WorkflowRunner(job_id=job_id)
        log.debug(
            f"Initialized WorkflowRunner for workflow {last_message.workflow_id} with job_id {job_id}"
        )

        # Update processing context with workflow_id
        processing_context.workflow_id = last_message.workflow_id

        request = RunJobRequest(
            workflow_id=last_message.workflow_id,
            messages=chat_history,
            graph=last_message.graph,
        )

        log.info(f"Running workflow for {last_message.workflow_id}")
        result = {}

        async for update in run_workflow(
            request,
            workflow_runner,
            processing_context,
        ):
            await self.send_message(update.model_dump())
            log.debug(f"Workflow update sent: {update.type}")
            if isinstance(update, OutputUpdate):
                result[update.node_name] = update.value

        # Signal completion
        await self.send_message({"type": "chunk", "content": "", "done": True})
        await self.send_message(
            self._create_response_message(result, last_message).model_dump()
        )

        # Always mark processing as complete
        self.is_processing = False

    def _create_response_message(self, result: dict, last_message: Message) -> Message:
        """Construct a response Message object from workflow results."""
        content = []
        for key, value in result.items():
            if isinstance(value, str):
                content.append(MessageTextContent(text=value))
            elif isinstance(value, list):
                content.append(MessageTextContent(text=" ".join(value)))
            elif isinstance(value, dict):
                if value.get("type") == "image":
                    content.append(MessageImageContent(image=ImageRef(**value)))
                elif value.get("type") == "video":
                    content.append(MessageVideoContent(video=VideoRef(**value)))
                elif value.get("type") == "audio":
                    content.append(MessageAudioContent(audio=AudioRef(**value)))
                else:
                    raise ValueError(f"Unknown type: {value}")
            else:
                raise ValueError(f"Unknown type: {type(value)} {value}")

        return Message(
            role="assistant",
            content=content,
            thread_id=last_message.thread_id,
            workflow_id=last_message.workflow_id,
            provider=last_message.provider,
            model=last_message.model,
            agent_mode=last_message.agent_mode or False,
            workflow_assistant=True,
        )


class WorkflowCreationProcessor(MessageProcessor):
    """
    Processor for workflow creation triggered by tools.

    This processor uses the GraphPlanner to create new workflows based
    on objectives provided through tool execution.
    """

    def __init__(self, provider: ChatProvider, user_id: Optional[str]):
        super().__init__()
        self.provider = provider
        self.user_id = user_id

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        objective: str,
        **kwargs,
    ):
        """Create a workflow using GraphPlanner."""
        try:
            from nodetool.agents.graph_planner import GraphPlanner

            # Get the current model from the last message
            if not chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = chat_history[-1]
            if not last_message.model:
                raise ValueError("No model specified in the current conversation")

            log.debug(f"Triggering workflow creation with model: {last_message.model}")

            planner = GraphPlanner(
                provider=self.provider,
                model=last_message.model,
                objective=objective,
                verbose=True,
            )

            # Send initial status
            await self.send_message(
                {
                    "type": "planning_update",
                    "phase": "Tool-Initiated Creation",
                    "status": "Starting",
                    "content": f"Creating workflow from tool request: {objective}",
                }
            )

            accumulated_content = ""
            workflow_graph = None

            # Execute graph planning and stream updates
            async for update in planner.create_graph(processing_context):
                if isinstance(update, PlanningUpdate):
                    await self.send_message(update.model_dump())

                    # Accumulate content for the final result
                    if update.content:
                        accumulated_content += (
                            f"[{update.phase}] {update.status}: {update.content}\n"
                        )

                elif isinstance(update, Chunk):
                    # If any chunks are emitted, send them too
                    await self.send_message(
                        {"type": "chunk", "content": update.content, "done": False}
                    )
                    accumulated_content += update.content

            # Get the generated graph
            if planner.graph:
                workflow_graph = planner.graph.model_dump()

                await self.send_message(
                    {
                        "type": "workflow_created",
                        "graph": workflow_graph,
                    }
                )
            else:
                raise ValueError("No graph was generated")
            await self.send_message(
                Message(
                    role="assistant",
                    content=accumulated_content,
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    workflow_assistant=True,
                ).model_dump()
            )
        except Exception as e:
            error_msg = f"Error creating workflow: {str(e)}"
            log.error(f"Error in workflow creation: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "workflow_creation_error",
                }
            )

            return {"success": False, "error": error_msg}

        finally:
            # Always mark processing as complete
            self.is_processing = False


class WorkflowEditingProcessor(MessageProcessor):
    """
    Processor for workflow editing triggered by tools.

    This processor uses the GraphPlanner to modify existing workflows based
    on objectives provided through tool execution.
    """

    def __init__(self, provider: ChatProvider, user_id: Optional[str]):
        super().__init__()
        self.provider = provider
        self.user_id = user_id

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        tools: Sequence[Tool],
        objective: str,
        workflow_id: Optional[str] = None,
        graph: Optional[Graph] = None,
        **kwargs,
    ):
        """Edit a workflow using GraphPlanner."""
        try:
            from nodetool.agents.graph_planner import GraphPlanner

            # Get the current model from the last message
            if not chat_history:
                raise ValueError("No chat history available to determine model")

            last_message = chat_history[-1]
            if not last_message.model:
                raise ValueError("No model specified in the current conversation")

            log.debug(f"Triggering workflow editing with model: {last_message.model}")

            planner = GraphPlanner(
                provider=self.provider,
                model=last_message.model,
                objective=objective,
                existing_graph=graph,
                verbose=True,
            )

            # Send initial status
            await self.send_message(
                {
                    "type": "planning_update",
                    "phase": "Tool-Initiated Editing",
                    "status": "Starting",
                    "content": f"Editing workflow from tool request: {objective}",
                }
            )

            accumulated_content = ""
            workflow_graph = None

            # Execute graph planning and stream updates
            async for update in planner.create_graph(processing_context):
                if isinstance(update, PlanningUpdate):
                    # Stream planning updates to client
                    await self.send_message(update.model_dump())

                    # Accumulate content for the final result
                    if update.content:
                        accumulated_content += (
                            f"[{update.phase}] {update.status}: {update.content}\n"
                        )

                elif isinstance(update, Chunk):
                    # If any chunks are emitted, send them too
                    await self.send_message(
                        {"type": "chunk", "content": update.content, "done": False}
                    )
                    accumulated_content += update.content

            # Get the generated graph
            if planner.graph:
                workflow_graph = planner.graph.model_dump()

                # Send the complete workflow graph
                await self.send_message(
                    {
                        "type": "workflow_updated",
                        "workflow_id": workflow_id,
                        "graph": workflow_graph,
                    }
                )
            else:
                raise ValueError("No graph was generated")

            await self.send_message(
                Message(
                    role="assistant",
                    content=accumulated_content,
                    thread_id=last_message.thread_id,
                    workflow_id=workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    workflow_assistant=True,
                ).model_dump()
            )

        except Exception as e:
            error_msg = f"Error editing workflow: {str(e)}"
            log.error(f"Error in workflow editing: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "workflow_editing_error",
                }
            )

            return {"success": False, "error": error_msg}

        finally:
            # Always mark processing as complete
            self.is_processing = False
