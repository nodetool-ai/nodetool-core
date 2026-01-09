"""
Chat Workflow Message Processor Module
=======================================

This module provides the processor for workflows with run_mode="chat".

For workflows designated as chat workflows, this processor:
1. Receives chat messages from the frontend
2. Runs the workflow with ChatInput node(s) set to message history
3. Returns workflow outputs as chat responses

The ChatInput node is mandatory for chat workflows and receives:
- Current message content
- Full message history for context

Architecture:
```
User Message -> ChatWorkflowMessageProcessor -> Run Workflow with ChatInput
                                              -> Return Workflow Outputs as Response
```

Usage:
------
To enable chat mode for a workflow:
1. Set the workflow's `run_mode` field to "chat" in the workflow model
2. Add one or more InputNode instances to the workflow with name "chat_input" or "messages"
3. The input nodes will receive the full chat history as a list of messages
4. The workflow's output nodes will be converted to chat responses

Example Workflow Structure:
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│ ChatInput   │─────>│ Processing   │─────>│ OutputNode  │
│ (messages)  │      │ Nodes        │      │ (response)  │
└─────────────┘      └──────────────┘      └─────────────┘
```

Message Format:
---------------
The ChatInput receives messages as a list of dictionaries:
```python
[
    {
        "role": "user",
        "content": "Hello",
        "created_at": "2024-01-01T00:00:00Z"
    },
    {
        "role": "assistant",
        "content": "Hi there!",
        "created_at": "2024-01-01T00:00:01Z"
    }
]
```
"""

import json
import uuid
from typing import Any, List, Optional

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    AudioRef,
    ImageRef,
    Message,
    MessageAudioContent,
    MessageImageContent,
    MessageTextContent,
    MessageVideoContent,
    VideoRef,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import OutputUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

from .message_processor import MessageProcessor

log = get_logger(__name__)


def _serialize_message(msg: Message) -> dict:
    """
    Serialize a Message object to a dictionary for workflow params.

    Preserves all message fields including rich content (images, videos, audio),
    thread_id, collections, input_files, etc.
    """
    msg_dict = msg.model_dump()

    return msg_dict


class ChatWorkflowMessageProcessor(MessageProcessor):
    """
    Processor for chat-enabled workflows (run_mode="chat").

    This processor handles messages for workflows that have been configured
    with run_mode="chat". It runs the workflow with ChatInput nodes set to
    the message history and streams results back to the client.

    Key Features:
    - Mandatory ChatInput node receives message history
    - Each message triggers a workflow execution
    - Workflow outputs are converted to chat responses
    - Streaming support for real-time updates
    """

    def __init__(self, user_id: Optional[str]):
        super().__init__()
        self.user_id = user_id

    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process messages for chat-enabled workflows."""
        job_id = str(uuid.uuid4())
        last_message = chat_history[-1]

        if last_message.workflow_id is None:
            raise ValueError("Workflow ID is required for chat workflow processing")

        processing_context.user_id = self.user_id or processing_context.user_id

        workflow_runner = WorkflowRunner(job_id=job_id)
        log.debug(f"Initialized WorkflowRunner for chat workflow {last_message.workflow_id} with job_id {job_id}")

        # Update processing context with workflow_id
        processing_context.workflow_id = last_message.workflow_id

        # Prepare workflow parameters
        # New interface: pass full message object and message history
        params = self._prepare_workflow_params(chat_history, last_message)

        request = RunJobRequest(
            workflow_id=last_message.workflow_id,
            messages=chat_history,
            params=params,
            graph=last_message.graph,
        )

        log.info(f"Running chat workflow {last_message.workflow_id} with {len(chat_history)} messages")
        result = {}
        workflow_id = last_message.workflow_id

        try:
            async for update in run_workflow(
                request,
                workflow_runner,
                processing_context,
            ):
                # Add job_id and workflow_id to all messages for UI visualization consistency
                # This matches the behavior of WebSocketRunner for normal workflows
                msg = update.model_dump()
                msg["job_id"] = job_id
                msg["workflow_id"] = workflow_id
                await self.send_message(msg)
                log.debug(f"Chat workflow update sent: {update.type}")
                if isinstance(update, OutputUpdate):
                    result[update.node_name] = update.value

            # Signal completion with job_id and workflow_id
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "job_id": job_id,
                    "workflow_id": workflow_id,
                    "thread_id": last_message.thread_id,
                }
            )
            response_msg = self._create_response_message(result, last_message).model_dump()
            response_msg["job_id"] = job_id
            response_msg["workflow_id"] = workflow_id
            await self.send_message(response_msg)
            log.info(f"Chat workflow {last_message.workflow_id} completed successfully with job_id {job_id}")

        except Exception as e:
            log.error(f"Error processing chat workflow: {e}", exc_info=True)
            await self.send_message(
                {
                    "type": "error",
                    "message": f"Error processing chat workflow: {str(e)}",
                    "job_id": job_id,
                    "workflow_id": workflow_id,
                    "thread_id": last_message.thread_id,
                }
            )
            # Send completion even on error with job_id and workflow_id
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "job_id": job_id,
                    "workflow_id": workflow_id,
                    "thread_id": last_message.thread_id,
                }
            )
            raise
        finally:
            # Always mark processing as complete
            self.is_processing = False

    def _prepare_workflow_params(self, chat_history: List[Message], last_message: Message) -> dict:
        """
        Prepare workflow parameters from chat history and message.

        New interface:
        - message: Full Message object with all fields (thread_id, collections, files, etc.)
        - messages: Full chat history as list of Message objects
        - chat_input/messages: Legacy support for existing workflows

        Args:
            chat_history: Full chat history including current message
            last_message: The most recent message

        Returns:
            dict: Parameters to pass to the workflow
        """
        params = {
            "message": last_message,
            "messages": chat_history,
        }

        log.debug(f"Prepared {len(chat_history)} messages for workflow, last message role={last_message.role}")

        # Legacy support: preserve old interface for backward compatibility
        chat_input_data = self._prepare_legacy_chat_input(chat_history)
        params["chat_input"] = chat_input_data
        params["messages"] = chat_input_data

        log.debug(f"Legacy chat_input prepared with {len(chat_input_data)} messages")
        return params

    def _prepare_legacy_chat_input(self, chat_history: List[Message]) -> List[dict]:
        """
        Legacy format for backward compatibility with existing workflows.

        This preserves the old text-only format for workflows that
        were built expecting chat_input/messages parameters.
        """
        messages_data = []
        for msg in chat_history:
            msg_data = {
                "role": msg.role,
                "content": self._extract_text_content(msg),
                "created_at": msg.created_at,
            }
            messages_data.append(msg_data)
        return messages_data

    def _extract_text_content(self, message: Message) -> str:
        """Extract text content from a message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            text_parts = []
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    text_parts.append(content_item.text)
            result = " ".join(text_parts) if text_parts else ""
            if not result:
                log.debug(f"No text content found in message {message.id}")
            return result
        log.debug(f"Empty or invalid content in message {message.id}")
        return ""

    def _create_response_message(self, result: dict, last_message: Message) -> Message:
        """Construct a response Message object from workflow results."""
        content = []

        for _key, value in result.items():
            if isinstance(value, str):
                content.append(MessageTextContent(text=value))
            elif isinstance(value, list):
                content.append(MessageTextContent(text=" ".join(str(v) for v in value)))
            elif isinstance(value, dict):
                asset_type = value.get("type")
                if asset_type in ("image", "video", "audio"):
                    asset_fields = {k: v for k, v in value.items() if k in ("uri", "asset_id", "data", "metadata")}
                    if asset_type == "image":
                        content.append(MessageImageContent(image=ImageRef(**asset_fields)))
                    elif asset_type == "video":
                        content.append(MessageVideoContent(video=VideoRef(**asset_fields)))
                    else:
                        content.append(MessageAudioContent(audio=AudioRef(**asset_fields)))
                else:
                    content.append(MessageTextContent(text=json.dumps(value)))
            else:
                content.append(MessageTextContent(text=str(value)))

        # If no content was generated, provide a default message
        if not content:
            content.append(MessageTextContent(text="Workflow completed successfully."))
            log.debug("No output generated from chat workflow, using default completion message")

        response = Message(
            role="assistant",
            content=content,
            thread_id=last_message.thread_id,
            workflow_id=last_message.workflow_id,
            provider=last_message.provider,
            model=last_message.model,
            agent_mode=last_message.agent_mode or False,
        )
        log.debug(f"Created response message for workflow {last_message.workflow_id} with {len(content)} content items")
        return response
