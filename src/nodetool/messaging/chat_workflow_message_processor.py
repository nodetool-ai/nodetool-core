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
from typing import List, Optional

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
from nodetool.workflows.types import OutputUpdate

from .message_processor import MessageProcessor

log = get_logger(__name__)


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
        from nodetool.workflows.run_job_request import RunJobRequest
        from nodetool.workflows.run_workflow import run_workflow
        from nodetool.workflows.workflow_runner import WorkflowRunner

        job_id = str(uuid.uuid4())
        last_message = chat_history[-1]
        
        # Workflow ID is required for chat workflows
        assert last_message.workflow_id is not None, "Workflow ID is required for chat workflow processing"

        workflow_runner = WorkflowRunner(job_id=job_id)
        log.debug(
            f"Initialized WorkflowRunner for chat workflow {last_message.workflow_id} with job_id {job_id}"
        )

        # Update processing context with workflow_id
        processing_context.workflow_id = last_message.workflow_id

        # Prepare workflow parameters for ChatInput nodes
        # The workflow should have at least one ChatInput node
        # We pass the full chat history as a parameter
        params = self._prepare_chat_input_params(chat_history)

        request = RunJobRequest(
            workflow_id=last_message.workflow_id,
            messages=chat_history,
            params=params,
            graph=last_message.graph,
        )

        log.info(f"Running chat workflow {last_message.workflow_id} with {len(chat_history)} messages")
        result = {}

        try:
            async for update in run_workflow(
                request,
                workflow_runner,
                processing_context,
            ):
                await self.send_message(update.model_dump())
                log.debug(f"Chat workflow update sent: {update.type}")
                if isinstance(update, OutputUpdate):
                    result[update.node_name] = update.value

            # Signal completion
            await self.send_message({"type": "chunk", "content": "", "done": True})
            await self.send_message(
                self._create_response_message(result, last_message).model_dump()
            )

        except Exception as e:
            log.error(f"Error processing chat workflow: {e}", exc_info=True)
            await self.send_message(
                {
                    "type": "error",
                    "message": f"Error processing chat workflow: {str(e)}",
                }
            )
            # Send completion even on error
            await self.send_message({"type": "chunk", "content": "", "done": True})
            raise
        finally:
            # Always mark processing as complete
            self.is_processing = False

    def _prepare_chat_input_params(self, chat_history: List[Message]) -> dict:
        """
        Prepare parameters for ChatInput nodes.

        For chat workflows, we need to provide the message history to ChatInput nodes.
        The convention is to use a parameter called "chat_input" or "messages"
        that contains the full chat history.

        Args:
            chat_history: Full chat history including current message

        Returns:
            dict: Parameters to pass to the workflow
        """
        # Convert messages to a serializable format for the workflow
        messages_data = [
            {
                "role": msg.role,
                "content": self._extract_text_content(msg),
                "created_at": msg.created_at,
            }
            for msg in chat_history
        ]

        # Return params with common ChatInput parameter names
        # The workflow author can use "chat_input" or "messages" as the input node name
        return {
            "chat_input": messages_data,
            "messages": messages_data,
        }

    def _extract_text_content(self, message: Message) -> str:
        """Extract text content from a message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list) and message.content:
            text_parts = []
            for content_item in message.content:
                if isinstance(content_item, MessageTextContent):
                    text_parts.append(content_item.text)
            return " ".join(text_parts) if text_parts else ""
        return ""

    def _create_response_message(self, result: dict, last_message: Message) -> Message:
        """Construct a response Message object from workflow results."""
        content = []
        
        for _key, value in result.items():
            if isinstance(value, str):
                content.append(MessageTextContent(text=value))
            elif isinstance(value, list):
                # Handle list of strings
                content.append(MessageTextContent(text=" ".join(str(v) for v in value)))
            elif isinstance(value, dict):
                # Handle asset references (images, videos, audio)
                if value.get("type") == "image":
                    content.append(MessageImageContent(image=ImageRef(**value)))
                elif value.get("type") == "video":
                    content.append(MessageVideoContent(video=VideoRef(**value)))
                elif value.get("type") == "audio":
                    content.append(MessageAudioContent(audio=AudioRef(**value)))
                else:
                    # For other dict types, convert to text
                    content.append(MessageTextContent(text=json.dumps(value)))
            else:
                # For other types, convert to string
                content.append(MessageTextContent(text=str(value)))

        # If no content was generated, provide a default message
        if not content:
            content.append(MessageTextContent(text="Workflow completed successfully."))

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
