"""
Workflow message processor module.

This module provides the processor for workflow execution messages.
"""

import uuid
from typing import Any

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
from nodetool.workflows.workflow_types import OutputUpdate

from .message_processor import MessageProcessor

log = get_logger(__name__)


def _serialize_message(msg: Message) -> dict[str, Any]:
    """
    Serialize a Message object to a dictionary for workflow params.

    Preserves all message fields including rich content (images, videos, audio),
    thread_id, collections, input_files, etc.
    """
    msg_dict = msg.model_dump()

    return msg_dict


class WorkflowMessageProcessor(MessageProcessor):
    """
    Processor for workflow execution messages.

    This processor handles messages that include a workflow_id, executing
    the workflow and streaming results back to the client.
    """

    def __init__(self, user_id: str | None):
        super().__init__()
        self.user_id = user_id

    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process messages for workflow execution."""
        from nodetool.workflows.run_job_request import RunJobRequest
        from nodetool.workflows.run_workflow import run_workflow
        from nodetool.workflows.workflow_runner import WorkflowRunner

        job_id = str(uuid.uuid4())
        last_message = chat_history[-1]
        assert last_message.workflow_id is not None, "Workflow ID is required"

        workflow_runner = WorkflowRunner(job_id=job_id)
        log.debug(f"Initialized WorkflowRunner for workflow {last_message.workflow_id} with job_id {job_id}")

        # Update processing context with workflow_id and user_id
        processing_context.workflow_id = last_message.workflow_id
        processing_context.user_id = self.user_id or processing_context.user_id

        # Prepare workflow parameters
        # New interface: pass full message object and message history
        params = {
            "message": _serialize_message(last_message),
            "messages": [_serialize_message(msg) for msg in chat_history],
        }

        request = RunJobRequest(
            workflow_id=last_message.workflow_id,
            messages=chat_history,
            params=params,
            graph=last_message.graph,
        )

        log.info(f"Running workflow for {last_message.workflow_id}")
        result = {}
        workflow_id = last_message.workflow_id

        try:
            async for update in run_workflow(
                request,
                workflow_runner,
                processing_context,
            ):
                # Add job_id and workflow_id to all messages for UI visualization consistency
                # This matches the behavior of UnifiedWebSocketRunner for normal workflows
                msg = update.model_dump()
                msg["job_id"] = job_id
                msg["workflow_id"] = workflow_id
                await self.send_message(msg)
                log.debug(f"Workflow update sent: {update.type}")
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
        except Exception as e:
            log.error(f"Error processing workflow: {e}", exc_info=True)
            await self.send_message(
                {
                    "type": "error",
                    "message": f"Error processing workflow: {str(e)}",
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

    def _create_response_message(self, result: dict[str, Any], last_message: Message) -> Message:
        """Construct a response Message object from workflow results."""
        content = []
        for _key, value in result.items():
            if value is None:
                continue
            if isinstance(value, str):
                content.append(MessageTextContent(text=value))
            elif isinstance(value, list):
                content.append(MessageTextContent(text=" ".join(str(v) for v in value)))
            elif isinstance(value, dict):
                if value.get("type") == "image":
                    content.append(MessageImageContent(image=ImageRef(**value)))
                elif value.get("type") == "video":
                    content.append(MessageVideoContent(video=VideoRef(**value)))
                elif value.get("type") == "audio":
                    content.append(MessageAudioContent(audio=AudioRef(**value)))
                else:
                    content.append(MessageTextContent(text=str(value)))
            elif isinstance(value, ImageRef):
                content.append(MessageImageContent(image=value))
            elif isinstance(value, VideoRef):
                content.append(MessageVideoContent(video=value))
            elif isinstance(value, AudioRef):
                content.append(MessageAudioContent(audio=value))
            else:
                content.append(MessageTextContent(text=str(value)))

        return Message(
            role="assistant",
            content=content,
            thread_id=last_message.thread_id,
            workflow_id=last_message.workflow_id,
            provider=last_message.provider,
            model=last_message.model,
            agent_mode=last_message.agent_mode or False,
        )
