"""
Workflow message processor module.

This module provides the processor for workflow execution messages.
"""

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
