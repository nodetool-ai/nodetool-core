"""
Manual Trigger Node
===================

This module provides a trigger that waits for manual input via the
workflow's input API. This enables interactive workflows where events
are pushed programmatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from nodetool.nodes.triggers.base import TriggerEvent, TriggerNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class ManualTrigger(TriggerNode):
    """
    Trigger node that waits for manual events pushed via the API.

    This trigger enables interactive workflows where events are pushed
    programmatically through the workflow runner's input API. Each event
    pushed to the trigger is emitted and processed by the workflow.

    This trigger is useful for:
    - Building chatbot-style workflows
    - Interactive processing pipelines
    - Manual batch processing
    - Testing and debugging workflows

    Example:
        Set up a manual trigger for chat messages:
        - External code pushes messages via the API
        - Each message triggers the workflow
        - Workflow processes and responds

    Usage:
        To push events to a ManualTrigger, use the WorkflowRunner's input API:

        ```python
        runner.push_input_value(
            input_name="trigger",  # The trigger node's name
            value={"message": "Hello"},
            source_handle="event"
        )
        ```
    """

    name: str = Field(
        default="manual_trigger",
        description="Name for this trigger (used in API calls)",
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="Timeout waiting for events (None = wait forever)",
        ge=0,
    )

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Initialize the manual trigger."""
        log.info(f"Setting up manual trigger: {self.name}")
        # Manual triggers don't need special setup
        pass

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        """Wait for a manually pushed event."""
        log.debug(f"Manual trigger waiting for event (timeout={self.timeout_seconds})")

        event = await self.get_event_from_queue(timeout=self.timeout_seconds)

        if event is None:
            if self.timeout_seconds is not None:
                log.info(f"Manual trigger timed out after {self.timeout_seconds}s")
            return None

        return event

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Clean up the manual trigger."""
        log.info(f"Manual trigger {self.name} stopping")

    def push_data(self, data: Any, event_type: str = "manual") -> None:
        """
        Convenience method to push data as an event.

        This wraps the data in a proper TriggerEvent structure.

        Args:
            data: The data to include in the event.
            event_type: The type of event (default: "manual").
        """
        event: TriggerEvent = {
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": self.name,
            "event_type": event_type,
        }
        self.push_event(event)
