"""
Base Trigger Node Implementation
================================

This module provides the TriggerNode base class that all trigger nodes
inherit from. It handles the core mechanics of blocking for events and
integrating with the workflow runner's infinite execution mode.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any, AsyncGenerator, ClassVar, TypedDict

from pydantic import Field

from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class TriggerEvent(TypedDict):
    """Standard event structure emitted by trigger nodes."""

    data: Any
    timestamp: str
    source: str
    event_type: str


class TriggerNode(BaseNode):
    """
    Base class for trigger nodes that enable infinite-running workflows.

    Trigger nodes are special streaming nodes that:
    1. Wait for external events (webhooks, file changes, timers, etc.)
    2. Emit event data when triggered
    3. Loop back to wait for the next event
    4. Only terminate when the workflow is explicitly stopped

    Subclasses must implement:
    - setup_trigger(): Initialize the event source
    - wait_for_event(): Block until an event occurs and return event data
    - cleanup_trigger(): Clean up the event source

    Attributes:
        _is_running: Flag to control the trigger loop
        _event_queue: Queue for receiving events from external sources
    """

    # Mark this as a streaming output node
    _layout: ClassVar[str] = "default"

    # Configuration
    max_events: int = Field(
        default=0,
        description="Maximum number of events to process (0 = unlimited)",
        ge=0,
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._is_running = False
        self._event_queue: asyncio.Queue[TriggerEvent | None] = asyncio.Queue()
        self._setup_complete = False
        self._loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def is_streaming_output(cls) -> bool:
        """Trigger nodes always produce streaming output."""
        return True

    @classmethod
    def is_cacheable(cls) -> bool:
        """Trigger nodes should never be cached."""
        return False

    @abstractmethod
    async def setup_trigger(self, context: ProcessingContext) -> None:
        """
        Initialize the trigger's event source.

        This method is called once when the workflow starts. Subclasses should
        set up any resources needed to receive events (start servers, register
        watchers, etc.).

        Args:
            context: The processing context for the workflow.
        """
        pass

    @abstractmethod
    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        """
        Wait for and return the next event.

        This method should block until an event is available or the trigger
        is stopped. Return None to signal that the trigger should stop.

        Args:
            context: The processing context for the workflow.

        Returns:
            The event data, or None to stop the trigger.
        """
        pass

    @abstractmethod
    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """
        Clean up trigger resources.

        This method is called when the workflow is stopping. Subclasses should
        release any resources acquired in setup_trigger().

        Args:
            context: The processing context for the workflow.
        """
        pass

    def stop(self) -> None:
        """Signal the trigger to stop processing events."""
        log.info(f"Stopping trigger {self.get_title()} ({self._id})")
        self._is_running = False
        # Push None to unblock wait_for_event if it's using the queue
        try:
            self._event_queue.put_nowait(None)
        except Exception:
            pass

    async def initialize(self, context: ProcessingContext, skip_cache: bool = False):
        """Initialize the trigger when the workflow starts."""
        await super().initialize(context, skip_cache)
        self._is_running = True
        self._setup_complete = False
        # Capture the event loop for thread-safe event pushing
        self._loop = asyncio.get_running_loop()

    async def finalize(self, context: ProcessingContext):
        """Clean up when the workflow is stopping."""
        log.info(f"Finalizing trigger {self.get_title()} ({self._id})")
        self.stop()
        if self._setup_complete:
            try:
                await self.cleanup_trigger(context)
            except Exception as e:
                log.error(f"Error cleaning up trigger: {e}")
        await super().finalize(context)

    class OutputType(TypedDict):
        event: TriggerEvent

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        """
        Main processing loop for the trigger.

        This method:
        1. Sets up the trigger
        2. Loops waiting for events
        3. Emits each event as it arrives
        4. Continues until stopped or max_events reached
        """
        log.info(f"Starting trigger {self.get_title()} ({self._id})")

        # Set up the trigger
        try:
            await self.setup_trigger(context)
            self._setup_complete = True
        except Exception as e:
            log.error(f"Failed to set up trigger: {e}")
            raise

        events_processed = 0

        while self._is_running:
            try:
                # Wait for the next event
                event = await self.wait_for_event(context)

                # None signals to stop
                if event is None:
                    log.info(f"Trigger {self.get_title()} received stop signal")
                    break

                # Emit the event
                log.debug(
                    f"Trigger {self.get_title()} emitting event: {event.get('event_type', 'unknown')}"
                )
                yield {"event": event}

                events_processed += 1

                # Check max_events limit
                if self.max_events > 0 and events_processed >= self.max_events:
                    log.info(
                        f"Trigger {self.get_title()} reached max_events ({self.max_events})"
                    )
                    break

            except asyncio.CancelledError:
                log.info(f"Trigger {self.get_title()} was cancelled")
                break
            except Exception as e:
                log.error(f"Error in trigger loop: {e}")
                # Continue processing unless it's a fatal error
                if not self._is_running:
                    break

        log.info(
            f"Trigger {self.get_title()} finished after {events_processed} events"
        )

    def push_event(self, event: TriggerEvent) -> None:
        """
        Push an event to the trigger's queue.

        This method is thread-safe and can be called from external sources
        (HTTP handlers, file watchers, etc.) to deliver events to the trigger.

        Args:
            event: The event to push.
        """
        try:
            if self._loop is not None and self._loop.is_running():
                # Thread-safe: schedule put_nowait on the event loop
                self._loop.call_soon_threadsafe(
                    self._event_queue.put_nowait, event
                )
            else:
                # Fallback for when running in same thread (tests)
                self._event_queue.put_nowait(event)
        except Exception as e:
            log.error(f"Failed to push event: {e}")

    async def get_event_from_queue(
        self, timeout: float | None = None
    ) -> TriggerEvent | None:
        """
        Wait for and retrieve an event from the queue.

        This is a helper method for subclasses that use the event queue.

        Args:
            timeout: Maximum time to wait in seconds, or None for no timeout.

        Returns:
            The event, or None if stopped or timeout reached.
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(
                    self._event_queue.get(), timeout=timeout
                )
            else:
                return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None
