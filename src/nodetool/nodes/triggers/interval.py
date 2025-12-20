"""
Interval Trigger Node
=====================

This module provides a trigger that fires at regular time intervals,
similar to a cron job but with more flexibility.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from nodetool.nodes.triggers.base import TriggerEvent, TriggerNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class IntervalTrigger(TriggerNode):
    """
    Trigger node that fires at regular time intervals.

    This trigger emits events at a configured interval, similar to a timer
    or scheduler. Each event contains:
    - The tick number (how many times the trigger has fired)
    - The current timestamp
    - The configured interval

    This trigger is useful for:
    - Periodic data collection or polling
    - Scheduled batch processing
    - Heartbeat or keepalive workflows
    - Time-based automation

    Example:
        Set up an interval trigger every 60 seconds:
        - Every minute, the workflow runs and processes data
        - Continues until the workflow is stopped
    """

    interval_seconds: float = Field(
        default=60.0,
        description="Interval between triggers in seconds",
        gt=0,
    )
    initial_delay_seconds: float = Field(
        default=0.0,
        description="Delay before the first trigger fires",
        ge=0,
    )
    emit_on_start: bool = Field(
        default=True,
        description="Whether to emit an event immediately on start",
    )
    include_drift_compensation: bool = Field(
        default=True,
        description="Compensate for execution time to maintain accurate intervals",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._tick_count = 0
        self._start_time: datetime | None = None

    async def setup_trigger(self, context: ProcessingContext) -> None:
        """Initialize the interval trigger."""
        log.info(
            f"Setting up interval trigger with {self.interval_seconds}s interval"
        )
        self._tick_count = 0
        self._start_time = datetime.now(timezone.utc)

        # Apply initial delay
        if self.initial_delay_seconds > 0:
            log.debug(f"Waiting {self.initial_delay_seconds}s initial delay")
            await asyncio.sleep(self.initial_delay_seconds)

    async def wait_for_event(self, context: ProcessingContext) -> TriggerEvent | None:
        """Wait for the next interval tick."""
        # Handle first tick
        if self._tick_count == 0:
            if self.emit_on_start:
                self._tick_count += 1
                return self._create_event()
            else:
                # Wait for first interval before first tick
                log.debug(f"Interval trigger waiting {self.interval_seconds:.2f}s for first tick")
                try:
                    await asyncio.sleep(self.interval_seconds)
                except asyncio.CancelledError:
                    return None

                if not self._is_running:
                    return None

                self._tick_count += 1
                return self._create_event()

        # Wait for the interval (subsequent ticks)
        if self.include_drift_compensation:
            # Calculate next tick time based on start time
            if self._start_time is None:
                self._start_time = datetime.now(timezone.utc)

            # For ticks after the first, calculate based on when we should fire next
            # If emit_on_start=False, first tick is at interval_seconds from start
            # So tick N (1-indexed) should fire at N * interval_seconds from start
            next_tick = self._tick_count * self.interval_seconds
            if self.initial_delay_seconds > 0:
                next_tick += self.initial_delay_seconds

            elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            wait_time = max(0.001, next_tick - elapsed)  # Minimum 1ms to prevent tight loops
        else:
            wait_time = self.interval_seconds

        log.debug(f"Interval trigger waiting {wait_time:.2f}s for next tick")

        try:
            await asyncio.sleep(wait_time)
        except asyncio.CancelledError:
            return None

        if not self._is_running:
            return None

        self._tick_count += 1
        return self._create_event()

    def _create_event(self) -> TriggerEvent:
        """Create an interval event."""
        now = datetime.now(timezone.utc)
        elapsed = 0.0
        if self._start_time:
            elapsed = (now - self._start_time).total_seconds()

        return {
            "data": {
                "tick": self._tick_count,
                "elapsed_seconds": elapsed,
                "interval_seconds": self.interval_seconds,
            },
            "timestamp": now.isoformat(),
            "source": "interval",
            "event_type": "tick",
        }

    async def cleanup_trigger(self, context: ProcessingContext) -> None:
        """Clean up the interval trigger."""
        log.info(
            f"Interval trigger stopping after {self._tick_count} ticks"
        )
