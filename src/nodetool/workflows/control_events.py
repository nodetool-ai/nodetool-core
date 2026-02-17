"""
Control Event System for Workflow Control Edges

This module defines the event-based control system for NodeTool workflows.
Controllers emit control events via async generators to trigger execution
of controlled nodes with transient property overrides.

Key Design Principles:
1. Events are Pydantic models for validation and serialization
2. target_node_id is NOT in the event - inferred from control edge
3. Properties in RunEvent are transient (applied only for that execution)
4. Events are queued FIFO when multiple controllers target the same node

Example Usage:
    async def gen_process(self, context):
        # Get info about controlled nodes
        controlled = context.get_controlled_nodes_info(self.id)

        # Emit control event to trigger execution
        yield {"__control__": RunEvent(properties={"threshold": 0.8})}
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ControlEvent(BaseModel):
    """
    Base class for control events.

    Uses discriminated unions for event type handling.
    Subclasses must define event_type as a Literal for proper deserialization.

    Note: target_node_id is NOT included in the event. The target is inferred
    from the control edge in the workflow graph. This prevents controllers from
    emitting events for nodes they don't control.
    """

    event_type: str

    class Config:
        frozen = True  # Events are immutable once created


class RunEvent(ControlEvent):
    """
    Event to trigger node execution with optional property overrides.

    Properties specified in this event are applied transiently - they override
    the node's current values for this execution only, then revert to original values.

    Args:
        properties: Dict of property names to values. Optional - can be empty
                   to trigger execution without changing properties.

    Example:
        # Trigger execution with property override
        yield {"__control__": RunEvent(properties={"threshold": 0.8})}

        # Trigger execution without changing properties
        yield {"__control__": RunEvent()}
    """

    event_type: Literal["run"] = "run"
    properties: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class StopEvent(ControlEvent):
    """
    Event to signal controlled node to stop gracefully.

    This event instructs the controlled node to finish processing
    any pending items and then terminate.

    Example:
        # Signal controlled node to stop
        yield {"__control__": StopEvent()}
    """

    event_type: Literal["stop"] = "stop"

    class Config:
        frozen = True


# Discriminated union for control event types
# Add new event types here as they are implemented
ControlEventUnion = RunEvent | StopEvent
