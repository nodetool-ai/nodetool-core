"""
Base Trigger Node Module
========================

This module provides the base class for all trigger nodes. Trigger nodes are
special nodes that wake up a workflow when external events occur.

Trigger nodes differ from regular nodes in that:
1. They are never cacheable (they respond to dynamic external events)
2. They typically produce output without consuming input from other nodes
3. They have a standardized interface for receiving event data
"""

from typing import Any, ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field

from nodetool.metadata.types import BaseType, Datetime
from nodetool.workflows.base_node import BaseNode


class TriggerEvent(BaseType):
    """
    Base type for all trigger events.
    
    All trigger event types should inherit from this class and add
    their specific fields.
    """
    type: Literal["trigger_event"] = "trigger_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    source: str = Field(default="", description="The source of the trigger event")
    event_type: str = Field(default="", description="The type of event that triggered")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event-specific payload data")


class TriggerNode(BaseNode):
    """
    Base class for all trigger nodes.
    
    Trigger nodes are special nodes that wake up a workflow when specific
    events occur from external sources (webhooks, emails, file changes, etc.).
    
    Subclasses should:
    1. Define the specific configuration properties needed for their trigger type
    2. Override `process()` or `gen_process()` to return the trigger event data
    3. Define an appropriate OutputType for their specific event data
    
    Attributes:
        enabled: Whether this trigger is active and should respond to events
    """
    
    enabled: bool = Field(
        default=True,
        description="Whether this trigger is enabled and should respond to events"
    )
    
    @classmethod
    def is_cacheable(cls) -> bool:
        """Trigger nodes are never cacheable since they respond to external events."""
        return False
    
    @classmethod
    def get_namespace(cls) -> str:
        """Return the namespace for trigger nodes."""
        return "triggers"
