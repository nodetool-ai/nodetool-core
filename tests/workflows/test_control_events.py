"""
Tests for the control event system.

This module tests the control event classes used for workflow control edges.
"""

import pytest

from nodetool.workflows.control_events import (
    ControlEvent,
    ControlEventUnion,
    RunEvent,
    StopEvent,
)


class TestControlEvent:
    """Test base ControlEvent class."""

    def test_control_event_creation(self):
        """Test that ControlEvent can be created."""
        event = ControlEvent(event_type="test")
        assert event.event_type == "test"

    def test_control_event_is_frozen(self):
        """Test that ControlEvent instances are immutable."""
        event = ControlEvent(event_type="test")
        with pytest.raises(ValueError):  # ValidationError for frozen models
            event.event_type = "modified"


class TestRunEvent:
    """Test RunEvent class."""

    def test_run_event_default_creation(self):
        """Test RunEvent creation with default values."""
        event = RunEvent()
        assert event.event_type == "run"
        assert event.properties == {}

    def test_run_event_with_properties(self):
        """Test RunEvent creation with properties."""
        props = {"threshold": 0.8, "iterations": 5}
        event = RunEvent(properties=props)
        assert event.event_type == "run"
        assert event.properties == props

    def test_run_event_is_frozen(self):
        """Test that RunEvent instances are immutable."""
        event = RunEvent(properties={"test": "value"})
        with pytest.raises(ValueError):
            event.properties = {}

    def test_run_event_properties_default_factory(self):
        """Test that properties default_factory creates new dict each time."""
        event1 = RunEvent()
        event2 = RunEvent()
        assert event1.properties is not event2.properties


class TestStopEvent:
    """Test StopEvent class."""

    def test_stop_event_creation(self):
        """Test StopEvent creation."""
        event = StopEvent()
        assert event.event_type == "stop"

    def test_stop_event_is_frozen(self):
        """Test that StopEvent instances are immutable."""
        event = StopEvent()
        with pytest.raises(ValueError):
            event.event_type = "run"


class TestControlEventUnion:
    """Test discriminated union of control events."""

    def test_run_event_in_union(self):
        """Test that RunEvent is part of ControlEventUnion."""
        event: ControlEventUnion = RunEvent(properties={"test": "value"})
        assert event.event_type == "run"
        assert isinstance(event, RunEvent)

    def test_stop_event_in_union(self):
        """Test that StopEvent is part of ControlEventUnion."""
        event: ControlEventUnion = StopEvent()
        assert event.event_type == "stop"
        assert isinstance(event, StopEvent)

    def test_event_type_discrimination(self):
        """Test that event_type field correctly discriminates between event types."""
        run_event = RunEvent()
        stop_event = StopEvent()

        assert run_event.event_type == "run"
        assert stop_event.event_type == "stop"
        assert run_event.event_type != stop_event.event_type


class TestEventSerialization:
    """Test event serialization and deserialization."""

    def test_run_event_to_dict(self):
        """Test RunEvent serialization to dict."""
        event = RunEvent(properties={"threshold": 0.8})
        data = event.model_dump()
        assert data["event_type"] == "run"
        assert data["properties"] == {"threshold": 0.8}

    def test_run_event_from_dict(self):
        """Test RunEvent deserialization from dict."""
        data = {"event_type": "run", "properties": {"threshold": 0.8}}
        event = RunEvent(**data)
        assert event.event_type == "run"
        assert event.properties == {"threshold": 0.8}

    def test_stop_event_to_dict(self):
        """Test StopEvent serialization to dict."""
        event = StopEvent()
        data = event.model_dump()
        assert data["event_type"] == "stop"

    def test_stop_event_from_dict(self):
        """Test StopEvent deserialization from dict."""
        data = {"event_type": "stop"}
        event = StopEvent(**data)
        assert event.event_type == "stop"

    def test_run_event_json_schema(self):
        """Test RunEvent JSON schema generation."""
        schema = RunEvent.model_json_schema()
        assert "properties" in schema
        assert "event_type" in schema["properties"]
        assert "properties" in schema["properties"]
        assert schema["properties"]["event_type"].get("const") == "run"
