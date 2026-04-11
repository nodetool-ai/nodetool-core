"""Tests for workflows/node_io.py - NodeInputs class.

Note: NodeOutputs is not tested here as it depends on workflow_runner which
was moved to TypeScript. The NodeInputs class is fully functional.
"""

from unittest.mock import MagicMock

import pytest

from nodetool.workflows.inbox import MessageEnvelope, NodeInbox
from nodetool.workflows.io import NodeInputs


@pytest.fixture
def mock_inbox():
    """Create a mock NodeInbox for testing."""
    inbox = MagicMock(spec=NodeInbox)

    # Setup async iterators
    async def mock_iter_input(name):
        if name == "empty":
            return
        elif name == "single":
            yield "value1"
        elif name == "multiple":
            yield "value1"
            yield "value2"

    async def mock_iter_input_with_envelope(name):
        if name == "empty":
            return
        elif name == "single":
            yield MessageEnvelope(
                data="value1",
                timestamp=1234567890,
                event_id="evt1",
            )
        elif name == "multiple":
            yield MessageEnvelope(
                data="value1",
                timestamp=1234567890,
                event_id="evt1",
            )
            yield MessageEnvelope(
                data="value2",
                timestamp=1234567891,
                event_id="evt2",
            )

    async def mock_iter_any():
        yield "handle1", "value1"
        yield "handle2", "value2"

    async def mock_iter_any_with_envelope():
        yield "handle1", MessageEnvelope(
            data="value1",
            timestamp=1234567890,
            event_id="evt1",
        )
        yield "handle2", MessageEnvelope(
            data="value2",
            timestamp=1234567891,
            event_id="evt2",
        )

    inbox.iter_input = mock_iter_input
    inbox.iter_input_with_envelope = mock_iter_input_with_envelope
    inbox.iter_any = mock_iter_any
    inbox.iter_any_with_envelope = mock_iter_any_with_envelope
    inbox.has_buffered = MagicMock(return_value=True)
    inbox.is_open = MagicMock(return_value=True)

    return inbox


class TestNodeInputs:
    """Tests for NodeInputs class."""

    def test_init(self, mock_inbox):
        """Test NodeInputs initialization."""
        inputs = NodeInputs(mock_inbox)
        assert inputs._inbox is mock_inbox

    @pytest.mark.asyncio
    async def test_first_returns_value(self, mock_inbox):
        """Test first() returns the first value from a handle."""
        inputs = NodeInputs(mock_inbox)
        result = await inputs.first("single")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_first_returns_default_for_empty_handle(self, mock_inbox):
        """Test first() returns default value when handle is empty."""
        inputs = NodeInputs(mock_inbox)
        result = await inputs.first("empty", default="default_value")
        assert result == "default_value"

    @pytest.mark.asyncio
    async def test_first_returns_none_when_no_default(self, mock_inbox):
        """Test first() returns None when handle is empty and no default."""
        inputs = NodeInputs(mock_inbox)
        result = await inputs.first("empty")
        assert result is None

    @pytest.mark.asyncio
    async def test_first_with_envelope(self, mock_inbox):
        """Test first_with_envelope() returns the first envelope."""
        inputs = NodeInputs(mock_inbox)
        result = await inputs.first_with_envelope("single")
        assert isinstance(result, MessageEnvelope)
        assert result.data == "value1"
        assert result.timestamp == 1234567890
        assert result.event_id == "evt1"

    @pytest.mark.asyncio
    async def test_stream_yields_values(self, mock_inbox):
        """Test stream() yields all values from a handle."""
        inputs = NodeInputs(mock_inbox)
        values = []
        async for value in inputs.stream("multiple"):
            values.append(value)
        assert values == ["value1", "value2"]

    @pytest.mark.asyncio
    async def test_stream_with_envelope_yields_envelopes(self, mock_inbox):
        """Test stream_with_envelope() yields all envelopes."""
        inputs = NodeInputs(mock_inbox)
        envelopes = []
        async for envelope in inputs.stream_with_envelope("multiple"):
            envelopes.append(envelope)
        assert len(envelopes) == 2
        assert envelopes[0].data == "value1"
        assert envelopes[1].data == "value2"

    @pytest.mark.asyncio
    async def test_any_yields_handle_value_tuples(self, mock_inbox):
        """Test any() yields (handle, value) tuples from all handles."""
        inputs = NodeInputs(mock_inbox)
        items = []
        async for handle, value in inputs.any():
            items.append((handle, value))
        assert items == [("handle1", "value1"), ("handle2", "value2")]

    @pytest.mark.asyncio
    async def test_any_with_envelope_yields_envelopes(self, mock_inbox):
        """Test any_with_envelope() yields (handle, envelope) tuples."""
        inputs = NodeInputs(mock_inbox)
        items = []
        async for handle, envelope in inputs.any_with_envelope():
            items.append((handle, envelope))
        assert len(items) == 2
        assert items[0][0] == "handle1"
        assert items[0][1].data == "value1"

    def test_has_buffered(self, mock_inbox):
        """Test has_buffered() checks inbox for buffered items."""
        inputs = NodeInputs(mock_inbox)
        mock_inbox.has_buffered.return_value = True
        assert inputs.has_buffered("test_handle") is True
        mock_inbox.has_buffered.assert_called_once_with("test_handle")

    def test_has_stream(self, mock_inbox):
        """Test has_stream() checks if handle has open upstream producers."""
        inputs = NodeInputs(mock_inbox)
        mock_inbox.is_open.return_value = True
        assert inputs.has_stream("test_handle") is True
        mock_inbox.is_open.assert_called_once_with("test_handle")
