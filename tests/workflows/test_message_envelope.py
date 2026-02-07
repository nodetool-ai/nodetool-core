"""Tests for MessageEnvelope functionality in NodeInbox.

These tests verify that:
1. Messages are wrapped in MessageEnvelope with auto-generated timestamp and event_id
2. Metadata can be attached to messages when putting
3. Backward compatibility is maintained (standard methods return unwrapped data)
4. Envelope access methods return full MessageEnvelope objects
5. Metadata propagates through the node graph
"""

import asyncio
from datetime import UTC, datetime, timezone

import pytest

from nodetool.workflows.inbox import MessageEnvelope, NodeInbox
from nodetool.workflows.io import NodeInputs


class TestMessageEnvelope:
    """Tests for the MessageEnvelope dataclass."""

    def test_envelope_creation_with_defaults(self):
        """Envelope should auto-generate timestamp and event_id."""
        envelope = MessageEnvelope(data="test_data")

        assert envelope.data == "test_data"
        assert envelope.metadata == {}
        assert isinstance(envelope.timestamp, datetime)
        assert envelope.timestamp.tzinfo == UTC
        assert isinstance(envelope.event_id, str)
        assert len(envelope.event_id) > 0

    def test_envelope_creation_with_metadata(self):
        """Envelope should accept custom metadata."""
        metadata = {"source": "test", "priority": 1}
        envelope = MessageEnvelope(data=123, metadata=metadata)

        assert envelope.data == 123
        assert envelope.metadata == {"source": "test", "priority": 1}

    def test_envelope_event_ids_are_unique(self):
        """Each envelope should have a unique event_id."""
        envelope1 = MessageEnvelope(data="a")
        envelope2 = MessageEnvelope(data="b")

        assert envelope1.event_id != envelope2.event_id


@pytest.mark.asyncio
class TestNodeInboxEnvelopeBasics:
    """Tests for basic NodeInbox envelope functionality."""

    async def test_put_wraps_item_in_envelope(self):
        """Items put into inbox should be wrapped in MessageEnvelope."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "test_value")
        inbox.mark_source_done("a")

        # Use envelope access method to verify wrapping
        envelopes = []
        async for envelope in inbox.iter_input_with_envelope("a"):
            envelopes.append(envelope)

        assert len(envelopes) == 1
        assert isinstance(envelopes[0], MessageEnvelope)
        assert envelopes[0].data == "test_value"

    async def test_put_with_metadata(self):
        """Metadata should be stored in the envelope."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        metadata = {"source": "producer_1", "batch_id": 42}
        await inbox.put("a", "value", metadata)
        inbox.mark_source_done("a")

        async for envelope in inbox.iter_input_with_envelope("a"):
            assert envelope.data == "value"
            assert envelope.metadata == {"source": "producer_1", "batch_id": 42}

    async def test_iter_input_returns_unwrapped_data(self):
        """Standard iter_input should return just the data for backward compatibility."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "value1")
        await inbox.put("a", "value2")
        inbox.mark_source_done("a")

        items = []
        async for item in inbox.iter_input("a"):
            items.append(item)

        # Should return raw data, not envelopes
        assert items == ["value1", "value2"]

    async def test_iter_any_returns_unwrapped_data(self):
        """Standard iter_any should return just the data for backward compatibility."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)
        inbox.add_upstream("b", 1)

        await inbox.put("a", "a_val")
        await inbox.put("b", "b_val")
        inbox.mark_source_done("a")
        inbox.mark_source_done("b")

        items = []
        async for handle, item in inbox.iter_any():
            items.append((handle, item))

        # Should return raw data, not envelopes
        assert items == [("a", "a_val"), ("b", "b_val")]


@pytest.mark.asyncio
class TestNodeInboxEnvelopeIterators:
    """Tests for envelope-aware iterator methods."""

    async def test_iter_input_with_envelope(self):
        """iter_input_with_envelope should yield full MessageEnvelope objects."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", 100, {"meta_key": "meta_value"})
        await inbox.put("a", 200, {"another_key": 123})
        inbox.mark_source_done("a")

        envelopes = []
        async for envelope in inbox.iter_input_with_envelope("a"):
            envelopes.append(envelope)

        assert len(envelopes) == 2

        assert envelopes[0].data == 100
        assert envelopes[0].metadata == {"meta_key": "meta_value"}
        assert isinstance(envelopes[0].timestamp, datetime)
        assert isinstance(envelopes[0].event_id, str)

        assert envelopes[1].data == 200
        assert envelopes[1].metadata == {"another_key": 123}

    async def test_iter_any_with_envelope(self):
        """iter_any_with_envelope should yield (handle, MessageEnvelope) tuples."""
        inbox = NodeInbox()
        inbox.add_upstream("x", 1)
        inbox.add_upstream("y", 1)

        await inbox.put("x", "x_data", {"from": "x"})
        await inbox.put("y", "y_data", {"from": "y"})
        inbox.mark_source_done("x")
        inbox.mark_source_done("y")

        results = []
        async for handle, envelope in inbox.iter_any_with_envelope():
            results.append((handle, envelope))

        assert len(results) == 2

        handle1, env1 = results[0]
        assert handle1 == "x"
        assert env1.data == "x_data"
        assert env1.metadata == {"from": "x"}

        handle2, env2 = results[1]
        assert handle2 == "y"
        assert env2.data == "y_data"
        assert env2.metadata == {"from": "y"}

    async def test_try_pop_any_returns_unwrapped_data(self):
        """try_pop_any should return unwrapped data for backward compatibility."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "popped_value", {"key": "value"})

        result = inbox.try_pop_any()
        assert result is not None
        handle, data = result
        assert handle == "a"
        assert data == "popped_value"  # Raw data, not envelope

    async def test_try_pop_any_with_envelope(self):
        """try_pop_any_with_envelope should return full MessageEnvelope."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "popped_value", {"key": "value"})

        result = inbox.try_pop_any_with_envelope()
        assert result is not None
        handle, envelope = result
        assert handle == "a"
        assert isinstance(envelope, MessageEnvelope)
        assert envelope.data == "popped_value"
        assert envelope.metadata == {"key": "value"}


@pytest.mark.asyncio
class TestNodeInputsEnvelope:
    """Tests for NodeInputs wrapper with envelope access."""

    async def test_first_returns_unwrapped_data(self):
        """NodeInputs.first should return just the data."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "first_value", {"meta": True})
        inbox.mark_source_done("a")

        inputs = NodeInputs(inbox)
        result = await inputs.first("a")

        assert result == "first_value"

    async def test_first_with_envelope(self):
        """NodeInputs.first_with_envelope should return the full envelope."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "first_value", {"meta": True})
        inbox.mark_source_done("a")

        inputs = NodeInputs(inbox)
        result = await inputs.first_with_envelope("a")

        assert isinstance(result, MessageEnvelope)
        assert result.data == "first_value"
        assert result.metadata == {"meta": True}

    async def test_stream_returns_unwrapped_data(self):
        """NodeInputs.stream should yield just the data."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "val1")
        await inbox.put("a", "val2")
        inbox.mark_source_done("a")

        inputs = NodeInputs(inbox)
        results = []
        async for item in inputs.stream("a"):
            results.append(item)

        assert results == ["val1", "val2"]

    async def test_stream_with_envelope(self):
        """NodeInputs.stream_with_envelope should yield full envelopes."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", "val1", {"index": 0})
        await inbox.put("a", "val2", {"index": 1})
        inbox.mark_source_done("a")

        inputs = NodeInputs(inbox)
        results = []
        async for envelope in inputs.stream_with_envelope("a"):
            results.append(envelope)

        assert len(results) == 2
        assert results[0].data == "val1"
        assert results[0].metadata == {"index": 0}
        assert results[1].data == "val2"
        assert results[1].metadata == {"index": 1}

    async def test_any_returns_unwrapped_data(self):
        """NodeInputs.any should yield (handle, data) tuples."""
        inbox = NodeInbox()
        inbox.add_upstream("x", 1)

        await inbox.put("x", "x_val")
        inbox.mark_source_done("x")

        inputs = NodeInputs(inbox)
        results = []
        async for handle, item in inputs.any():
            results.append((handle, item))

        assert results == [("x", "x_val")]

    async def test_any_with_envelope(self):
        """NodeInputs.any_with_envelope should yield (handle, envelope) tuples."""
        inbox = NodeInbox()
        inbox.add_upstream("x", 1)

        await inbox.put("x", "x_val", {"key": "value"})
        inbox.mark_source_done("x")

        inputs = NodeInputs(inbox)
        results = []
        async for handle, envelope in inputs.any_with_envelope():
            results.append((handle, envelope))

        assert len(results) == 1
        handle, envelope = results[0]
        assert handle == "x"
        assert isinstance(envelope, MessageEnvelope)
        assert envelope.data == "x_val"
        assert envelope.metadata == {"key": "value"}


@pytest.mark.asyncio
class TestEnvelopeTimestampAndEventId:
    """Tests for envelope timestamp and event_id behavior."""

    async def test_envelopes_have_increasing_timestamps(self):
        """Envelopes created in sequence should have non-decreasing timestamps."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        await inbox.put("a", 1)
        await asyncio.sleep(0.01)  # Small delay
        await inbox.put("a", 2)
        inbox.mark_source_done("a")

        envelopes = []
        async for envelope in inbox.iter_input_with_envelope("a"):
            envelopes.append(envelope)

        assert len(envelopes) == 2
        assert envelopes[0].timestamp <= envelopes[1].timestamp

    async def test_all_event_ids_unique_in_stream(self):
        """All event_ids in a stream should be unique."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)

        for i in range(10):
            await inbox.put("a", i)
        inbox.mark_source_done("a")

        event_ids = set()
        async for envelope in inbox.iter_input_with_envelope("a"):
            event_ids.add(envelope.event_id)

        assert len(event_ids) == 10  # All unique


@pytest.mark.asyncio
class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with existing code."""

    async def test_existing_iter_input_pattern_unchanged(self):
        """The classic iter_input pattern should work exactly as before."""
        inbox = NodeInbox()
        inbox.add_upstream("input", 1)

        await inbox.put("input", 1)
        await inbox.put("input", 2)
        await inbox.put("input", 3)
        inbox.mark_source_done("input")

        received = []
        async for item in inbox.iter_input("input"):
            received.append(item)

        assert received == [1, 2, 3]

    async def test_existing_iter_any_pattern_unchanged(self):
        """The classic iter_any pattern should work exactly as before."""
        inbox = NodeInbox()
        inbox.add_upstream("a", 1)
        inbox.add_upstream("b", 1)

        await inbox.put("a", "a1")
        await inbox.put("b", "b1")
        await inbox.put("a", "a2")
        inbox.mark_source_done("a")
        await inbox.put("b", "b2")
        inbox.mark_source_done("b")

        items = []
        async for handle, item in inbox.iter_any():
            items.append((handle, item))

        assert items == [("a", "a1"), ("b", "b1"), ("a", "a2"), ("b", "b2")]
