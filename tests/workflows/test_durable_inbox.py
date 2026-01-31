"""
Tests for the DurableInbox class.
"""

import uuid

import pytest

from nodetool.workflows.durable_inbox import DurableInbox


def unique_id() -> str:
    """Generate a unique identifier for test isolation."""
    return str(uuid.uuid4())[:8]


class TestDurableInboxGenerateMessageId:
    """Tests for the generate_message_id static method."""

    def test_generate_message_id_deterministic(self):
        """Test that generate_message_id returns deterministic IDs."""
        id1 = DurableInbox.generate_message_id("run1", "node1", "handle1", 1)
        id2 = DurableInbox.generate_message_id("run1", "node1", "handle1", 1)
        assert id1 == id2

    def test_generate_message_id_different_for_different_inputs(self):
        """Test that different inputs produce different IDs."""
        id1 = DurableInbox.generate_message_id("run1", "node1", "handle1", 1)
        id2 = DurableInbox.generate_message_id("run1", "node1", "handle1", 2)
        id3 = DurableInbox.generate_message_id("run1", "node1", "handle2", 1)
        id4 = DurableInbox.generate_message_id("run1", "node2", "handle1", 1)
        id5 = DurableInbox.generate_message_id("run2", "node1", "handle1", 1)

        # All IDs should be unique
        ids = [id1, id2, id3, id4, id5]
        assert len(set(ids)) == 5

    def test_generate_message_id_length(self):
        """Test that generated IDs have expected length."""
        id1 = DurableInbox.generate_message_id("run1", "node1", "handle1", 1)
        assert len(id1) == 16  # SHA256 truncated to 16 chars

    def test_generate_message_id_alphanumeric(self):
        """Test that generated IDs contain only hex characters."""
        id1 = DurableInbox.generate_message_id("run1", "node1", "handle1", 1)
        assert all(c in "0123456789abcdef" for c in id1)


class TestDurableInboxInit:
    """Tests for DurableInbox initialization."""

    def test_init_stores_run_id_and_node_id(self):
        """Test that init properly stores run_id and node_id."""
        inbox = DurableInbox(run_id="test-run-123", node_id="test-node-456")
        assert inbox.run_id == "test-run-123"
        assert inbox.node_id == "test-node-456"


@pytest.mark.asyncio
class TestDurableInboxAppend:
    """Tests for the append method."""

    async def test_append_creates_message(self):
        """Test that append creates a new message."""
        run_id = f"test-run-{unique_id()}"
        node_id = f"test-node-{unique_id()}"
        inbox = DurableInbox(run_id=run_id, node_id=node_id)
        message = await inbox.append(
            handle="input",
            payload={"key": "value"},
        )

        assert message is not None
        assert message.run_id == run_id
        assert message.node_id == node_id
        assert message.handle == "input"
        assert message.status == "pending"
        assert message.payload_json == {"key": "value"}

    async def test_append_with_custom_message_id(self):
        """Test that append respects custom message_id."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")
        message = await inbox.append(
            handle="input",
            payload={"data": 123},
            message_id="custom-msg-id",
        )

        assert message.message_id == "custom-msg-id"

    async def test_append_is_idempotent(self):
        """Test that appending duplicate message_id returns existing message."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # First append
        msg1 = await inbox.append(
            handle="input",
            payload={"data": 1},
            message_id=f"idempotent-msg-{unique_id()}",
        )

        # Second append with same message_id
        msg2 = await inbox.append(
            handle="input",
            payload={"data": 2},  # Different payload
            message_id=msg1.message_id,  # Use same message_id
        )

        # Should return the same message
        assert msg1.id == msg2.id
        assert msg1.payload_json == {"data": 1}  # Original payload preserved

    async def test_append_increments_sequence(self):
        """Test that append increments sequence numbers."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        msg1 = await inbox.append(handle="input", payload={"seq": 1})
        msg2 = await inbox.append(handle="input", payload={"seq": 2})
        msg3 = await inbox.append(handle="input", payload={"seq": 3})

        assert msg1.msg_seq == 1
        assert msg2.msg_seq == 2
        assert msg3.msg_seq == 3


@pytest.mark.asyncio
class TestDurableInboxGetPending:
    """Tests for the get_pending method."""

    async def test_get_pending_returns_pending_messages(self):
        """Test that get_pending returns only pending messages."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create some messages
        await inbox.append(handle="input", payload={"msg": 1})
        await inbox.append(handle="input", payload={"msg": 2})
        await inbox.append(handle="input", payload={"msg": 3})

        # Get pending messages
        pending = await inbox.get_pending(handle="input")

        assert len(pending) == 3
        assert all(m.status == "pending" for m in pending)

    async def test_get_pending_respects_limit(self):
        """Test that get_pending respects the limit parameter."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create 5 messages
        for i in range(5):
            await inbox.append(handle="input", payload={"msg": i})

        # Get only 2
        pending = await inbox.get_pending(handle="input", limit=2)

        assert len(pending) == 2

    async def test_get_pending_respects_min_seq(self):
        """Test that get_pending respects min_seq parameter."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create 5 messages
        for i in range(5):
            await inbox.append(handle="input", payload={"msg": i})

        # Get messages with seq >= 3
        pending = await inbox.get_pending(handle="input", min_seq=3)

        assert len(pending) == 3
        assert all(m.msg_seq >= 3 for m in pending)


@pytest.mark.asyncio
class TestDurableInboxMarkConsumed:
    """Tests for the mark_consumed method."""

    async def test_mark_consumed_updates_status(self):
        """Test that mark_consumed updates message status."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create a message
        msg = await inbox.append(handle="input", payload={"data": 1})
        assert msg.status == "pending"

        # Mark as consumed
        await inbox.mark_consumed(msg)

        # Verify status changed
        assert msg.status == "consumed"
        assert msg.consumed_at is not None


@pytest.mark.asyncio
class TestDurableInboxGetMaxSeq:
    """Tests for the get_max_seq method."""

    async def test_get_max_seq_returns_zero_when_empty(self):
        """Test that get_max_seq returns 0 when no messages exist."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        max_seq = await inbox.get_max_seq(handle="input")

        assert max_seq == 0

    async def test_get_max_seq_returns_highest_sequence(self):
        """Test that get_max_seq returns the highest sequence number."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create 3 messages
        await inbox.append(handle="input", payload={"msg": 1})
        await inbox.append(handle="input", payload={"msg": 2})
        await inbox.append(handle="input", payload={"msg": 3})

        max_seq = await inbox.get_max_seq(handle="input")

        assert max_seq == 3


@pytest.mark.asyncio
class TestDurableInboxCleanupConsumed:
    """Tests for the cleanup_consumed method."""

    async def test_cleanup_consumed_deletes_consumed_messages(self):
        """Test that cleanup_consumed removes consumed messages."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create and consume messages
        msg1 = await inbox.append(handle="input", payload={"msg": 1})
        msg2 = await inbox.append(handle="input", payload={"msg": 2})
        await inbox.append(handle="input", payload={"msg": 3})  # msg3 is still pending

        await inbox.mark_consumed(msg1)
        await inbox.mark_consumed(msg2)

        # Cleanup consumed messages with seq < 3
        deleted = await inbox.cleanup_consumed(handle="input", older_than_seq=3)

        assert deleted == 2

    async def test_cleanup_consumed_does_not_delete_pending(self):
        """Test that cleanup_consumed does not delete pending messages."""
        inbox = DurableInbox(run_id=f"test-run-{unique_id()}", node_id=f"test-node-{unique_id()}")

        # Create messages (all pending)
        await inbox.append(handle="input", payload={"msg": 1})
        await inbox.append(handle="input", payload={"msg": 2})

        # Try to cleanup - should not delete anything
        deleted = await inbox.cleanup_consumed(handle="input", older_than_seq=10)

        assert deleted == 0
