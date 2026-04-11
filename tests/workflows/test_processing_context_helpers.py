"""Tests for ProcessingContext helper methods.

Note: Some tests avoid the set() method which has a missing _persist_variable_if_needed method.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import ProcessingMessage


def _make_context(env: dict | None = None) -> ProcessingContext:
    return ProcessingContext(environment=env or {})


class TestProcessingContextBasicOperations:
    """Tests for basic ProcessingContext operations."""

    def test_get_with_default(self):
        """Test get() with default value."""
        context = _make_context()
        assert context.get("nonexistent", "default") == "default"

    def test_get_without_default_returns_none(self):
        """Test get() without default returns None for missing keys."""
        context = _make_context()
        assert context.get("nonexistent") is None

    def test_environment_variable_access(self):
        """Test environment variables are accessible."""
        env = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        context = _make_context(env)

        assert context.environment == env
        assert context.environment["TEST_VAR"] == "test_value"

    def test_user_id_and_workflow_id(self):
        """Test user_id and workflow_id properties."""
        context = _make_context()

        # Default values
        assert context.user_id == ""
        assert context.workflow_id == ""

    def test_is_cancelled(self):
        """Test is_cancelled property."""
        context = _make_context()

        # Default is not cancelled
        assert context.is_cancelled is False


class TestProcessingContextCopy:
    """Tests for ProcessingContext.copy() method."""

    def test_copy(self):
        """Test copy() creates a shallow copy."""
        context = _make_context({"ENV_VAR": "value"})

        copied = context.copy()

        # Check environment is copied
        assert copied.environment == {"ENV_VAR": "value"}

        # Check it's a different instance
        assert copied is not context


class TestProcessingContextMessages:
    """Tests for ProcessingContext message passing."""

    def test_has_messages_initially_false(self):
        """Test has_messages() is False initially."""
        context = _make_context()
        assert context.has_messages() is False

    def test_post_message_and_has_messages(self):
        """Test post_message() and has_messages() for message passing."""
        context = _make_context()

        # Initially no messages
        assert context.has_messages() is False

        # Post a message
        msg = ProcessingMessage(type="test", data={"key": "value"})
        context.post_message(msg)

        # Now has messages
        assert context.has_messages() is True

    def test_post_multiple_messages(self):
        """Test posting multiple messages."""
        context = _make_context()

        msg1 = ProcessingMessage(type="test1", data={"key": "value1"})
        msg2 = ProcessingMessage(type="test2", data={"key": "value2"})

        context.post_message(msg1)
        context.post_message(msg2)

        assert context.has_messages() is True

    @pytest.mark.asyncio
    async def test_pop_message_async_when_no_messages(self):
        """Test pop_message_async() returns None when no messages."""
        context = _make_context()

        result = await context.pop_message_async()
        assert result is None

    @pytest.mark.asyncio
    async def test_pop_message_async_returns_messages_in_order(self):
        """Test pop_message_async() returns messages in FIFO order."""
        context = _make_context()

        msg1 = ProcessingMessage(type="test1", data={"order": 1})
        msg2 = ProcessingMessage(type="test2", data={"order": 2})

        context.post_message(msg1)
        context.post_message(msg2)

        # Pop messages
        result1 = await context.pop_message_async()
        result2 = await context.pop_message_async()
        result3 = await context.pop_message_async()

        assert result1.type == "test1"
        assert result2.type == "test2"
        assert result3 is None  # No more messages


class TestProcessingContextCaching:
    """Tests for ProcessingContext caching functionality."""

    def test_generate_node_cache_key(self):
        """Test generate_node_cache_key() creates consistent keys."""
        from nodetool.workflows.base_node import BaseNode

        context = _make_context()

        class TestNode(BaseNode):
            value: str = "test"

        node = TestNode()
        key1 = context.generate_node_cache_key(node)
        key2 = context.generate_node_cache_key(node)

        # Same node should generate same key
        assert key1 == key2
        assert key1 is not None

    def test_generate_node_cache_key_different_params(self):
        """Test generate_node_cache_key() creates different keys for different params."""
        from nodetool.workflows.base_node import BaseNode

        context = _make_context()

        class TestNode(BaseNode):
            value: str = "test"

        node1 = TestNode(value="value1")
        node2 = TestNode(value="value2")

        key1 = context.generate_node_cache_key(node1)
        key2 = context.generate_node_cache_key(node2)

        # Different params should generate different keys
        assert key1 != key2

    def test_get_cached_result_miss(self):
        """Test get_cached_result() returns None when cache miss."""
        from nodetool.workflows.base_node import BaseNode

        context = _make_context()

        class TestNode(BaseNode):
            value: str = "test"

        node = TestNode()
        result = context.get_cached_result(node)

        assert result is None
