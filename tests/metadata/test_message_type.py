"""Tests for the Message type from nodetool.metadata.types."""

import pytest

from nodetool.metadata.types import Message


class TestMessageIsEmpty:
    """Test cases for the Message.is_empty() method."""

    def test_default_message_is_empty(self):
        """Test that a default-initialized Message is considered empty."""
        message = Message()
        assert message.is_empty() is True
        assert message.is_set() is False

    def test_message_with_role_only_is_not_empty(self):
        """Test that a Message with only a role is not considered empty."""
        message = Message(role="user")
        assert message.is_empty() is False
        assert message.is_set() is True

    def test_message_with_content_only_is_not_empty(self):
        """Test that a Message with only content is not considered empty."""
        message = Message(content="Hello, world!")
        assert message.is_empty() is False
        assert message.is_set() is True

    def test_message_with_role_and_content_is_not_empty(self):
        """Test that a Message with both role and content is not considered empty."""
        message = Message(role="user", content="Hello, world!")
        assert message.is_empty() is False
        assert message.is_set() is True

    def test_message_with_empty_string_content_is_empty(self):
        """Test that a Message with empty string content is considered empty."""
        message = Message(content="")
        assert message.is_empty() is True
        assert message.is_set() is False

    def test_message_with_empty_list_content_is_empty(self):
        """Test that a Message with empty list content is considered empty."""
        message = Message(content=[])
        assert message.is_empty() is True
        assert message.is_set() is False

    def test_message_with_list_content_is_not_empty(self):
        """Test that a Message with non-empty list content is not considered empty."""
        from nodetool.metadata.types import MessageTextContent

        message = Message(content=[MessageTextContent(text="Hello")])
        assert message.is_empty() is False
        assert message.is_set() is True

    def test_message_with_empty_role_and_none_content(self):
        """Test that a Message with empty role and None content is empty."""
        message = Message(role="", content=None)
        assert message.is_empty() is True
        assert message.is_set() is False

    def test_message_with_metadata_only_is_empty(self):
        """Test that a Message with only metadata (no role/content) is empty."""
        message = Message(thread_id="thread123", id="msg123")
        assert message.is_empty() is True
        assert message.is_set() is False

    def test_assistant_message_is_not_empty(self):
        """Test that an assistant message with content is not empty."""
        message = Message(role="assistant", content="I can help you with that.")
        assert message.is_empty() is False
        assert message.is_set() is True

    def test_system_message_is_not_empty(self):
        """Test that a system message with content is not empty."""
        message = Message(role="system", content="You are a helpful assistant.")
        assert message.is_empty() is False
        assert message.is_set() is True
