"""
Tests for trigger nodes.

This module tests the trigger node implementations including:
- TriggerNode base class
- WebhookTrigger
- EmailTrigger
- FolderWatchTrigger
- TelegramTrigger
- WhatsAppTrigger
"""

from datetime import datetime
from typing import Any
from unittest.mock import Mock

import pytest

from nodetool.metadata.types import Datetime
from nodetool.nodes.triggers.base import TriggerNode, TriggerEvent
from nodetool.nodes.triggers.webhook import WebhookTrigger, WebhookEvent
from nodetool.nodes.triggers.email import EmailTrigger, EmailEvent
from nodetool.nodes.triggers.folder import (
    FolderWatchTrigger,
    FileEvent,
    FileChangeType,
)
from nodetool.nodes.triggers.telegram import (
    TelegramTrigger,
    TelegramEvent,
    TelegramUser,
    TelegramChat,
    TelegramMessage,
)
from nodetool.nodes.triggers.whatsapp import (
    WhatsAppTrigger,
    WhatsAppEvent,
    WhatsAppMessageType,
    WhatsAppContact,
    WhatsAppMessage,
)
from nodetool.workflows.base_node import NODE_BY_TYPE


class TestTriggerNode:
    """Tests for the TriggerNode base class."""

    def test_trigger_node_not_cacheable(self):
        """Trigger nodes should never be cacheable."""
        assert TriggerNode.is_cacheable() is False

    def test_trigger_node_namespace(self):
        """Trigger nodes should be in the 'triggers' namespace."""
        assert TriggerNode.get_namespace() == "triggers"


class TestWebhookTrigger:
    """Tests for the WebhookTrigger node."""

    def test_webhook_trigger_registered(self):
        """WebhookTrigger should be registered in NODE_BY_TYPE."""
        assert "triggers.WebhookTrigger" in NODE_BY_TYPE

    def test_webhook_trigger_properties(self):
        """WebhookTrigger should have expected properties."""
        props = {p.name for p in WebhookTrigger.properties()}
        expected = {
            "enabled",
            "secret",
            "allowed_methods",
            "method",
            "path",
            "headers",
            "query_params",
            "body",
            "content_type",
        }
        assert expected.issubset(props)

    def test_webhook_trigger_outputs(self):
        """WebhookTrigger should have expected outputs."""
        outputs = {o.name for o in WebhookTrigger.outputs()}
        assert "event" in outputs

    @pytest.mark.asyncio
    async def test_webhook_trigger_process(self, context):
        """WebhookTrigger process should return WebhookEvent."""
        trigger = WebhookTrigger(
            id="test-webhook",
            method="POST",
            path="/webhook/test",
            headers={"Content-Type": "application/json"},
            query_params={"foo": "bar"},
            body={"message": "hello"},
            content_type="application/json",
        )
        
        result = await trigger.process(context)
        
        assert "event" in result
        event = result["event"]
        assert isinstance(event, WebhookEvent)
        assert event.method == "POST"
        assert event.path == "/webhook/test"
        assert event.body == {"message": "hello"}
        assert event.query_params == {"foo": "bar"}

    def test_webhook_trigger_not_cacheable(self):
        """WebhookTrigger should not be cacheable."""
        assert WebhookTrigger.is_cacheable() is False


class TestEmailTrigger:
    """Tests for the EmailTrigger node."""

    def test_email_trigger_registered(self):
        """EmailTrigger should be registered in NODE_BY_TYPE."""
        assert "triggers.EmailTrigger" in NODE_BY_TYPE

    def test_email_trigger_properties(self):
        """EmailTrigger should have expected properties."""
        props = {p.name for p in EmailTrigger.properties()}
        expected = {
            "enabled",
            "host",
            "port",
            "username",
            "password",
            "use_ssl",
            "folder",
            "from_filter",
            "subject_filter",
            "sender",
            "subject",
            "body",
            "date",
            "message_id",
        }
        assert expected.issubset(props)

    def test_email_trigger_outputs(self):
        """EmailTrigger should have expected outputs."""
        outputs = {o.name for o in EmailTrigger.outputs()}
        assert "event" in outputs

    @pytest.mark.asyncio
    async def test_email_trigger_process(self, context):
        """EmailTrigger process should return EmailEvent."""
        now = Datetime.from_datetime(datetime.now())
        trigger = EmailTrigger(
            id="test-email",
            sender="test@example.com",
            subject="Test Subject",
            body="Test body content",
            date=now,
            message_id="12345",
        )
        
        result = await trigger.process(context)
        
        assert "event" in result
        event = result["event"]
        assert isinstance(event, EmailEvent)
        assert event.email.sender == "test@example.com"
        assert event.email.subject == "Test Subject"
        assert event.email.body == "Test body content"

    def test_email_trigger_not_cacheable(self):
        """EmailTrigger should not be cacheable."""
        assert EmailTrigger.is_cacheable() is False


class TestFolderWatchTrigger:
    """Tests for the FolderWatchTrigger node."""

    def test_folder_watch_trigger_registered(self):
        """FolderWatchTrigger should be registered in NODE_BY_TYPE."""
        assert "triggers.FolderWatchTrigger" in NODE_BY_TYPE

    def test_folder_watch_trigger_properties(self):
        """FolderWatchTrigger should have expected properties."""
        props = {p.name for p in FolderWatchTrigger.properties()}
        expected = {
            "enabled",
            "folder_path",
            "recursive",
            "patterns",
            "watch_created",
            "watch_modified",
            "watch_deleted",
            "watch_moved",
            "file_path",
            "file_name",
            "change_type",
            "old_path",
            "is_directory",
        }
        assert expected.issubset(props)

    def test_folder_watch_trigger_outputs(self):
        """FolderWatchTrigger should have expected outputs."""
        outputs = {o.name for o in FolderWatchTrigger.outputs()}
        assert "event" in outputs
        assert "file" in outputs

    @pytest.mark.asyncio
    async def test_folder_watch_trigger_process(self, context):
        """FolderWatchTrigger process should return FileEvent."""
        trigger = FolderWatchTrigger(
            id="test-folder",
            file_path="/tmp/test/newfile.txt",
            file_name="newfile.txt",
            change_type=FileChangeType.CREATED,
            is_directory=False,
        )
        
        result = await trigger.process(context)
        
        assert "event" in result
        assert "file" in result
        event = result["event"]
        assert isinstance(event, FileEvent)
        assert event.file_path == "/tmp/test/newfile.txt"
        assert event.file_name == "newfile.txt"
        assert event.change_type == FileChangeType.CREATED
        assert event.is_directory is False

    def test_folder_watch_trigger_not_cacheable(self):
        """FolderWatchTrigger should not be cacheable."""
        assert FolderWatchTrigger.is_cacheable() is False


class TestTelegramTrigger:
    """Tests for the TelegramTrigger node."""

    def test_telegram_trigger_registered(self):
        """TelegramTrigger should be registered in NODE_BY_TYPE."""
        assert "triggers.TelegramTrigger" in NODE_BY_TYPE

    def test_telegram_trigger_properties(self):
        """TelegramTrigger should have expected properties."""
        props = {p.name for p in TelegramTrigger.properties()}
        expected = {
            "enabled",
            "bot_token",
            "allowed_users",
            "allowed_chats",
            "message_text",
            "message_id",
            "user_id",
            "username",
            "first_name",
            "last_name",
            "chat_id",
            "chat_type",
            "chat_title",
            "update_id",
            "has_photo",
            "has_document",
            "message_date",
        }
        assert expected.issubset(props)

    def test_telegram_trigger_outputs(self):
        """TelegramTrigger should have expected outputs."""
        outputs = {o.name for o in TelegramTrigger.outputs()}
        assert "event" in outputs
        assert "text" in outputs
        assert "user_id" in outputs
        assert "chat_id" in outputs

    @pytest.mark.asyncio
    async def test_telegram_trigger_process(self, context):
        """TelegramTrigger process should return TelegramEvent."""
        trigger = TelegramTrigger(
            id="test-telegram",
            message_text="Hello from Telegram!",
            message_id=12345,
            user_id=67890,
            username="testuser",
            first_name="Test",
            last_name="User",
            chat_id=111222,
            chat_type="private",
            update_id=1,
        )
        
        result = await trigger.process(context)
        
        assert "event" in result
        assert "text" in result
        assert "user_id" in result
        assert "chat_id" in result
        
        event = result["event"]
        assert isinstance(event, TelegramEvent)
        assert event.message.text == "Hello from Telegram!"
        assert event.message.user.user_id == 67890
        assert event.message.chat.chat_id == 111222
        
        assert result["text"] == "Hello from Telegram!"
        assert result["user_id"] == 67890
        assert result["chat_id"] == 111222

    def test_telegram_trigger_not_cacheable(self):
        """TelegramTrigger should not be cacheable."""
        assert TelegramTrigger.is_cacheable() is False


class TestWhatsAppTrigger:
    """Tests for the WhatsAppTrigger node."""

    def test_whatsapp_trigger_registered(self):
        """WhatsAppTrigger should be registered in NODE_BY_TYPE."""
        assert "triggers.WhatsAppTrigger" in NODE_BY_TYPE

    def test_whatsapp_trigger_properties(self):
        """WhatsAppTrigger should have expected properties."""
        props = {p.name for p in WhatsAppTrigger.properties()}
        expected = {
            "enabled",
            "access_token",
            "phone_number_id",
            "verify_token",
            "allowed_numbers",
            "message_text",
            "message_id",
            "message_type",
            "sender_wa_id",
            "sender_name",
            "sender_phone",
            "conversation_id",
            "media_url",
            "media_mime_type",
            "caption",
            "message_timestamp",
        }
        assert expected.issubset(props)

    def test_whatsapp_trigger_outputs(self):
        """WhatsAppTrigger should have expected outputs."""
        outputs = {o.name for o in WhatsAppTrigger.outputs()}
        assert "event" in outputs
        assert "text" in outputs
        assert "sender_id" in outputs
        assert "message_type" in outputs

    @pytest.mark.asyncio
    async def test_whatsapp_trigger_process(self, context):
        """WhatsAppTrigger process should return WhatsAppEvent."""
        trigger = WhatsAppTrigger(
            id="test-whatsapp",
            message_text="Hello from WhatsApp!",
            message_id="wamid.123456",
            message_type=WhatsAppMessageType.TEXT,
            sender_wa_id="1234567890",
            sender_name="Test User",
            sender_phone="+1234567890",
            conversation_id="conv123",
        )
        
        result = await trigger.process(context)
        
        assert "event" in result
        assert "text" in result
        assert "sender_id" in result
        assert "message_type" in result
        
        event = result["event"]
        assert isinstance(event, WhatsAppEvent)
        assert event.message.text == "Hello from WhatsApp!"
        assert event.message.sender.wa_id == "1234567890"
        assert event.message.message_type == WhatsAppMessageType.TEXT
        
        assert result["text"] == "Hello from WhatsApp!"
        assert result["sender_id"] == "1234567890"
        assert result["message_type"] == WhatsAppMessageType.TEXT

    def test_whatsapp_trigger_not_cacheable(self):
        """WhatsAppTrigger should not be cacheable."""
        assert WhatsAppTrigger.is_cacheable() is False


class TestTriggerEventTypes:
    """Tests for trigger event types."""

    def test_trigger_event_base_type(self):
        """TriggerEvent should be a valid BaseType."""
        event = TriggerEvent(
            source="test",
            event_type="test_event",
            payload={"key": "value"},
        )
        assert event.type == "trigger_event"
        assert event.source == "test"
        assert event.event_type == "test_event"

    def test_webhook_event_type(self):
        """WebhookEvent should have correct type."""
        event = WebhookEvent(
            method="POST",
            path="/test",
            headers={},
            query_params={},
            body={},
            content_type="application/json",
        )
        assert event.type == "webhook_event"

    def test_email_event_type(self):
        """EmailEvent should have correct type."""
        event = EmailEvent()
        assert event.type == "email_event"

    def test_file_event_type(self):
        """FileEvent should have correct type."""
        event = FileEvent(
            file_path="/test/file.txt",
            change_type=FileChangeType.CREATED,
            file_name="file.txt",
        )
        assert event.type == "file_event"

    def test_telegram_event_type(self):
        """TelegramEvent should have correct type."""
        event = TelegramEvent()
        assert event.type == "telegram_event"

    def test_whatsapp_event_type(self):
        """WhatsAppEvent should have correct type."""
        event = WhatsAppEvent()
        assert event.type == "whatsapp_event"
