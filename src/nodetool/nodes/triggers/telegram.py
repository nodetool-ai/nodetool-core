"""
Telegram Trigger Node
=====================

This module provides the TelegramTrigger node that receives messages from
Telegram bots and triggers workflow execution.

The Telegram trigger:
1. Receives messages sent to a configured Telegram bot
2. Parses message content, sender info, and chat details
3. Emits message data to downstream nodes

Usage:
    Configure with a Telegram bot token. When messages are sent to the bot,
    the trigger will emit TelegramEvent objects containing the message data.
"""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import Field

from nodetool.metadata.types import BaseType, Datetime, ImageRef
from nodetool.nodes.triggers.base import TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class TelegramUser(BaseType):
    """
    Represents a Telegram user.
    
    Attributes:
        user_id: Telegram user ID
        username: Telegram username (without @)
        first_name: User's first name
        last_name: User's last name (optional)
        is_bot: Whether the user is a bot
    """
    type: Literal["telegram_user"] = "telegram_user"
    user_id: int = Field(default=0, description="Telegram user ID")
    username: str = Field(default="", description="Telegram username")
    first_name: str = Field(default="", description="User's first name")
    last_name: str = Field(default="", description="User's last name")
    is_bot: bool = Field(default=False, description="Whether the user is a bot")


class TelegramChat(BaseType):
    """
    Represents a Telegram chat (private, group, or channel).
    
    Attributes:
        chat_id: Telegram chat ID
        chat_type: Type of chat (private, group, supergroup, channel)
        title: Chat title (for groups/channels)
    """
    type: Literal["telegram_chat"] = "telegram_chat"
    chat_id: int = Field(default=0, description="Telegram chat ID")
    chat_type: str = Field(default="private", description="Type of chat")
    title: str = Field(default="", description="Chat title")


class TelegramMessage(BaseType):
    """
    Represents a Telegram message.
    
    Attributes:
        message_id: Unique message identifier
        text: Message text content
        user: Sender information
        chat: Chat information
        date: Message timestamp
        reply_to_message_id: ID of the message being replied to
    """
    type: Literal["telegram_message"] = "telegram_message"
    message_id: int = Field(default=0, description="Message ID")
    text: str = Field(default="", description="Message text content")
    user: TelegramUser = Field(default_factory=TelegramUser, description="Sender info")
    chat: TelegramChat = Field(default_factory=TelegramChat, description="Chat info")
    date: Datetime = Field(default_factory=Datetime, description="Message timestamp")
    reply_to_message_id: int | None = Field(default=None, description="Reply-to message ID")


class TelegramEvent(BaseType):
    """
    Represents a Telegram trigger event.
    
    Attributes:
        message: The received Telegram message
        update_id: Telegram update ID
        has_photo: Whether the message contains a photo
        has_document: Whether the message contains a document
    """
    type: Literal["telegram_event"] = "telegram_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    message: TelegramMessage = Field(default_factory=TelegramMessage)
    update_id: int = Field(default=0, description="Telegram update ID")
    has_photo: bool = Field(default=False, description="Message has photo attachment")
    has_document: bool = Field(default=False, description="Message has document attachment")


class TelegramTrigger(TriggerNode):
    """
    Trigger node that receives messages from a Telegram bot.
    
    This node receives messages sent to a configured Telegram bot and
    triggers workflow execution. It's useful for building chatbot workflows
    and automation triggered by Telegram messages.
    
    telegram, bot, message, chat, trigger, notification
    
    Attributes:
        bot_token: Telegram bot API token from BotFather
        allowed_users: List of allowed user IDs (empty = allow all)
        allowed_chats: List of allowed chat IDs (empty = allow all)
    """
    
    bot_token: str = Field(
        default="",
        description="Telegram bot API token from @BotFather"
    )
    allowed_users: list[int] = Field(
        default=[],
        description="List of allowed user IDs (empty allows all)"
    )
    allowed_chats: list[int] = Field(
        default=[],
        description="List of allowed chat IDs (empty allows all)"
    )
    
    # Input fields populated when triggered
    message_text: str = Field(default="", description="Message text content")
    message_id: int = Field(default=0, description="Message ID")
    user_id: int = Field(default=0, description="Sender's user ID")
    username: str = Field(default="", description="Sender's username")
    first_name: str = Field(default="", description="Sender's first name")
    last_name: str = Field(default="", description="Sender's last name")
    chat_id: int = Field(default=0, description="Chat ID")
    chat_type: str = Field(default="private", description="Chat type")
    chat_title: str = Field(default="", description="Chat title")
    update_id: int = Field(default=0, description="Update ID")
    has_photo: bool = Field(default=False, description="Has photo attachment")
    has_document: bool = Field(default=False, description="Has document attachment")
    message_date: Datetime = Field(default_factory=Datetime, description="Message date")

    class OutputType(TypedDict):
        event: TelegramEvent
        text: str
        user_id: int
        chat_id: int

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Process the Telegram trigger and emit the event data.
        
        The input fields are expected to be populated by the workflow runner
        when a message is received from Telegram.
        """
        user = TelegramUser(
            user_id=self.user_id,
            username=self.username,
            first_name=self.first_name,
            last_name=self.last_name,
            is_bot=False,
        )
        
        chat = TelegramChat(
            chat_id=self.chat_id,
            chat_type=self.chat_type,
            title=self.chat_title,
        )
        
        message = TelegramMessage(
            message_id=self.message_id,
            text=self.message_text,
            user=user,
            chat=chat,
            date=self.message_date,
        )
        
        event = TelegramEvent(
            timestamp=Datetime.from_datetime(datetime.now()),
            message=message,
            update_id=self.update_id,
            has_photo=self.has_photo,
            has_document=self.has_document,
        )
        
        return {
            "event": event,
            "text": self.message_text,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
        }
