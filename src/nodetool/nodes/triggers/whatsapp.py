"""
WhatsApp Trigger Node
=====================

This module provides the WhatsAppTrigger node that receives messages from
WhatsApp and triggers workflow execution.

The WhatsApp trigger:
1. Receives messages via WhatsApp Business API webhook
2. Parses message content, sender info, and conversation details
3. Emits message data to downstream nodes

Usage:
    Configure with WhatsApp Business API credentials. When messages are
    received, the trigger will emit WhatsAppEvent objects containing the
    message data.
"""

from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import Field

from nodetool.metadata.types import BaseType, Datetime, ImageRef, AudioRef
from nodetool.nodes.triggers.base import TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class WhatsAppMessageType(str, Enum):
    """Types of WhatsApp messages."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"
    STICKER = "sticker"
    REACTION = "reaction"


class WhatsAppContact(BaseType):
    """
    Represents a WhatsApp contact.
    
    Attributes:
        wa_id: WhatsApp ID (phone number)
        name: Contact's display name
        phone_number: Phone number with country code
    """
    type: Literal["whatsapp_contact"] = "whatsapp_contact"
    wa_id: str = Field(default="", description="WhatsApp ID")
    name: str = Field(default="", description="Contact's name")
    phone_number: str = Field(default="", description="Phone number")


class WhatsAppMessage(BaseType):
    """
    Represents a WhatsApp message.
    
    Attributes:
        message_id: Unique message identifier
        message_type: Type of message (text, image, audio, etc.)
        text: Message text content (for text messages)
        timestamp: Message timestamp
        sender: Sender contact information
        media_url: URL to media content (for media messages)
    """
    type: Literal["whatsapp_message"] = "whatsapp_message"
    message_id: str = Field(default="", description="Message ID")
    message_type: WhatsAppMessageType = Field(
        default=WhatsAppMessageType.TEXT,
        description="Type of message"
    )
    text: str = Field(default="", description="Message text content")
    timestamp: Datetime = Field(default_factory=Datetime, description="Message timestamp")
    sender: WhatsAppContact = Field(default_factory=WhatsAppContact, description="Sender info")
    media_url: str = Field(default="", description="URL to media content")
    media_mime_type: str = Field(default="", description="MIME type of media")
    caption: str = Field(default="", description="Media caption")


class WhatsAppEvent(BaseType):
    """
    Represents a WhatsApp trigger event.
    
    Attributes:
        message: The received WhatsApp message
        conversation_id: Conversation/thread ID
        is_business_message: Whether from a business account
    """
    type: Literal["whatsapp_event"] = "whatsapp_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    message: WhatsAppMessage = Field(default_factory=WhatsAppMessage)
    conversation_id: str = Field(default="", description="Conversation ID")
    is_business_message: bool = Field(default=False, description="From business account")


class WhatsAppTrigger(TriggerNode):
    """
    Trigger node that receives messages from WhatsApp.
    
    This node receives messages via the WhatsApp Business API webhook and
    triggers workflow execution. It's useful for building chatbot workflows
    and automation triggered by WhatsApp messages.
    
    whatsapp, message, chat, trigger, notification, business
    
    Attributes:
        access_token: WhatsApp Business API access token
        phone_number_id: WhatsApp Business phone number ID
        verify_token: Webhook verification token
        allowed_numbers: List of allowed phone numbers (empty = allow all)
    """
    
    access_token: str = Field(
        default="",
        description="WhatsApp Business API access token"
    )
    phone_number_id: str = Field(
        default="",
        description="WhatsApp Business phone number ID"
    )
    verify_token: str = Field(
        default="",
        description="Webhook verification token"
    )
    allowed_numbers: list[str] = Field(
        default=[],
        description="List of allowed phone numbers (empty allows all)"
    )
    
    # Input fields populated when triggered
    message_text: str = Field(default="", description="Message text content")
    message_id: str = Field(default="", description="Message ID")
    message_type: WhatsAppMessageType = Field(
        default=WhatsAppMessageType.TEXT,
        description="Type of message received"
    )
    sender_wa_id: str = Field(default="", description="Sender's WhatsApp ID")
    sender_name: str = Field(default="", description="Sender's name")
    sender_phone: str = Field(default="", description="Sender's phone number")
    conversation_id: str = Field(default="", description="Conversation ID")
    media_url: str = Field(default="", description="URL to media content")
    media_mime_type: str = Field(default="", description="MIME type of media")
    caption: str = Field(default="", description="Media caption")
    message_timestamp: Datetime = Field(default_factory=Datetime, description="Message timestamp")

    class OutputType(TypedDict):
        event: WhatsAppEvent
        text: str
        sender_id: str
        message_type: WhatsAppMessageType

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Process the WhatsApp trigger and emit the event data.
        
        The input fields are expected to be populated by the workflow runner
        when a message is received from WhatsApp.
        """
        from datetime import datetime
        
        sender = WhatsAppContact(
            wa_id=self.sender_wa_id,
            name=self.sender_name,
            phone_number=self.sender_phone,
        )
        
        message = WhatsAppMessage(
            message_id=self.message_id,
            message_type=self.message_type,
            text=self.message_text,
            timestamp=self.message_timestamp,
            sender=sender,
            media_url=self.media_url,
            media_mime_type=self.media_mime_type,
            caption=self.caption,
        )
        
        event = WhatsAppEvent(
            timestamp=Datetime.from_datetime(datetime.now()),
            message=message,
            conversation_id=self.conversation_id,
            is_business_message=False,
        )
        
        return {
            "event": event,
            "text": self.message_text,
            "sender_id": self.sender_wa_id,
            "message_type": self.message_type,
        }
