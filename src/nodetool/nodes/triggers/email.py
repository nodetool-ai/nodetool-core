"""
Email Trigger Node
==================

This module provides the EmailTrigger node that monitors an email inbox
via IMAP and triggers workflow execution when new emails arrive.

The email trigger:
1. Connects to an IMAP server using provided credentials
2. Monitors a specified folder for new messages
3. Emits email data to downstream nodes when new emails arrive

Usage:
    Configure the IMAP connection settings and optionally filter criteria.
    The trigger will emit EmailEvent objects for each new email received.
"""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import Field

from nodetool.metadata.types import BaseType, Datetime, Email
from nodetool.nodes.triggers.base import TriggerNode
from nodetool.workflows.processing_context import ProcessingContext


class EmailEvent(BaseType):
    """
    Represents an email received via the email trigger.
    
    Attributes:
        email: The email data (sender, subject, body, etc.)
        folder: The folder the email was received in
        is_unread: Whether the email was unread when received
    """
    type: Literal["email_event"] = "email_event"
    timestamp: Datetime = Field(default_factory=Datetime)
    email: Email = Field(default_factory=Email, description="The received email")
    folder: str = Field(default="INBOX", description="The folder the email was received in")
    is_unread: bool = Field(default=True, description="Whether the email was unread")


class EmailTrigger(TriggerNode):
    """
    Trigger node that monitors an email inbox via IMAP.
    
    This node connects to an IMAP server and triggers when new emails
    arrive matching the specified criteria. It's useful for building
    email-driven workflows.
    
    email, imap, inbox, message, trigger, notification
    
    Attributes:
        host: IMAP server hostname
        port: IMAP server port (default: 993 for SSL)
        username: Email account username
        password: Email account password or app password
        use_ssl: Whether to use SSL/TLS connection
        folder: Mailbox folder to monitor (default: INBOX)
        from_filter: Optional filter for sender address
        subject_filter: Optional filter for subject line
    """
    
    host: str = Field(
        default="imap.gmail.com",
        description="IMAP server hostname"
    )
    port: int = Field(
        default=993,
        description="IMAP server port"
    )
    username: str = Field(
        default="",
        description="Email account username"
    )
    password: str = Field(
        default="",
        description="Email account password or app-specific password"
    )
    use_ssl: bool = Field(
        default=True,
        description="Use SSL/TLS for connection"
    )
    folder: str = Field(
        default="INBOX",
        description="Mailbox folder to monitor"
    )
    from_filter: str = Field(
        default="",
        description="Only trigger on emails from this address (optional)"
    )
    subject_filter: str = Field(
        default="",
        description="Only trigger on emails containing this subject text (optional)"
    )
    
    # Input fields populated when triggered
    sender: str = Field(default="", description="Email sender address")
    subject: str = Field(default="", description="Email subject")
    body: str = Field(default="", description="Email body content")
    date: Datetime = Field(default_factory=Datetime, description="Email date")
    message_id: str = Field(default="", description="Email message ID")

    class OutputType(TypedDict):
        event: EmailEvent

    async def process(self, context: ProcessingContext) -> OutputType:
        """
        Process the email trigger and emit the event data.
        
        The input fields are expected to be populated by the workflow runner
        when a new email matching the criteria is received.
        """
        email = Email(
            id=self.message_id,
            sender=self.sender,
            subject=self.subject,
            body=self.body,
            date=self.date,
        )
        
        event = EmailEvent(
            timestamp=Datetime.from_datetime(datetime.now()),
            email=email,
            folder=self.folder,
            is_unread=True,
        )
        
        return {"event": event}
