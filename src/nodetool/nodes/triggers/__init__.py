"""
Trigger Nodes Module
===================

This module provides trigger nodes that can wake up a workflow when specific
events occur. Trigger nodes are designed to listen for events from various
sources and emit data that initiates workflow execution.

Supported Triggers:
- WebhookTrigger: Receives HTTP webhooks and emits the request data
- EmailTrigger: Monitors email inbox via IMAP and triggers on new emails
- FolderWatchTrigger: Watches a folder for file system changes
- TelegramTrigger: Receives messages from Telegram bots
- WhatsAppTrigger: Receives messages from WhatsApp

Trigger nodes are special nodes that:
1. Are not cacheable (they respond to external events)
2. Implement `is_streaming_input()` returning False since they are producers
3. Typically have configuration properties for the event source
4. Emit structured data containing the event payload
"""

from nodetool.nodes.triggers.base import TriggerNode
from nodetool.nodes.triggers.webhook import WebhookTrigger
from nodetool.nodes.triggers.email import EmailTrigger
from nodetool.nodes.triggers.folder import FolderWatchTrigger
from nodetool.nodes.triggers.telegram import TelegramTrigger
from nodetool.nodes.triggers.whatsapp import WhatsAppTrigger

__all__ = [
    "TriggerNode",
    "WebhookTrigger",
    "EmailTrigger",
    "FolderWatchTrigger",
    "TelegramTrigger",
    "WhatsAppTrigger",
]
