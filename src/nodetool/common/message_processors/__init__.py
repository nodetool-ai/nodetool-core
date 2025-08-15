"""
Message processors package.

This package contains processors for handling different types of chat messages
in the WebSocket chat system.
"""

from .base import MessageProcessor
from .regular_chat import RegularChatProcessor, REGULAR_SYSTEM_PROMPT
from .help import HelpMessageProcessor
from .agent import AgentMessageProcessor
from .workflow import WorkflowMessageProcessor

__all__ = [
    "MessageProcessor",
    "RegularChatProcessor",
    "HelpMessageProcessor", 
    "AgentMessageProcessor",
    "WorkflowMessageProcessor",
    "REGULAR_SYSTEM_PROMPT",
]