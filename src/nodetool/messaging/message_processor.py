"""
Base message processor module.

This module provides the abstract base class for all message processors
in the WebSocket chat system.

Architecture Overview
=====================

Message processors are the core abstraction for handling chat messages.
They receive chat history and context, process the request, and stream
responses back to the client via a message queue.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Message Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   WebSocket  │───>│  BaseChatRunner  │───>│ MessageProcessor │   │
│  │   Message    │    │                  │    │   (Abstract)     │   │
│  └──────────────┘    └──────────────────┘    └────────┬─────────┘   │
│                                                       │              │
│                    ┌──────────────────────────────────┼──────────┐   │
│                    │                                  │          │   │
│                    ▼                                  ▼          ▼   │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌──────────────┐  │
│  │ ClaudeAgentProcessor│  │ HelpMessageProc.   │  │ RegularChat  │  │
│  │ (Agent Mode)        │  │ (Workflow Help)    │  │ Processor    │  │
│  └─────────────────────┘  └────────────────────┘  └──────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Processor Types
===============

1. **RegularChatProcessor**: Standard chat completion without tools
2. **HelpMessageProcessor**: Workflow assistance with node/example search
3. **ClaudeAgentMessageProcessor**: Full agent mode with Claude SDK
4. **ClaudeAgentHelpMessageProcessor**: Help mode using Claude SDK

Message Flow
============

```
Client ──WebSocket──> Runner ──> Processor.process()
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                        ▼                           ▼
                  [AI Provider]              [Tool Execution]
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                              send_message()
                                      │
                                      ▼
                              message_queue
                                      │
Client <──WebSocket──────────────────-┘
```
"""

import asyncio
from abc import ABC, abstractmethod
from asyncio import Queue
from typing import Any, Dict, List, Optional

from nodetool.metadata.types import Message
from nodetool.workflows.processing_context import ProcessingContext


class MessageProcessor(ABC):
    """
    Abstract base class for message processors.

    Each processor handles a specific type of message processing scenario
    and manages its own queue for sending messages back to the client.

    Responsibilities
    ----------------
    - Process incoming chat messages with appropriate AI provider
    - Manage tool execution and result handling
    - Stream responses back to client via message queue
    - Handle cancellation gracefully

    Subclass Contract
    -----------------
    Subclasses must implement the `process()` method to handle messages.
    They should:
    - Call `send_message()` to stream chunks/updates to client
    - Check `is_cancelled()` periodically for graceful shutdown
    - Handle exceptions and send error messages to client

    Attributes
    ----------
    message_queue : Queue[Dict[str, Any]]
        Async queue for outgoing messages to the client
    is_processing : bool
        Flag indicating active processing state
    _cancelled : bool
        Internal cancellation flag
    """

    def __init__(self):
        self.message_queue: Queue[dict[str, Any]] = Queue()
        self.is_processing = True
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the current processing."""
        self._cancelled = True
        self.is_processing = False

    def is_cancelled(self) -> bool:
        """Check if processing has been cancelled."""
        return self._cancelled

    def reset_cancellation(self):
        """Reset cancellation state for reuse."""
        self._cancelled = False
        self.is_processing = True

    @abstractmethod
    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ) -> Message | None:
        """
        Process messages and return the assistant's response.

        Args:
            chat_history: The complete chat history
            processing_context: Context for processing including user information
            tools: Available tools for the processor to use
            **kwargs: Additional processor-specific parameters

        Returns:
            Optional[Message]: The assistant's response message, or None if processing is streamed
        """
        pass

    async def send_message(self, message: dict[str, Any]):
        """
        Add a message to the queue for sending to the client.

        Args:
            message: The message dictionary to send
        """
        await self.message_queue.put(message)

    async def get_message(self) -> dict[str, Any] | None:
        """
        Get the next message from the queue.

        Returns:
            The next message or None if queue is empty
        """
        try:
            return self.message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def has_messages(self) -> bool:
        """Check if there are messages in the queue."""
        return not self.message_queue.empty()
