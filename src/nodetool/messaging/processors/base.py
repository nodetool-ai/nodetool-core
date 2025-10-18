"""
Base message processor module.

This module provides the abstract base class for all message processors
in the WebSocket chat system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from asyncio import Queue

from nodetool.metadata.types import Message
from nodetool.workflows.processing_context import ProcessingContext


class MessageProcessor(ABC):
    """
    Abstract base class for message processors.

    Each processor handles a specific type of message processing scenario
    and manages its own queue for sending messages back to the client.
    """

    def __init__(self):
        self.message_queue: Queue[Dict[str, Any]] = Queue()
        self.is_processing = True

    @abstractmethod
    async def process(
        self,
        chat_history: List[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ) -> Message:
        """
        Process messages and return the assistant's response.

        Args:
            chat_history: The complete chat history
            processing_context: Context for processing including user information
            tools: Available tools for the processor to use
            **kwargs: Additional processor-specific parameters

        Returns:
            Message: The assistant's response message
        """
        pass

    async def send_message(self, message: Dict[str, Any]):
        """
        Add a message to the queue for sending to the client.

        Args:
            message: The message dictionary to send
        """
        await self.message_queue.put(message)

    async def get_message(self) -> Optional[Dict[str, Any]]:
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
