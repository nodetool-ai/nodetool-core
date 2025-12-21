"""
Durable Inbox Implementation
============================

Provides durable message delivery with idempotency guarantees for node-to-node
communication. Wraps RunInboxMessage model to provide a high-level API.

Key Features:
- Idempotent message delivery (deterministic message_ids)
- At-least-once semantics (simple, efficient)
- Monotonic sequencing per (run_id, node_id, handle)
- Support for large payloads via external references
- Cross-process coordination safe

Usage:
------
```python
# Create durable inbox for a node
inbox = DurableInbox(run_id="job-123", node_id="node-1")

# Append messages (idempotent)
await inbox.append(handle="input", message_id="msg-1", payload={"data": 123})

# Get pending messages
messages = await inbox.get_pending(handle="input", limit=100)

# Mark messages as consumed
for msg in messages:
    # Process message...
    await inbox.mark_consumed(msg)
```

Architecture:
-------------
This implements Phase 3 of the architectural refactor:
- Messages stored in run_inbox_messages table (source of truth)
- Duplicate message_ids are silently ignored
- Simple offset-based consumption (pending → consumed)
- Can be upgraded to exactly-once with claims later if needed
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.run_inbox_message import RunInboxMessage, MessageStatus

log = get_logger(__name__)


class DurableInbox:
    """
    Durable inbox for node message delivery with idempotency.
    
    Provides at-least-once delivery semantics using the run_inbox_messages table.
    Messages are identified by deterministic message_id to prevent duplicates.
    
    Args:
        run_id: The workflow run ID
        node_id: The node ID receiving messages
    """
    
    def __init__(self, run_id: str, node_id: str):
        self.run_id = run_id
        self.node_id = node_id
    
    @staticmethod
    def generate_message_id(run_id: str, node_id: str, handle: str, seq: int) -> str:
        """
        Generate a deterministic message ID.
        
        Args:
            run_id: The workflow run ID
            node_id: The node ID
            handle: The input handle name
            seq: The message sequence number
            
        Returns:
            A deterministic message ID string
        """
        key = f"{run_id}:{node_id}:{handle}:{seq}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def append(
        self,
        handle: str,
        payload: Any,
        message_id: Optional[str] = None,
        payload_ref: Optional[str] = None,
    ) -> RunInboxMessage:
        """
        Append a message to the inbox (idempotent).
        
        If message_id already exists, returns the existing message without error.
        
        Args:
            handle: Input handle name
            payload: Message payload (will be JSON-serialized)
            message_id: Optional deterministic message ID (generated if not provided)
            payload_ref: Optional reference to external storage for large payloads
            
        Returns:
            The created or existing RunInboxMessage
        """
        # Get next sequence number
        next_seq = await self._get_next_seq(handle)
        
        # Generate message_id if not provided
        if message_id is None:
            message_id = self.generate_message_id(
                self.run_id, self.node_id, handle, next_seq
            )
        
        # Check if message already exists (idempotency)
        existing = await RunInboxMessage.find_one({"message_id": message_id})
        if existing:
            log.debug(f"Message {message_id} already exists (idempotent)")
            return existing
        
        # Serialize payload
        payload_json = json.dumps(payload) if payload is not None else None
        
        # Detect large payloads (>1MB) and warn
        if payload_json and len(payload_json) > 1_000_000:
            log.warning(
                f"Large payload ({len(payload_json)} bytes) in inbox message. "
                f"Consider using payload_ref for large data."
            )
        
        # Create message
        message = RunInboxMessage(
            run_id=self.run_id,
            node_id=self.node_id,
            handle=handle,
            message_id=message_id,
            msg_seq=next_seq,
            payload_json=payload_json,
            payload_ref=payload_ref,
            status="pending",
            created_at=datetime.now(),
        )
        
        await message.save()
        log.debug(
            f"Appended message {message_id} to inbox "
            f"({self.run_id}/{self.node_id}/{handle}), seq={next_seq}"
        )
        
        return message
    
    async def get_pending(
        self,
        handle: str,
        limit: int = 100,
        min_seq: int = 0,
    ) -> list[RunInboxMessage]:
        """
        Get pending messages for a handle in sequence order.
        
        Args:
            handle: Input handle name
            limit: Maximum number of messages to return
            min_seq: Minimum sequence number (for cursor-based pagination)
            
        Returns:
            List of pending messages in sequence order
        """
        messages = await RunInboxMessage.find(
            {
                "run_id": self.run_id,
                "node_id": self.node_id,
                "handle": handle,
                "status": "pending",
                "msg_seq": {"$gte": min_seq},
            },
            sort=[("msg_seq", 1)],
            limit=limit,
        )
        
        return messages
    
    async def mark_consumed(self, message: RunInboxMessage) -> None:
        """
        Mark a message as consumed.
        
        Args:
            message: The message to mark as consumed
        """
        message.status = "consumed"
        message.consumed_at = datetime.now()
        await message.save()
        
        log.debug(
            f"Marked message {message.message_id} as consumed "
            f"({self.run_id}/{self.node_id}/{message.handle}), seq={message.msg_seq}"
        )
    
    async def get_max_seq(self, handle: str) -> int:
        """
        Get the maximum sequence number for a handle.
        
        Args:
            handle: Input handle name
            
        Returns:
            Maximum sequence number (0 if no messages exist)
        """
        messages = await RunInboxMessage.find(
            {
                "run_id": self.run_id,
                "node_id": self.node_id,
                "handle": handle,
            },
            sort=[("msg_seq", -1)],
            limit=1,
        )
        
        if messages:
            return messages[0].msg_seq
        return 0
    
    async def _get_next_seq(self, handle: str) -> int:
        """
        Get the next sequence number for a handle.
        
        Args:
            handle: Input handle name
            
        Returns:
            Next sequence number
        """
        max_seq = await self.get_max_seq(handle)
        return max_seq + 1
    
    async def cleanup_consumed(self, handle: str, older_than_seq: int) -> int:
        """
        Clean up consumed messages older than a given sequence.
        
        This helps manage storage growth by removing old consumed messages.
        
        Args:
            handle: Input handle name
            older_than_seq: Delete consumed messages with seq < this value
            
        Returns:
            Number of messages deleted
        """
        # Find consumed messages to delete
        messages = await RunInboxMessage.find(
            {
                "run_id": self.run_id,
                "node_id": self.node_id,
                "handle": handle,
                "status": "consumed",
                "msg_seq": {"$lt": older_than_seq},
            }
        )
        
        # Delete them
        count = 0
        for msg in messages:
            await msg.delete()
            count += 1
        
        if count > 0:
            log.info(
                f"Cleaned up {count} consumed messages from inbox "
                f"({self.run_id}/{self.node_id}/{handle})"
            )
        
        return count
