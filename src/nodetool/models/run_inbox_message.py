"""
RunInboxMessage model - durable inbox for node message delivery.

This table provides idempotent message delivery with support for
at-least-once or exactly-once semantics depending on configuration.
"""

from datetime import datetime, timedelta
from typing import Any, Literal

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid

MessageStatus = Literal["pending", "claimed", "consumed"]


@DBIndex(columns=["run_id", "node_id", "handle", "msg_seq"], name="idx_inbox_run_node_handle_seq")
@DBIndex(columns=["run_id", "node_id", "handle", "status"], name="idx_inbox_run_node_handle_status")
@DBIndex(columns=["message_id"], unique=True, name="idx_inbox_message_id")
class RunInboxMessage(DBModel):
    """
    Durable inbox message for node-to-node communication.

    Provides idempotent message delivery with configurable semantics:
    - At-least-once: Use pending/consumed status with offsets
    - Exactly-once: Use pending/claimed/consumed with TTL claims

    Key properties:
    - Unique message_id prevents duplicates
    - Monotonic msg_seq per (run_id, node_id, handle)
    - Claim TTL prevents duplicate processing
    - Payload can be inline or external reference
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "run_inbox_messages",
            "primary_key": "id",
        }

    id: str = DBField(hash_key=True, default_factory=create_time_ordered_uuid)

    # Message identification
    message_id: str = DBField()  # Unique, deterministic ID for idempotency
    run_id: str = DBField()
    node_id: str = DBField()
    handle: str = DBField()  # Input handle name

    # Sequencing
    msg_seq: int = DBField()  # Monotonic per (run_id, node_id, handle)

    # Payload
    payload_json: dict[str, Any] = DBField(default_factory=dict)
    payload_ref: str | None = DBField(default=None)  # External storage reference for large payloads

    # Status and consumption
    status: str = DBField()  # pending | claimed | consumed
    claim_worker_id: str | None = DBField(default=None)
    claim_expires_at: datetime | None = DBField(default=None)
    consumed_at: datetime | None = DBField(default=None)

    # Timestamps
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    def before_save(self):
        """Update timestamp before saving."""
        self.updated_at = datetime.now()

    @classmethod
    async def get_next_seq(cls, run_id: str, node_id: str, handle: str) -> int:
        """
        Get next sequence number for a (run_id, node_id, handle) tuple.

        Args:
            run_id: The workflow run identifier
            node_id: The node identifier
            handle: The input handle name

        Returns:
            Next available sequence number
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup(
                [Field("run_id").equals(run_id), Field("node_id").equals(node_id), Field("handle").equals(handle)],
                LogicalOperator.AND,
            )
        )

        results, _ = await adapter.query(
            condition=condition,
            order_by="msg_seq",
            reverse=True,
            limit=1,
            columns=["msg_seq"],
        )
        if not results:
            return 0
        return results[0]["msg_seq"] + 1

    @classmethod
    async def append_message(
        cls,
        run_id: str,
        node_id: str,
        handle: str,
        message_id: str,
        payload: dict[str, Any],
        payload_ref: str | None = None,
    ) -> "RunInboxMessage | None":
        """
        Append a message to the inbox (idempotent).

        Args:
            run_id: The workflow run identifier
            node_id: The node identifier
            handle: The input handle name
            message_id: Unique message ID (idempotency key)
            payload: Message payload (small messages)
            payload_ref: External storage reference (large messages)

        Returns:
            Created message or None if duplicate (idempotent)
        """
        # Check if message already exists
        existing = await cls.get_by_message_id(message_id)
        if existing:
            return existing

        # Get next sequence number
        seq = await cls.get_next_seq(run_id, node_id, handle)

        # Create message
        message = cls(
            id=create_time_ordered_uuid(),
            message_id=message_id,
            run_id=run_id,
            node_id=node_id,
            handle=handle,
            msg_seq=seq,
            payload_json=payload if payload_ref is None else {},
            payload_ref=payload_ref,
            status="pending",
        )

        try:
            await message.save()
            return message
        except Exception as e:
            # Handle uniqueness constraint violation (race condition)
            # Try to fetch existing message
            existing = await cls.get_by_message_id(message_id)
            if existing:
                return existing
            raise e

    @classmethod
    async def get_by_message_id(cls, message_id: str) -> "RunInboxMessage | None":
        """Get message by unique message_id."""
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import Field

        results, _ = await adapter.query(
            condition=Field("message_id").equals(message_id),
            limit=1,
        )
        if not results:
            return None
        return cls.from_dict(results[0])

    @classmethod
    async def get_pending_messages(
        cls,
        run_id: str,
        node_id: str,
        handle: str,
        limit: int = 100,
    ) -> list["RunInboxMessage"]:
        """
        Get pending messages for a node's input handle.

        Args:
            run_id: The workflow run identifier
            node_id: The node identifier
            handle: The input handle name
            limit: Maximum messages to return

        Returns:
            List of pending messages ordered by sequence
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        # Get pending or expired claims
        now = datetime.now()

        condition = ConditionBuilder(
            ConditionGroup(
                [
                    Field("run_id").equals(run_id),
                    Field("node_id").equals(node_id),
                    Field("handle").equals(handle),
                    Field("status").equals("pending"),
                ],
                LogicalOperator.AND,
            )
        )

        results, _ = await adapter.query(
            condition=condition,
            order_by="msg_seq",
            limit=limit,
        )

        messages = [cls.from_dict(row) for row in results]

        # Also include expired claims
        claimed_condition = ConditionBuilder(
            ConditionGroup(
                [
                    Field("run_id").equals(run_id),
                    Field("node_id").equals(node_id),
                    Field("handle").equals(handle),
                    Field("status").equals("claimed"),
                ],
                LogicalOperator.AND,
            )
        )

        claimed_results, _ = await adapter.query(
            condition=claimed_condition,
            order_by="msg_seq",
            limit=limit,
        )

        # Filter expired claims
        for row in claimed_results:
            msg = cls.from_dict(row)
            if msg.claim_expires_at and msg.claim_expires_at < now:
                messages.append(msg)

        # Sort by sequence
        messages.sort(key=lambda m: m.msg_seq)
        return messages[:limit]

    async def claim(self, worker_id: str, ttl_seconds: int = 30) -> bool:
        """
        Claim this message for processing (exactly-once semantics).

        Args:
            worker_id: Identifier of the worker claiming this message
            ttl_seconds: How long the claim is valid

        Returns:
            True if successfully claimed, False if already claimed
        """
        if self.status != "pending":
            # Check if our claim expired
            if self.status == "claimed" and self.claim_expires_at:
                if datetime.now() >= self.claim_expires_at:
                    # Claim expired, can reclaim
                    pass
                else:
                    return False
            else:
                return False

        self.status = "claimed"
        self.claim_worker_id = worker_id
        self.claim_expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        await self.save()
        return True

    async def mark_consumed(self):
        """Mark message as consumed."""
        self.status = "consumed"
        self.consumed_at = datetime.now()
        await self.save()

    def get_payload(self) -> dict[str, Any]:
        """
        Get message payload (handles both inline and external).

        Returns:
            Message payload dictionary
        """
        if self.payload_ref:
            # TODO: Load from external storage
            # For now, raise error
            raise NotImplementedError("External payload storage not yet implemented")
        return self.payload_json

    @classmethod
    async def find_one(cls, query: dict[str, Any]) -> "RunInboxMessage | None":
        """
        Find a single message matching the query.

        Args:
            query: Query dictionary with field -> value mappings

        Returns:
            Matching message or None if not found
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        conditions = [Field(k).equals(v) for k, v in query.items()]
        condition = ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND))  # type: ignore[arg-type]

        results, _ = await adapter.query(
            condition=condition,
            limit=1,
        )
        if not results:
            return None
        return cls.from_dict(results[0])

    @classmethod
    async def find(
        cls,
        query: dict[str, Any],
        sort: list[tuple[str, int]] | None = None,
        limit: int = 100,
    ) -> list["RunInboxMessage"]:
        """
        Find messages matching the query.

        Args:
            query: Query dictionary with field -> value mappings
            sort: List of (field, direction) tuples for sorting (1=asc, -1=desc)
            limit: Maximum number of results to return

        Returns:
            List of matching messages
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        conditions = []
        for key, value in query.items():
            if isinstance(value, dict) and "$gte" in value:
                conditions.append(Field(key).greater_than_or_equal(value["$gte"]))
            elif isinstance(value, dict) and "$lt" in value:
                conditions.append(Field(key).less_than(value["$lt"]))
            else:
                conditions.append(Field(key).equals(value))

        condition = ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND))  # type: ignore[arg-type]

        order_by = None
        reverse = False
        if sort:
            for field, direction in sort:
                order_by = field
                reverse = direction < 0
                break

        results, _ = await adapter.query(
            condition=condition,
            order_by=order_by,
            reverse=reverse,
            limit=limit,
        )

        return [cls.from_dict(row) for row in results]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunInboxMessage":
        """Create RunInboxMessage from dictionary."""
        return cls(
            id=data.get("id", create_time_ordered_uuid()),
            message_id=data["message_id"],
            run_id=data["run_id"],
            node_id=data["node_id"],
            handle=data["handle"],
            msg_seq=data["msg_seq"],
            payload_json=data.get("payload_json", {}),
            payload_ref=data.get("payload_ref"),
            status=data.get("status", "pending"),
            claim_worker_id=data.get("claim_worker_id"),
            claim_expires_at=data.get("claim_expires_at"),
            consumed_at=data.get("consumed_at"),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
        )
