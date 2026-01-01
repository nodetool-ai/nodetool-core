"""
TriggerInput model - durable storage for trigger events.

This table stores trigger inputs that wake up suspended trigger workflows.
Provides idempotent delivery and cross-process wake-up coordination.
"""

from datetime import datetime
from typing import Any

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid


@DBIndex(columns=["run_id", "node_id", "processed"], name="idx_trigger_input_run_node_processed")
@DBIndex(columns=["input_id"], unique=True, name="idx_trigger_input_id")
class TriggerInput(DBModel):
    """
    Durable trigger input for workflow wake-up.

    Stores external trigger events that should wake up suspended trigger workflows.
    Provides idempotent delivery and cursor support for ordered processing.

    Key properties:
    - Unique input_id prevents duplicate delivery
    - Processed flag tracks consumption
    - Cursor support for ordered trigger sources
    - Cross-process coordination safe
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "trigger_inputs",
            "primary_key": "id",
        }

    id: str = DBField(hash_key=True, default_factory=create_time_ordered_uuid)

    # Identification
    input_id: str = DBField()  # Unique ID for idempotency
    run_id: str = DBField()
    node_id: str = DBField()

    # Payload
    payload_json: dict[str, Any] = DBField(default_factory=dict)

    # Processing state
    processed: bool = DBField(default=False)
    processed_at: datetime | None = DBField(default=None)

    # Optional cursor for ordered triggers
    cursor: str | None = DBField(default=None)

    # Timestamps
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    def before_save(self):
        """Update timestamp before saving."""
        self.updated_at = datetime.now()

    @classmethod
    async def add_trigger_input(
        cls,
        run_id: str,
        node_id: str,
        input_id: str,
        payload: dict[str, Any],
        cursor: str | None = None,
    ) -> "TriggerInput | None":
        """
        Add a trigger input (idempotent).

        Args:
            run_id: The workflow run identifier
            node_id: The trigger node identifier
            input_id: Unique input ID (idempotency key)
            payload: Trigger event data
            cursor: Optional cursor for ordered processing

        Returns:
            Created TriggerInput or None if duplicate
        """
        # Check if input already exists
        existing = await cls.get_by_input_id(input_id)
        if existing:
            return existing

        # Create trigger input
        trigger_input = cls(
            id=create_time_ordered_uuid(),
            input_id=input_id,
            run_id=run_id,
            node_id=node_id,
            payload_json=payload,
            cursor=cursor,
            processed=False,
        )

        try:
            await trigger_input.save()
            return trigger_input
        except Exception as e:
            # Handle uniqueness constraint violation
            existing = await cls.get_by_input_id(input_id)
            if existing:
                return existing
            raise e

    @classmethod
    async def get_by_input_id(cls, input_id: str) -> "TriggerInput | None":
        """Get trigger input by unique input_id."""
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import Field

        results, _ = await adapter.query(
            condition=Field("input_id").equals(input_id),
            limit=1,
        )
        if not results:
            return None
        return cls.from_dict(results[0])

    @classmethod
    async def get_pending_inputs(
        cls,
        run_id: str,
        node_id: str,
        limit: int = 100,
    ) -> list["TriggerInput"]:
        """
        Get pending (unprocessed) trigger inputs for a node.

        Args:
            run_id: The workflow run identifier
            node_id: The trigger node identifier
            limit: Maximum inputs to return

        Returns:
            List of unprocessed trigger inputs ordered by creation time
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import ConditionBuilder, ConditionGroup, Field, LogicalOperator

        condition = ConditionBuilder(
            ConditionGroup([
                Field("run_id").equals(run_id),
                Field("node_id").equals(node_id),
                Field("processed").equals(False)
            ], LogicalOperator.AND)
        )

        results, _ = await adapter.query(
            condition=condition,
            order_by="created_at",
            limit=limit,
        )

        return [cls.from_dict(row) for row in results]

    async def mark_processed(self):
        """Mark this trigger input as processed."""
        self.processed = True
        self.processed_at = datetime.now()
        await self.save()

    @classmethod
    async def get_runs_with_pending_inputs(cls, limit: int = 100) -> list[tuple[str, str]]:
        """
        Get (run_id, node_id) tuples for runs with pending trigger inputs.

        This is used by the wake-up service to find suspended workflows that
        need to be resumed.

        Args:
            limit: Maximum runs to return

        Returns:
            List of (run_id, node_id) tuples
        """
        adapter = await cls.adapter()
        from nodetool.models.condition_builder import Field

        condition = Field("processed").equals(False)

        results, _ = await adapter.query(
            condition=condition,
            limit=limit,
            columns=["run_id", "node_id"],
        )

        # Deduplicate by (run_id, node_id)
        seen = set()
        runs = []
        for row in results:
            key = (row["run_id"], row["node_id"])
            if key not in seen:
                seen.add(key)
                runs.append(key)

        return runs

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TriggerInput":
        """Create TriggerInput from dictionary."""
        return cls(
            id=data.get("id", create_time_ordered_uuid()),
            input_id=data["input_id"],
            run_id=data["run_id"],
            node_id=data["node_id"],
            payload_json=data.get("payload_json", {}),
            processed=data.get("processed", False),
            processed_at=data.get("processed_at"),
            cursor=data.get("cursor"),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
        )
