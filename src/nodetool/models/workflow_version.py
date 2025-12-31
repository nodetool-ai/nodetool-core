from datetime import datetime
from typing import Any, Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import (
    DBField,
    DBIndex,
    DBModel,
    create_time_ordered_uuid,
)
from nodetool.models.condition_builder import (
    ConditionBuilder,
    ConditionGroup,
    Field,
    LogicalOperator,
)

"""
Defines the WorkflowVersion database model.

Represents a version/snapshot of a workflow in the nodetool system,
capturing the workflow's graph and metadata at a specific point in time.
"""

log = get_logger(__name__)


@DBIndex(columns=["workflow_id"])
class WorkflowVersion(DBModel):
    """Database model representing a version of a nodetool workflow."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for workflow versions."""
        return {
            "table_name": "nodetool_workflow_versions",
        }

    id: str = DBField(hash_key=True)
    workflow_id: str = DBField(default="")
    user_id: str = DBField(default="")
    version: int = DBField(default=1)
    created_at: datetime = DBField(default_factory=datetime.now)
    name: str = DBField(default="")
    description: str | None = DBField(default="")
    graph: dict = DBField(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """
        Create a new WorkflowVersion object from a dictionary.
        """
        return cls(
            id=data.get("id", ""),
            workflow_id=data.get("workflow_id", ""),
            user_id=data.get("user_id", ""),
            version=data.get("version", 1),
            created_at=data.get("created_at", datetime.now()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            graph=data.get(
                "graph",
                {
                    "nodes": [],
                    "edges": [],
                },
            ),
        )

    @classmethod
    async def create(
        cls,
        workflow_id: str,
        user_id: str,
        graph: dict[str, Any],
        name: str = "",
        description: str = "",
        **kwargs,
    ):  # type: ignore
        """
        Create a new workflow version in the database.
        """
        # Get the next version number
        next_version = await cls.get_next_version(workflow_id)

        return await super().create(
            id=create_time_ordered_uuid(),
            workflow_id=workflow_id,
            user_id=user_id,
            version=next_version,
            name=name,
            description=description,
            graph=graph,
            **kwargs,
        )

    @classmethod
    async def get_next_version(cls, workflow_id: str) -> int:
        """
        Get the next version number for a workflow.
        """
        latest = await cls.get_latest_version(workflow_id)
        return (latest.version + 1) if latest else 1

    @classmethod
    async def get_latest_version(cls, workflow_id: str) -> Optional["WorkflowVersion"]:
        """
        Get the latest version of a workflow.
        """
        conditions = [Field("workflow_id").equals(workflow_id)]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="version",
            reverse=True,
            limit=1,
        )
        return WorkflowVersion.from_dict(results[0]) if results else None

    @classmethod
    async def get_by_version(
        cls, workflow_id: str, version: int
    ) -> Optional["WorkflowVersion"]:
        """
        Get a specific version of a workflow.
        """
        conditions = [
            Field("workflow_id").equals(workflow_id),
            Field("version").equals(version),
        ]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            limit=1,
        )
        return WorkflowVersion.from_dict(results[0]) if results else None

    @classmethod
    async def paginate(
        cls,
        workflow_id: str,
        limit: int = 100,
        start_key: Optional[str] = None,
    ) -> tuple[list["WorkflowVersion"], str]:
        """Paginate through versions for a specific workflow.

        Args:
            workflow_id: The ID of the workflow to get versions for.
            limit: Maximum number of versions to return.
            start_key: The ID of the version to start pagination after (exclusive).

        Returns:
            A tuple containing a list of WorkflowVersion objects and the ID of the
            last evaluated version (or an empty string if it's the last page).
        """
        conditions = [Field("workflow_id").equals(workflow_id)]

        if start_key:
            conditions.append(Field("id").greater_than(start_key))

        adapter = await cls.adapter()
        results, last_evaluated_key = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="version",
            reverse=True,
            limit=limit + 1,
        )

        versions = [WorkflowVersion.from_dict(row) for row in results[:limit]]

        return versions, last_evaluated_key if len(results) > limit else ""
