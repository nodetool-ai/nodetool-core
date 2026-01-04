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


@DBIndex(columns=["workflow_id", "save_type", "created_at"])
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
    description: str = DBField(default="")
    graph: dict = DBField(default_factory=dict)
    save_type: str = DBField(default="manual")
    autosave_metadata: dict = DBField(default_factory=dict)

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
            save_type=data.get("save_type", "manual"),
            autosave_metadata=data.get("autosave_metadata", {}),
        )

    @classmethod
    async def create(
        cls,
        workflow_id: str,
        user_id: str,
        graph: dict[str, Any],
        name: str = "",
        description: str = "",
        save_type: str = "manual",
        autosave_metadata: dict[str, Any] | None = None,
        **kwargs,
    ):  # type: ignore
        """
        Create a new workflow version in the database.
        """
        next_version = await cls.get_next_version(workflow_id)

        return await super().create(
            id=create_time_ordered_uuid(),
            workflow_id=workflow_id,
            user_id=user_id,
            version=next_version,
            name=name,
            description=description,
            graph=graph,
            save_type=save_type,
            autosave_metadata=autosave_metadata or {},
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
    async def get_by_version(cls, workflow_id: str, version: int) -> Optional["WorkflowVersion"]:
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
        start_version: Optional[int] = None,
    ) -> tuple[list["WorkflowVersion"], Optional[int]]:
        """Paginate through versions for a specific workflow.

        Args:
            workflow_id: The ID of the workflow to get versions for.
            limit: Maximum number of versions to return.
            start_version: The version number to start pagination after (exclusive).
                Results are ordered by version descending, so this should be the
                lowest version number from the previous page.

        Returns:
            A tuple containing a list of WorkflowVersion objects and the version number
            of the last returned version (or None if it's the last page).
        """
        conditions = [Field("workflow_id").equals(workflow_id)]

        if start_version is not None:
            conditions.append(Field("version").less_than(start_version))

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="version",
            reverse=True,
            limit=limit + 1,
        )

        versions = [WorkflowVersion.from_dict(row) for row in results[:limit]]

        # Return the last version number for pagination if there are more results
        next_cursor = versions[-1].version if len(results) > limit and versions else None

        return versions, next_cursor

    @classmethod
    async def get_latest_autosave(cls, workflow_id: str) -> Optional["WorkflowVersion"]:
        """
        Get the latest autosave version of a workflow.
        """
        conditions = [
            Field("workflow_id").equals(workflow_id),
            Field("save_type").equals("autosave"),
        ]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="created_at",
            reverse=True,
            limit=1,
        )
        return WorkflowVersion.from_dict(results[0]) if results else None

    @classmethod
    async def get_autosave_versions(cls, workflow_id: str, limit: int = 100) -> list["WorkflowVersion"]:
        """
        Get all autosave versions for a workflow, ordered by creation time descending.
        """
        conditions = [
            Field("workflow_id").equals(workflow_id),
            Field("save_type").equals("autosave"),
        ]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="created_at",
            reverse=True,
            limit=limit,
        )
        return [WorkflowVersion.from_dict(row) for row in results]

    @classmethod
    async def count_autosaves(cls, workflow_id: str) -> int:
        """
        Count the number of autosave versions for a workflow.
        """
        conditions = [
            Field("workflow_id").equals(workflow_id),
            Field("save_type").equals("autosave"),
        ]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["id"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            limit=1000,
        )
        return len(results)

    @classmethod
    async def delete_old_autosaves(
        cls,
        workflow_id: str,
        keep_count: int,
        older_than: Optional[datetime] = None,
    ) -> int:
        """
        Delete old autosave versions, keeping the most recent ones.

        Args:
            workflow_id: The workflow ID to clean up
            keep_count: Maximum number of autosaves to keep
            older_than: If provided, also delete autosaves older than this datetime

        Returns:
            Number of deleted versions
        """
        autosaves = await cls.get_autosave_versions(workflow_id, limit=1000)

        if not autosaves:
            return 0

        versions_to_delete: list[str] = []

        for autosave in autosaves[keep_count:]:
            if older_than is None or autosave.created_at < older_than:
                versions_to_delete.append(autosave.id)

        if not versions_to_delete:
            return 0

        for version_id in versions_to_delete:
            await cls.delete(id=version_id)

        return len(versions_to_delete)

    @classmethod
    async def delete_by_id(cls, version_id: str) -> bool:
        """
        Delete a specific version by ID.
        """
        try:
            await cls.delete(id=version_id)
            return True
        except Exception:
            return False
