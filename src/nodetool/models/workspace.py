"""
Defines the Workspace database model.

Represents a user-defined workspace directory for file I/O operations in workflows.
"""

import os
from datetime import datetime
from typing import Optional

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

log = get_logger(__name__)


@DBIndex(columns=["user_id"])
class Workspace(DBModel):
    """Database model representing a user-defined workspace directory."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for workspaces."""
        return {
            "table_name": "nodetool_workspaces",
        }

    id: str = DBField(hash_key=True)
    user_id: str = DBField(default="")
    name: str = DBField(default="")
    path: str = DBField(default="")
    is_default: bool = DBField(default=False)
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    def before_save(self):
        """Updates the `updated_at` timestamp before saving."""
        self.updated_at = datetime.now()

    def is_accessible(self) -> bool:
        """
        Check if the workspace path exists and is writable.

        Returns:
            bool: True if the path exists and is writable, False otherwise.
        """
        if not self.path:
            return False
        return os.path.isdir(self.path) and os.access(self.path, os.W_OK)

    def validate_path(self) -> bool:
        """
        Validate the workspace path.

        Returns:
            bool: True if the path is valid (absolute, exists, and writable).
        """
        if not self.path:
            return False
        if not os.path.isabs(self.path):
            return False
        return self.is_accessible()

    @classmethod
    async def find(cls, user_id: str, workspace_id: str) -> Optional["Workspace"]:
        """Find a workspace by ID, respecting user ownership.

        Args:
            user_id: The ID of the user attempting to find the workspace.
            workspace_id: The ID of the workspace to find.

        Returns:
            The Workspace object if found and owned by user, otherwise None.
        """
        workspace = await cls.get(workspace_id)
        return workspace if workspace and workspace.user_id == user_id else None

    @classmethod
    async def create(
        cls,
        user_id: str,
        name: str,
        path: str,
        is_default: bool = False,
        **kwargs,
    ) -> "Workspace":
        """
        Create a new workspace in the database.

        Args:
            user_id: The user ID.
            name: Display name for the workspace.
            path: Absolute path to the workspace directory.
            is_default: Whether this is the default workspace for the user.

        Returns:
            The created Workspace instance.

        Raises:
            ValueError: If the path is not absolute or doesn't exist.
        """
        # Validate path is absolute
        if not os.path.isabs(path):
            raise ValueError(f"Workspace path must be absolute: {path}")

        # Validate path exists and is a directory
        if not os.path.isdir(path):
            raise ValueError(f"Workspace path does not exist or is not a directory: {path}")

        # Validate path is writable
        if not os.access(path, os.W_OK):
            raise ValueError(f"Workspace path is not writable: {path}")

        # If this is set as default, unset other defaults for this user
        if is_default:
            await cls._unset_other_defaults(user_id)

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            name=name,
            path=path,
            is_default=is_default,
            **kwargs,
        )

    @classmethod
    async def _unset_other_defaults(cls, user_id: str) -> None:
        """Unset is_default for all other workspaces belonging to this user."""
        workspaces, _ = await cls.paginate(user_id=user_id)
        for ws in workspaces:
            if ws.is_default:
                ws.is_default = False
                await ws.save()

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        limit: int = 100,
        start_key: Optional[str] = None,
    ) -> tuple[list["Workspace"], str]:
        """Paginate through workspaces for a user.

        Args:
            user_id: Only workspaces owned by this user are returned.
            limit: Maximum number of workspaces to return.
            start_key: The ID of the workspace to start pagination after (exclusive).

        Returns:
            A tuple containing a list of Workspace objects and the ID of the
            last evaluated workspace (or an empty string if it's the last page).
        """
        conditions = [Field("user_id").equals(user_id)]

        if start_key:
            conditions.append(Field("id").greater_than(start_key))

        adapter = await cls.adapter()
        results, last_evaluated_key = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="updated_at",
            reverse=True,
            limit=limit + 1,
        )

        workspaces = [Workspace(**row) for row in results[:limit]]
        return workspaces, last_evaluated_key if len(results) > limit else ""

    @classmethod
    async def get_default(cls, user_id: str) -> Optional["Workspace"]:
        """Get the default workspace for a user.

        Args:
            user_id: The user ID.

        Returns:
            The default Workspace for the user, or None if none is set.
        """
        conditions = [
            Field("user_id").equals(user_id),
            Field("is_default").equals(True),
        ]

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            limit=1,
        )

        return Workspace(**results[0]) if results else None

    @classmethod
    async def has_linked_workflows(cls, workspace_id: str) -> bool:
        """Check if any workflows are linked to this workspace.

        Args:
            workspace_id: The workspace ID to check.

        Returns:
            True if workflows are linked, False otherwise.
        """
        from nodetool.models.workflow import Workflow

        conditions = [Field("workspace_id").equals(workspace_id)]
        adapter = await Workflow.adapter()
        results, _ = await adapter.query(
            columns=["id"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            limit=1,
        )
        return len(results) > 0
