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
from nodetool.types.api_graph import Graph as APIGraph
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph

"""
Defines the Workflow database model.

Represents a workflow in the nodetool system, including its structure (graph),
metadata (name, description, tags), user association, access control, and settings.
"""

log = get_logger(__name__)


@DBIndex(columns=["user_id"])
class Workflow(DBModel):
    """Database model representing a nodetool workflow."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for workflows."""
        return {
            "table_name": "nodetool_workflows",
        }

    id: str = DBField(hash_key=True)
    user_id: str = DBField(default="")
    access: str = DBField(default="private")
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)
    name: str = DBField(default="")
    tool_name: str | None = DBField(default=None)
    tags: list[str] | None = DBField(default_factory=list)
    description: str | None = DBField(default="")
    package_name: str | None = DBField(default="")
    thumbnail: str | None = DBField(default=None)
    thumbnail_url: str | None = DBField(default=None)
    graph: dict = DBField(default_factory=dict)
    settings: dict[str, Any] | None = DBField(default_factory=dict)
    receive_clipboard: bool | None = DBField(default=False)
    run_mode: str | None = DBField(default=None)

    def before_save(self):
        """Updates the `updated_at` timestamp before saving."""
        self.updated_at = datetime.now()

    async def save(self):
        """
        Save a workflow and auto-start trigger workflows when running in the server.
        """
        await super().save()
        await self._maybe_autostart_trigger()
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """
        Create a new Workflow object from a dictionary.
        """
        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            access=data.get("access", ""),
            created_at=data.get("created_at", datetime.now()),
            updated_at=data.get("updated_at", datetime.now()),
            name=data.get("name", ""),
            tool_name=data.get("tool_name"),
            package_name=data.get("package_name", ""),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            thumbnail=data.get("thumbnail"),
            settings=data.get("settings", {}),
            graph=data.get(
                "graph",
                {
                    "nodes": [],
                    "edges": [],
                },
            ),
            run_mode=data.get("run_mode"),
        )

    @classmethod
    async def find(cls, user_id: str, workflow_id: str):
        """Find a workflow by ID, respecting user ownership and access level.

        Args:
            user_id: The ID of the user attempting to find the workflow.
            workflow_id: The ID of the workflow to find.

        Returns:
            The Workflow object if found and accessible, otherwise None.
        """
        workflow = await cls.get(workflow_id)
        return workflow if workflow and (workflow.user_id == user_id or workflow.access == "public") else None

    @classmethod
    async def create(cls, user_id: str, name: str, graph: dict[str, Any], **kwargs):  # type: ignore
        """
        Create a new image in the database.
        """

        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            name=name,
            graph=graph,
            **kwargs,
        )

    def has_trigger_nodes(self) -> bool:
        """Check if the workflow graph contains trigger nodes."""
        if not self.graph or "nodes" not in self.graph:
            return False

        for node in self.graph.get("nodes", []):
            node_type = node.get("type", "")
            # Check if node type contains triggers namespace
            if "triggers." in node_type:
                return True

        return False

    async def _maybe_autostart_trigger(self) -> None:
        if self.run_mode != "trigger":
            return
        if not self.has_trigger_nodes():
            return

        try:
            from nodetool.workflows.trigger_workflow_manager import (
                TriggerWorkflowManager,
            )
        except Exception as exc:
            log.debug("TriggerWorkflowManager unavailable: %s", exc)
            return

        manager = TriggerWorkflowManager.get_instance()
        if getattr(manager, "_watchdog_task", None) is None:
            return

        try:
            await manager.start_trigger_workflow(self, self.user_id)
        except Exception as exc:
            log.error("Failed to auto-start trigger workflow %s: %s", self.id, exc)

    @classmethod
    async def paginate(
        cls,
        user_id: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        columns: list[str] | None = None,
        run_mode: str | None = None,
    ) -> tuple[list["Workflow"], str]:
        """Paginate through workflows, optionally filtering by user.

        Args:
            user_id: If provided, only workflows owned by this user are returned.
                     If None, only public workflows are returned.
            limit: Maximum number of workflows to return.
            start_key: The ID of the workflow to start pagination after (exclusive).
            columns: Specific columns to select. If None, all columns are selected.
            run_mode: Optional run mode to filter by.
        Returns:
            A tuple containing a list of Workflow objects and the ID of the
            last evaluated workflow (or an empty string if it's the last page).

        Raises:
            ValueError: If invalid column names are provided.
        """
        allowed_columns = {
            "id",
            "name",
            "description",
            "thumbnail",
            "thumbnail_url",
            "access",
            "user_id",
            "created_at",
            "updated_at",
            "tags",
            "graph",
            "settings",
            "run_mode",
        }

        # Validate and sanitize column names
        if columns:
            invalid_columns = set(columns) - allowed_columns
            if invalid_columns:
                raise ValueError(f"Invalid columns requested: {invalid_columns}")
            sanitized_columns = [col for col in columns if col in allowed_columns]
        else:
            sanitized_columns = list(allowed_columns)

        conditions = []

        if user_id is None:
            conditions.append(Field("access").equals("public"))
        else:
            conditions.append(Field("user_id").equals(user_id))

        if start_key:
            conditions.append(Field("id").greater_than(start_key))

        if run_mode:
            conditions.append(Field("run_mode").equals(run_mode))

        adapter = await cls.adapter()
        results, last_evaluated_key = await adapter.query(
            columns=sanitized_columns,
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="updated_at",
            reverse=True,
            limit=limit + 1,
        )

        workflows = [Workflow.from_dict(row) for row in results[:limit]]

        return workflows, last_evaluated_key if len(results) > limit else ""

    @classmethod
    async def paginate_tools(cls, user_id: str, limit: int = 100, start_key: str | None = None):
        """Paginate through tools, optionally filtering by user."""
        conditions = []

        conditions.append(Field("user_id").equals(user_id))
        conditions.append(Field("run_mode").equals("tool"))
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

        tools = [Workflow.from_dict(row) for row in results[:limit]]
        tools = [tool for tool in tools if tool.tool_name is not None]

        return tools, last_evaluated_key if len(results) > limit else ""

    @classmethod
    async def find_by_tool_name(cls, user_id: str, tool_name: str):
        """Find a workflow by tool name."""
        conditions = []

        conditions.append(Field("user_id").equals(user_id))
        conditions.append(Field("tool_name").equals(tool_name))
        conditions.append(Field("run_mode").equals("tool"))

        adapter = await cls.adapter()
        results, _ = await adapter.query(
            columns=["*"],
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
        )
        return Workflow.from_dict(results[0]) if results else None

    def get_api_graph(self) -> APIGraph:
        """
        Returns the graph object for the workflow.
        """
        return APIGraph(
            nodes=self.graph["nodes"],
            edges=self.graph["edges"],
        )

    def get_graph(self) -> Graph:
        """
        Returns the graph object for the workflow.
        """
        return Graph(
            nodes=[
                node
                for node in [BaseNode.from_dict(node, skip_errors=True)[0] for node in self.graph["nodes"]]
                if node is not None
            ],
            edges=self.graph["edges"],
        )
