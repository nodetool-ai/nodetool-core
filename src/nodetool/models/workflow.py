from datetime import datetime
from typing import Any, Optional
from nodetool.models.condition_builder import (
    ConditionBuilder,
    ConditionGroup,
    Field,
    LogicalOperator,
)
from nodetool.types.graph import Graph as APIGraph
from nodetool.workflows.graph import Graph
from nodetool.workflows.base_node import BaseNode

from nodetool.models.base_model import (
    DBModel,
    DBField,
    DBIndex,
    create_time_ordered_uuid,
)


"""
Defines the Workflow database model.

Represents a workflow in the nodetool system, including its structure (graph),
metadata (name, description, tags), user association, access control, and settings.
"""


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
            package_name=data.get("package_name", ""),
            tags=data.get("tags", []),
            description=data.get("description", ""),
            thumbnail=data.get("thumbnail", None),
            settings=data.get("settings", {}),
            graph=data.get(
                "graph",
                {
                    "nodes": [],
                    "edges": [],
                },
            ),
            run_mode=data.get("run_mode", None),
        )

    @classmethod
    def find(cls, user_id: str, workflow_id: str):
        """Find a workflow by ID, respecting user ownership and access level.

        Args:
            user_id: The ID of the user attempting to find the workflow.
            workflow_id: The ID of the workflow to find.

        Returns:
            The Workflow object if found and accessible, otherwise None.
        """
        workflow = cls.get(workflow_id)
        return (
            workflow
            if workflow and (workflow.user_id == user_id or workflow.access == "public")
            else None
        )

    @classmethod
    def create(cls, user_id: str, name: str, graph: dict[str, Any], **kwargs):
        """
        Create a new image in the database.
        """

        return super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            name=name,
            graph=graph,
            **kwargs,
        )

    @classmethod
    def paginate(
        cls,
        user_id: str | None = None,
        limit: int = 100,
        start_key: Optional[str] = None,
        columns: list[str] | None = None,
    ) -> tuple[list["Workflow"], str]:
        """Paginate through workflows, optionally filtering by user.

        Args:
            user_id: If provided, only workflows owned by this user are returned.
                     If None, only public workflows are returned.
            limit: Maximum number of workflows to return.
            start_key: The ID of the workflow to start pagination after (exclusive).
            columns: Specific columns to select. If None, all columns are selected.

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

        results, last_evaluated_key = cls.adapter().query(
            columns=sanitized_columns,
            condition=ConditionBuilder(ConditionGroup(conditions, LogicalOperator.AND)),
            order_by="updated_at",
            reverse=True,
            limit=limit + 1,
        )

        workflows = [Workflow.from_dict(row) for row in results[:limit]]

        return workflows, last_evaluated_key if len(results) > limit else ""

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
                node for node in [
                    BaseNode.from_dict(node, skip_errors=True)[0]
                    for node in self.graph["nodes"]
                ] if node is not None
            ],
            edges=self.graph["edges"],
        )
