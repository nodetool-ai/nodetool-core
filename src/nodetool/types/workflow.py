from typing import Any, List
from nodetool.types.graph import Graph
from pydantic import BaseModel


class Workflow(BaseModel):
    id: str
    access: str
    created_at: str
    updated_at: str
    name: str
    description: str
    tags: list[str] | None = None
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    graph: Graph
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    settings: dict[str, str | bool | int | float | None] | None = None
    package_name: str | None = None
    path: str | None = None


class WorkflowRequest(BaseModel):
    name: str
    package_name: str | None = None
    path: str | None = None
    tags: list[str] | None = None
    description: str | None = None
    thumbnail: str | None = None
    thumbnail_url: str | None = None
    access: str
    graph: Graph | None = None
    comfy_workflow: dict[str, Any] | None = None
    settings: dict[str, str | bool | int | float | None] | None = None


class WorkflowList(BaseModel):
    next: str | None
    workflows: List[Workflow]
