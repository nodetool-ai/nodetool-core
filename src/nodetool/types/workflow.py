from typing import Any, List

from pydantic import BaseModel

from nodetool.types.api_graph import Graph


class Workflow(BaseModel):
    id: str
    access: str
    created_at: str
    updated_at: str
    name: str
    tool_name: str | None = None
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
    run_mode: str | None = None
    workspace_id: str | None = None
    required_providers: list[str] | None = None
    required_models: list[str] | None = None


class WorkflowRequest(BaseModel):
    name: str
    tool_name: str | None = None
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
    run_mode: str | None = None
    workspace_id: str | None = None


class WorkflowList(BaseModel):
    next: str | None
    workflows: List[Workflow]


class WorkflowTool(BaseModel):
    name: str
    tool_name: str | None = None
    description: str | None = None


class WorkflowToolList(BaseModel):
    next: str | None
    workflows: List[WorkflowTool]


class WorkflowVersion(BaseModel):
    """Represents a version/snapshot of a workflow."""

    id: str
    workflow_id: str
    version: int
    created_at: str
    name: str
    description: str = ""
    graph: Graph
    save_type: str = "manual"
    autosave_metadata: dict[str, Any] = {}


class WorkflowVersionList(BaseModel):
    """List of workflow versions with pagination support."""

    next: str | None
    versions: List[WorkflowVersion]


class CreateWorkflowVersionRequest(BaseModel):
    """Request to create a new workflow version."""

    name: str = ""
    description: str = ""


class AutosaveWorkflowRequest(BaseModel):
    """Request to autosave a workflow version."""

    save_type: str = "autosave"
    description: str = ""
    force: bool = False
    client_id: str | None = None


class AutosaveResponse(BaseModel):
    """Response from an autosave request."""

    version: WorkflowVersion | None = None
    message: str
    skipped: bool = False
