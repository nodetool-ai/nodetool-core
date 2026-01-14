from typing import Literal

from pydantic import BaseModel


class JobCancelledException(Exception):
    pass


class Job(BaseModel):
    id: str
    job_type: str
    status: str
    workflow_id: str
    started_at: str
    finished_at: str | None
    error: str | None
    cost: float | None


class JobRequest(BaseModel):
    workflow_id: str
    params: dict


class RunStateInfo(BaseModel):
    """Run state info for WebSocket messages."""

    status: str
    suspended_node_id: str | None = None
    suspension_reason: str | None = None
    error_message: str | None = None
    execution_strategy: str | None = None
    is_resumable: bool = False


class JobUpdate(BaseModel):
    type: Literal["job_update"] = "job_update"
    status: str
    job_id: str | None = None
    workflow_id: str | None = None
    message: str | None = None
    result: dict | None = None
    error: str | None = None
    traceback: str | None = None
    run_state: RunStateInfo | None = None


class JobList(BaseModel):
    next: str | None
    jobs: list[Job]
