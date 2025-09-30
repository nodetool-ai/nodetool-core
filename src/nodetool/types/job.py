from typing import Literal, Optional, List
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


class JobUpdate(BaseModel):
    type: Literal["job_update"] = "job_update"
    status: str
    job_id: str | None = None
    workflow_id: str | None = None
    message: str | None = None
    result: dict | None = None
    error: str | None = None


class JobList(BaseModel):
    next: Optional[str]
    jobs: List[Job]
