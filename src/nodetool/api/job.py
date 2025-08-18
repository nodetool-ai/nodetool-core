#!/usr/bin/env python

from fastapi import APIRouter, Depends, HTTPException
from nodetool.api.utils import current_user

from nodetool.types.job import (
    Job,
    JobList,
    JobUpdate,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.common.environment import Environment

from nodetool.models.job import Job as JobModel


log = Environment.get_logger()
router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def from_model(job: JobModel):
    return Job(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        workflow_id=job.workflow_id,
        started_at=job.started_at.isoformat(),
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
        error=job.error,
        cost=job.cost,
    )


@router.get("/{id}")
async def get(id: str, user: str = Depends(current_user)) -> Job:
    """
    Returns the status of a job.
    """
    job = await JobModel.find(user, id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    else:
        if job.user_id != user:
            raise HTTPException(status_code=403, detail="Forbidden")
        else:
            return from_model(job)


@router.get("/")
async def index(
    workflow_id: str | None = None,
    cursor: str | None = None,
    page_size: int | None = None,
    user: str = Depends(current_user),
) -> JobList:
    """
    Returns all assets for a given user or workflow.
    """
    if page_size is None:
        page_size = 10

    jobs, next_cursor = await JobModel.paginate(
        user_id=user, workflow_id=workflow_id, limit=page_size, start_key=cursor
    )

    return JobList(next=next_cursor, jobs=[from_model(job) for job in jobs])


@router.put("/{id}")
async def update(id: str, req: JobUpdate, user: str = Depends(current_user)) -> Job:
    """
    Update a job.
    """
    job = await JobModel.find(user, id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    else:
        if job.user_id != user:
            raise HTTPException(status_code=403, detail="Forbidden")
        else:
            job.status = req.status
            job.error = req.error
            await job.save()
            return from_model(job)


@router.post("/")
async def create(
    job_request: RunJobRequest,
    user: str = Depends(current_user),
):

    job = await JobModel.create(
        job_type=job_request.job_type,
        workflow_id=job_request.workflow_id,
        user_id=user,
        graph=job_request.graph.model_dump() if job_request.graph else None,
        status="running",
    )

    return job
