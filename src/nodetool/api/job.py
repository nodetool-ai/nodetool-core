import asyncio
from datetime import UTC, datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.job import Job
from nodetool.workflows.job_execution_manager import JobExecutionManager

log = get_logger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobResponse(BaseModel):
    id: str
    user_id: str
    job_type: str
    status: str
    workflow_id: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    cost: Optional[float] = None

    class Config:
        from_attributes = True


class BackgroundJobResponse(BaseModel):
    job_id: str
    status: str
    workflow_id: str
    created_at: Optional[datetime] = None
    is_running: bool
    is_completed: bool


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    next_start_key: Optional[str] = None


@router.get("/", response_model=JobListResponse)
async def list_jobs(
    user_id: str = Depends(current_user),
    workflow_id: Optional[str] = None,
    limit: int = 100,
    start_key: Optional[str] = None,
):
    """
    List jobs for the current user.

    Args:
        user_id: Current authenticated user ID
        workflow_id: Optional workflow ID to filter by
        limit: Maximum number of jobs to return
        start_key: Pagination start key

    Returns:
        List of jobs
    """
    jobs, next_start_key = await Job.paginate(
        user_id=user_id, workflow_id=workflow_id, limit=limit, start_key=start_key
    )

    # Reconcile DB status with the background manager for this page of jobs
    await reconcile_jobs_for_user(user_id, jobs)

    log.info(
        "Jobs API list_jobs",
        extra={
            "user_id": user_id,
            "workflow_id": workflow_id,
            "limit": limit,
            "start_key": start_key,
            "job_count": len(jobs),
            "job_ids": [job.id for job in jobs],
        },
    )

    return JobListResponse(
        jobs=[
            JobResponse(
                id=job.id,
                user_id=job.user_id,
                job_type=job.job_type,
                status=job.status,
                workflow_id=job.workflow_id,
                started_at=job.started_at,
                finished_at=job.finished_at,
                error=job.error,
                cost=job.cost,
            )
            for job in jobs
        ],
        next_start_key=next_start_key,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, user_id: str = Depends(current_user)):
    """
    Get a specific job by ID.

    Args:
        job_id: Job ID
        user_id: Current authenticated user ID

    Returns:
        Job details
    """
    job = await Job.find(user_id=user_id, job_id=job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        id=job.id,
        user_id=job.user_id,
        job_type=job.job_type,
        status=job.status,
        workflow_id=job.workflow_id,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        cost=job.cost,
    )


@router.get("/running/all", response_model=List[BackgroundJobResponse])
async def list_running_jobs(user_id: str = Depends(current_user)):
    """
    List all currently running background jobs for the current user.

    Args:
        user_id: Current authenticated user ID

    Returns:
        List of running background jobs
    """
    job_manager = JobExecutionManager.get_instance()
    bg_jobs = job_manager.list_jobs(user_id=user_id)

    return [
        BackgroundJobResponse(
            job_id=job.job_id,
            status=job.status,
            workflow_id=job.request.workflow_id,
            created_at=job.created_at,
            is_running=job.is_running(),
            is_completed=job.is_completed(),
        )
        for job in bg_jobs
    ]


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, user_id: str = Depends(current_user)) -> BackgroundJobResponse:
    """
    Cancel a running job.

    Args:
        job_id: Job ID to cancel
        user_id: Current authenticated user ID

    Returns:
        Success message
    """
    # Verify the job belongs to the user
    job = await Job.find(user_id=user_id, job_id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Cancel the background job if it's running
    job_manager = JobExecutionManager.get_instance()
    cancelled = await job_manager.cancel_job(job_id)

    status = "cancelled" if cancelled else "not_found_or_completed"
    return BackgroundJobResponse(
        job_id=job_id,
        status=status,
        workflow_id=job.workflow_id,
        created_at=job.started_at,
        is_running=False,
        is_completed=not cancelled,
    )


class TriggerWorkflowResponse(BaseModel):
    workflow_id: str
    job_id: Optional[str] = None
    status: str
    is_running: bool


class TriggerWorkflowListResponse(BaseModel):
    workflows: List[TriggerWorkflowResponse]


@router.get("/triggers/running", response_model=TriggerWorkflowListResponse)
async def list_running_trigger_workflows(user_id: str = Depends(current_user)):
    """
    List all currently running trigger workflows.

    Args:
        user_id: Current authenticated user ID

    Returns:
        List of running trigger workflows
    """
    from nodetool.workflows.trigger_workflow_manager import TriggerWorkflowManager

    trigger_manager = TriggerWorkflowManager.get_instance()
    running = trigger_manager.list_running_workflows()

    return TriggerWorkflowListResponse(
        workflows=[
            TriggerWorkflowResponse(
                workflow_id=workflow_id,
                job_id=job.job_id,
                status=job.status,
                is_running=job.is_running(),
            )
            for workflow_id, job in running.items()
        ]
    )


@router.post("/triggers/{workflow_id}/start", response_model=TriggerWorkflowResponse)
async def start_trigger_workflow(workflow_id: str, user_id: str = Depends(current_user)):
    """
    Start a trigger workflow in the background.

    Args:
        workflow_id: Workflow ID to start
        user_id: Current authenticated user ID

    Returns:
        Trigger workflow status
    """
    from nodetool.models.workflow import Workflow as WorkflowModel
    from nodetool.workflows.trigger_workflow_manager import (
        TriggerWorkflowManager,
        workflow_has_trigger_nodes,
    )

    workflow = await WorkflowModel.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    if not workflow_has_trigger_nodes(workflow):
        raise HTTPException(status_code=400, detail="Workflow does not contain trigger nodes")

    trigger_manager = TriggerWorkflowManager.get_instance()
    job = await trigger_manager.start_trigger_workflow(workflow, user_id)

    if job:
        return TriggerWorkflowResponse(
            workflow_id=workflow_id,
            job_id=job.job_id,
            status=job.status,
            is_running=job.is_running(),
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to start trigger workflow")


@router.post("/triggers/{workflow_id}/stop", response_model=TriggerWorkflowResponse)
async def stop_trigger_workflow(workflow_id: str, user_id: str = Depends(current_user)):
    """
    Stop a running trigger workflow.

    Args:
        workflow_id: Workflow ID to stop
        user_id: Current authenticated user ID

    Returns:
        Trigger workflow status
    """
    from nodetool.models.workflow import Workflow as WorkflowModel
    from nodetool.workflows.trigger_workflow_manager import TriggerWorkflowManager

    workflow = await WorkflowModel.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    trigger_manager = TriggerWorkflowManager.get_instance()
    stopped = await trigger_manager.stop_trigger_workflow(workflow_id)

    return TriggerWorkflowResponse(
        workflow_id=workflow_id,
        job_id=None,
        status="stopped" if stopped else "not_running",
        is_running=False,
    )


async def reconcile_jobs_for_user(user_id: str, jobs: List[Job]) -> None:
    """
    Ensure job status reflects the background execution manager.
    Syncs completed/failed states from the background manager; marks missing handles as failed.
    """
    job_manager = JobExecutionManager.get_instance()
    bg_jobs = {job.job_id: job for job in job_manager.list_jobs(user_id=user_id)}

    updates = []
    for job in jobs:
        bg_job = bg_jobs.get(job.id)
        if bg_job is None and job.status in {"running", "starting", "queued"}:
            job.status = "failed"
            job.error = job.error or "Reconciled: execution handle missing"
            job.finished_at = job.finished_at or datetime.now(UTC)
            updates.append(job.save())
        elif bg_job is not None and bg_job.is_completed():
            new_status = getattr(bg_job, "status", "completed")
            if job.status != new_status or job.finished_at is None:
                job.status = new_status
                job.error = job.error or bg_job.error
                job.finished_at = job.finished_at or datetime.now(UTC)
                updates.append(job.save())
        elif bg_job is not None and not bg_job.is_running():
            # Not running but not completed: mark as failed
            if job.status in {"running", "starting", "queued"}:
                job.status = "failed"
                job.error = job.error or "Reconciled: execution handle stopped unexpectedly"
                job.finished_at = job.finished_at or datetime.now(UTC)
                updates.append(job.save())

    if updates:
        await asyncio.gather(*updates)
