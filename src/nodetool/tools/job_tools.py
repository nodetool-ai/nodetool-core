"""Job and run management tools.

These tools provide functionality for managing workflow executions.
"""

from __future__ import annotations

from typing import Any

from nodetool.models.job import Job as JobModel
from nodetool.models.run_event import RunEvent
from nodetool.workflows.job_execution_manager import JobExecutionManager


class JobTools:
    """Job management tools."""

    @staticmethod
    async def list_jobs(
        workflow_id: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        List jobs for the user, optionally filtered by workflow.

        Args:
            workflow_id: Optional workflow ID to filter by
            limit: Maximum number of jobs to return
            start_key: Pagination start key

        Returns:
            Dictionary containing jobs and pagination cursor
        """
        jobs, next_start_key = await JobModel.paginate(
            user_id=user_id,
            workflow_id=workflow_id,
            limit=limit,
            start_key=start_key,
        )

        return {
            "jobs": [
                {
                    "id": job.id,
                    "user_id": job.user_id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "workflow_id": job.workflow_id,
                    "started_at": job.started_at.isoformat() if job.started_at else "",
                    "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                    "error": job.error,
                    "cost": job.cost,
                }
                for job in jobs
            ],
            "next_start_key": next_start_key,
        }

    @staticmethod
    async def get_job(job_id: str, user_id: str = "1") -> dict[str, Any]:
        """
        Get a job by ID for the user.

        Args:
            job_id: The job ID
            user_id: User ID (default: "1")

        Returns:
            Job details
        """
        job = await JobModel.find(user_id=user_id, job_id=job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        return {
            "id": job.id,
            "user_id": job.user_id,
            "job_type": job.job_type,
            "status": job.status,
            "workflow_id": job.workflow_id,
            "started_at": job.started_at.isoformat() if job.started_at else "",
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "error": job.error,
            "cost": job.cost,
        }

    @staticmethod
    async def get_job_logs(job_id: str, limit: int = 200, user_id: str = "1") -> dict[str, Any]:
        """
        Get logs for a job, preferring live logs for running jobs.

        Args:
            job_id: The job ID
            limit: Maximum number of logs to return
            user_id: User ID (default: "1")

        Returns:
            Job logs
        """
        job = await JobModel.find(user_id=user_id, job_id=job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        manager = JobExecutionManager.get_instance()
        live = manager.get_job(job_id)
        logs = live.get_live_logs(limit=limit) if live is not None else (job.logs or [])[: max(0, limit)]

        return {"job_id": job_id, "logs": logs}

    @staticmethod
    async def start_background_job(
        workflow_id: str,
        params: dict[str, Any] | None = None,
        execution_strategy: str = "threaded",
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Start running a workflow in the background.

        Args:
            workflow_id: The workflow ID to run
            params: Optional input parameters
            execution_strategy: Execution strategy (default: "threaded")
            user_id: User ID (default: "1")

        Returns:
            Job information including job_id and status
        """
        from nodetool.models.workflow import Workflow as WorkflowModel
        from nodetool.workflows.processing_context import (
            AssetOutputMode,
            ProcessingContext,
        )
        from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest

        workflow = await WorkflowModel.find(user_id, workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        try:
            strategy = ExecutionStrategy(execution_strategy)
        except ValueError as exc:
            raise ValueError(f"Invalid execution_strategy: {execution_strategy}") from exc

        request = RunJobRequest(
            user_id=user_id,
            workflow_id=workflow_id,
            params=params or {},
            graph=workflow.get_api_graph(),
            execution_strategy=strategy,
        )
        context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)

        manager = JobExecutionManager.get_instance()
        job = await manager.start_job(request, context)
        return {"job_id": job.job_id, "status": job.status, "workflow_id": workflow_id}

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all job tool functions."""
        return {
            "list_jobs": JobTools.list_jobs,
            "get_job": JobTools.get_job,
            "get_job_logs": JobTools.get_job_logs,
            "start_background_job": JobTools.start_background_job,
        }
