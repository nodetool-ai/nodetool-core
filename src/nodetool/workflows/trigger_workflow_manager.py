"""
Trigger Workflow Manager
========================

This module provides the TriggerWorkflowManager which is responsible for
starting and managing workflows that contain trigger nodes. These workflows
run in the background indefinitely until explicitly stopped.
"""

import asyncio
from typing import Dict, Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import ExecutionStrategy, RunJobRequest

log = get_logger(__name__)

# Default interval for the watchdog to check job health (in seconds)
DEFAULT_WATCHDOG_INTERVAL = 30


def workflow_has_trigger_nodes(workflow: WorkflowModel) -> bool:
    """
    Check if a workflow contains any trigger nodes.

    Args:
        workflow: The workflow model to check.

    Returns:
        True if the workflow contains trigger nodes, False otherwise.
    """
    if not workflow.graph or "nodes" not in workflow.graph:
        return False

    for node in workflow.graph.get("nodes", []):
        node_type = node.get("type", "")
        # Check if the node type is a trigger node
        if "triggers." in node_type:
            return True

    return False


class TriggerWorkflowManager:
    """
    Singleton manager for starting and managing trigger-based workflows.

    This manager:
    - Starts trigger workflows in the background on server startup
    - Tracks running trigger workflows
    - Provides APIs to start/stop trigger workflows
    - Monitors running jobs and restarts them if they die unexpectedly
    """

    _instance: Optional["TriggerWorkflowManager"] = None
    _initialized: bool = False

    def __init__(self) -> None:
        """Initialize the TriggerWorkflowManager instance."""
        # Only initialize once for singleton
        if not TriggerWorkflowManager._initialized:
            self._running_workflows: Dict[str, JobExecution] = {}
            self._workflow_metadata: Dict[str, dict] = {}  # Store workflow info for restarts
            self._watchdog_task: Optional[asyncio.Task] = None
            self._watchdog_interval = DEFAULT_WATCHDOG_INTERVAL
            TriggerWorkflowManager._initialized = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TriggerWorkflowManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = TriggerWorkflowManager()
        return cls._instance

    async def _start_single_job(
        self,
        workflow: WorkflowModel,
        user_id: str,
        auth_token: str = "local_token",
    ) -> Optional[JobExecution]:
        """
        Start a single trigger workflow job.

        This is the core method for starting a job. It creates the job request,
        processing context, and starts the job through the JobExecutionManager.

        Args:
            workflow: The workflow model to start.
            user_id: The user ID for the workflow.
            auth_token: Authentication token for API calls.

        Returns:
            JobExecution if started successfully, None otherwise.
        """
        try:
            # Create the job request
            request = RunJobRequest(
                workflow_id=workflow.id,
                user_id=user_id,
                params={},
                auth_token=auth_token,
                execution_strategy=ExecutionStrategy.THREADED,
            )

            # Create processing context
            context = ProcessingContext()

            # Start the job using JobExecutionManager
            job_manager = JobExecutionManager.get_instance()
            job = await job_manager.start_job(request, context)

            log.info(f"Started job {job.job_id} for trigger workflow {workflow.id}")
            return job

        except Exception as e:
            log.error(f"Failed to start job for trigger workflow {workflow.id}: {e}")
            return None

    async def start_trigger_workflow(
        self,
        workflow: WorkflowModel,
        user_id: str,
        auth_token: str = "local_token",
    ) -> Optional[JobExecution]:
        """
        Start a trigger workflow in the background.

        Args:
            workflow: The workflow model to start.
            user_id: The user ID for the workflow.
            auth_token: Authentication token for API calls.

        Returns:
            JobExecution if started successfully, None otherwise.
        """
        if workflow.id in self._running_workflows:
            job = self._running_workflows[workflow.id]
            if job.is_running():
                log.info(f"Trigger workflow {workflow.id} is already running")
                return job

        if not workflow_has_trigger_nodes(workflow):
            log.warning(f"Workflow {workflow.id} has no trigger nodes, not starting")
            return None

        log.info(f"Starting trigger workflow: {workflow.name} ({workflow.id})")

        job = await self._start_single_job(workflow, user_id, auth_token)

        if job:
            # Track the running workflow
            self._running_workflows[workflow.id] = job
            # Store metadata for potential restarts
            self._workflow_metadata[workflow.id] = {
                "user_id": user_id,
                "auth_token": auth_token,
                "workflow_name": workflow.name,
            }

            log.info(f"Started trigger workflow {workflow.id} with job {job.job_id}")

        return job

    async def stop_trigger_workflow(self, workflow_id: str) -> bool:
        """
        Stop a running trigger workflow.

        Args:
            workflow_id: The workflow ID to stop.

        Returns:
            True if stopped successfully, False otherwise.
        """
        if workflow_id not in self._running_workflows:
            log.warning(f"Trigger workflow {workflow_id} is not running")
            return False

        job = self._running_workflows[workflow_id]

        try:
            job_manager = JobExecutionManager.get_instance()
            cancelled = await job_manager.cancel_job(job.job_id)

            if cancelled:
                log.info(f"Stopped trigger workflow {workflow_id}")
                del self._running_workflows[workflow_id]
                # Remove metadata as workflow is intentionally stopped
                self._workflow_metadata.pop(workflow_id, None)
                return True
            else:
                log.warning(f"Failed to cancel trigger workflow {workflow_id}")
                return False

        except Exception as e:
            log.error(f"Error stopping trigger workflow {workflow_id}: {e}")
            return False

    def get_running_workflow(self, workflow_id: str) -> Optional[JobExecution]:
        """Get a running trigger workflow by ID."""
        return self._running_workflows.get(workflow_id)

    def list_running_workflows(self) -> Dict[str, JobExecution]:
        """List all running trigger workflows."""
        return self._running_workflows.copy()

    def is_workflow_running(self, workflow_id: str) -> bool:
        """Check if a trigger workflow is running."""
        if workflow_id not in self._running_workflows:
            return False
        job = self._running_workflows[workflow_id]
        return job.is_running()

    async def _restart_workflow(self, workflow_id: str) -> bool:
        """
        Restart a trigger workflow that has died.

        Args:
            workflow_id: The workflow ID to restart.

        Returns:
            True if restarted successfully, False otherwise.
        """
        metadata = self._workflow_metadata.get(workflow_id)
        if not metadata:
            log.warning(f"No metadata found for workflow {workflow_id}, cannot restart")
            return False

        try:
            async with ResourceScope():
                workflow = await WorkflowModel.get(workflow_id)
                if not workflow:
                    log.error(f"Workflow {workflow_id} not found, cannot restart")
                    self._workflow_metadata.pop(workflow_id, None)
                    return False

                # Remove old job reference
                self._running_workflows.pop(workflow_id, None)

                # Start a new job
                job = await self._start_single_job(
                    workflow,
                    metadata["user_id"],
                    metadata["auth_token"],
                )

                if job:
                    self._running_workflows[workflow_id] = job
                    log.info(f"Successfully restarted trigger workflow {workflow_id}")
                    return True
                else:
                    log.error(f"Failed to restart trigger workflow {workflow_id}")
                    return False

        except Exception as e:
            log.error(f"Error restarting trigger workflow {workflow_id}: {e}")
            return False

    async def _watchdog_loop(self):
        """
        Watchdog loop that periodically checks job health and restarts dead jobs.

        This runs as a background task and monitors all registered trigger workflows.
        If a job has died unexpectedly, it will be restarted.
        """
        log.info(f"Starting trigger workflow watchdog (interval: {self._watchdog_interval}s)")

        while True:
            try:
                await asyncio.sleep(self._watchdog_interval)

                # Check all tracked workflows
                workflows_to_restart = []

                for workflow_id, job in list(self._running_workflows.items()):
                    if not job.is_running() and not job.is_completed():
                        # Job died unexpectedly
                        log.warning(
                            f"Trigger workflow {workflow_id} (job {job.job_id}) died unexpectedly, status: {job.status}"
                        )
                        workflows_to_restart.append(workflow_id)
                    elif job.is_completed():
                        # Job completed (shouldn't happen for trigger workflows)
                        log.warning(f"Trigger workflow {workflow_id} (job {job.job_id}) completed unexpectedly")
                        workflows_to_restart.append(workflow_id)

                # Restart dead workflows
                for workflow_id in workflows_to_restart:
                    log.info(f"Attempting to restart trigger workflow {workflow_id}")
                    await self._restart_workflow(workflow_id)

            except asyncio.CancelledError:
                log.info("Trigger workflow watchdog cancelled")
                break
            except Exception as e:
                log.error(f"Error in trigger workflow watchdog: {e}")
                # Continue running despite errors

    async def start_watchdog(self, interval: int = DEFAULT_WATCHDOG_INTERVAL):
        """
        Start the watchdog task that monitors job health.

        Args:
            interval: How often to check job health (in seconds).
        """
        if self._watchdog_task is not None and not self._watchdog_task.done():
            log.info("Watchdog is already running")
            return

        self._watchdog_interval = interval
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        log.info("Trigger workflow watchdog started")

    async def stop_watchdog(self):
        """Stop the watchdog task."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None
            log.info("Trigger workflow watchdog stopped")

    async def start_all_trigger_workflows(self) -> int:
        """
        Start all workflows with run_mode="trigger" for all users.

        This method queries for all trigger workflows across all users and
        starts each one under the owner's user_id.

        Returns:
            Number of workflows started.
        """
        started = 0

        try:
            async with ResourceScope():
                # Get all trigger workflows (no user filter to get all users)
                # We need to query without user_id to get all trigger workflows
                workflows, _ = await WorkflowModel.paginate(
                    user_id=None,  # Get all users' workflows
                    limit=1000,
                    run_mode="trigger",
                )

                log.info(f"Found {len(workflows)} trigger workflows across all users")

                for workflow in workflows:
                    if workflow_has_trigger_nodes(workflow):
                        # Start workflow under the owner's user_id
                        job = await self.start_trigger_workflow(
                            workflow,
                            user_id=workflow.user_id,  # Use workflow owner's user_id
                        )
                        if job:
                            started += 1

        except Exception as e:
            log.error(f"Error starting trigger workflows: {e}")

        log.info(f"Started {started} trigger workflows")
        return started

    async def stop_all_trigger_workflows(self) -> int:
        """
        Stop all running trigger workflows.

        Returns:
            Number of workflows stopped.
        """
        stopped = 0
        workflow_ids = list(self._running_workflows.keys())

        for workflow_id in workflow_ids:
            if await self.stop_trigger_workflow(workflow_id):
                stopped += 1

        log.info(f"Stopped {stopped} trigger workflows")
        return stopped

    async def shutdown(self):
        """Shutdown the trigger workflow manager."""
        log.info("Shutting down TriggerWorkflowManager")
        await self.stop_watchdog()
        await self.stop_all_trigger_workflows()
        log.info("TriggerWorkflowManager shutdown complete")
