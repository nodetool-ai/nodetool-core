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
from nodetool.workflows.run_job_request import RunJobRequest, ExecutionStrategy

log = get_logger(__name__)


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
        if node_type.startswith("nodetool.nodes.triggers."):
            return True

    return False


class TriggerWorkflowManager:
    """
    Singleton manager for starting and managing trigger-based workflows.

    This manager:
    - Starts trigger workflows in the background on server startup
    - Tracks running trigger workflows
    - Provides APIs to start/stop trigger workflows
    """

    _instance: Optional["TriggerWorkflowManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._running_workflows: Dict[str, JobExecution] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "TriggerWorkflowManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = TriggerWorkflowManager()
        return cls._instance

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

            # Track the running workflow
            self._running_workflows[workflow.id] = job

            log.info(f"Started trigger workflow {workflow.id} with job {job.job_id}")
            return job

        except Exception as e:
            log.error(f"Failed to start trigger workflow {workflow.id}: {e}")
            return None

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

    async def start_all_trigger_workflows(self, user_id: str) -> int:
        """
        Start all workflows with run_mode="trigger" for a user.

        Args:
            user_id: The user ID to start workflows for. Required parameter.

        Returns:
            Number of workflows started.
        """
        started = 0

        try:
            async with ResourceScope():
                # Get all trigger workflows
                workflows, _ = await WorkflowModel.paginate(
                    user_id=user_id,
                    limit=1000,
                    run_mode="trigger",
                )

                log.info(f"Found {len(workflows)} trigger workflows for user {user_id}")

                for workflow in workflows:
                    if workflow_has_trigger_nodes(workflow):
                        job = await self.start_trigger_workflow(workflow, user_id)
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
        await self.stop_all_trigger_workflows()
        log.info("TriggerWorkflowManager shutdown complete")
