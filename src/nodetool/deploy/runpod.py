"""
RunPod deployment implementation for NodeTool.

This module handles deployment to RunPod serverless infrastructure, including:
- Docker image building and pushing
- RunPod template creation/update
- Serverless endpoint creation
- Endpoint management
"""

from typing import Dict, Any, Optional
import logging

from nodetool.config.deployment import (
    RunPodDeployment,
    DeploymentStatus,
)
from nodetool.deploy.state import StateManager
from nodetool.deploy.deploy_to_runpod import deploy_to_runpod as legacy_deploy_to_runpod

logger = logging.getLogger(__name__)


class RunPodDeployer:
    """
    Handles deployment to RunPod serverless infrastructure.

    This class orchestrates the entire RunPod deployment process including:
    - Docker image building and pushing
    - Template creation/update
    - Endpoint creation and configuration
    - State management
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: RunPodDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the RunPod deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: RunPod deployment configuration
            state_manager: State manager instance (optional)
        """
        self.deployment_name = deployment_name
        self.deployment = deployment
        self.state_manager = state_manager or StateManager()

    def plan(self) -> Dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        Returns:
            Dictionary describing planned changes
        """
        plan = {
            "deployment_name": self.deployment_name,
            "type": "runpod",
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)

        # Check if this is initial deployment
        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial deployment - will create all resources")
            plan["will_create"].extend(
                [
                    "Docker image",
                    "RunPod template",
                    "RunPod serverless endpoint",
                ]
            )
        else:
            # Check for configuration changes
            # TODO: Implement more granular change detection
            plan["changes"].append("Configuration may have changed")
            plan["will_update"].append("RunPod endpoint configuration")

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply the deployment to RunPod.

        Args:
            dry_run: If True, only show what would be done without executing

        Returns:
            Dictionary with deployment results
        """
        if dry_run:
            return self.plan()

        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            # Update state to deploying
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DEPLOYING.value
            )

            results["steps"].append("Starting RunPod deployment...")

            # Prepare environment variables
            env = (
                dict(self.deployment.environment) if self.deployment.environment else {}
            )

            # Convert GPU types to list
            gpu_types = (
                tuple(self.deployment.gpu_types) if self.deployment.gpu_types else ()
            )

            # Convert data centers to list
            data_centers = (
                tuple(self.deployment.data_centers)
                if self.deployment.data_centers
                else ()
            )

            # Call legacy deploy function
            # TODO: Refactor this to use direct API calls instead of legacy function
            legacy_deploy_to_runpod(
                docker_username=self.deployment.docker.username,
                docker_registry=self.deployment.docker.registry,
                image_name=self.deployment.image.name,
                tag=self.deployment.image.tag,
                platform="linux/amd64",  # RunPod requires amd64
                template_name=self.deployment.template_name or self.deployment_name,
                skip_build=False,
                skip_push=False,
                skip_template=False,
                skip_endpoint=False,
                compute_type=self.deployment.compute_type,
                gpu_types=gpu_types,
                gpu_count=self.deployment.gpu_count,
                data_centers=data_centers,
                workers_min=self.deployment.workers_min,
                workers_max=self.deployment.workers_max,
                idle_timeout=self.deployment.idle_timeout,
                execution_timeout=self.deployment.execution_timeout,
                flashboot=self.deployment.flashboot,
                network_volume_id=self.deployment.network_volume_id,
                name=self.deployment_name,
                env=env,
            )

            results["steps"].append("RunPod deployment completed")

            # Update state with success
            # Note: The legacy function doesn't return endpoint details,
            # so we can't store them in state yet
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.ACTIVE.value,
                    "template_name": self.deployment.template_name
                    or self.deployment_name,
                },
            )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))

            # Update state with error
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.ERROR.value
            )

            raise

        return results

    def status(self) -> Dict[str, Any]:
        """
        Get current deployment status.

        Returns:
            Dictionary with current status information
        """
        status_info = {
            "deployment_name": self.deployment_name,
            "type": "runpod",
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed")
            status_info["template_name"] = state.get("template_name")
            status_info["pod_id"] = state.get("pod_id")

        # TODO: Query RunPod API for live status

        return status_info

    def logs(
        self,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """
        Get logs from RunPod endpoint.

        Args:
            service: Not used for RunPod (kept for interface compatibility)
            follow: Not supported for RunPod serverless
            tail: Number of lines to show

        Returns:
            Log output as string

        Raises:
            NotImplementedError: RunPod serverless doesn't provide log access
        """
        raise NotImplementedError(
            "RunPod serverless endpoints don't provide direct log access. "
            "Check logs via RunPod web console or API."
        )

    def destroy(self) -> Dict[str, Any]:
        """
        Destroy the deployment (delete endpoint).

        Returns:
            Dictionary with destruction results
        """
        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            results["steps"].append("Destroying RunPod endpoint...")

            # TODO: Implement endpoint deletion via RunPod API
            # For now, user must delete manually via RunPod console
            results["steps"].append(
                "⚠️  RunPod endpoint deletion must be done manually via RunPod console"
            )
            results["steps"].append(
                f"Visit https://www.runpod.io/console/serverless and delete endpoint '{self.deployment_name}'"
            )

            # Update state
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DESTROYED.value
            )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results
