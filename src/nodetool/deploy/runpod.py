"""
RunPod deployment implementation for NodeTool.

This module handles deployment to RunPod serverless infrastructure, including:
- Docker image building and pushing
- RunPod template creation/update
- Serverless endpoint creation
- Endpoint management
"""

import logging
from typing import Any, Optional

from nodetool.config.deployment import (
    DeploymentStatus,
    RunPodDeployment,
)
from nodetool.deploy.deploy_to_runpod import deploy_to_runpod
from nodetool.deploy.state import StateManager

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

    def plan(self) -> dict[str, Any]:
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

    def apply(self, dry_run: bool = False) -> dict[str, Any]:
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
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DEPLOYING.value)

            results["steps"].append("Starting RunPod deployment...")

            # Prepare environment variables
            env = dict(self.deployment.environment) if self.deployment.environment else {}
            template_name = self.deployment.template_name or self.deployment_name

            deploy_kwargs = {
                "deployment": self.deployment,
                "docker_username": self.deployment.docker.username,
                "docker_registry": self.deployment.docker.registry,
                "image_name": self.deployment.image.name,
                "tag": self.deployment.image.tag,
                "template_name": template_name,
                "platform": self.deployment.platform,
                "gpu_types": tuple(self.deployment.gpu_types),
                "gpu_count": self.deployment.gpu_count,
                "data_centers": tuple(self.deployment.data_centers),
                "workers_min": self.deployment.workers_min,
                "workers_max": self.deployment.workers_max,
                "idle_timeout": self.deployment.idle_timeout,
                "execution_timeout": self.deployment.execution_timeout,
                "flashboot": self.deployment.flashboot,
                "env": env,
                "skip_build": False,
                "skip_push": False,
                "skip_template": False,
                "skip_endpoint": False,
                "name": self.deployment_name,
            }

            # Call legacy deploy function
            deploy_to_runpod(**deploy_kwargs)  # type: ignore[arg-type]

            results["steps"].append("RunPod deployment completed")

            # Update state with success
            # Note: The legacy function doesn't return endpoint details,
            # so we can't store them in state yet
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.ACTIVE.value,
                    "template_name": template_name,
                },
            )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))

            # Update state with error
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.ERROR.value)

            raise

        return results

    def status(self) -> dict[str, Any]:
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
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["template_name"] = state.get("template_name", "unknown")
            status_info["pod_id"] = state.get("pod_id", "unknown")

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
            "RunPod serverless endpoints don't provide direct log access. Check logs via RunPod web console or API."
        )

    def destroy(self) -> dict[str, Any]:
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
            results["steps"].append("⚠️  RunPod endpoint deletion must be done manually via RunPod console")
            results["steps"].append(
                f"Visit https://www.runpod.io/console/serverless and delete endpoint '{self.deployment_name}'"
            )

            # Update state
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DESTROYED.value)

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results
