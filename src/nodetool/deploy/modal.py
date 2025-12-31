"""
Modal deployment implementation for NodeTool.

This module handles deployment to Modal serverless infrastructure, including:
- Function deployment with configurable resources
- GPU support
- Secret management
- App lifecycle management
"""

import logging
from typing import Any, Dict, Optional

from nodetool.config.deployment import (
    DeploymentStatus,
    ModalDeployment,
)
from nodetool.deploy.modal_api import (
    delete_modal_app,
    deploy_modal_app,
    get_modal_app_status,
    get_modal_logs,
    stop_modal_app,
)
from nodetool.deploy.state import StateManager

logger = logging.getLogger(__name__)


class ModalDeployer:
    """
    Handles deployment to Modal serverless infrastructure.

    This class orchestrates the entire Modal deployment process including:
    - App creation and configuration
    - Function deployment
    - Resource configuration (CPU, memory, GPU)
    - State management
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: ModalDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the Modal deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: Modal deployment configuration
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
            "type": "modal",
            "app_name": self.deployment.app_name,
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
                    f"Modal app: {self.deployment.app_name}",
                    f"Modal function: {self.deployment.function_name}",
                ]
            )
        else:
            # Check for configuration changes
            plan["changes"].append("Configuration may have changed")
            plan["will_update"].append(f"Modal app: {self.deployment.app_name}")

        # Add resource summary
        resources = self.deployment.resources
        plan["resources"] = {
            "cpu": resources.cpu,
            "memory": f"{resources.memory}MB",
            "gpu": f"{resources.gpu.type} x{resources.gpu.count}" if resources.gpu else None,
            "timeout": f"{resources.timeout}s",
        }

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply the deployment to Modal.

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

            results["steps"].append("Starting Modal deployment...")

            # Prepare deployment parameters
            image_config = {
                "base_image": self.deployment.image.base_image,
                "pip_packages": self.deployment.image.pip_packages,
                "apt_packages": self.deployment.image.apt_packages,
                "dockerfile": self.deployment.image.dockerfile,
                "context_dir": self.deployment.image.context_dir,
            }

            resources = self.deployment.resources

            # Deploy the app
            deploy_result = deploy_modal_app(
                app_name=self.deployment.app_name,
                function_name=self.deployment.function_name,
                image_config=image_config,
                cpu=resources.cpu,
                memory=resources.memory,
                gpu_type=resources.gpu.type if resources.gpu else None,
                gpu_count=resources.gpu.count if resources.gpu else 1,
                timeout=resources.timeout,
                container_idle_timeout=resources.container_idle_timeout,
                allow_concurrent_inputs=resources.allow_concurrent_inputs,
                environment=dict(self.deployment.environment) if self.deployment.environment else None,
                secrets=list(self.deployment.secrets) if self.deployment.secrets else None,
                region=self.deployment.region,
            )

            results["steps"].append("Modal deployment completed")
            results["app_id"] = deploy_result.get("app_id")
            results["function_url"] = deploy_result.get("function_url")

            # Update state with success
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.ACTIVE.value,
                    "app_name": self.deployment.app_name,
                    "app_id": deploy_result.get("app_id"),
                    "function_url": deploy_result.get("function_url"),
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
            "type": "modal",
            "app_name": self.deployment.app_name,
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["app_id"] = state.get("app_id", "unknown")
            status_info["function_url"] = state.get("function_url", "unknown")

        # Try to get live status from Modal
        try:
            live_status = get_modal_app_status(self.deployment.app_name)
            status_info["live_status"] = live_status.get("status")
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(
        self,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """
        Get logs from Modal app.

        Args:
            service: Not used for Modal (kept for interface compatibility)
            follow: Not supported for Modal (logs available via CLI)
            tail: Number of lines to show

        Returns:
            Log output as string (with instructions for accessing logs)
        """
        return get_modal_logs(self.deployment.app_name, tail=tail)

    def destroy(self) -> Dict[str, Any]:
        """
        Destroy the deployment (delete/stop the Modal app).

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
            results["steps"].append(f"Stopping Modal app: {self.deployment.app_name}...")

            # Stop/delete the app
            delete_result = delete_modal_app(self.deployment.app_name)

            if delete_result.get("status") == "deleted":
                results["steps"].append("Modal app stopped successfully")
            elif delete_result.get("status") == "not_found":
                results["steps"].append("Modal app not found (may already be deleted)")
            else:
                results["errors"].append(delete_result.get("error", "Unknown error"))
                results["status"] = "error"

            # Update state
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DESTROYED.value
            )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results
