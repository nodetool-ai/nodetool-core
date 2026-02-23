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
from nodetool.deploy.runpod_api import (
    get_runpod_endpoint_by_name,
    get_runpod_template_by_name,
)
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
            changes, will_update = self._check_changes(current_state)
            plan["changes"].extend(changes)
            plan["will_update"].extend(will_update)

        return plan

    def _check_changes(self, current_state: dict[str, Any]) -> tuple[list[str], list[str]]:
        """
        Check for changes between current configuration and RunPod state.

        Args:
            current_state: The current state from the state manager

        Returns:
            Tuple of (changes list, will_update list)
        """
        changes = []
        will_update = []

        # Get endpoint name and template name
        endpoint_name = self.deployment_name
        template_name = self.deployment.template_name or self.deployment_name

        try:
            # Fetch current endpoint state
            endpoint = get_runpod_endpoint_by_name(endpoint_name)
            if not endpoint:
                changes.append(f"Endpoint '{endpoint_name}' not found on RunPod (will be created)")
                will_update.append("Create endpoint")
                return changes, will_update

            # Fetch current template state
            # Note: The endpoint stores templateId, but we need to check the template definition
            # or the template name we expect to use.
            template = get_runpod_template_by_name(template_name)
            if not template:
                changes.append(f"Template '{template_name}' not found on RunPod (will be created)")
                will_update.append("Create template")
                # If template is missing, we likely need to update endpoint too
                will_update.append("Update endpoint with new template")
                return changes, will_update

            # --- Check Endpoint Configuration ---

            # Workers
            if endpoint.get("workersMin") != self.deployment.workers_min:
                changes.append(
                    f"Min workers changed: {endpoint.get('workersMin')} -> {self.deployment.workers_min}"
                )
                will_update.append("Min workers")

            if endpoint.get("workersMax") != self.deployment.workers_max:
                changes.append(
                    f"Max workers changed: {endpoint.get('workersMax')} -> {self.deployment.workers_max}"
                )
                will_update.append("Max workers")

            # Idle timeout
            if endpoint.get("idleTimeout") != self.deployment.idle_timeout:
                changes.append(
                    f"Idle timeout changed: {endpoint.get('idleTimeout')} -> {self.deployment.idle_timeout}"
                )
                will_update.append("Idle timeout")

            # GPU configuration
            # self.deployment.gpu_types is a list of strings
            # endpoint['gpuIds'] is typically a string (e.g. "AMPERE_16") or comma-separated string
            desired_gpu_ids = ",".join(self.deployment.gpu_types) if self.deployment.gpu_types else "AMPERE_24"
            current_gpu_ids = endpoint.get("gpuIds", "")

            # Normalize for comparison
            if current_gpu_ids != desired_gpu_ids:
                changes.append(f"GPU types changed: {current_gpu_ids} -> {desired_gpu_ids}")
                will_update.append("GPU configuration")

            # GPU count (if applicable)
            desired_gpu_count = self.deployment.gpu_count or 1
            current_gpu_count = endpoint.get("gpuCount", 1)
            if current_gpu_count != desired_gpu_count:
                changes.append(f"GPU count changed: {current_gpu_count} -> {desired_gpu_count}")
                will_update.append("GPU count")

            # Flashboot
            # Endpoint might not return flashboot if false, or as boolean
            current_flashboot = endpoint.get("flashboot", False)
            if current_flashboot != self.deployment.flashboot:
                changes.append(f"Flashboot changed: {current_flashboot} -> {self.deployment.flashboot}")
                will_update.append("Flashboot setting")

            # --- Check Template Configuration (Image & Env) ---

            # Image
            # Template returns imageName, e.g. "repo/image:tag"
            desired_image = self.deployment.image.full_name
            current_image = template.get("imageName", "")

            if current_image != desired_image:
                changes.append(f"Docker image changed: {current_image} -> {desired_image}")
                will_update.append("Update template image")
                will_update.append("Redeploy endpoint")

            # Environment Variables
            # self.deployment.environment is a dict
            # template['env'] is likely a list of dicts [{'key': 'K', 'value': 'V'}] or a dict
            # Based on runpod_api.py usage, it seems to handle it as dict in some places,
            # but API responses are often lists. We'll handle both.
            desired_env = self.deployment.environment or {}

            # Add PORT/PORT_HEALTH which are auto-added
            desired_env_check = desired_env.copy()
            desired_env_check["PORT"] = "8000"
            desired_env_check["PORT_HEALTH"] = "8000"

            current_env_raw = template.get("env", [])
            current_env = {}

            if isinstance(current_env_raw, list):
                for item in current_env_raw:
                    if "key" in item and "value" in item:
                        current_env[item["key"]] = item["value"]
            elif isinstance(current_env_raw, dict):
                current_env = current_env_raw

            # Compare env vars
            env_changes = []
            for key, value in desired_env_check.items():
                if str(current_env.get(key)) != str(value):
                    env_changes.append(key)

            # Check for removed env vars (ignoring system ones if possible, but strict check is safer)
            for key in current_env:
                if key not in desired_env_check:
                    # RunPod might add system env vars, so maybe we only care about missing/changed user vars
                    # But for now, let's just check if user vars are present and correct
                    pass

            if env_changes:
                changes.append(f"Environment variables changed: {', '.join(env_changes)}")
                will_update.append("Update template environment")
                will_update.append("Redeploy endpoint")

        except Exception as e:
            logger.warning(f"Failed to check RunPod state: {e}")
            changes.append("Could not verify current RunPod state (API error)")
            will_update.append("Force update (precautionary)")

        if not changes:
            changes.append("No configuration changes detected")

        return changes, will_update

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
