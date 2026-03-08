"""
Google Cloud Run deployment implementation for NodeTool.

This module handles deployment to Google Cloud Run, including:
- Docker image building and pushing to GCR/Artifact Registry
- Cloud Run service deployment
- Service management
- IAM permission setup
"""

import logging
from typing import Any, Optional

from nodetool.config.deployment import (
    DeploymentStatus,
    GCPDeployment,
)
from nodetool.deploy.deploy_to_gcp import (
    delete_gcp_service,
    deploy_to_gcp,
    get_gcp_default_env,
    list_gcp_services,
)
from nodetool.deploy.google_cloud_run_api import get_cloud_run_service
from nodetool.deploy.state import StateManager

logger = logging.getLogger(__name__)


class GCPDeployer:
    """
    Handles deployment to Google Cloud Run.

    This class orchestrates the entire GCP deployment process including:
    - Docker image building and pushing
    - Cloud Run service deployment
    - Service configuration
    - State management
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: GCPDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the GCP deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: GCP deployment configuration
            state_manager: State manager instance (optional)
        """
        self.deployment_name = deployment_name
        self.deployment = deployment
        self.state_manager = state_manager or StateManager()

    def _detect_changes(self, current_service: dict[str, Any]) -> list[str]:
        """
        Detect changes between local configuration and remote service.

        Args:
            current_service: Remote service configuration (from gcloud describe)

        Returns:
            List of detected changes
        """
        changes = []
        spec = current_service.get("spec", {}).get("template", {}).get("spec", {})
        metadata = current_service.get("spec", {}).get("template", {}).get("metadata", {})
        annotations = metadata.get("annotations", {})
        containers = spec.get("containers", [{}])
        container = containers[0] if containers else {}
        resources = container.get("resources", {}).get("limits", {})

        # 1. Resources (CPU, Memory)
        remote_cpu = resources.get("cpu", "")
        # Normalize remote CPU (1000m -> 1)
        if remote_cpu.endswith("m"):
            remote_cpu_val = int(remote_cpu[:-1]) / 1000
            remote_cpu_str = str(int(remote_cpu_val)) if remote_cpu_val.is_integer() else str(remote_cpu_val)
        else:
            remote_cpu_str = remote_cpu

        if str(self.deployment.resources.cpu) != remote_cpu_str:
            changes.append(f"CPU: {remote_cpu_str} -> {self.deployment.resources.cpu}")

        remote_memory = resources.get("memory", "")
        if self.deployment.resources.memory != remote_memory:
            changes.append(f"Memory: {remote_memory} -> {self.deployment.resources.memory}")

        # 2. GPU
        remote_gpu_count = annotations.get("run.googleapis.com/gpu", "0")
        expected_gpu_count = self.deployment.resources.gpu_count or (1 if self.deployment.resources.gpu_type else 0)
        if str(expected_gpu_count) != str(remote_gpu_count):
            changes.append(f"GPU Count: {remote_gpu_count} -> {expected_gpu_count}")

        # 3. Scaling
        remote_min = annotations.get("autoscaling.knative.dev/minScale", "0")
        if str(self.deployment.resources.min_instances) != str(remote_min):
            changes.append(f"Min Instances: {remote_min} -> {self.deployment.resources.min_instances}")

        remote_max = annotations.get("autoscaling.knative.dev/maxScale", "0")
        if str(self.deployment.resources.max_instances) != str(remote_max):
            changes.append(f"Max Instances: {remote_max} -> {self.deployment.resources.max_instances}")

        # 4. Concurrency & Timeout
        remote_concurrency = spec.get("containerConcurrency")
        if remote_concurrency and str(self.deployment.resources.concurrency) != str(remote_concurrency):
            changes.append(f"Concurrency: {remote_concurrency} -> {self.deployment.resources.concurrency}")

        remote_timeout = spec.get("timeoutSeconds")
        if remote_timeout and str(self.deployment.resources.timeout) != str(remote_timeout):
            changes.append(f"Timeout: {remote_timeout}s -> {self.deployment.resources.timeout}s")

        # 5. Service Account
        remote_sa = spec.get("serviceAccountName")
        if self.deployment.iam.service_account and self.deployment.iam.service_account != remote_sa:
            changes.append(f"Service Account: {remote_sa} -> {self.deployment.iam.service_account}")

        # 6. Environment Variables
        remote_env_list = container.get("env", [])
        remote_env = {item["name"]: item.get("value", "") for item in remote_env_list}

        expected_env = get_gcp_default_env(self.deployment)
        env_changes = []
        for k, v in expected_env.items():
            if k not in remote_env:
                env_changes.append(f"added {k}")
            elif remote_env[k] != str(v):
                env_changes.append(f"changed {k}")

        if env_changes:
            changes.append(f"Environment variables: {', '.join(env_changes)}")

        return changes

    def plan(self) -> dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        Returns:
            Dictionary describing planned changes
        """
        plan = {
            "deployment_name": self.deployment_name,
            "type": "gcp",
            "project": self.deployment.project_id,
            "region": self.deployment.region,
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)

        # Check if service exists remotely
        try:
            remote_service = get_cloud_run_service(
                self.deployment.service_name,
                self.deployment.region,
                self.deployment.project_id,
            )
        except Exception:
            remote_service = None

        # Always plan to build/push Docker image (generic action)
        # Note: If we wanted to be smarter, we'd check if image needs building,
        # but that requires deeper inspection. For now, we always list it.
        # However, for clarity in "changes", we only list "Service Config" changes below.
        plan["will_create"].append("Docker image (if changed)")

        if not current_state or not current_state.get("last_deployed") or not remote_service:
            plan["changes"].append("Initial deployment - will create all resources")
            plan["will_create"].append(f"Cloud Run service: {self.deployment.service_name}")
        else:
            # Check for configuration changes
            detected_changes = self._detect_changes(remote_service)
            if detected_changes:
                plan["changes"].extend(detected_changes)
                plan["will_update"].append(f"Cloud Run service: {self.deployment.service_name}")
            else:
                plan["changes"].append("No configuration changes detected")

        return plan

    def apply(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Apply the deployment to Google Cloud Run.

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

            results["steps"].append("Starting Google Cloud Run deployment...")

            # Prepare environment variables
            env: dict[str, str] = {}

            # Call deploy function
            deploy_to_gcp(
                deployment=self.deployment,
                env=env,
                skip_build=False,
                skip_push=False,
                skip_deploy=False,
                skip_permission_setup=False,
            )

            results["steps"].append("Google Cloud Run deployment completed")

            # Update state with success
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.SERVING.value,
                    "service_name": self.deployment.service_name,
                    "project": self.deployment.project_id,
                    "region": self.deployment.region,
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
            "type": "gcp",
            "project": self.deployment.project_id,
            "region": self.deployment.region,
            "service_name": self.deployment.service_name,
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed")

        # Try to get live status from Cloud Run
        try:
            services = list_gcp_services(region=self.deployment.region, project_id=self.deployment.project_id)

            # Find our service
            for service in services:
                if service.get("metadata", {}).get("name") == self.deployment.service_name:
                    status_info["live_status"] = service.get("status", {})
                    status_info["url"] = service.get("status", {}).get("url")
                    break

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
        Get logs from Cloud Run service.

        Args:
            service: Not used for GCP (kept for interface compatibility)
            follow: Follow log output (not recommended for programmatic use)
            tail: Number of lines to show

        Returns:
            Log output as string
        """
        import subprocess

        # Use gcloud to fetch logs
        cmd = [
            "gcloud",
            "logging",
            "read",
            f'resource.type="cloud_run_revision" AND resource.labels.service_name="{self.deployment.service_name}"',
            "--project",
            self.deployment.project_id,
            "--limit",
            str(tail),
            "--format",
            "value(timestamp,textPayload)",
        ]

        if follow:
            cmd.append("--follow")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=None if follow else 30,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch logs: {e.stderr}") from e

    def destroy(self) -> dict[str, Any]:
        """
        Destroy the deployment (delete Cloud Run service).

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
            results["steps"].append(f"Deleting Cloud Run service: {self.deployment.service_name}...")

            success = delete_gcp_service(
                service_name=self.deployment.service_name,
                region=self.deployment.region,
                project_id=self.deployment.project_id,
            )

            if success:
                results["steps"].append("Service deleted successfully")
            else:
                results["errors"].append("Failed to delete service")
                results["status"] = "error"

            # Update state
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DESTROYED.value)

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results
