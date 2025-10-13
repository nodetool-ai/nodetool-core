"""
Google Cloud Run deployment implementation for NodeTool.

This module handles deployment to Google Cloud Run, including:
- Docker image building and pushing to GCR/Artifact Registry
- Cloud Run service deployment
- Service management
- IAM permission setup
"""

from typing import Dict, Any, Optional
import logging

from nodetool.config.deployment import (
    GCPDeployment,
    DeploymentStatus,
)
from nodetool.deploy.state import StateManager
from nodetool.deploy.deploy_to_gcp import (
    deploy_to_gcp as legacy_deploy_to_gcp,
    delete_gcp_service,
    list_gcp_services,
)

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

    def plan(self) -> Dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        Returns:
            Dictionary describing planned changes
        """
        plan = {
            "deployment_name": self.deployment_name,
            "type": "gcp",
            "project": self.deployment.project,
            "region": self.deployment.region,
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
                    f"Cloud Run service: {self.deployment.service_name}",
                ]
            )
        else:
            # Check for configuration changes
            # TODO: Implement more granular change detection
            plan["changes"].append("Configuration may have changed")
            plan["will_update"].append(
                f"Cloud Run service: {self.deployment.service_name}"
            )

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
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
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DEPLOYING.value
            )

            results["steps"].append("Starting Google Cloud Run deployment...")

            # Prepare environment variables
            env = (
                dict(self.deployment.environment) if self.deployment.environment else {}
            )

            # Call legacy deploy function
            # TODO: Refactor this to use direct API calls instead of legacy function
            legacy_deploy_to_gcp(
                service_name=self.deployment.service_name,
                project_id=self.deployment.project,
                region=self.deployment.region,
                registry=self.deployment.registry,
                cpu=self.deployment.cpu,
                memory=self.deployment.memory,
                gpu_type=self.deployment.gpu_type,
                gpu_count=self.deployment.gpu_count,
                min_instances=self.deployment.min_instances,
                max_instances=self.deployment.max_instances,
                concurrency=self.deployment.concurrency,
                timeout=self.deployment.timeout,
                allow_unauthenticated=self.deployment.allow_unauthenticated,
                env=env,
                docker_username=self.deployment.docker.username
                if self.deployment.docker
                else None,
                docker_registry=self.deployment.docker.registry
                if self.deployment.docker
                else "docker.io",
                image_name=self.deployment.image.name,
                tag=self.deployment.image.tag,
                platform="linux/amd64",  # Cloud Run requires amd64
                skip_build=False,
                skip_push=False,
                skip_deploy=False,
                skip_permission_setup=False,
                service_account=self.deployment.service_account,
                gcs_bucket=self.deployment.gcs_bucket,
                gcs_mount_path=self.deployment.gcs_mount_path,
            )

            results["steps"].append("Google Cloud Run deployment completed")

            # Update state with success
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.SERVING.value,
                    "service_name": self.deployment.service_name,
                    "project": self.deployment.project,
                    "region": self.deployment.region,
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
            "type": "gcp",
            "project": self.deployment.project,
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
            services = list_gcp_services(
                region=self.deployment.region, project_id=self.deployment.project
            )

            # Find our service
            for service in services:
                if (
                    service.get("metadata", {}).get("name")
                    == self.deployment.service_name
                ):
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
            self.deployment.project,
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
            raise RuntimeError(f"Failed to fetch logs: {e.stderr}")

    def destroy(self) -> Dict[str, Any]:
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
            results["steps"].append(
                f"Deleting Cloud Run service: {self.deployment.service_name}..."
            )

            success = delete_gcp_service(
                service_name=self.deployment.service_name,
                region=self.deployment.region,
                project_id=self.deployment.project,
            )

            if success:
                results["steps"].append("Service deleted successfully")
            else:
                results["errors"].append("Failed to delete service")
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
