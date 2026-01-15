"""
AWS App Runner deployment implementation for NodeTool.

This module handles deployment to AWS App Runner, including:
- Docker image building and pushing to ECR
- App Runner service deployment
- Service management
- IAM role setup
"""

import logging
import subprocess
from typing import Any, Dict, Optional

from nodetool.config.deployment import (
    AWSDeployment,
    DeploymentStatus,
)
from nodetool.deploy.deploy_to_aws import (
    delete_aws_service,
    deploy_to_aws,
    list_aws_services,
)
from nodetool.deploy.state import StateManager

logger = logging.getLogger(__name__)


class AWSDeployer:
    """
    Handles deployment to AWS App Runner.

    This class orchestrates the entire AWS deployment process including:
    - Docker image building and pushing
    - App Runner service deployment
    - Service configuration
    - State management
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: AWSDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the AWS deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: AWS deployment configuration
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
            "type": "aws",
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
                    "ECR repository",
                    f"App Runner service: {self.deployment.service_name}",
                ]
            )
        else:
            # Check for configuration changes
            # TODO: Implement more granular change detection
            plan["changes"].append("Configuration may have changed")
            plan["will_update"].append(f"App Runner service: {self.deployment.service_name}")

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply the deployment to AWS App Runner.

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

            results["steps"].append("Starting AWS App Runner deployment...")

            # Prepare environment variables
            env = dict(self.deployment.environment) if self.deployment.environment else {}

            # Call deploy function
            deploy_to_aws(
                deployment=self.deployment,
                env=env,
                skip_build=False,
                skip_push=False,
                skip_deploy=False,
            )

            results["steps"].append("AWS App Runner deployment completed")

            # Update state with success
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.SERVING.value,
                    "service_name": self.deployment.service_name,
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

    def status(self) -> Dict[str, Any]:
        """
        Get current deployment status.

        Returns:
            Dictionary with current status information
        """
        status_info = {
            "deployment_name": self.deployment_name,
            "type": "aws",
            "region": self.deployment.region,
            "service_name": self.deployment.service_name,
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed")

        # Try to get live status from App Runner
        try:
            services = list_aws_services(region=self.deployment.region)

            # Find our service
            for service in services:
                if service.get("ServiceName") == self.deployment.service_name:
                    status_info["live_status"] = service.get("Status")
                    status_info["url"] = f"https://{service.get('ServiceUrl')}" if service.get("ServiceUrl") else None
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
        Get logs from App Runner service.

        Args:
            service: Not used for AWS (kept for interface compatibility)
            follow: Follow log output (not recommended for programmatic use)
            tail: Number of lines to show

        Returns:
            Log output as string
        """
        # Use AWS CLI to fetch logs from CloudWatch
        # App Runner logs go to CloudWatch Logs
        log_group = f"/aws/apprunner/{self.deployment.service_name}"

        cmd = [
            "aws",
            "logs",
            "tail",
            log_group,
            "--region",
            self.deployment.region,
            "--since",
            "1h",
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

    def destroy(self) -> Dict[str, Any]:
        """
        Destroy the deployment (delete App Runner service).

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
            results["steps"].append(f"Deleting App Runner service: {self.deployment.service_name}...")

            success = delete_aws_service(
                service_name=self.deployment.service_name,
                region=self.deployment.region,
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
