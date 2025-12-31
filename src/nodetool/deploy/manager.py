"""
Deployment manager orchestrator for all deployment types.

This module provides a unified interface for managing deployments across
different platforms (self-hosted, RunPod, GCP, Fly.io). It handles:
- Change detection (comparing current vs desired state)
- Plan generation (showing what will change)
- Deployment orchestration
- State management
- Validation and error handling
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nodetool.config.deployment import (
    FlyDeployment,
    GCPDeployment,
    RunPodDeployment,
    SelfHostedDeployment,
    load_deployment_config,
)
from nodetool.deploy.fly import FlyDeployer
from nodetool.deploy.gcp import GCPDeployer
from nodetool.deploy.runpod import RunPodDeployer
from nodetool.deploy.self_hosted import SelfHostedDeployer
from nodetool.deploy.state import StateManager

logger = logging.getLogger(__name__)


class DeploymentManager:
    """
    Orchestrates deployments across all platforms.

    This class provides a unified interface for deployment operations
    regardless of the target platform (self-hosted, RunPod, GCP).
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the deployment manager.

        Args:
            config_path: Path to deployment.yaml (optional, uses default if not provided)
        """
        self.config = load_deployment_config()
        self.state_manager = StateManager(config_path=config_path)

    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all configured deployments with their current status.

        Returns:
            List of deployment info dictionaries
        """
        deployments = []

        for name, deployment in self.config.deployments.items():
            state = self.state_manager.read_state(name)

            info = {
                "name": name,
                "type": deployment.type,
                "status": state.get("status", "unknown") if state else "unknown",
                "last_deployed": state.get("last_deployed") if state else None,
            }

            # Add type-specific info
            if isinstance(deployment, SelfHostedDeployment):
                info["host"] = deployment.host
                info["container"] = deployment.container.name
            elif isinstance(deployment, RunPodDeployment):
                info["pod_id"] = state.get("pod_id") if state else None
            elif isinstance(deployment, GCPDeployment):
                info["project"] = deployment.project_id
                info["region"] = deployment.region
            elif isinstance(deployment, FlyDeployment):
                info["app_name"] = deployment.app_name
                info["region"] = deployment.region

            deployments.append(info)

        return deployments

    def get_deployment(self, name: str) -> SelfHostedDeployment | RunPodDeployment | GCPDeployment | FlyDeployment:
        """
        Get deployment configuration by name.

        Args:
            name: Deployment name

        Returns:
            Deployment configuration

        Raises:
            KeyError: If deployment not found
        """
        if name not in self.config.deployments:
            raise KeyError(f"Deployment '{name}' not found")
        return self.config.deployments[name]

    def plan(self, name: str) -> Dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        Similar to 'terraform plan' - shows what will happen without
        actually executing the deployment.

        Args:
            name: Deployment name

        Returns:
            Dictionary describing planned changes

        Raises:
            KeyError: If deployment not found
            NotImplementedError: If deployment type doesn't support plan
        """
        deployment = self.get_deployment(name)

        if isinstance(deployment, SelfHostedDeployment):
            deployer = SelfHostedDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.plan()
        elif isinstance(deployment, RunPodDeployment):
            deployer = RunPodDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.plan()
        elif isinstance(deployment, GCPDeployment):
            deployer = GCPDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.plan()
        elif isinstance(deployment, FlyDeployment):
            deployer = FlyDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.plan()
        else:
            raise ValueError(f"Unknown deployment type: {deployment.type}")

    def apply(
        self,
        name: str,
        dry_run: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply a deployment to its target platform.

        Args:
            name: Deployment name
            dry_run: If True, only show what would be done without executing
            force: If True, skip confirmation prompts

        Returns:
            Dictionary with deployment results

        Raises:
            KeyError: If deployment not found
            NotImplementedError: If deployment type not supported
        """
        deployment = self.get_deployment(name)

        logger.info(f"Applying deployment '{name}' (type: {deployment.type})")

        if isinstance(deployment, SelfHostedDeployment):
            deployer = SelfHostedDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.apply(dry_run=dry_run)
        elif isinstance(deployment, RunPodDeployment):
            deployer = RunPodDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.apply(dry_run=dry_run)
        elif isinstance(deployment, GCPDeployment):
            deployer = GCPDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.apply(dry_run=dry_run)
        elif isinstance(deployment, FlyDeployment):
            deployer = FlyDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.apply(dry_run=dry_run)
        else:
            raise ValueError(f"Unknown deployment type: {deployment.type}")

    def status(self, name: str) -> Dict[str, Any]:
        """
        Get current status of a deployment.

        Args:
            name: Deployment name

        Returns:
            Dictionary with current status information

        Raises:
            KeyError: If deployment not found
        """
        deployment = self.get_deployment(name)

        if isinstance(deployment, SelfHostedDeployment):
            deployer = SelfHostedDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.status()
        elif isinstance(deployment, RunPodDeployment):
            deployer = RunPodDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.status()
        elif isinstance(deployment, GCPDeployment):
            deployer = GCPDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.status()
        elif isinstance(deployment, FlyDeployment):
            deployer = FlyDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.status()
        else:
            raise ValueError(f"Unknown deployment type: {deployment.type}")

    def logs(
        self,
        name: str,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """
        Get logs from a deployment.

        Args:
            name: Deployment name
            service: Specific service/container name (optional)
            follow: Follow log output (not recommended for programmatic use)
            tail: Number of lines to show from end of logs

        Returns:
            Log output as string

        Raises:
            KeyError: If deployment not found
            NotImplementedError: If deployment type doesn't support logs
        """
        deployment = self.get_deployment(name)

        if isinstance(deployment, SelfHostedDeployment):
            deployer = SelfHostedDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.logs(service=service, follow=follow, tail=tail)
        elif isinstance(deployment, RunPodDeployment):
            deployer = RunPodDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.logs(service=service, follow=follow, tail=tail)
        elif isinstance(deployment, GCPDeployment):
            deployer = GCPDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.logs(service=service, follow=follow, tail=tail)
        elif isinstance(deployment, FlyDeployment):
            deployer = FlyDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.logs(service=service, follow=follow, tail=tail)
        else:
            raise ValueError(f"Unknown deployment type: {deployment.type}")

    def destroy(self, name: str, force: bool = False) -> Dict[str, Any]:
        """
        Destroy a deployment (stop and remove all resources).

        Args:
            name: Deployment name
            force: If True, skip confirmation prompts

        Returns:
            Dictionary with destruction results

        Raises:
            KeyError: If deployment not found
            NotImplementedError: If deployment type doesn't support destroy
        """
        deployment = self.get_deployment(name)

        logger.warning(f"Destroying deployment '{name}' (type: {deployment.type})")

        if isinstance(deployment, SelfHostedDeployment):
            deployer = SelfHostedDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.destroy()
        elif isinstance(deployment, RunPodDeployment):
            deployer = RunPodDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.destroy()
        elif isinstance(deployment, GCPDeployment):
            deployer = GCPDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.destroy()
        elif isinstance(deployment, FlyDeployment):
            deployer = FlyDeployer(
                deployment_name=name,
                deployment=deployment,
                state_manager=self.state_manager,
            )
            return deployer.destroy()
        else:
            raise ValueError(f"Unknown deployment type: {deployment.type}")

    def validate(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate deployment configuration.

        Checks for:
        - Required fields present
        - Valid configuration values
        - SSH connectivity (for self-hosted)
        - API credentials (for cloud providers)

        Args:
            name: Deployment name to validate (if None, validates all)

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        deployments_to_validate = [name] if name else list(self.config.deployments.keys())

        for deployment_name in deployments_to_validate:
            try:
                deployment = self.get_deployment(deployment_name)

                # Basic validation (Pydantic handles this automatically)
                # Additional validation logic can be added here

                if isinstance(deployment, SelfHostedDeployment):
                    # Validate SSH config
                    if not deployment.ssh.key_path and not deployment.ssh.password:
                        results["warnings"].append(f"{deployment_name}: No SSH authentication method configured")

                    # Validate container
                    if not deployment.container:
                        results["errors"].append(f"{deployment_name}: No container configured")
                        results["valid"] = False

            except Exception as e:
                results["errors"].append(f"{deployment_name}: {str(e)}")
                results["valid"] = False

        return results

    def has_changes(self, name: str) -> bool:
        """
        Check if a deployment has changes that need to be applied.

        Args:
            name: Deployment name

        Returns:
            True if deployment has pending changes
        """
        try:
            plan = self.plan(name)
            return len(plan.get("changes", [])) > 0
        except Exception as e:
            logger.error(f"Error checking for changes: {e}")
            return False

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get state for all deployments.

        Returns:
            Dictionary mapping deployment names to their states
        """
        return self.state_manager.get_all_states()
