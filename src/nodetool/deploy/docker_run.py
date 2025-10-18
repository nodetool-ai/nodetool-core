"""
Docker run command generation for self-hosted deployments.

This module generates docker run commands from deployment configuration,
supporting GPU assignments, volume mounts, and environment variables.
"""

import hashlib
from typing import List
import json

from nodetool.config.deployment import (
    SelfHostedDeployment,
)


class DockerRunGenerator:
    """
    Generates docker run command from deployment settings.

    This class handles the conversion of NodeTool deployment configuration
    into a docker run command suitable for single container deployment.
    """

    def __init__(self, deployment: SelfHostedDeployment):
        """
        Initialize the docker run generator.

        Args:
            deployment: Self-hosted deployment configuration
        """
        self.deployment = deployment
        self.container = deployment.container

    def generate_command(self) -> str:
        """
        Generate docker run command as a string.

        Returns:
            Docker run command string
        """
        parts = ["docker run"]

        # Detached mode
        parts.append("-d")

        # Container name
        container_name = f"nodetool-{self.container.name}"
        parts.append(f"--name {container_name}")

        # Restart policy
        parts.append("--restart unless-stopped")

        # Port mapping
        parts.append(f"-p {self.container.port}:8000")

        # Volume mounts
        for volume in self._build_volumes():
            parts.append(f"-v {volume}")

        # Environment variables
        for env in self._build_environment():
            parts.append(f"-e {env}")

        # GPU configuration
        if self.container.gpu:
            gpu_args = self._build_gpu_args()
            parts.extend(gpu_args)

        # Health check
        healthcheck = (
            '--health-cmd="curl -f http://localhost:8000/health || exit 1" '
            "--health-interval=30s "
            "--health-timeout=10s "
            "--health-retries=3 "
            "--health-start-period=40s"
        )
        parts.append(healthcheck)

        # Image name
        parts.append(self.deployment.image.full_name)

        return " \\\n  ".join(parts)

    def generate_hash(self) -> str:
        """
        Generate a hash of the docker run configuration.

        This can be used to detect changes in the configuration.

        Returns:
            SHA256 hash of the docker run configuration
        """
        # Create a deterministic representation of the configuration
        config_dict = {
            "image": self.deployment.image.full_name,
            "container_name": self.container.name,
            "port": self.container.port,
            "volumes": self._build_volumes(),
            "environment": sorted(self._build_environment()),
            "gpu": self.container.gpu,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_container_name(self) -> str:
        """
        Get the full container name.

        Returns:
            Container name with 'nodetool-' prefix
        """
        return f"nodetool-{self.container.name}"

    def _build_volumes(self) -> List[str]:
        """
        Build volume mounts for the container.

        Returns:
            List of volume mount strings
        """
        volumes = []

        # Workspace volume (read-write)
        workspace_path = self.deployment.paths.workspace
        volumes.append(f"{workspace_path}:/workspace")

        # HuggingFace cache volume (read-only)
        volumes.append(f"{self.deployment.paths.hf_cache}:/hf-cache:ro")

        return volumes

    def _build_environment(self) -> List[str]:
        """
        Build environment variables for the container.

        Returns:
            List of environment variable strings in KEY=value format
        """
        # Start with container environment
        env = dict(self.container.environment) if self.container.environment else {}

        # Add container-specific settings
        env["PORT"] = "8000"
        env["NODETOOL_API_URL"] = f"http://localhost:{self.container.port}"

        # Set database path to workspace (mounted volume)
        env["DB_PATH"] = "/workspace/nodetool.db"

        # Set HuggingFace cache to mounted volume
        env["HF_HOME"] = "/hf-cache"

        # Add workflow IDs if specified
        if self.container.workflows:
            env["NODETOOL_WORKFLOWS"] = ",".join(self.container.workflows)

        # Add worker authentication token (for self-hosted deployments)
        if self.deployment.worker_auth_token:
            env["WORKER_AUTH_TOKEN"] = self.deployment.worker_auth_token

        # Convert to KEY=value format
        return [f"{key}={value}" for key, value in env.items()]

    def _build_gpu_args(self) -> List[str]:
        """
        Build GPU arguments for Docker.

        Returns:
            List of GPU-related docker arguments
        """
        if not self.container.gpu:
            return []

        # Parse GPU specification
        # Can be single GPU ("0") or multiple GPUs ("0,1")
        gpu_ids = self.container.gpu.strip()

        # Use --gpus flag with device specification
        # Format: --gpus '"device=0,1"' for multiple GPUs
        # Format: --gpus '"device=0"' for single GPU
        return [f"--gpus '\"device={gpu_ids}\"'"]


def generate_docker_run_command(deployment: SelfHostedDeployment) -> str:
    """
    Generate docker run command from deployment configuration.

    Args:
        deployment: Self-hosted deployment configuration

    Returns:
        Generated docker run command as string
    """
    generator = DockerRunGenerator(deployment)
    return generator.generate_command()


def get_docker_run_hash(deployment: SelfHostedDeployment) -> str:
    """
    Get hash of the docker run configuration for change detection.

    Args:
        deployment: Self-hosted deployment configuration

    Returns:
        SHA256 hash of the docker run configuration
    """
    generator = DockerRunGenerator(deployment)
    return generator.generate_hash()


def get_container_name(deployment: SelfHostedDeployment) -> str:
    """
    Get the container name for the deployment.

    Args:
        deployment: Self-hosted deployment configuration

    Returns:
        Container name
    """
    generator = DockerRunGenerator(deployment)
    return generator.get_container_name()
