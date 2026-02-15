"""
Docker run command generation for self-hosted deployments.

This module generates docker run commands from deployment configuration,
supporting GPU assignments, volume mounts, and environment variables.
"""

import hashlib
import json

from nodetool.config.deployment import SelfHostedDeployment

INTERNAL_API_PORT = 7777
APP_ENV_PORT = 8000


class DockerRunGenerator:
    """
    Generates docker run command from deployment settings.

    This class handles the conversion of NodeTool deployment configuration
    into a docker run command suitable for single container deployment.
    """

    def __init__(self, deployment: SelfHostedDeployment, runtime_command: str = "docker"):
        """
        Initialize the docker run generator.

        Args:
            deployment: Self-hosted deployment configuration
        """
        self.deployment = deployment
        self.container = deployment.container
        self.runtime_command = runtime_command

    def generate_command(self) -> str:
        """
        Generate docker run command as a string.

        Returns:
            Docker run command string
        """
        parts = [f"{self.runtime_command} run"]

        # Detached mode
        parts.append("-d")

        # Container name
        container_name = f"nodetool-{self.container.name}"
        parts.append(f"--name {container_name}")

        # Restart policy
        parts.append("--restart unless-stopped")

        # Port mapping
        host_port = self._resolve_host_port()
        parts.append(f"-p {host_port}:{INTERNAL_API_PORT}")

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
            f'--health-cmd="curl -f http://localhost:{INTERNAL_API_PORT}/health || exit 1" '
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
            "port": self._resolve_host_port(),
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

    def _build_volumes(self) -> list[str]:
        """
        Build volume mounts for the container.

        Returns:
            List of volume mount strings
        """
        volumes = []

        # Workspace volume (read-write)
        workspace_path = self.deployment.paths.workspace
        volumes.append(f"{workspace_path}:/workspace")

        # HuggingFace cache volume
        # If persistent_paths is configured, mount as read-write for model downloads
        persistent_paths = self.deployment.persistent_paths
        if persistent_paths:
            # For persistent deployments, hf_cache must be writable to allow model downloads
            volumes.append(f"{self.deployment.paths.hf_cache}:/hf-cache")
        else:
            # Default: read-only for safety when not using persistent storage
            volumes.append(f"{self.deployment.paths.hf_cache}:/hf-cache:ro")

        return volumes

    def _build_environment(self) -> list[str]:
        """
        Build environment variables for the container.

        Returns:
            List of environment variable strings in KEY=value format
        """
        # Start with container environment
        env = dict(self.container.environment) if self.container.environment else {}

        # Add container-specific settings
        env["PORT"] = str(APP_ENV_PORT)
        env["NODETOOL_API_URL"] = f"http://localhost:{self.container.port}"
        env["NODETOOL_SERVER_MODE"] = "private"

        # Configure paths from persistent_paths if available
        persistent_paths = self.deployment.persistent_paths
        if persistent_paths:
            # Use persistent_paths for all storage configuration
            env["USERS_FILE"] = persistent_paths.users_file
            env["DB_PATH"] = persistent_paths.db_path
            env["CHROMA_PATH"] = persistent_paths.chroma_path
            env["HF_HOME"] = persistent_paths.hf_cache
            env["ASSET_BUCKET"] = persistent_paths.asset_bucket
            # Enable multi_user auth when persistent_paths is configured
            env["AUTH_PROVIDER"] = "multi_user"
        else:
            # Fallback to default paths
            env["DB_PATH"] = "/workspace/nodetool.db"
            env["HF_HOME"] = "/hf-cache"
            env.setdefault("AUTH_PROVIDER", "static")

        # Add workflow IDs if specified
        if self.container.workflows:
            env["NODETOOL_WORKFLOWS"] = ",".join(self.container.workflows)

        # Add authentication token for self-hosted deployments.
        if self.deployment.server_auth_token:
            env["SERVER_AUTH_TOKEN"] = self.deployment.server_auth_token

        # Convert to KEY=value format
        return [f"{key}={value}" for key, value in env.items()]

    def _resolve_host_port(self) -> int:
        """Return the host port to expose for this container."""
        host_port = self.container.port or APP_ENV_PORT
        if host_port == INTERNAL_API_PORT:
            return APP_ENV_PORT
        return host_port

    def _build_gpu_args(self) -> list[str]:
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
