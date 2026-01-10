"""
Docker Compose file generation for self-hosted deployments.

This module generates docker-compose.yml files from deployment configuration,
supporting multi-container setups with GPU assignments, volume mounts, and
environment variables.
"""

import hashlib
from typing import Any, Dict, List, Optional

import yaml

from nodetool.config.deployment import (
    ContainerConfig,
    SelfHostedDeployment,
)

# Self-hosted runtime listens on this internal port by default.
INTERNAL_API_PORT = 7777
APP_ENV_PORT = 8000


class ComposeGenerator:
    """
    Generates docker-compose.yml configuration from deployment settings.

    This class handles the conversion of NodeTool deployment configuration
    into a docker-compose.yml file suitable for deployment.
    """

    def __init__(self, deployment: SelfHostedDeployment):
        """
        Initialize the compose generator.

        Args:
            deployment: Self-hosted deployment configuration
        """
        self.deployment = deployment

    def generate(self) -> str:
        """
        Generate docker-compose.yml content as a string.

        Returns:
            YAML string containing docker-compose configuration
        """
        compose_dict = self._build_compose_dict()
        return yaml.dump(
            compose_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def generate_hash(self) -> str:
        """
        Generate a hash of the compose configuration.

        This can be used to detect changes in the configuration.

        Returns:
            SHA256 hash of the compose configuration
        """
        content = self.generate()
        return hashlib.sha256(content.encode()).hexdigest()

    def _build_compose_dict(self) -> dict[str, Any]:
        """
        Build the compose dictionary structure.

        Returns:
            Dictionary representing docker-compose.yml structure
        """
        compose: dict[str, Any] = {
            "version": "3.8",
            "services": {},
        }

        # Generate service for the container
        container = self.deployment.container
        service_name = self._sanitize_service_name(container.name)
        compose["services"][service_name] = self._build_service(container)

        return compose

    def _sanitize_service_name(self, name: str) -> str:
        """
        Sanitize container name for use as docker-compose service name.

        Args:
            name: Original container name

        Returns:
            Sanitized service name
        """
        # Replace invalid characters with hyphens
        sanitized = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        # Ensure it starts with alphanumeric
        if sanitized and not sanitized[0].isalnum():
            sanitized = "c" + sanitized
        return sanitized.lower()

    def _build_service(self, container: ContainerConfig) -> dict[str, Any]:
        """
        Build service definition for a container.

        Args:
            container: Container configuration

        Returns:
            Service definition dictionary
        """
        service: dict[str, Any] = {
            "image": self.deployment.image.full_name,
            "container_name": f"nodetool-{container.name}",
            "ports": [f"{container.port}:{INTERNAL_API_PORT}"],
            "volumes": self._build_volumes(container),
            "environment": self._build_environment(container),
            "restart": "unless-stopped",
            "healthcheck": {
                "test": [
                    "CMD",
                    "curl",
                    "-f",
                    f"http://localhost:{INTERNAL_API_PORT}/health",
                ],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "40s",
            },
        }

        # Add GPU configuration if specified
        if container.gpu:
            service["deploy"] = self._build_deploy_config(container)

        return service

    def _build_volumes(self, container: ContainerConfig) -> list[str]:
        """
        Build volume mounts for a container.

        Args:
            container: Container configuration

        Returns:
            List of volume mount strings
        """
        volumes = []

        # Workspace volume (read-write)
        volumes.append(f"{self.deployment.paths.workspace}:/workspace")

        # HuggingFace cache volume (read-only, shared)
        volumes.append(f"{self.deployment.paths.hf_cache}:/hf-cache:ro")

        return volumes

    def _build_environment(self, container: ContainerConfig) -> list[str]:
        """
        Build environment variables for a container.

        Args:
            container: Container configuration

        Returns:
            List of environment variable strings in KEY=value format
        """
        # Start with container environment (already includes defaults via merge)
        env = dict(container.environment) if container.environment else {}

        # Add container-specific settings
        env["PORT"] = str(APP_ENV_PORT)
        env["NODETOOL_API_URL"] = f"http://localhost:{container.port}"

        # Add workflow IDs if specified
        if container.workflows:
            env["NODETOOL_WORKFLOWS"] = ",".join(container.workflows)

        # Convert to KEY=value format
        return [f"{key}={value}" for key, value in env.items()]

    def _build_deploy_config(self, container: ContainerConfig) -> dict[str, Any]:
        """
        Build deployment configuration for GPU resources.

        Args:
            container: Container configuration

        Returns:
            Deploy configuration dictionary
        """
        if not container.gpu:
            return {}

        # Parse GPU specification
        # Can be single GPU ("0") or multiple GPUs ("0,1")
        gpu_ids = [g.strip() for g in container.gpu.split(",")]

        return {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "device_ids": gpu_ids,
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        }


def generate_compose_file(deployment: SelfHostedDeployment, output_path: str | None = None) -> str:
    """
    Generate docker-compose.yml file from deployment configuration.

    Args:
        deployment: Self-hosted deployment configuration
        output_path: Optional path to write the compose file to

    Returns:
        Generated docker-compose.yml content as string
    """
    generator = ComposeGenerator(deployment)
    content = generator.generate()

    if output_path:
        with open(output_path, "w") as f:
            f.write(content)

    return content


def get_compose_hash(deployment: SelfHostedDeployment) -> str:
    """
    Get hash of the compose configuration for change detection.

    Args:
        deployment: Self-hosted deployment configuration

    Returns:
        SHA256 hash of the compose configuration
    """
    generator = ComposeGenerator(deployment)
    return generator.generate_hash()
