"""
Deployment configuration management for NodeTool.

This module provides a Terraform-like deployment configuration system where all
deployments (self-hosted, RunPod, GCP) are managed through a single deployment.yaml file.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator

from nodetool.config.settings import get_system_file_path


class DeploymentType(str, Enum):
    """Supported deployment types."""

    SELF_HOSTED = "self-hosted"
    RUNPOD = "runpod"
    GCP = "gcp"


class DeploymentStatus(str, Enum):
    """Deployment status values."""

    UNKNOWN = "unknown"
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    ACTIVE = "active"
    SERVING = "serving"
    ERROR = "error"
    STOPPED = "stopped"
    DESTROYED = "destroyed"


# ============================================================================
# Self-Hosted Deployment Models
# ============================================================================


class SSHConfig(BaseModel):
    """SSH connection configuration."""

    user: str = Field(..., description="SSH username")
    key_path: Optional[str] = Field(
        None, description="Path to SSH private key (default: ~/.ssh/id_rsa)"
    )
    password: Optional[str] = Field(
        None, description="SSH password (not recommended, use keys)"
    )
    port: int = Field(22, description="SSH port")

    @field_validator("key_path")
    @classmethod
    def expand_key_path(cls, v: Optional[str]) -> Optional[str]:
        """Expand ~ in key path."""
        if v:
            return str(Path(v).expanduser())
        return v


class ContainerConfig(BaseModel):
    """Docker container configuration."""

    name: str = Field(..., description="Container name")
    port: int = Field(..., description="Port to expose")
    gpu: Optional[str] = Field(
        None, description="GPU device ID(s) (e.g., '0' or '0,1')"
    )
    environment: Optional[Dict[str, str]] = Field(
        None, description="Environment variables for the container"
    )
    workflows: Optional[List[str]] = Field(
        None, description="Workflow IDs to run in this container"
    )


class SelfHostedPaths(BaseModel):
    """Paths on the remote self-hosted server."""

    workspace: str = Field(
        "/data/workspace", description="Container workspace directory"
    )
    hf_cache: str = Field(
        "/data/hf-cache", description="Shared HuggingFace cache directory"
    )


class SelfHostedState(BaseModel):
    """Runtime state for self-hosted deployment."""

    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    container_id: Optional[str] = None
    container_name: Optional[str] = None


class ImageConfig(BaseModel):
    """Docker image configuration."""

    name: str = Field(..., description="Image name (e.g., nodetool/nodetool)")
    tag: str = Field("latest", description="Image tag")
    registry: str = Field("docker.io", description="Docker registry")

    @property
    def full_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.name}:{self.tag}"


class SelfHostedDeployment(BaseModel):
    """Self-hosted deployment configuration for a single container."""

    type: Literal[DeploymentType.SELF_HOSTED] = DeploymentType.SELF_HOSTED
    enabled: bool = Field(True, description="Whether this deployment is enabled")
    host: str = Field(..., description="Remote host address (IP or hostname)")
    ssh: SSHConfig
    paths: SelfHostedPaths = Field(default_factory=SelfHostedPaths)
    image: ImageConfig
    container: ContainerConfig = Field(..., description="Container configuration")
    state: SelfHostedState = Field(default_factory=SelfHostedState)


# ============================================================================
# RunPod Deployment Models
# ============================================================================


class RunPodBuildConfig(BaseModel):
    """Docker build configuration for RunPod."""

    platform: str = Field("linux/amd64", description="Docker build platform")
    no_cache: bool = Field(False, description="Disable build cache")


class RunPodImageConfig(BaseModel):
    """Docker image configuration for RunPod."""

    name: str = Field(..., description="Image name")
    tag: str = Field(..., description="Image tag")
    registry: str = Field("docker.io", description="Docker registry")
    build: RunPodBuildConfig = Field(default_factory=RunPodBuildConfig)

    @property
    def full_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.name}:{self.tag}"


class RunPodTemplateConfig(BaseModel):
    """RunPod template configuration."""

    name: str = Field(..., description="Template name")
    gpu_types: List[str] = Field(default_factory=list, description="Allowed GPU types")
    data_centers: List[str] = Field(
        default_factory=list, description="Preferred data center locations"
    )
    network_volume_id: Optional[str] = Field(
        None, description="Network volume ID to attach"
    )
    allowed_cuda_versions: List[str] = Field(
        default_factory=list, description="Allowed CUDA versions"
    )


class RunPodEndpointConfig(BaseModel):
    """RunPod endpoint configuration."""

    name: str = Field(..., description="Endpoint name")
    workers_min: int = Field(0, description="Minimum number of workers")
    workers_max: int = Field(3, description="Maximum number of workers")
    idle_timeout: int = Field(
        60, description="Seconds before scaling down idle workers"
    )
    execution_timeout: Optional[int] = Field(
        None, description="Maximum execution time in milliseconds"
    )
    flashboot: bool = Field(False, description="Enable flashboot for faster startup")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs per worker")


class RunPodState(BaseModel):
    """Runtime state for RunPod deployment."""

    template_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    last_build_hash: Optional[str] = Field(None, description="Hash of last built image")


class RunPodDeployment(BaseModel):
    """RunPod serverless deployment configuration."""

    type: Literal[DeploymentType.RUNPOD] = DeploymentType.RUNPOD
    enabled: bool = Field(True, description="Whether this deployment is enabled")
    image: RunPodImageConfig
    template: RunPodTemplateConfig
    endpoint: RunPodEndpointConfig
    workflows: List[str] = Field(
        default_factory=list, description="Workflow IDs to deploy"
    )
    state: RunPodState = Field(default_factory=RunPodState)


# ============================================================================
# GCP Deployment Models
# ============================================================================


class GCPBuildConfig(BaseModel):
    """Docker build configuration for GCP."""

    platform: str = Field("linux/amd64", description="Docker build platform")


class GCPImageConfig(BaseModel):
    """Docker image configuration for GCP Cloud Run."""

    registry: str = Field(
        "us-docker.pkg.dev", description="Container registry (e.g., us-docker.pkg.dev)"
    )
    repository: str = Field(
        ..., description="Full repository path (e.g., project/repo/image)"
    )
    tag: str = Field(..., description="Image tag")
    build: GCPBuildConfig = Field(default_factory=GCPBuildConfig)

    @property
    def full_name(self) -> str:
        """Get full image name with registry and tag."""
        return f"{self.registry}/{self.repository}:{self.tag}"


class GCPResourceConfig(BaseModel):
    """Cloud Run resource configuration."""

    cpu: str = Field("4", description="CPU allocation (1, 2, 4, 6, 8)")
    memory: str = Field("16Gi", description="Memory allocation (e.g., 16Gi)")
    min_instances: int = Field(0, description="Minimum number of instances")
    max_instances: int = Field(3, description="Maximum number of instances")
    concurrency: int = Field(80, description="Maximum concurrent requests per instance")
    timeout: int = Field(3600, description="Request timeout in seconds")


class GCPStorageConfig(BaseModel):
    """Cloud Run storage configuration."""

    gcs_bucket: Optional[str] = Field(None, description="GCS bucket name")
    gcs_mount_path: str = Field(
        "/mnt/gcs", description="Container path to mount GCS bucket"
    )


class GCPIAMConfig(BaseModel):
    """Cloud Run IAM configuration."""

    service_account: Optional[str] = Field(
        None, description="Service account email to run the service"
    )
    allow_unauthenticated: bool = Field(
        False, description="Allow unauthenticated access"
    )


class GCPState(BaseModel):
    """Runtime state for GCP Cloud Run deployment."""

    service_url: Optional[str] = None
    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    revision: Optional[str] = Field(None, description="Current Cloud Run revision")


class GCPDeployment(BaseModel):
    """Google Cloud Run deployment configuration."""

    type: Literal[DeploymentType.GCP] = DeploymentType.GCP
    enabled: bool = Field(True, description="Whether this deployment is enabled")
    project_id: str = Field(..., description="GCP project ID")
    region: str = Field("us-central1", description="GCP region")
    service_name: str = Field(..., description="Cloud Run service name")
    image: GCPImageConfig
    resources: GCPResourceConfig = Field(default_factory=GCPResourceConfig)
    storage: Optional[GCPStorageConfig] = None
    iam: GCPIAMConfig = Field(default_factory=GCPIAMConfig)
    workflows: List[str] = Field(
        default_factory=list, description="Workflow IDs to deploy"
    )
    state: GCPState = Field(default_factory=GCPState)


# ============================================================================
# Main Configuration Models
# ============================================================================


class DefaultsConfig(BaseModel):
    """Default environment variables applied to all deployments."""

    chat_provider: str = Field("llama_cpp", description="Default chat provider")
    default_model: str = Field("", description="Default model name")
    log_level: str = Field("INFO", description="Default log level")
    remote_auth: bool = Field(False, description="Enable remote authentication")
    # Can add more defaults as needed
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional default environment variables"
    )


class DeploymentConfig(BaseModel):
    """Main deployment configuration."""

    version: str = Field("1.0", description="Configuration schema version")
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    deployments: Dict[
        str, Union[SelfHostedDeployment, RunPodDeployment, GCPDeployment]
    ] = Field(default_factory=dict, description="Deployment definitions")

    @field_validator("deployments")
    @classmethod
    def validate_deployment_types(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each deployment has a valid type."""
        for name, deployment in v.items():
            if not hasattr(deployment, "type"):
                raise ValueError(f"Deployment '{name}' missing 'type' field")
        return v


# ============================================================================
# Configuration Loading and Saving
# ============================================================================

DEPLOYMENT_CONFIG_FILE = "deployment.yaml"


def get_deployment_config_path() -> Path:
    """Get the path to the deployment configuration file."""
    config_dir = get_system_file_path("config")
    return Path(config_dir) / DEPLOYMENT_CONFIG_FILE


def load_deployment_config() -> DeploymentConfig:
    """
    Load deployment configuration from deployment.yaml.

    Returns:
        DeploymentConfig: The loaded configuration.

    Raises:
        FileNotFoundError: If deployment.yaml doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
        ValidationError: If the configuration is invalid.
    """
    config_path = get_deployment_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Deployment configuration not found at {config_path}. "
            f"Run 'nodetool deploy init' to create it."
        )

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return DeploymentConfig()

    return DeploymentConfig.model_validate(data)


def save_deployment_config(config: DeploymentConfig) -> None:
    """
    Save deployment configuration to deployment.yaml.

    This performs an atomic write by writing to a temporary file first,
    then renaming it to prevent corruption.

    Args:
        config: The deployment configuration to save.
    """
    config_path = get_deployment_config_path()

    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for YAML serialization
    data = config.model_dump(mode="json", exclude_none=True)

    # Write to temporary file first (atomic operation)
    temp_path = config_path.with_suffix(".tmp")

    try:
        with open(temp_path, "w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        # Atomic rename
        temp_path.replace(config_path)

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def init_deployment_config() -> DeploymentConfig:
    """
    Initialize a new deployment configuration file with defaults.

    Returns:
        DeploymentConfig: The newly created configuration.

    Raises:
        FileExistsError: If deployment.yaml already exists.
    """
    config_path = get_deployment_config_path()

    if config_path.exists():
        raise FileExistsError(
            f"Deployment configuration already exists at {config_path}"
        )

    # Create default configuration
    config = DeploymentConfig()

    # Save it
    save_deployment_config(config)

    return config


def merge_defaults_with_env(
    defaults: DefaultsConfig, deployment_env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Merge default environment variables with deployment-specific overrides.

    Args:
        defaults: Default configuration
        deployment_env: Deployment-specific environment variables

    Returns:
        Dictionary of merged environment variables
    """
    env = {}

    # Add defaults
    env["CHAT_PROVIDER"] = defaults.chat_provider
    env["DEFAULT_MODEL"] = defaults.default_model
    env["LOG_LEVEL"] = defaults.log_level
    env["REMOTE_AUTH"] = str(defaults.remote_auth).lower()

    # Add extra defaults
    env.update(defaults.extra)

    # Override with deployment-specific values
    if deployment_env:
        env.update(deployment_env)

    return env
