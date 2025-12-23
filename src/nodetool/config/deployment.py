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
from pydantic import BaseModel, Field, field_validator, model_validator

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
    key_path: Optional[str] = Field(None, description="Path to SSH private key (default: ~/.ssh/id_rsa)")
    password: Optional[str] = Field(None, description="SSH password (not recommended, use keys)")
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
    gpu: Optional[str] = Field(None, description="GPU device ID(s) (e.g., '0' or '0,1')")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables for the container")
    workflows: Optional[List[str]] = Field(None, description="Workflow IDs to run in this container")


class SelfHostedPaths(BaseModel):
    """Paths on the remote self-hosted server."""

    workspace: str = "/data/workspace"
    hf_cache: str = "/data/hf-cache"


class SelfHostedState(BaseModel):
    """Runtime state for self-hosted deployment."""

    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    container_id: Optional[str] = None
    container_name: Optional[str] = None
    url: Optional[str] = None
    proxy_run_hash: Optional[str] = None
    proxy_bearer_token: Optional[str] = None


class ImageConfig(BaseModel):
    """Docker image configuration."""

    name: str = Field(..., description="Image name (e.g., nodetool/nodetool)")
    tag: str = Field("latest", description="Image tag")
    registry: str = Field("docker.io", description="Docker registry")

    @property
    def full_name(self) -> str:
        """Get full image name with tag."""
        return f"{self.name}:{self.tag}"


class ServiceSpec(BaseModel):
    """Service definition managed by the self-hosted proxy."""

    name: str = Field(..., description="Unique service identifier")
    path: str = Field(..., description="Path prefix to proxy (e.g., /app)")
    image: str = Field(..., description="Docker image for the service")
    auth_token: Optional[str] = Field(default=None, description="Bearer token for upstream service authentication")
    environment: Optional[Dict[str, str]] = Field(default=None, description="Environment variables for the service")
    volumes: Optional[Dict[str, str | Dict[str, str]]] = Field(
        default=None,
        description="Volume mounts (host -> container or detailed dict with bind/mode)",
    )
    mem_limit: Optional[str] = Field(default=None, description="Memory limit (e.g., 512m, 1g)")
    cpus: Optional[float] = Field(default=None, description="CPU quota in cores (e.g., 0.5, 1.0)")


class ProxySpec(BaseModel):
    """Proxy container specification for self-hosted deployments."""

    image: str = Field(..., description="Proxy image (e.g., nodetool/proxy:1.0.0)")
    listen_http: int = Field(80, ge=1, le=65535, description="HTTP port for ACME and health checks")
    listen_https: int = Field(443, ge=1, le=65535, description="HTTPS port for proxied traffic")
    domain: str = Field(..., description="Public domain served by the proxy")
    email: str = Field(..., description="Email for ACME/Let's Encrypt registration")
    tls_certfile: Optional[str] = Field(default=None, description="Path to TLS certificate (inside container)")
    tls_keyfile: Optional[str] = Field(default=None, description="Path to TLS private key (inside container)")
    local_tls_certfile: Optional[str] = Field(
        default=None,
        description="Local path to TLS certificate copied to remote host before deployment",
    )
    local_tls_keyfile: Optional[str] = Field(
        default=None,
        description="Local path to TLS private key copied to remote host before deployment",
    )
    acme_webroot: str = Field(
        "/var/www/acme",
        description="Webroot directory for HTTP-01 challenges (inside container)",
    )
    docker_network: str = Field("nodetool-net", description="Docker network shared between proxy and services")
    connect_mode: Literal["docker_dns", "host_port"] = Field(
        "docker_dns",
        description="How the proxy connects to services",
    )
    http_redirect_to_https: bool = Field(True, description="Redirect HTTP traffic to HTTPS (except ACME)")
    bearer_token: Optional[str] = Field(
        default=None,
        description="Bearer token for proxy authentication (auto-generated if omitted)",
    )
    idle_timeout: int = Field(
        300,
        ge=30,
        description="Seconds before idle services are stopped by the proxy",
    )
    log_level: str = Field("INFO", description="Log level for proxy process")
    services: List[ServiceSpec] = Field(..., description="List of services managed by the proxy")
    auto_certbot: bool = Field(
        False,
        description="When true, run certbot on the remote host to obtain/renew TLS certificates",
    )


class SelfHostedDeployment(BaseModel):
    """Self-hosted deployment configuration for a single container."""

    type: Literal[DeploymentType.SELF_HOSTED] = DeploymentType.SELF_HOSTED
    enabled: bool = Field(True, description="Whether this deployment is enabled")
    host: str = Field(..., description="Remote host address (IP or hostname)")
    ssh: SSHConfig
    paths: SelfHostedPaths = SelfHostedPaths()
    image: ImageConfig
    container: ContainerConfig = Field(..., description="Container configuration")
    worker_auth_token: Optional[str] = Field(
        None,
        description="Authentication token for worker API (auto-generated if not set)",
    )
    proxy: Optional[ProxySpec] = Field(default=None, description="Proxy container specification")
    state: SelfHostedState = Field(default_factory=SelfHostedState)

    @model_validator(mode="after")
    def _ensure_proxy(self) -> "SelfHostedDeployment":
        """Provide a minimal proxy specification when omitted for backward compatibility."""
        if self.proxy is None:
            default_service = ServiceSpec(
                name=self.container.name,
                path="/",
                image="nodetool/nodetool:latest",
            )
            self.proxy = ProxySpec(
                image="nodetool/proxy:latest",
                domain=self.host,
                email="admin@example.com",
                services=[default_service],
            )
        return self

    def get_server_url(self) -> str:
        """Get the server URL for this deployment."""
        if self.proxy:
            has_tls = bool(self.proxy.tls_certfile and self.proxy.tls_keyfile)
            scheme = "https" if has_tls else "http"
            port = self.proxy.listen_https if has_tls else self.proxy.listen_http
            host = self.proxy.domain or self.host
            if (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
                return f"{scheme}://{host}"
            return f"{scheme}://{host}:{port}"
        return f"http://{self.host}:{self.container.port}"


# ============================================================================
# RunPod Deployment Models
# ============================================================================


class RunPodBuildConfig(BaseModel):
    """Docker build configuration for RunPod."""

    platform: str = "linux/amd64"
    no_cache: bool = False


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
    data_centers: List[str] = Field(default_factory=list, description="Preferred data center locations")
    network_volume_id: Optional[str] = Field(None, description="Network volume ID to attach")
    allowed_cuda_versions: List[str] = Field(default_factory=list, description="Allowed CUDA versions")


class RunPodEndpointConfig(BaseModel):
    """RunPod endpoint configuration."""

    name: str = Field(..., description="Endpoint name")
    workers_min: int = Field(0, description="Minimum number of workers")
    workers_max: int = Field(3, description="Maximum number of workers")
    idle_timeout: int = Field(60, description="Seconds before scaling down idle workers")
    execution_timeout: Optional[int] = Field(None, description="Maximum execution time in milliseconds")
    flashboot: bool = Field(False, description="Enable flashboot for faster startup")
    gpu_count: Optional[int] = Field(None, description="Number of GPUs per worker")


class RunPodState(BaseModel):
    """Runtime state for RunPod deployment."""

    template_id: Optional[str] = None
    endpoint_id: Optional[str] = None
    endpoint_url: Optional[str] = None
    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    last_build_hash: Optional[str] = None


class RunPodDockerConfig(BaseModel):
    """Docker configuration for RunPod."""

    username: Optional[str] = None
    registry: str = "docker.io"


class RunPodDeployment(BaseModel):
    """RunPod serverless deployment configuration."""

    type: Literal[DeploymentType.RUNPOD] = DeploymentType.RUNPOD
    enabled: bool = Field(True, description="Whether this deployment is enabled")
    image: RunPodImageConfig
    gpu_types: List[str] = Field(default_factory=list, description="Allowed GPU types")
    gpu_count: Optional[int] = None
    cpu_flavors: List[str] = Field(default_factory=list, description="Allowed CPU flavors")
    vcpu_count: Optional[int] = None
    data_centers: List[str] = Field(default_factory=list, description="Preferred data center locations")
    network_volume_id: Optional[str] = Field(None, description="Network volume ID to attach")
    allowed_cuda_versions: List[str] = Field(default_factory=list, description="Allowed CUDA versions")
    docker: RunPodDockerConfig = RunPodDockerConfig()
    platform: str = "linux/amd64"
    template_name: Optional[str] = None
    compute_type: str = "GPU"
    workers_min: int = 0
    workers_max: int = 3
    idle_timeout: int = 5
    execution_timeout: Optional[int] = None
    flashboot: bool = False
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables for the deployment")
    workflows: List[str] = Field(default_factory=list, description="Workflow IDs to deploy")
    state: RunPodState = Field(default=RunPodState())

    def get_server_url(self) -> Optional[str]:
        """Get the server URL for this deployment."""
        return self.state.endpoint_url


# ============================================================================
# GCP Deployment Models
# ============================================================================


class GCPBuildConfig(BaseModel):
    """Docker build configuration for GCP."""

    platform: str = "linux/amd64"


class GCPImageConfig(BaseModel):
    """Docker image configuration for GCP Cloud Run."""

    registry: str = Field("us-docker.pkg.dev", description="Container registry (e.g., us-docker.pkg.dev)")
    repository: str = Field(..., description="Full repository path (e.g., project/repo/image)")
    tag: str = Field(..., description="Image tag")
    build: GCPBuildConfig = Field(default_factory=GCPBuildConfig)

    @property
    def full_name(self) -> str:
        """Get full image name with registry and tag."""
        return f"{self.registry}/{self.repository}:{self.tag}"


class GCPResourceConfig(BaseModel):
    """Cloud Run resource configuration."""

    cpu: str = "4"
    memory: str = "16Gi"
    min_instances: int = 0
    max_instances: int = 3
    concurrency: int = 80
    timeout: int = 3600


class GCPStorageConfig(BaseModel):
    """Cloud Run storage configuration."""

    gcs_bucket: Optional[str] = Field(None, description="GCS bucket name")
    gcs_mount_path: str = Field("/mnt/gcs", description="Container path to mount GCS bucket")


class GCPIAMConfig(BaseModel):
    """Cloud Run IAM configuration."""

    service_account: Optional[str] = None
    allow_unauthenticated: bool = False


class GCPState(BaseModel):
    """Runtime state for GCP Cloud Run deployment."""

    service_url: Optional[str] = None
    last_deployed: Optional[datetime] = None
    status: DeploymentStatus = DeploymentStatus.UNKNOWN
    revision: Optional[str] = None


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
    workflows: List[str] = Field(default_factory=list, description="Workflow IDs to deploy")
    state: GCPState = Field(default_factory=GCPState)

    def get_server_url(self) -> Optional[str]:
        """Get the server URL for this deployment."""
        return self.state.service_url


# ============================================================================
# Main Configuration Models
# ============================================================================


class DefaultsConfig(BaseModel):
    """Default environment variables applied to all deployments."""

    chat_provider: str = "llama_cpp"
    default_model: str = ""
    log_level: str = "INFO"
    auth_provider: str = "local"  # none, local, static, supabase
    # Can add more defaults as needed
    extra: Dict[str, Any] = {}


class DeploymentConfig(BaseModel):
    """Main deployment configuration."""

    version: str = "1.0"
    defaults: DefaultsConfig = DefaultsConfig()
    deployments: Dict[str, SelfHostedDeployment | RunPodDeployment | GCPDeployment] = {}

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
    return get_system_file_path(DEPLOYMENT_CONFIG_FILE)


def load_deployment_config() -> DeploymentConfig:
    """
    Load deployment configuration from deployment.yaml.

    Automatically generates and saves worker_auth_token for self-hosted
    deployments that don't have one.

    Returns:
        DeploymentConfig: The loaded configuration.

    Raises:
        FileNotFoundError: If deployment.yaml doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
        ValidationError: If the configuration is invalid.
    """
    import secrets

    config_path = get_deployment_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Deployment configuration not found at {config_path}. Run 'nodetool deploy init' to create it."
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return DeploymentConfig()

    config = DeploymentConfig.model_validate(data)

    # Auto-generate worker_auth_token for self-hosted deployments that don't have one
    config_updated = False
    for _name, deployment in config.deployments.items():
        if isinstance(deployment, SelfHostedDeployment):
            if not deployment.worker_auth_token:
                deployment.worker_auth_token = secrets.token_urlsafe(32)
                config_updated = True

    # Save the config if we generated any tokens
    if config_updated:
        save_deployment_config(config)

    return config


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
        raise FileExistsError(f"Deployment configuration already exists at {config_path}")

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
    env["AUTH_PROVIDER"] = defaults.auth_provider

    # Add extra defaults
    env.update(defaults.extra)

    # Override with deployment-specific values
    if deployment_env:
        env.update(deployment_env)

    return env
