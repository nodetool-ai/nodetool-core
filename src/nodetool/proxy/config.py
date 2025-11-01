"""
Configuration loader and validator for the async reverse proxy.

Handles loading YAML configuration files and validating service/global settings.
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class ServiceConfig(BaseModel):
    """Configuration for a single proxied service."""

    # All containers listen on port 8000 internally
    INTERNAL_PORT: ClassVar[int] = 8000

    name: str = Field(..., description="Unique service name")
    path: str = Field(..., description="Path prefix for routing (e.g., /app1)")
    image: str = Field(..., description="Docker image name (e.g., myregistry/myapp:latest)")
    host_port: Optional[int] = Field(None, ge=1, le=65535, description="Fixed host port (optional)")
    auth_token: Optional[str] = Field(None, description="Bearer token for upstream service authentication")
    environment: Optional[Dict[str, str]] = Field(None, description="Environment variables for container")
    volumes: Optional[Dict[str, Union[str, Dict[str, str]]]] = Field(None, description="Volume mounts for container")
    mem_limit: Optional[str] = Field(None, description="Memory limit (e.g., '1g', '512m')")
    cpus: Optional[float] = Field(None, gt=0, description="CPU limit (e.g., 0.5, 1.0)")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path is normalized (starts with /, no trailing / except for root)."""
        if not v:
            v = "/"
        if not v.startswith("/"):
            v = "/" + v
        if v != "/" and v.endswith("/"):
            v = v.rstrip("/")
        return v

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Basic image name validation."""
        if not v or not v.strip():
            raise ValueError("image cannot be empty")
        return v.strip()

    @property
    def internal_port(self) -> int:
        """Expose the internal container port used by the proxy."""
        return self.INTERNAL_PORT


class GlobalConfig(BaseModel):
    """Global proxy configuration."""

    domain: str = Field(..., description="Domain name for TLS certificates")
    email: str = Field(..., description="Email for Let's Encrypt notifications")
    bearer_token: str = Field(..., description="Bearer token for authentication")
    idle_timeout: int = Field(300, ge=30, description="Idle timeout in seconds before stopping containers")
    listen_http: int = Field(80, ge=1, le=65535, description="HTTP port (for ACME)")
    listen_https: int = Field(443, ge=1, le=65535, description="HTTPS port")
    acme_webroot: str = Field("/var/www/acme", description="Path to ACME webroot directory")
    tls_certfile: Optional[str] = Field(None, description="Path to TLS certificate file")
    tls_keyfile: Optional[str] = Field(None, description="Path to TLS key file")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    docker_network: str = Field("nodetool-net", description="Docker network for proxy/services")
    connect_mode: Literal["docker_dns", "host_port"] = Field(
        "docker_dns", description="How proxy connects to services (Docker DNS or host port)"
    )
    http_redirect_to_https: bool = Field(True, description="Redirect HTTP to HTTPS (except ACME paths)")

    @field_validator("bearer_token")
    @classmethod
    def validate_bearer(cls, v: str) -> str:
        """Ensure bearer token is not empty."""
        if not v or not v.strip():
            raise ValueError("bearer_token cannot be empty")
        return v.strip()


class ProxyConfig(BaseModel):
    """Complete proxy configuration."""

    global_: GlobalConfig = Field(alias="global", description="Global configuration")
    services: List[ServiceConfig] = Field(..., description="List of services to proxy")

    @field_validator("services")
    @classmethod
    def validate_services(cls, v: List[ServiceConfig]) -> List[ServiceConfig]:
        """Validate service configurations."""
        if not v:
            raise ValueError("At least one service must be defined")

        # Check for duplicate names
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Service names must be unique")

        # Check for duplicate paths (unless it's intended longest-prefix matching)
        paths = [s.path for s in v]
        # Warn about overlapping paths (but don't fail - longest-prefix matching handles this)

        return v

    class Config:
        """Pydantic model config."""

        populate_by_name = True


def load_config(config_path: str) -> ProxyConfig:
    """
    Load and validate proxy configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        ProxyConfig instance with validated settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If configuration is invalid.
        yaml.YAMLError: If YAML is malformed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    if not raw_config:
        raise ValueError("Config file is empty")

    try:
        config = ProxyConfig(**raw_config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    return config


def load_config_with_env(config_path: str) -> ProxyConfig:
    """
    Load configuration and override values from environment variables.

    Environment variable naming convention:
    - PROXY_GLOBAL_{key} for global settings
    - PROXY_SERVICE_{service_name}_{key} for service settings

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        ProxyConfig with environment variable overrides applied.
    """
    config = load_config(config_path)

    # Override global settings from environment
    env_domain = os.getenv("PROXY_GLOBAL_DOMAIN")
    if env_domain:
        config.global_.domain = env_domain

    env_email = os.getenv("PROXY_GLOBAL_EMAIL")
    if env_email:
        config.global_.email = env_email

    env_bearer = os.getenv("PROXY_GLOBAL_BEARER_TOKEN")
    if env_bearer:
        config.global_.bearer_token = env_bearer

    env_idle = os.getenv("PROXY_GLOBAL_IDLE_TIMEOUT")
    if env_idle:
        config.global_.idle_timeout = int(env_idle)

    env_network = os.getenv("PROXY_GLOBAL_DOCKER_NETWORK")
    if env_network:
        config.global_.docker_network = env_network

    env_connect_mode = os.getenv("PROXY_GLOBAL_CONNECT_MODE")
    if env_connect_mode and env_connect_mode in {"docker_dns", "host_port"}:
        config.global_.connect_mode = env_connect_mode  # type: ignore[assignment]

    env_http_redirect = os.getenv("PROXY_GLOBAL_HTTP_REDIRECT_TO_HTTPS")
    if env_http_redirect:
        config.global_.http_redirect_to_https = env_http_redirect.lower() in {
            "1",
            "true",
            "yes",
        }

    return config
