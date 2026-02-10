"""
Configuration loader and validator for the async reverse proxy.

Handles loading YAML configuration files and validating service/global settings.
"""

import ipaddress
import os
from pathlib import Path
from typing import ClassVar, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ServiceConfig(BaseModel):
    """Configuration for a single proxied service."""

    # Proxy-managed containers listen on port 8000 internally
    INTERNAL_PORT: ClassVar[int] = 8000

    name: str = Field(..., description="Unique service name")
    path: str = Field(..., description="Path prefix for routing (e.g., /app1)")
    image: str = Field(..., description="Docker image name (e.g., myregistry/myapp:latest)")
    host_port: Optional[int] = Field(None, ge=1, le=65535, description="Fixed host port (optional)")
    auth_token: Optional[str] = Field(None, description="Bearer token for upstream service authentication")
    environment: Optional[dict[str, str]] = Field(None, description="Environment variables for container")
    volumes: Optional[dict[str, str | dict[str, str]]] = Field(None, description="Volume mounts for container")
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
    trusted_proxies: list[str] = Field(
        default_factory=list,
        description=(
            "List of trusted proxy IP addresses or CIDR ranges. "
            "When a request comes from a trusted proxy, the real client IP "
            "is extracted from X-Forwarded-For header. Empty list means no proxies are trusted."
        ),
    )

    @field_validator("bearer_token")
    @classmethod
    def validate_bearer(cls, v: str) -> str:
        """Ensure bearer token is not empty."""
        if not v or not v.strip():
            raise ValueError("bearer_token cannot be empty")
        return v.strip()

    @field_validator("trusted_proxies")
    @classmethod
    def validate_trusted_proxies(cls, v: list[str]) -> list[str]:
        """Validate that all entries are valid IP addresses or CIDR ranges.

        Note: CIDR ranges are parsed with strict=False, which means that
        '192.168.1.5/24' is treated as '192.168.1.0/24' (the entire subnet).
        Single IP addresses like '10.0.0.1' are valid and treated as /32.
        """
        validated = []
        for entry in v:
            entry = entry.strip()
            if not entry:
                continue
            try:
                # Try parsing as network (covers both single IPs and CIDR notation)
                # strict=False allows single IPs and normalizes network addresses
                ipaddress.ip_network(entry, strict=False)
                validated.append(entry)
            except ValueError as e:
                raise ValueError(f"Invalid IP address or CIDR range: {entry!r}") from e
        return validated


class ProxyConfig(BaseModel):
    """Complete proxy configuration."""

    model_config = ConfigDict(populate_by_name=True)

    global_: GlobalConfig = Field(alias="global", description="Global configuration")
    services: list[ServiceConfig] = Field(..., description="List of services to proxy")

    @field_validator("services")
    @classmethod
    def validate_services(cls, v: list[ServiceConfig]) -> list[ServiceConfig]:
        """Validate service configurations."""
        if not v:
            raise ValueError("At least one service must be defined")

        # Check for duplicate names
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError("Service names must be unique")

        return v


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
        with open(config_file, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}") from e

    if not raw_config:
        raise ValueError("Config file is empty")

    try:
        config = ProxyConfig(**raw_config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}") from e

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

    # PROXY_GLOBAL_TRUSTED_PROXIES: comma-separated list of IPs or CIDR ranges
    # Note: Uses strict=False so single IPs and network ranges are both accepted.
    # Invalid entries are silently skipped for resilience during startup.
    env_trusted_proxies = os.getenv("PROXY_GLOBAL_TRUSTED_PROXIES")
    if env_trusted_proxies:
        proxies = [p.strip() for p in env_trusted_proxies.split(",") if p.strip()]
        validated_proxies = []
        for proxy in proxies:
            try:
                ipaddress.ip_network(proxy, strict=False)
                validated_proxies.append(proxy)
            except ValueError:
                # Skip invalid entries for resilience during startup
                pass
        config.global_.trusted_proxies = validated_proxies

    return config


def is_ip_trusted(client_ip: str, trusted_proxies: list[str]) -> bool:
    """
    Check if a client IP is in the list of trusted proxies.

    Args:
        client_ip: The IP address to check (from request.client.host).
        trusted_proxies: List of trusted IP addresses or CIDR ranges.
            Uses strict=False for network parsing, so single IPs and
            CIDR ranges are both accepted.

    Returns:
        True if the client IP is trusted, False otherwise.
    """
    if not trusted_proxies:
        return False

    try:
        client_addr = ipaddress.ip_address(client_ip)
    except ValueError:
        return False

    for proxy in trusted_proxies:
        try:
            # strict=False allows both single IPs (10.0.0.1) and CIDR notation
            network = ipaddress.ip_network(proxy, strict=False)
            if client_addr in network:
                return True
        except ValueError:
            continue

    return False


def get_real_client_ip(
    request_client_ip: str,
    x_forwarded_for: Optional[str],
    trusted_proxies: list[str],
) -> str:
    """
    Extract the real client IP address, considering trusted proxies.

    When the request comes from a trusted proxy, the real client IP is extracted
    from the X-Forwarded-For header. The rightmost IP that is NOT a trusted proxy
    is considered the real client IP.

    Security note: X-Forwarded-For can be spoofed by clients. Only trust this header
    when the immediate connecting IP (request_client_ip) is in the trusted_proxies list.

    Args:
        request_client_ip: The immediate client IP from the connection (request.client.host).
        x_forwarded_for: Value of the X-Forwarded-For header (comma-separated IPs).
        trusted_proxies: List of trusted proxy IP addresses or CIDR ranges.

    Returns:
        The real client IP address.
    """
    # If no trusted proxies configured, always use the direct client IP
    if not trusted_proxies:
        return request_client_ip

    # If the connecting IP is not trusted, don't trust X-Forwarded-For
    if not is_ip_trusted(request_client_ip, trusted_proxies):
        return request_client_ip

    # If no X-Forwarded-For header, use the direct client IP
    if not x_forwarded_for:
        return request_client_ip

    # Parse X-Forwarded-For header (format: "client, proxy1, proxy2")
    # IPs are in order from original client to most recent proxy
    forwarded_ips = [ip.strip() for ip in x_forwarded_for.split(",") if ip.strip()]

    if not forwarded_ips:
        return request_client_ip

    # Walk backwards through the chain to find the first non-trusted IP
    # This is the rightmost IP that is not a known proxy
    for ip in reversed(forwarded_ips):
        if not is_ip_trusted(ip, trusted_proxies):
            return ip

    # All IPs in the chain are trusted proxies, use the leftmost (original) IP
    return forwarded_ips[0]
