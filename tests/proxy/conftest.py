"""
Pytest fixtures and configuration for proxy tests.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from nodetool.proxy.config import GlobalConfig, ProxyConfig, ServiceConfig


@pytest.fixture
def sample_global_config() -> GlobalConfig:
    """Create a sample global configuration."""
    return GlobalConfig(
        domain="example.com",
        email="admin@example.com",
        bearer_token="test-token-123",
        idle_timeout=300,
        listen_http=80,
        listen_https=443,
        acme_webroot="/tmp/acme",
        tls_certfile="/etc/letsencrypt/live/example.com/fullchain.pem",
        tls_keyfile="/etc/letsencrypt/live/example.com/privkey.pem",
        docker_network="nodetool-net-test",
        connect_mode="docker_dns",
        http_redirect_to_https=True,
        log_level="INFO",
    )


@pytest.fixture
def sample_services() -> list[ServiceConfig]:
    """Create sample service configurations."""
    return [
        ServiceConfig(
            name="app1",
            path="/app1",
            image="nginx:latest",
        ),
        ServiceConfig(
            name="app2",
            path="/app2",
            image="python:3.11",
            host_port=18002,
        ),
        ServiceConfig(
            name="api",
            path="/api",
            image="fastapi-app:latest",
            environment={"DEBUG": "true"},
        ),
    ]


@pytest.fixture
def sample_proxy_config(
    sample_global_config: GlobalConfig,
    sample_services: list[ServiceConfig],
) -> ProxyConfig:
    """Create a sample proxy configuration."""
    return ProxyConfig(
        **{
            "global": sample_global_config.model_dump(),
            "services": [s.model_dump() for s in sample_services],
        }
    )


@pytest.fixture
def config_yaml_content(sample_proxy_config: ProxyConfig) -> str:
    """Create YAML content for a proxy config file."""
    data = {
        "global": {
            "domain": sample_proxy_config.global_.domain,
            "email": sample_proxy_config.global_.email,
            "bearer_token": sample_proxy_config.global_.bearer_token,
            "idle_timeout": sample_proxy_config.global_.idle_timeout,
            "listen_http": sample_proxy_config.global_.listen_http,
            "listen_https": sample_proxy_config.global_.listen_https,
            "acme_webroot": sample_proxy_config.global_.acme_webroot,
            "tls_certfile": sample_proxy_config.global_.tls_certfile,
            "tls_keyfile": sample_proxy_config.global_.tls_keyfile,
            "docker_network": sample_proxy_config.global_.docker_network,
            "connect_mode": sample_proxy_config.global_.connect_mode,
            "http_redirect_to_https": sample_proxy_config.global_.http_redirect_to_https,
            "log_level": sample_proxy_config.global_.log_level,
        },
        "services": [
            {
                "name": s.name,
                "path": s.path,
                "image": s.image,
                **({"host_port": s.host_port} if s.host_port else {}),
                **({"environment": s.environment} if s.environment else {}),
            }
            for s in sample_proxy_config.services
        ],
    }
    return yaml.dump(data)


@pytest.fixture
def temp_config_file(config_yaml_content: str) -> Path:
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml_content)
        return Path(f.name)


@pytest.fixture(autouse=True)
def cleanup_temp_files(temp_config_file: Path):
    """Clean up temporary files after tests."""
    yield
    if temp_config_file.exists():
        temp_config_file.unlink()
