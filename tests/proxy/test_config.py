"""
Tests for proxy configuration loading and validation.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nodetool.proxy.config import (
    GlobalConfig,
    ProxyConfig,
    ServiceConfig,
    load_config,
    load_config_with_env,
)


class TestServiceConfig:
    """Tests for ServiceConfig validation."""

    def test_valid_service_config(self):
        """Test creating valid service configuration."""
        svc = ServiceConfig(
            name="app1",
            path="/app1",
            image="nginx:latest",
        )
        assert svc.name == "app1"
        assert svc.path == "/app1"
        assert svc.image == "nginx:latest"
        assert ServiceConfig.INTERNAL_PORT == 8000
        assert svc.host_port is None

    def test_service_path_normalization(self):
        """Test that service paths are normalized."""
        # Path without leading slash should be added
        svc = ServiceConfig(
            name="app",
            path="app",
            image="nginx:latest",
        )
        assert svc.path == "/app"

        # Trailing slash should be removed (except for root)
        svc = ServiceConfig(
            name="app",
            path="/app/",
            image="nginx:latest",
        )
        assert svc.path == "/app"

        # Root path should remain as /
        svc = ServiceConfig(
            name="root",
            path="/",
            image="nginx:latest",
        )
        assert svc.path == "/"

    def test_service_with_host_port(self):
        """Test service configuration with fixed host port."""
        svc = ServiceConfig(
            name="app",
            path="/app",
            image="nginx:latest",
            host_port=18000,
        )
        assert svc.host_port == 18000

    def test_service_with_environment(self):
        """Test service configuration with environment variables."""
        env = {"DEBUG": "true", "LOG_LEVEL": "info"}
        svc = ServiceConfig(
            name="app",
            path="/app",
            image="nginx:latest",
            environment=env,
        )
        assert svc.environment == env

    def test_empty_image_rejected(self):
        """Test that empty image name is rejected."""
        with pytest.raises(ValueError):
            ServiceConfig(
                name="app",
                path="/app",
                image="",
            )


class TestGlobalConfig:
    """Tests for GlobalConfig validation."""

    def test_valid_global_config(self):
        """Test creating valid global configuration."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
        )
        assert cfg.domain == "example.com"
        assert cfg.email == "admin@example.com"
        assert cfg.bearer_token == "secret123"
        assert cfg.idle_timeout == 300  # Default
        assert cfg.listen_http == 80  # Default
        assert cfg.listen_https == 443  # Default
        assert cfg.docker_network == "nodetool-net"
        assert cfg.connect_mode == "docker_dns"
        assert cfg.http_redirect_to_https is True

    def test_empty_bearer_token_rejected(self):
        """Test that empty bearer token is rejected."""
        with pytest.raises(ValueError):
            GlobalConfig(
                domain="example.com",
                email="admin@example.com",
                bearer_token="",  # Invalid
            )

    def test_custom_idle_timeout(self):
        """Test custom idle timeout."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
            idle_timeout=600,
        )
        assert cfg.idle_timeout == 600

    def test_invalid_idle_timeout(self):
        """Test that invalid idle timeout is rejected."""
        with pytest.raises(ValueError):
            GlobalConfig(
                domain="example.com",
                email="admin@example.com",
                bearer_token="secret123",
                idle_timeout=10,  # Too small (< 30)
            )

    def test_invalid_connect_mode_rejected(self):
        """Ensure invalid connect_mode raises validation error."""
        with pytest.raises(ValueError):
            GlobalConfig(
                domain="example.com",
                email="admin@example.com",
                bearer_token="secret123",
                connect_mode="invalid-mode",  # type: ignore[arg-type]
            )


class TestProxyConfig:
    """Tests for ProxyConfig validation."""

    def test_valid_proxy_config(self, sample_proxy_config: ProxyConfig):
        """Test creating valid proxy configuration."""
        assert sample_proxy_config.global_.domain == "example.com"
        assert len(sample_proxy_config.services) == 3
        assert sample_proxy_config.services[0].name == "app1"

    def test_duplicate_service_names_rejected(self, sample_global_config: GlobalConfig):
        """Test that duplicate service names are rejected."""
        services = [
            {"name": "app1", "path": "/app1", "image": "nginx"},
            {"name": "app1", "path": "/app2", "image": "nginx"},
        ]
        with pytest.raises(ValueError, match="unique"):
            ProxyConfig(
                **{
                    "global": sample_global_config.model_dump(),
                    "services": services,
                }
            )

    def test_empty_services_rejected(self, sample_global_config: GlobalConfig):
        """Test that empty services list is rejected."""
        with pytest.raises(ValueError, match="At least one service"):
            ProxyConfig(
                **{
                    "global": sample_global_config.model_dump(),
                    "services": [],
                }
            )


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, temp_config_file: Path):
        """Test loading valid configuration from YAML file."""
        config = load_config(str(temp_config_file))
        assert config.global_.domain == "example.com"
        assert len(config.services) == 3

    def test_load_nonexistent_file(self):
        """Test loading nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="empty"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_config_structure(self):
        """Test loading file with invalid config structure."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"invalid": "structure"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestLoadConfigWithEnv:
    """Tests for load_config_with_env function."""

    def test_env_override_domain(self, temp_config_file: Path):
        """Test that environment variable can override domain."""
        os.environ["PROXY_GLOBAL_DOMAIN"] = "override.com"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.domain == "override.com"
        finally:
            del os.environ["PROXY_GLOBAL_DOMAIN"]

    def test_env_override_bearer_token(self, temp_config_file: Path):
        """Test that environment variable can override bearer token."""
        os.environ["PROXY_GLOBAL_BEARER_TOKEN"] = "env-token-override"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.bearer_token == "env-token-override"
        finally:
            del os.environ["PROXY_GLOBAL_BEARER_TOKEN"]

    def test_env_override_idle_timeout(self, temp_config_file: Path):
        """Test that environment variable can override idle timeout."""
        os.environ["PROXY_GLOBAL_IDLE_TIMEOUT"] = "600"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.idle_timeout == 600
        finally:
            del os.environ["PROXY_GLOBAL_IDLE_TIMEOUT"]

    def test_env_override_network(self, temp_config_file: Path):
        """Test environment override for docker network."""
        os.environ["PROXY_GLOBAL_DOCKER_NETWORK"] = "override-net"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.docker_network == "override-net"
        finally:
            del os.environ["PROXY_GLOBAL_DOCKER_NETWORK"]

    def test_env_override_connect_mode(self, temp_config_file: Path):
        """Test environment override for connect mode."""
        os.environ["PROXY_GLOBAL_CONNECT_MODE"] = "host_port"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.connect_mode == "host_port"
        finally:
            del os.environ["PROXY_GLOBAL_CONNECT_MODE"]

    def test_env_override_http_redirect(self, temp_config_file: Path):
        """Test environment override for HTTP redirect flag."""
        os.environ["PROXY_GLOBAL_HTTP_REDIRECT_TO_HTTPS"] = "false"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.http_redirect_to_https is False
        finally:
            del os.environ["PROXY_GLOBAL_HTTP_REDIRECT_TO_HTTPS"]

    def test_no_env_override(self, temp_config_file: Path):
        """Test loading without environment overrides."""
        # Ensure env vars don't exist
        for key in ["PROXY_GLOBAL_DOMAIN", "PROXY_GLOBAL_BEARER_TOKEN"]:
            if key in os.environ:
                del os.environ[key]

        config = load_config_with_env(str(temp_config_file))
        assert config.global_.domain == "example.com"
        assert config.global_.bearer_token == "test-token-123"
