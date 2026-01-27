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
    get_real_client_ip,
    is_ip_trusted,
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid YAML"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_empty_file(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="empty"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_config_structure(self):
        """Test loading file with invalid config structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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

    def test_env_override_trusted_proxies(self, temp_config_file: Path):
        """Test that environment variable can override trusted_proxies."""
        os.environ["PROXY_GLOBAL_TRUSTED_PROXIES"] = "10.0.0.1,192.168.1.0/24"
        try:
            config = load_config_with_env(str(temp_config_file))
            assert config.global_.trusted_proxies == ["10.0.0.1", "192.168.1.0/24"]
        finally:
            del os.environ["PROXY_GLOBAL_TRUSTED_PROXIES"]

    def test_env_override_trusted_proxies_with_invalid(self, temp_config_file: Path):
        """Test that invalid entries in trusted_proxies env var are skipped."""
        os.environ["PROXY_GLOBAL_TRUSTED_PROXIES"] = "10.0.0.1,invalid,192.168.1.0/24"
        try:
            config = load_config_with_env(str(temp_config_file))
            # Invalid entry should be skipped
            assert config.global_.trusted_proxies == ["10.0.0.1", "192.168.1.0/24"]
        finally:
            del os.environ["PROXY_GLOBAL_TRUSTED_PROXIES"]


class TestTrustedProxiesConfig:
    """Tests for trusted_proxies configuration."""

    def test_trusted_proxies_default_empty(self):
        """Test that trusted_proxies defaults to empty list."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
        )
        assert cfg.trusted_proxies == []

    def test_trusted_proxies_with_single_ip(self):
        """Test trusted_proxies with a single IP address."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
            trusted_proxies=["10.0.0.1"],
        )
        assert cfg.trusted_proxies == ["10.0.0.1"]

    def test_trusted_proxies_with_cidr(self):
        """Test trusted_proxies with CIDR notation."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
            trusted_proxies=["192.168.1.0/24", "10.0.0.0/8"],
        )
        assert cfg.trusted_proxies == ["192.168.1.0/24", "10.0.0.0/8"]

    def test_trusted_proxies_with_ipv6(self):
        """Test trusted_proxies with IPv6 addresses."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
            trusted_proxies=["::1", "2001:db8::/32"],
        )
        assert cfg.trusted_proxies == ["::1", "2001:db8::/32"]

    def test_trusted_proxies_invalid_ip_rejected(self):
        """Test that invalid IP address is rejected."""
        with pytest.raises(ValueError, match="Invalid IP address or CIDR range"):
            GlobalConfig(
                domain="example.com",
                email="admin@example.com",
                bearer_token="secret123",
                trusted_proxies=["not-an-ip"],
            )

    def test_trusted_proxies_invalid_cidr_rejected(self):
        """Test that invalid CIDR notation is rejected."""
        with pytest.raises(ValueError, match="Invalid IP address or CIDR range"):
            GlobalConfig(
                domain="example.com",
                email="admin@example.com",
                bearer_token="secret123",
                trusted_proxies=["192.168.1.0/99"],  # Invalid prefix
            )

    def test_trusted_proxies_empty_entries_filtered(self):
        """Test that empty entries are filtered out."""
        cfg = GlobalConfig(
            domain="example.com",
            email="admin@example.com",
            bearer_token="secret123",
            trusted_proxies=["10.0.0.1", "", "  ", "192.168.1.1"],
        )
        assert cfg.trusted_proxies == ["10.0.0.1", "192.168.1.1"]


class TestIsIpTrusted:
    """Tests for is_ip_trusted function."""

    def test_empty_trusted_list(self):
        """Test that empty trusted list returns False."""
        assert is_ip_trusted("10.0.0.1", []) is False

    def test_exact_ip_match(self):
        """Test exact IP address matching."""
        assert is_ip_trusted("10.0.0.1", ["10.0.0.1"]) is True
        assert is_ip_trusted("10.0.0.2", ["10.0.0.1"]) is False

    def test_cidr_match(self):
        """Test CIDR range matching."""
        trusted = ["192.168.1.0/24"]
        assert is_ip_trusted("192.168.1.1", trusted) is True
        assert is_ip_trusted("192.168.1.254", trusted) is True
        assert is_ip_trusted("192.168.2.1", trusted) is False

    def test_multiple_ranges(self):
        """Test matching against multiple trusted ranges."""
        trusted = ["10.0.0.0/8", "192.168.0.0/16"]
        assert is_ip_trusted("10.1.2.3", trusted) is True
        assert is_ip_trusted("192.168.100.50", trusted) is True
        assert is_ip_trusted("172.16.0.1", trusted) is False

    def test_ipv6_match(self):
        """Test IPv6 address matching."""
        assert is_ip_trusted("::1", ["::1"]) is True
        assert is_ip_trusted("2001:db8::1", ["2001:db8::/32"]) is True
        assert is_ip_trusted("2001:db9::1", ["2001:db8::/32"]) is False

    def test_invalid_client_ip(self):
        """Test that invalid client IP returns False."""
        assert is_ip_trusted("not-an-ip", ["10.0.0.1"]) is False


class TestGetRealClientIp:
    """Tests for get_real_client_ip function."""

    def test_no_trusted_proxies(self):
        """Test that without trusted proxies, direct IP is returned."""
        result = get_real_client_ip("1.2.3.4", "5.6.7.8, 9.10.11.12", [])
        assert result == "1.2.3.4"

    def test_untrusted_connection(self):
        """Test that untrusted connection IP ignores X-Forwarded-For."""
        result = get_real_client_ip(
            "1.2.3.4",  # Not in trusted list
            "5.6.7.8, 9.10.11.12",
            ["10.0.0.1"],  # Different from connection IP
        )
        assert result == "1.2.3.4"

    def test_trusted_proxy_single_hop(self):
        """Test single hop through trusted proxy."""
        # Connection from trusted proxy, X-Forwarded-For has real client
        result = get_real_client_ip(
            "10.0.0.1",  # Trusted proxy
            "1.2.3.4",  # Real client
            ["10.0.0.1"],
        )
        assert result == "1.2.3.4"

    def test_trusted_proxy_multiple_hops(self):
        """Test multiple hops through trusted proxies."""
        # Format: client, proxy1, proxy2 (rightmost is most recent)
        result = get_real_client_ip(
            "10.0.0.2",  # Trusted proxy (most recent)
            "1.2.3.4, 10.0.0.1",  # client, then first proxy
            ["10.0.0.1", "10.0.0.2"],
        )
        assert result == "1.2.3.4"

    def test_no_x_forwarded_for(self):
        """Test trusted proxy without X-Forwarded-For header."""
        result = get_real_client_ip(
            "10.0.0.1",
            None,
            ["10.0.0.1"],
        )
        assert result == "10.0.0.1"

    def test_empty_x_forwarded_for(self):
        """Test trusted proxy with empty X-Forwarded-For header."""
        result = get_real_client_ip(
            "10.0.0.1",
            "",
            ["10.0.0.1"],
        )
        assert result == "10.0.0.1"

    def test_all_ips_trusted(self):
        """Test when all IPs in chain are trusted proxies."""
        result = get_real_client_ip(
            "10.0.0.3",
            "10.0.0.1, 10.0.0.2",
            ["10.0.0.0/8"],
        )
        # When all are trusted, return leftmost (original)
        assert result == "10.0.0.1"

    def test_cidr_trusted_proxy(self):
        """Test trusted proxy matching with CIDR range."""
        result = get_real_client_ip(
            "10.0.1.50",  # In 10.0.0.0/8 range
            "1.2.3.4",
            ["10.0.0.0/8"],
        )
        assert result == "1.2.3.4"

    def test_mixed_ipv4_ipv6(self):
        """Test with IPv6 trusted proxy."""
        result = get_real_client_ip(
            "::1",  # Trusted IPv6 proxy
            "1.2.3.4",
            ["::1"],
        )
        assert result == "1.2.3.4"

    def test_spoofed_header_blocked(self):
        """Test that spoofed X-Forwarded-For is blocked when not from trusted proxy."""
        # Attacker connects directly and tries to spoof the header
        result = get_real_client_ip(
            "1.2.3.4",  # Attacker's real IP (not trusted)
            "fake.internal.ip",  # Spoofed header
            ["10.0.0.1"],  # Trusted proxy is different
        )
        # Should return the real connection IP, not the spoofed one
        assert result == "1.2.3.4"
