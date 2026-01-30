"""Tests for OpenClaw configuration."""

import os

import pytest

from nodetool.integrations.openclaw.config import OpenClawConfig


def test_openclaw_config_default_values():
    """Test that OpenClaw config loads with default values."""
    config = OpenClawConfig()

    assert config.gateway_url == "https://gateway.openclaw.ai"
    assert config.node_name == "nodetool-core"
    assert config.enabled is False  # Default is disabled
    assert config.auto_register is True
    assert config.heartbeat_interval == 60
    assert config.max_concurrent_tasks == 10


def test_openclaw_config_from_environment(monkeypatch):
    """Test that OpenClaw config reads from environment variables."""
    monkeypatch.setenv("OPENCLAW_ENABLED", "true")
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "https://test-gateway.example.com")
    monkeypatch.setenv("OPENCLAW_NODE_ID", "test-node-123")
    monkeypatch.setenv("OPENCLAW_NODE_NAME", "test-node")
    monkeypatch.setenv("OPENCLAW_HEARTBEAT_INTERVAL", "30")
    monkeypatch.setenv("OPENCLAW_MAX_CONCURRENT_TASKS", "5")

    # Clear singleton to force re-initialization
    OpenClawConfig._instance = None

    config = OpenClawConfig()

    assert config.enabled is True
    assert config.gateway_url == "https://test-gateway.example.com"
    assert config.node_id == "test-node-123"
    assert config.node_name == "test-node"
    assert config.heartbeat_interval == 30
    assert config.max_concurrent_tasks == 5


def test_openclaw_config_singleton():
    """Test that OpenClawConfig is a singleton."""
    config1 = OpenClawConfig()
    config2 = OpenClawConfig()

    assert config1 is config2


def test_openclaw_config_is_enabled():
    """Test the is_enabled class method."""
    # Reset singleton
    OpenClawConfig._instance = None

    # Disabled by default
    assert OpenClawConfig.is_enabled() is False


def test_openclaw_config_get_uptime():
    """Test that uptime is calculated correctly."""
    import time

    OpenClawConfig._start_time = time.time() - 10  # 10 seconds ago
    uptime = OpenClawConfig.get_uptime()

    assert uptime >= 10
    assert uptime < 11  # Should be around 10 seconds


def test_openclaw_config_node_endpoint_construction(monkeypatch):
    """Test that node endpoint is constructed from NODETOOL_API_URL."""
    monkeypatch.setenv("NODETOOL_API_URL", "http://localhost:8888")

    # Clear singleton
    OpenClawConfig._instance = None

    config = OpenClawConfig()

    assert config.node_endpoint == "http://localhost:8888/openclaw"
