"""Configuration for OpenClaw integration.

This module handles configuration for connecting to and registering with
the OpenClaw Gateway.
"""

import os
import platform
import time
from typing import Optional

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class OpenClawConfig:
    """Configuration for OpenClaw node integration."""

    _instance: Optional["OpenClawConfig"] = None
    _start_time: float = time.time()

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize OpenClaw configuration from environment variables."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Gateway configuration
        self.gateway_url: str = os.environ.get(
            "OPENCLAW_GATEWAY_URL", "https://gateway.openclaw.ai"
        )
        self.gateway_token: Optional[str] = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        self.enabled: bool = os.environ.get("OPENCLAW_ENABLED", "false").lower() == "true"

        # Node identification
        self.node_id: str = os.environ.get("OPENCLAW_NODE_ID", self._generate_node_id())
        self.node_name: str = os.environ.get("OPENCLAW_NODE_NAME", "nodetool-core")
        self.node_version: str = self._get_node_version()

        # Node endpoint configuration
        self.node_endpoint: Optional[str] = os.environ.get("OPENCLAW_NODE_ENDPOINT")
        if not self.node_endpoint:
            # Try to construct from NODETOOL_API_URL
            api_url = Environment.get("NODETOOL_API_URL", "http://localhost:7777")
            self.node_endpoint = f"{api_url}/openclaw"

        # Registration settings
        self.auto_register: bool = (
            os.environ.get("OPENCLAW_AUTO_REGISTER", "true").lower() == "true"
        )
        self.heartbeat_interval: int = int(
            os.environ.get("OPENCLAW_HEARTBEAT_INTERVAL", "60")
        )

        # Task execution settings
        self.max_concurrent_tasks: int = int(
            os.environ.get("OPENCLAW_MAX_CONCURRENT_TASKS", "10")
        )
        self.task_timeout: int = int(os.environ.get("OPENCLAW_TASK_TIMEOUT", "300"))

        log.info(
            "OpenClaw configuration loaded: enabled=%s, gateway_url=%s, node_id=%s",
            self.enabled,
            self.gateway_url,
            self.node_id,
        )

    def _generate_node_id(self) -> str:
        """Generate a unique node ID based on hostname and process ID."""
        hostname = platform.node()
        pid = os.getpid()
        return f"nodetool-{hostname}-{pid}"

    def _get_node_version(self) -> str:
        """Get the version of nodetool-core."""
        try:
            from importlib.metadata import version

            return version("nodetool-core")
        except Exception:
            return "0.0.0-dev"

    @classmethod
    def get_uptime(cls) -> float:
        """Get the uptime of this node in seconds."""
        return time.time() - cls._start_time

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if OpenClaw integration is enabled."""
        config = cls()
        return config.enabled

    @classmethod
    def get_instance(cls) -> "OpenClawConfig":
        """Get the singleton instance."""
        return cls()
