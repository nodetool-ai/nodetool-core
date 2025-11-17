"""
Async Docker reverse proxy with on-demand container management.

This module provides a FastAPI-based reverse proxy that:
- Routes HTTP requests to Docker containers
- Starts containers on-demand when first accessed
- Stops idle containers after a timeout
- Supports Let's Encrypt ACME for TLS certificates
- Streams requests and responses without buffering
- Authenticates all protected endpoints with Bearer tokens
"""

from nodetool.proxy.config import (
    GlobalConfig,
    ProxyConfig,
    ServiceConfig,
    load_config,
    load_config_with_env,
)
from nodetool.proxy.docker_manager import DockerManager, ServiceRuntime
from nodetool.proxy.filters import (
    EXCLUDED_HEADERS,
    HOP_BY_HOP_HEADERS,
    filter_headers,
    filter_request_headers,
    filter_response_headers,
)
from nodetool.proxy.server import (
    AsyncReverseProxy,
    create_proxy_app,
    run_proxy_app,
)

__all__ = [
    "EXCLUDED_HEADERS",
    "HOP_BY_HOP_HEADERS",
    "AsyncReverseProxy",
    "DockerManager",
    "GlobalConfig",
    "ProxyConfig",
    "ServiceConfig",
    "ServiceRuntime",
    "create_proxy_app",
    "filter_headers",
    "filter_request_headers",
    "filter_response_headers",
    "load_config",
    "load_config_with_env",
    "run_proxy_app",
]
