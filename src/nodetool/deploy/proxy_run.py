"""
Utilities for generating the docker run command used by the self-hosted proxy container.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nodetool.config.deployment import SelfHostedDeployment


class ProxyRunGenerator:
    """Generate docker run command lines for the proxy container."""

    def __init__(self, deployment: SelfHostedDeployment):
        self.deployment = deployment
        self.proxy = deployment.proxy

    def generate_command(self) -> str:
        """Render the docker run command for the proxy container."""
        parts: list[str] = ["docker run", "-d"]

        container_name = self.get_container_name()
        parts.extend(
            [
                f"--name {container_name}",
                "--restart unless-stopped",
                f"-p {self.proxy.listen_http}:{self.proxy.listen_http}",
            ]
        )

        # Only expose HTTPS port when TLS is configured
        if self.proxy.tls_certfile and self.proxy.tls_keyfile:
            parts.append(f"-p {self.proxy.listen_https}:{self.proxy.listen_https}")

        workspace = Path(self.deployment.paths.workspace)
        proxy_config_path = workspace / "proxy" / "proxy.yaml"
        acme_path = workspace / "acme"

        parts.extend(
            [
                "-v /var/run/docker.sock:/var/run/docker.sock",
                f"-v {proxy_config_path}:{Path('/etc/nodetool/proxy.yaml')}:ro",
                f"-v {acme_path}:{Path(self.proxy.acme_webroot)}",
            ]
        )

        if self.proxy.tls_certfile and self.proxy.tls_keyfile:
            cert_parent = Path(self.proxy.tls_certfile).parent
            key_parent = Path(self.proxy.tls_keyfile).parent
            mount_root = os.path.commonpath([str(cert_parent), str(key_parent)])
            parts.append(f"-v {mount_root}:{mount_root}:ro")

        parts.append("--group-add $(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 0)")
        parts.append(f"--network {self.proxy.docker_network}")

        health_cmd = (
            '--health-cmd="curl -fsS http://localhost/healthz || exit 1" '
            "--health-interval=30s "
            "--health-timeout=5s "
            "--health-retries=3 "
            "--health-start-period=10s"
        )
        parts.append(health_cmd)

        parts.append(self.proxy.image)
        parts.append("--config /etc/nodetool/proxy.yaml")

        return " \\\n  ".join(parts)

    def get_container_name(self) -> str:
        """Return the deterministic proxy container name."""
        return f"nodetool-proxy-{self.deployment.container.name}"

    def generate_hash(self) -> str:
        """Compute a hash for the docker run configuration."""
        data = {
            "image": self.proxy.image,
            "listen_http": self.proxy.listen_http,
            "listen_https": self.proxy.listen_https,
            "domain": self.proxy.domain,
            "tls_certfile": self.proxy.tls_certfile,
            "tls_keyfile": self.proxy.tls_keyfile,
            "docker_network": self.proxy.docker_network,
            "connect_mode": self.proxy.connect_mode,
        }
        payload = repr(sorted(data.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
