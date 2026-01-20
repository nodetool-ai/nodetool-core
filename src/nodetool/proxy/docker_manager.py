"""
Docker container lifecycle management for the async reverse proxy.

Handles starting, stopping, and monitoring containers with async-friendly interfaces.
"""

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any, Dict, Optional

from docker.errors import APIError, NotFound

import docker
from nodetool.proxy.config import ServiceConfig

log = logging.getLogger(__name__)


class ServiceRuntime:
    """Runtime state for a single service."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.last_access: float = 0.0
        self.host_port: Optional[int] = None  # cached published host port


class DockerManager:
    """Manages Docker container lifecycle with async support."""

    def __init__(
        self,
        idle_timeout: int = 300,
        network_name: Optional[str] = None,
        connect_mode: str = "docker_dns",
    ):
        """
        Initialize the Docker manager.

        Args:
            idle_timeout: Idle timeout in seconds before stopping containers.
            network_name: Optional Docker network to ensure and use for managed containers.
            connect_mode: How the proxy reaches services ("docker_dns" or "host_port").
        """
        self.docker = docker.from_env()  # type: ignore[attr-defined]
        self.idle_timeout = idle_timeout
        self.runtime: dict[str, ServiceRuntime] = {}
        self.idle_task: Optional[asyncio.Task] = None
        self.network_name = network_name
        self.connect_mode = connect_mode
        self.network = None

        # Verify Docker connectivity
        try:
            self.docker.ping()
            log.info("Docker connection established")
        except Exception as e:
            log.error(f"Failed to connect to Docker: {e}")
            raise

    async def initialize(self):
        """Initialize the Docker manager (async startup)."""

        # Ensure network exists (if configured)
        if self.network_name:

            def _ensure_network():
                assert self.network_name
                try:
                    return self.docker.networks.get(self.network_name)
                except NotFound:
                    log.info("Creating Docker network %s", self.network_name)
                    return self.docker.networks.create(
                        self.network_name,
                        driver="bridge",
                    )

            self.network = await asyncio.to_thread(_ensure_network)
            log.info("Using Docker network: %s", self.network_name)

        # Start idle reaper task
        self.idle_task = asyncio.create_task(self._idle_reaper())
        log.info("Docker manager initialized")

    async def shutdown(self):
        """Shutdown the Docker manager and clean up resources."""
        if self.idle_task:
            self.idle_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.idle_task
        log.info("Docker manager shutdown")

    def register_service(self, name: str) -> ServiceRuntime:
        """
        Register a service for runtime tracking.

        Args:
            name: Service name.

        Returns:
            ServiceRuntime instance for this service.
        """
        if name not in self.runtime:
            self.runtime[name] = ServiceRuntime()
        return self.runtime[name]

    async def ensure_running(self, service: ServiceConfig) -> int:
        """
        Ensure container for service is running and return the published host port.

        Args:
            service: Service configuration.

        Returns:
            Published host port for the service's container.

        Raises:
            RuntimeError: If container cannot be started or port cannot be determined.
        """
        name = service.name
        internal_port = ServiceConfig.INTERNAL_PORT
        desired_host_port = int(service.host_port) if service.host_port else None
        publish_host_port = self.connect_mode == "host_port"

        def _ensure() -> int:
            """Blocking function to ensure container is running."""
            try:
                container = self.docker.containers.get(name)
            except NotFound:
                container = None

            if container is None:
                # Create + start new container
                port_spec = {f"{internal_port}/tcp": desired_host_port or None}
                ports_arg = port_spec if publish_host_port else None
                nano_cpus = int(service.cpus * 1_000_000_000) if service.cpus else None
                # Normalize volume spec for docker-py
                volumes_arg = None
                if service.volumes:
                    volumes_arg = {}
                    for host, target in service.volumes.items():
                        if isinstance(target, dict):
                            bind = target.get("bind")
                            mode = target.get("mode", "rw")
                        else:
                            parts = str(target).split(":")
                            bind = parts[0]
                            mode = parts[1] if len(parts) > 1 else "rw"
                        if not bind:
                            raise RuntimeError(f"Invalid volume mapping for {name}: {host} -> {target}")
                        volumes_arg[host] = {"bind": bind, "mode": mode}

                try:
                    container = self.docker.containers.run(
                        service.image,
                        name=name,
                        ports=ports_arg,
                        detach=True,
                        environment=service.environment or {},
                        volumes=volumes_arg,
                        mem_limit=service.mem_limit,
                        nano_cpus=nano_cpus,
                        labels={
                            "com.nodetool.managed": "true",
                            "com.nodetool.service": name,
                        },
                        network=self.network_name if self.network_name else None,
                    )
                    log.info(f"Started container: {name}")
                except APIError as e:
                    log.error(f"Failed to create/start container {name}: {e}")
                    raise RuntimeError(f"Failed to start container {name}: {e}") from e
            else:
                container.reload()
                if container.status != "running":
                    try:
                        container.start()
                        log.info(f"Restarted container: {name}")
                    except APIError as e:
                        log.error(f"Failed to restart container {name}: {e}")
                        raise RuntimeError(f"Failed to restart container {name}: {e}") from e

            # Ensure container is attached to the managed network when configured
            if self.network_name:
                try:
                    container.reload()
                    networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
                    if self.network_name not in networks and self.network is not None:
                        self.network.connect(container)
                except APIError as e:
                    log.warning(
                        "Unable to connect container %s to network %s: %s",
                        name,
                        self.network_name,
                        e,
                    )

            # Wait for application to be ready by checking logs for startup message
            log.debug(f"Waiting for {name} to be ready...")
            self._wait_for_startup(container, timeout=30)

            # Inspect container to find published host port
            container.reload()
            port_map = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
            key = f"{internal_port}/tcp"

            if publish_host_port:
                if key not in port_map or not port_map[key]:
                    raise RuntimeError(f"Container {name} has no published host port for {key}. Port map: {port_map}")
                host_port = int(port_map[key][0]["HostPort"])
                return host_port

            # For docker_dns mode, no host port is published. Return internal port.
            return internal_port

        # Execute blocking operation in thread pool
        host_port = await asyncio.to_thread(_ensure)
        log.debug(f"Service {name} running on port {host_port}")
        return host_port

    def _wait_for_startup(self, container, timeout: int = 30) -> None:
        """
        Wait for container application to be ready by checking port connectivity.

        Args:
            container: Docker container object.
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If container fails to start or port is not accessible within timeout.
        """
        import socket

        start_time = time.time()
        internal_port = ServiceConfig.INTERNAL_PORT

        while time.time() - start_time < timeout:
            try:
                container.reload()

                # Check if container exited with error
                if container.status != "running":
                    error_logs = container.logs(stderr=True).decode("utf-8")
                    raise RuntimeError(f"Container {container.name} exited: {error_logs[-500:]}")

                # Get the host port to test connectivity
                port_map = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
                key = f"{internal_port}/tcp"

                # If port is published to host, test connectivity
                if port_map.get(key):
                    host_port = int(port_map[key][0]["HostPort"])

                    # Try to connect to the port
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    try:
                        result = sock.connect_ex(("127.0.0.1", host_port))
                        sock.close()

                        if result == 0:
                            log.debug(f"Container {container.name} is ready (port {host_port} accepting connections)")
                            return
                    except Exception:
                        pass
                    finally:
                        sock.close()

                # For docker_dns mode or if port check fails, fall back to container running check
                # If container has been running for 2 seconds, consider it started
                if container.status == "running":
                    container_start = container.attrs.get("State", {}).get("StartedAt")
                    if container_start:
                        # Container is running, give it a brief moment then accept
                        time.sleep(0.5)
                        container.reload()
                        if container.status == "running":
                            log.debug(f"Container {container.name} is ready (running)")
                            return

                time.sleep(0.3)
            except Exception as e:
                if "exited" not in str(e).lower():
                    log.debug(f"Error checking startup: {e}")
                time.sleep(0.3)

        # Final check - if container is still running, accept it
        container.reload()
        if container.status == "running":
            log.debug(f"Container {container.name} is running after timeout")
            return

        logs = container.logs(tail=50).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Container {container.name} did not start within {timeout}s. Last logs: {logs[-500:]}")

    async def stop_container_if_running(self, name: str) -> bool:
        """
        Stop container if it's currently running (graceful).

        Args:
            name: Container/service name.

        Returns:
            True if container was stopped, False if not found or already stopped.
        """

        def _stop() -> bool:
            try:
                container = self.docker.containers.get(name)
                container.reload()

                if container.status == "running":
                    try:
                        container.stop(timeout=10)
                        log.info(f"Stopped container: {name}")
                        return True
                    except APIError as e:
                        log.error(f"Failed to stop container {name}: {e}")
                        return False
                return False
            except NotFound:
                return False

        return await asyncio.to_thread(_stop)

    async def get_container_status(self, name: str) -> dict[str, Any]:
        """
        Get current container status.

        Args:
            name: Container/service name.

        Returns:
            Dict with status information.
        """

        def _get_status() -> dict[str, Any]:
            try:
                container = self.docker.containers.get(name)
                container.reload()
                port_map = container.attrs.get("NetworkSettings", {}).get("Ports") or {}
                return {
                    "status": container.status,
                    "port_map": port_map,
                }
            except NotFound:
                return {"status": "not_created", "port_map": {}}

        return await asyncio.to_thread(_get_status)

    async def _idle_reaper(self):
        """Background task to stop idle containers."""
        while True:
            try:
                now = time.time()

                for name, rt in self.runtime.items():
                    # Skip if never accessed
                    if not rt.last_access:
                        continue

                    # Check if idle
                    if (now - rt.last_access) > self.idle_timeout:
                        if await self.stop_container_if_running(name):
                            rt.host_port = None  # Clear cached port
                            log.info(f"Stopped idle container: {name}")

                # Check every 30 seconds
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                log.debug("Idle reaper cancelled")
                break
            except Exception as e:
                log.exception(f"Idle reaper error: {e}")
                await asyncio.sleep(10)
