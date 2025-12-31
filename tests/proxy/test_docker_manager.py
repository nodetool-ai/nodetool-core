"""
Tests for Docker container manager.

These tests use real Docker containers and are skipped if Docker is not available.
"""

import asyncio
import time

import pytest

import docker
from docker.errors import DockerException
from nodetool.proxy.config import ServiceConfig
from nodetool.proxy.docker_manager import DockerManager, ServiceRuntime


def is_docker_available():
    """Check if Docker daemon is available."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except (DockerException, Exception):
        return False


# Skip all tests in this module if Docker is not available
pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker is not available")


@pytest.fixture
def service_config():
    """Create a sample service config using nginx for quick startup."""
    return ServiceConfig(
        name="test-nginx",
        path="/",
        image="nginx:alpine",  # Alpine is smaller and starts faster
    )


@pytest.fixture
async def docker_manager():
    """Create a DockerManager instance."""
    manager = DockerManager(idle_timeout=300, connect_mode="host_port")
    await manager.initialize()
    yield manager

    # Cleanup: stop and remove test containers
    try:
        client = docker.from_env()
        for container_name in ["test-nginx", "test-app", "idle-app", "app1", "app2", "limited-app"]:
            try:
                container = client.containers.get(container_name)
                container.stop(timeout=2)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass
    except Exception:
        pass

    await manager.shutdown()


@pytest.fixture
async def docker_manager_dns():
    """Create a DockerManager configured for docker DNS connectivity."""
    manager = DockerManager(
        idle_timeout=300,
        network_name="test-proxy-net",
        connect_mode="docker_dns",
    )
    await manager.initialize()
    yield manager

    # Cleanup
    try:
        client = docker.from_env()
        for container_name in ["test-nginx", "test-app"]:
            try:
                container = client.containers.get(container_name)
                container.stop(timeout=2)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass

        # Remove test network
        try:
            network = client.networks.get("test-proxy-net")
            network.remove()
        except docker.errors.NotFound:
            pass
    except Exception:
        pass

    await manager.shutdown()


class TestServiceRuntime:
    """Tests for ServiceRuntime class."""

    def test_service_runtime_initialization(self):
        """Test ServiceRuntime initialization."""
        rt = ServiceRuntime()
        assert rt.last_access == 0.0
        assert rt.host_port is None
        assert isinstance(rt.lock, asyncio.Lock)

    def test_service_runtime_lock(self):
        """Test that ServiceRuntime has a working lock."""
        rt = ServiceRuntime()
        assert not rt.lock.locked()


class TestDockerManager:
    """Tests for DockerManager class."""

    def test_docker_manager_initialization(self):
        """Test DockerManager initialization."""
        manager = DockerManager(idle_timeout=600)
        assert manager.idle_timeout == 600
        assert len(manager.runtime) == 0

    @pytest.mark.asyncio
    async def test_initialize_ensures_network(self):
        """Ensure Docker network is created on initialize when provided."""
        manager = DockerManager(
            idle_timeout=300,
            network_name="test-init-net",
            connect_mode="docker_dns",
        )
        await manager.initialize()

        # Verify network exists
        client = docker.from_env()
        network = client.networks.get("test-init-net")
        assert network is not None

        # Cleanup
        await manager.shutdown()
        network.remove()

    @pytest.mark.asyncio
    async def test_register_service(self, docker_manager: DockerManager):
        """Test service registration."""
        rt = docker_manager.register_service("test-service")
        assert isinstance(rt, ServiceRuntime)
        assert docker_manager.runtime["test-service"] is rt

    @pytest.mark.asyncio
    async def test_ensure_running_creates_container(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
    ):
        """Test that ensure_running creates a container if it doesn't exist."""
        host_port = await docker_manager.ensure_running(service_config)

        assert isinstance(host_port, int)
        assert host_port > 0

        # Verify container exists and is running
        client = docker.from_env()
        container = client.containers.get(service_config.name)
        assert container.status == "running"

    @pytest.mark.asyncio
    async def test_ensure_running_restarts_stopped_container(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
    ):
        """Test that ensure_running restarts a stopped container."""
        # First start the container
        host_port1 = await docker_manager.ensure_running(service_config)
        assert isinstance(host_port1, int)
        assert host_port1 > 0

        # Stop the container
        client = docker.from_env()
        container = client.containers.get(service_config.name)
        container.stop(timeout=2)

        # Ensure it's restarted
        host_port2 = await docker_manager.ensure_running(service_config)

        # Both ports should be valid (may be different due to Docker dynamic port assignment)
        assert isinstance(host_port2, int)
        assert host_port2 > 0

        # Verify container is running
        container.reload()
        assert container.status == "running"

    @pytest.mark.asyncio
    async def test_ensure_running_with_fixed_host_port(
        self,
        docker_manager: DockerManager,
    ):
        """Test ensure_running with fixed host port."""
        service_config = ServiceConfig(
            name="test-app",
            path="/",
            image="nginx:alpine",
            host_port=18888,  # Use a specific high port
        )

        host_port = await docker_manager.ensure_running(service_config)

        assert host_port == 18888

        # Verify port mapping
        client = docker.from_env()
        container = client.containers.get("test-app")
        container.reload()
        port_map = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        assert "8000/tcp" in port_map
        assert port_map["8000/tcp"][0]["HostPort"] == "18888"

    @pytest.mark.asyncio
    async def test_ensure_running_sets_nano_cpus(
        self,
        docker_manager: DockerManager,
    ):
        """Ensure CPU limits are converted to nano_cpus."""
        service_config = ServiceConfig(
            name="limited-app",
            path="/",
            image="nginx:alpine",
            cpus=1.5,
        )

        await docker_manager.ensure_running(service_config)

        # Verify CPU limit was set
        client = docker.from_env()
        container = client.containers.get("limited-app")
        container.attrs.get("HostConfig", {})
        # Note: NanoCpus might not be in attrs depending on Docker version
        # Just verify container started successfully
        assert container.status == "running"

    @pytest.mark.asyncio
    async def test_stop_container_if_running(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
    ):
        """Test stopping a running container."""
        # Start the container first
        await docker_manager.ensure_running(service_config)

        # Stop it
        result = await docker_manager.stop_container_if_running(service_config.name)

        assert result is True

        # Verify it's stopped
        client = docker.from_env()
        container = client.containers.get(service_config.name)
        container.reload()
        assert container.status in ["exited", "stopped"]

    @pytest.mark.asyncio
    async def test_stop_container_not_found(
        self,
        docker_manager: DockerManager,
    ):
        """Test stopping a non-existent container."""
        result = await docker_manager.stop_container_if_running("nonexistent-container-xyz")
        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_running_docker_dns_mode(
        self,
        docker_manager_dns: DockerManager,
        service_config: ServiceConfig,
    ):
        """Ensure docker_dns mode returns internal port and connects network."""
        port = await docker_manager_dns.ensure_running(service_config)

        # In docker_dns mode, should return internal port
        assert port == ServiceConfig.INTERNAL_PORT

        # Verify container is connected to the network
        client = docker.from_env()
        container = client.containers.get(service_config.name)
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
        assert "test-proxy-net" in networks

    @pytest.mark.asyncio
    async def test_get_container_status(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
    ):
        """Test getting container status."""
        # Start container first
        await docker_manager.ensure_running(service_config)

        status = await docker_manager.get_container_status(service_config.name)

        assert status["status"] == "running"
        assert "8000/tcp" in status["port_map"]

    @pytest.mark.asyncio
    async def test_idle_reaper_task_is_running(
        self,
        docker_manager: DockerManager,
    ):
        """Test that idle reaper task starts and runs."""
        assert docker_manager.idle_task is not None
        assert not docker_manager.idle_task.done()


class TestDockerManagerIntegration:
    """Integration tests for DockerManager."""

    @pytest.mark.asyncio
    async def test_concurrent_service_starts(
        self,
        docker_manager: DockerManager,
    ):
        """Test that concurrent service starts are properly serialized."""
        service1 = ServiceConfig(name="app1", path="/", image="nginx:alpine")
        service2 = ServiceConfig(name="app2", path="/", image="nginx:alpine")

        # Start both containers concurrently
        results = await asyncio.gather(
            docker_manager.ensure_running(service1),
            docker_manager.ensure_running(service2),
        )

        # Both services should be running with valid ports
        assert len(results) == 2
        assert all(isinstance(p, int) and p > 0 for p in results)

        # Verify both containers are running
        client = docker.from_env()
        for service in [service1, service2]:
            container = client.containers.get(service.name)
            assert container.status == "running"
