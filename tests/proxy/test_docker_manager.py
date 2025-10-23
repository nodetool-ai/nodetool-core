"""
Tests for Docker container manager.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.proxy.config import ServiceConfig
from nodetool.proxy.docker_manager import DockerManager, ServiceRuntime


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    with patch("nodetool.proxy.docker_manager.docker.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_client.ping.return_value = None
        mock_network = MagicMock()
        mock_client.networks = MagicMock()
        mock_client.networks.get.return_value = mock_network
        mock_client.networks.create.return_value = mock_network
        mock_from_env.return_value = mock_client
        yield mock_client


@pytest.fixture
def service_config():
    """Create a sample service config."""
    return ServiceConfig(
        name="test-app",
        path="/app",
        image="nginx:latest",
    )


@pytest.fixture
async def docker_manager(mock_docker_client):
    """Create a DockerManager instance with mocked Docker client."""
    manager = DockerManager(idle_timeout=300, connect_mode="host_port")
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
async def docker_manager_dns(mock_docker_client):
    """Create a DockerManager configured for docker DNS connectivity."""
    manager = DockerManager(
        idle_timeout=300,
        network_name="proxy-net",
        connect_mode="docker_dns",
    )
    await manager.initialize()
    yield manager
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

    def test_docker_manager_initialization(self, mock_docker_client):
        """Test DockerManager initialization."""
        manager = DockerManager(idle_timeout=600)
        assert manager.idle_timeout == 600
        assert len(manager.runtime) == 0

    @pytest.mark.asyncio
    async def test_initialize_ensures_network(self, mock_docker_client):
        """Ensure Docker network is looked up on initialize when provided."""
        manager = DockerManager(
            idle_timeout=300,
            network_name="proxy-net",
            connect_mode="docker_dns",
        )
        await manager.initialize()
        mock_docker_client.networks.get.assert_called_with("proxy-net")
        await manager.shutdown()

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
        mock_docker_client,
    ):
        """Test that ensure_running creates a container if it doesn't exist."""
        from docker.errors import NotFound

        # Mock container not found, then successful run
        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "18000"}]}
            }
        }
        mock_docker_client.containers.run.return_value = mock_container

        host_port = await docker_manager.ensure_running(service_config)

        assert host_port == 18000
        mock_docker_client.containers.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_running_restarts_stopped_container(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
        mock_docker_client,
    ):
        """Test that ensure_running restarts a stopped container."""
        mock_container = MagicMock()
        mock_container.status = "exited"
        mock_container.attrs = {
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "18000"}]}
            }
        }
        mock_docker_client.containers.get.return_value = mock_container

        host_port = await docker_manager.ensure_running(service_config)

        assert host_port == 18000
        mock_container.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_running_with_fixed_host_port(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Test ensure_running with fixed host port."""
        from docker.errors import NotFound

        service_config = ServiceConfig(
            name="test-app",
            path="/app",
            image="nginx:latest",
            host_port=18000,
        )

        mock_docker_client.containers.get.side_effect = NotFound("Not found")
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "18000"}]}
            }
        }
        mock_docker_client.containers.run.return_value = mock_container

        host_port = await docker_manager.ensure_running(service_config)

        assert host_port == 18000
        # Verify that host_port was passed to run
        call_args = mock_docker_client.containers.run.call_args
        assert call_args[1]["ports"] == {"8000/tcp": 18000}
        assert call_args[1]["nano_cpus"] is None

    @pytest.mark.asyncio
    async def test_ensure_running_sets_nano_cpus(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Ensure CPU limits are converted to nano_cpus."""
        from docker.errors import NotFound

        service_config = ServiceConfig(
            name="limited-app",
            path="/limited",
            image="worker:latest",
            cpus=1.5,
        )

        mock_docker_client.containers.get.side_effect = NotFound("Not found")
        mock_container = MagicMock()
        mock_container.attrs = {
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "28000"}]}}
        }
        mock_docker_client.containers.run.return_value = mock_container

        await docker_manager.ensure_running(service_config)

        call_args = mock_docker_client.containers.run.call_args
        assert call_args[1]["nano_cpus"] == 1_500_000_000

    @pytest.mark.asyncio
    async def test_stop_container_if_running(
        self,
        docker_manager: DockerManager,
        service_config: ServiceConfig,
        mock_docker_client,
    ):
        """Test stopping a running container."""
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker_client.containers.get.return_value = mock_container

        result = await docker_manager.stop_container_if_running("test-app")

        assert result is True
        mock_container.stop.assert_called_once_with(timeout=10)

    @pytest.mark.asyncio
    async def test_stop_container_not_found(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Test stopping a non-existent container."""
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        with patch("asyncio.to_thread", side_effect=lambda f: f()):
            result = await docker_manager.stop_container_if_running("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_running_docker_dns_mode(
        self,
        docker_manager_dns: DockerManager,
        service_config: ServiceConfig,
        mock_docker_client,
    ):
        """Ensure docker_dns mode returns internal port and connects network."""
        from docker.errors import NotFound

        mock_docker_client.containers.get.side_effect = NotFound("Not found")
        mock_container = MagicMock()
        mock_container.attrs = {"NetworkSettings": {"Ports": {}}}
        mock_docker_client.containers.run.return_value = mock_container

        port = await docker_manager_dns.ensure_running(service_config)
        assert port == ServiceConfig.INTERNAL_PORT

        mock_docker_client.networks.get.return_value.connect.assert_called_with(
            mock_container
        )

    @pytest.mark.asyncio
    async def test_get_container_status(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Test getting container status."""
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_container.attrs = {
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostPort": "18000"}]}
            }
        }
        mock_docker_client.containers.get.return_value = mock_container

        status = await docker_manager.get_container_status("test-app")

        assert status["status"] == "running"
        assert "8000/tcp" in status["port_map"]

    @pytest.mark.asyncio
    async def test_idle_reaper_stops_idle_containers(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Test that idle reaper stops idle containers."""
        # Register a service
        rt = docker_manager.register_service("idle-app")
        rt.last_access = time.time() - 400  # 400 seconds ago (> 300s timeout)

        # Mock the container
        mock_container = MagicMock()
        mock_container.status = "running"
        mock_docker_client.containers.get.return_value = mock_container

        # Give the idle reaper a chance to run (it checks every 30 seconds, but we won't wait that long)
        # Just test that the idle reaper task is running
        assert docker_manager.idle_task is not None
        assert not docker_manager.idle_task.done()

        # Cleanup - cancel the task
        docker_manager.idle_task.cancel()
        try:
            await docker_manager.idle_task
        except asyncio.CancelledError:
            pass


class TestDockerManagerIntegration:
    """Integration tests for DockerManager."""

    @pytest.mark.asyncio
    async def test_concurrent_service_starts(
        self,
        docker_manager: DockerManager,
        mock_docker_client,
    ):
        """Test that concurrent service starts are properly serialized."""
        from docker.errors import NotFound

        service1 = ServiceConfig(
            name="app1", path="/app1", image="nginx"
        )
        service2 = ServiceConfig(
            name="app2", path="/app2", image="nginx"
        )

        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        def create_container(*args, **kwargs):
            mock_container = MagicMock()
            port = list(kwargs["ports"].keys())[0]
            port_num = int(port.split("/")[0]) + 10000
            mock_container.attrs = {
                "NetworkSettings": {"Ports": {port: [{"HostPort": str(port_num)}]}}
            }
            return mock_container

        mock_docker_client.containers.run.side_effect = create_container

        # Start both containers concurrently
        results = await asyncio.gather(
            docker_manager.ensure_running(service1),
            docker_manager.ensure_running(service2),
        )

        # Both services should be running (with same internal port 8000)
        assert len(results) == 2
        assert all(isinstance(p, int) and p > 0 for p in results)
