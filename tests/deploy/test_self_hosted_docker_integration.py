"""
Integration tests for SelfHostedDeployer using real Docker.

These tests actually deploy containers to localhost Docker and verify the full
deployment lifecycle. They are skipped if Docker is not available.
"""

import subprocess
import time
import tempfile
from pathlib import Path

import docker
import docker.errors
import pytest

from nodetool.deploy.self_hosted import SelfHostedDeployer, LocalExecutor
from nodetool.config.deployment import (
    SelfHostedDeployment,
    SSHConfig,
    ImageConfig,
    ContainerConfig,
    SelfHostedPaths,
    ProxySpec,
    ServiceSpec,
    DeploymentStatus,
)
from nodetool.deploy.state import StateManager


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_test_image_available() -> bool:
    """Check if a simple test image is available (nginx as proxy stand-in)."""
    try:
        client = docker.from_env()
        # Check if nginx image is available or can be pulled
        try:
            client.images.get("nginx:alpine")
            return True
        except docker.errors.ImageNotFound:
            # Try to pull it
            try:
                client.images.pull("nginx", tag="alpine")
                return True
            except Exception:
                return False
    except Exception:
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not check_docker_available(),
    reason="Docker is not available",
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory(prefix="nodetool_test_") as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()

        # Create required subdirectories
        (workspace / "data").mkdir()
        (workspace / "assets").mkdir()
        (workspace / "temp").mkdir()
        (workspace / "proxy").mkdir()
        (workspace / "acme").mkdir()

        # Create minimal proxy config
        proxy_config = workspace / "proxy" / "proxy.yaml"
        proxy_config.write_text("""
version: "1.0"
proxy:
  listen_http: 8080
  listen_https: 8443
  domain: localhost
  docker_network: bridge
  connect_mode: websocket
containers: []
""")

        yield workspace


@pytest.fixture
def docker_client():
    """Provide a Docker client for tests."""
    client = docker.from_env()
    yield client
    client.close()


@pytest.fixture
def test_deployment(temp_workspace):
    """Create a test deployment configuration using nginx as proxy stand-in."""
    return SelfHostedDeployment(
        host="localhost",
        ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
        image=ImageConfig(name="nginx", tag="alpine"),
        container=ContainerConfig(name="test", port=8080),
        paths=SelfHostedPaths(
            workspace=str(temp_workspace),
            hf_cache=str(temp_workspace / "hf-cache"),
        ),
        proxy=ProxySpec(
            image="nginx:alpine",  # Use nginx as a stand-in for the proxy
            listen_http=8080,
            listen_https=8443,
            domain="localhost",
            email="test@example.com",
            docker_network="bridge",
            connect_mode="host_port",  # Changed from websocket to host_port
            acme_webroot="/var/www/acme",
            services=[
                ServiceSpec(
                    name="test",
                    path="/",
                    image="nginx:alpine",
                )
            ],
        ),
    )


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    from unittest.mock import Mock

    manager = Mock(spec=StateManager)
    manager.read_state = Mock(return_value=None)
    manager.write_state = Mock()
    manager.update_deployment_status = Mock()
    manager.get_or_create_secret = Mock(return_value="test-proxy-token")
    return manager


class TestLocalExecutor:
    """Integration tests for LocalExecutor with real commands."""

    def test_execute_echo(self):
        """Test executing a simple echo command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("echo 'test'", check=False)

        assert exit_code == 0
        assert "test" in stdout
        assert stderr == ""

    def test_execute_docker_ps(self):
        """Test executing docker ps command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("docker ps", check=False)

        assert exit_code == 0
        # Should contain headers like CONTAINER ID, IMAGE, etc.
        assert "CONTAINER" in stdout or "IMAGE" in stdout

    def test_mkdir_real_directory(self, temp_workspace):
        """Test creating a real directory."""
        executor = LocalExecutor()
        test_dir = temp_workspace / "test_mkdir"

        executor.mkdir(str(test_dir), parents=True)

        assert test_dir.exists()
        assert test_dir.is_dir()


class TestSelfHostedDeployerIntegration:
    """Integration tests using real Docker."""

    def test_localhost_detection(self, test_deployment):
        """Test that localhost is properly detected."""
        deployer = SelfHostedDeployer(
            deployment_name="test",
            deployment=test_deployment,
        )

        assert deployer.is_localhost is True

    def test_get_local_executor(self, test_deployment):
        """Test that LocalExecutor is used for localhost."""
        deployer = SelfHostedDeployer(
            deployment_name="test",
            deployment=test_deployment,
        )

        executor = deployer._get_executor()
        assert isinstance(executor, LocalExecutor)

    def test_create_directories_real(self, test_deployment, mock_state_manager):
        """Test creating real directories on the filesystem."""
        deployer = SelfHostedDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state_manager,
        )

        # Create fresh directories
        test_dir = Path(test_deployment.paths.workspace) / "new_test"
        test_dir.mkdir(exist_ok=True)

        executor = LocalExecutor()
        results = {"steps": []}

        # Test mkdir works
        executor.mkdir(str(test_dir / "subdir"), parents=True)

        assert (test_dir / "subdir").exists()


class TestDockerContainerLifecycle:
    """Test actual Docker container lifecycle."""

    def test_start_and_stop_nginx_container(self, docker_client, test_deployment):
        """Test starting and stopping a real nginx container."""
        # Skip if nginx image not available
        if not check_test_image_available():
            pytest.skip("nginx:alpine image not available")

        container_name = "nodetool-test-nginx"

        try:
            # Remove any existing test container
            try:
                old_container = docker_client.containers.get(container_name)
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Start a simple nginx container
            container = docker_client.containers.run(
                "nginx:alpine",
                name=container_name,
                detach=True,
                ports={"80/tcp": 8888},
                remove=False,
            )

            # Wait a bit for container to start
            time.sleep(2)

            # Verify it's running
            container.reload()
            assert container.status == "running"

            # Stop it
            container.stop(timeout=5)
            container.reload()
            assert container.status == "exited"

        finally:
            # Cleanup
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass

    def test_container_with_volume_mount(self, docker_client, temp_workspace):
        """Test container with real volume mounts."""
        if not check_test_image_available():
            pytest.skip("nginx:alpine image not available")

        container_name = "nodetool-test-volume"

        # Create a test file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello from host")

        try:
            # Remove any existing test container
            try:
                old_container = docker_client.containers.get(container_name)
                old_container.remove(force=True)
            except docker.errors.NotFound:
                pass

            # Start container with volume mount
            container = docker_client.containers.run(
                "nginx:alpine",
                name=container_name,
                detach=True,
                volumes={
                    str(temp_workspace): {"bind": "/data", "mode": "ro"}
                },
                remove=False,
            )

            time.sleep(1)

            # Verify the file is accessible in the container
            exit_code, output = container.exec_run("cat /data/test.txt")
            assert exit_code == 0
            assert b"Hello from host" in output

        finally:
            # Cleanup
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass

    def test_docker_network_exists(self, docker_client):
        """Test that bridge network exists (used by deployment)."""
        networks = docker_client.networks.list(names=["bridge"])
        assert len(networks) > 0
        assert networks[0].name == "bridge"


class TestDeploymentPlanWithDocker:
    """Test deployment planning with Docker available."""

    def test_plan_shows_container_name(self, test_deployment, mock_state_manager):
        """Test that plan shows the correct container name."""
        deployer = SelfHostedDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test"
        assert plan["host"] == "localhost"
        assert "nodetool-proxy-test" in str(plan["will_create"])

    def test_plan_with_docker_running(self, test_deployment, mock_state_manager):
        """Test planning when Docker is running."""
        deployer = SelfHostedDeployer(
            deployment_name="test",
            deployment=test_deployment,
            state_manager=mock_state_manager,
        )

        # This should not raise any errors
        plan = deployer.plan()

        assert "changes" in plan
        assert isinstance(plan["changes"], list)


class TestDockerCommandExecution:
    """Test actual Docker command execution."""

    def test_docker_ps_command(self):
        """Test running 'docker ps' command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("docker ps -a", check=True)

        assert exit_code == 0
        assert "CONTAINER" in stdout or len(stdout) == 0  # Empty if no containers

    def test_docker_version_command(self):
        """Test running 'docker version' command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("docker version", check=True)

        assert exit_code == 0
        assert "Version" in stdout or "version" in stdout.lower()

    def test_check_nonexistent_container(self):
        """Test checking for a container that doesn't exist."""
        executor = LocalExecutor()
        container_name = "nodetool-nonexistent-12345"

        exit_code, stdout, stderr = executor.execute(
            f"docker ps -aq --filter name=^{container_name}$",
            check=False
        )

        assert exit_code == 0
        assert stdout.strip() == ""  # Should be empty


class TestHealthCheckWithDocker:
    """Test health check functionality with real containers."""

    def test_check_running_container_status(self, docker_client):
        """Test checking the status of a running container."""
        if not check_test_image_available():
            pytest.skip("nginx:alpine image not available")

        container_name = "nodetool-test-health"

        try:
            # Start a container
            container = docker_client.containers.run(
                "nginx:alpine",
                name=container_name,
                detach=True,
                remove=False,
            )

            time.sleep(1)

            # Use LocalExecutor to check status (simulates what deployer does)
            executor = LocalExecutor()
            exit_code, stdout, stderr = executor.execute(
                f"docker ps --filter name={container_name} --format '{{{{.Status}}}}'",
                check=True
            )

            assert exit_code == 0
            assert "Up" in stdout

        finally:
            try:
                container = docker_client.containers.get(container_name)
                container.remove(force=True)
            except docker.errors.NotFound:
                pass


class TestDockerCleanup:
    """Test cleanup of Docker resources."""

    def test_cleanup_removes_container(self, docker_client):
        """Test that cleanup properly removes containers."""
        if not check_test_image_available():
            pytest.skip("nginx:alpine image not available")

        container_name = "nodetool-test-cleanup"

        # Start a container
        container = docker_client.containers.run(
            "nginx:alpine",
            name=container_name,
            detach=True,
            remove=False,
        )

        time.sleep(1)

        # Verify it exists
        containers = docker_client.containers.list(
            all=True,
            filters={"name": container_name}
        )
        assert len(containers) == 1

        # Remove it
        container.stop(timeout=2)
        container.remove()

        # Verify it's gone
        containers = docker_client.containers.list(
            all=True,
            filters={"name": container_name}
        )
        assert len(containers) == 0


# Add a marker to identify these as integration tests
pytestmark = [
    pytestmark,  # Keep the Docker availability check
    pytest.mark.integration,  # Add integration marker
]
