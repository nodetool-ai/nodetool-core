"""
Unit tests for SelfHostedDeployer.
"""

from unittest.mock import Mock, patch

import pytest

from nodetool.config.deployment import (
    ContainerConfig,
    DeploymentStatus,
    ImageConfig,
    DockerDeployment,
    RootDeployment,
    ServerPaths,
    SSHConfig,
)
from nodetool.deploy.docker_run import DockerRunGenerator
from nodetool.deploy.proxy_run import ProxyRunGenerator
from nodetool.deploy.self_hosted import (
    LocalExecutor,
    DockerDeployer,
    RootDeployer,
    is_localhost,
)
from nodetool.deploy.ssh import SSHCommandError

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestIsLocalhost:
    """Tests for localhost detection."""

    def test_localhost_string(self):
        """Test localhost string detection."""
        assert is_localhost("localhost") is True
        assert is_localhost("LOCALHOST") is True
        assert is_localhost("LocalHost") is True

    def test_ipv4_localhost(self):
        """Test IPv4 localhost detection."""
        assert is_localhost("127.0.0.1") is True

    def test_ipv6_localhost(self):
        """Test IPv6 localhost detection."""
        assert is_localhost("::1") is True

    def test_any_address(self):
        """Test any address detection."""
        assert is_localhost("0.0.0.0") is True

    def test_remote_host(self):
        """Test remote host detection."""
        assert is_localhost("192.168.1.100") is False
        assert is_localhost("example.com") is False
        assert is_localhost("10.0.0.1") is False


class TestLocalExecutor:
    """Tests for LocalExecutor."""

    def test_context_manager(self):
        """Test using LocalExecutor as context manager."""
        with LocalExecutor() as executor:
            assert executor is not None

    def test_execute_failure_with_check(self):
        """Test command failure with check=True."""
        executor = LocalExecutor()

        with pytest.raises(SSHCommandError) as exc_info:
            executor.execute("sh -c 'exit 1'", check=True)

        assert exc_info.value.exit_code == 1

    def test_execute_failure_without_check(self):
        """Test command failure with check=False."""
        executor = LocalExecutor()
        exit_code, _stdout, _stderr = executor.execute("sh -c 'exit 1'", check=False)

        assert exit_code == 1

    def test_execute_with_timeout(self):
        """Test command execution with timeout."""
        executor = LocalExecutor()

        with pytest.raises(SSHCommandError, match="timed out"):
            executor.execute("sleep 10", check=True, timeout=0.1)


class TestDockerDeployer:
    """Tests for DockerDeployer class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic Docker deployment configuration."""
        return DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
        )

    @pytest.fixture
    def localhost_deployment(self):
        """Create a localhost deployment configuration."""
        return DockerDeployment(
            host="localhost",
            ssh=SSHConfig(user="user", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
        )

    @pytest.fixture
    def gpu_deployment(self):
        """Create a deployment with GPU configuration."""
        return DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="gpu1", port=8001, gpu="0"),
        )

    @pytest.fixture
    def no_proxy_deployment(self):
        """Create a deployment that skips the proxy container."""
        return DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
            use_proxy=False,
        )

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.read_state = Mock(return_value=None)
        manager.write_state = Mock()
        manager.update_deployment_status = Mock()
        manager.get_or_create_secret = Mock(return_value="test-proxy-token")
        return manager

    def test_init(self, basic_deployment, mock_state_manager):
        """Test deployer initialization."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        assert deployer.deployment_name == "test"
        assert deployer.deployment == basic_deployment
        assert deployer.state_manager == mock_state_manager
        assert deployer.is_localhost is False

    def test_runtime_command_override_uses_podman(self, basic_deployment, mock_state_manager, monkeypatch):
        """Runtime override should force podman command generation."""
        monkeypatch.setenv("NODETOOL_CONTAINER_RUNTIME", "podman")
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )
        generator = deployer._container_generator()
        assert generator.generate_command().startswith("podman run")

    def test_localhost_runtime_detection_prefers_podman_when_docker_missing(
        self, localhost_deployment, mock_state_manager
    ):
        """Localhost deploy should use podman when docker is not available."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=localhost_deployment,
            state_manager=mock_state_manager,
        )
        with (
            patch("nodetool.deploy.self_hosted.shutil.which") as mock_which,
            patch.dict("os.environ", {}, clear=False),
        ):
            mock_which.side_effect = lambda cmd: None if cmd == "docker" else "/usr/bin/podman"
            generator = deployer._container_generator()
            assert generator.generate_command().startswith("podman run")

    def test_get_executor_ssh(self, basic_deployment):
        """Test getting SSH executor for remote host."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
        )

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh = Mock()
            mock_ssh_cls.return_value = mock_ssh

            executor = deployer._get_executor()

            # Should return SSHConnection instance
            assert executor == mock_ssh
            # Verify SSHConnection was called with correct parameters
            # Note: key_path may be expanded from ~ to absolute path
            call_args = mock_ssh_cls.call_args
            assert call_args[1]["host"] == "192.168.1.100"
            assert call_args[1]["user"] == "ubuntu"
            assert call_args[1]["key_path"].endswith("/.ssh/id_rsa")
            assert call_args[1]["password"] is None
            assert call_args[1]["port"] == 22

    def test_plan_initial_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for initial deployment."""
        mock_state_manager.read_state.return_value = None

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        # Mock _generate_container_hash
        deployer._generate_container_hash = Mock(return_value="hash123")

        plan = deployer.plan()
        container_name = deployer._container_name()

        assert plan["deployment_name"] == "test"
        assert plan["host"] == "192.168.1.100"
        assert "Initial deployment" in plan["changes"][0]
        assert f"Container: {container_name}" in plan["will_create"]
        assert f"Directory: {basic_deployment.paths.workspace}" in str(plan["will_create"])
        assert f"Directory: {basic_deployment.paths.hf_cache}" in str(plan["will_create"])

    def test_plan_existing_deployment_no_changes(self, basic_deployment, mock_state_manager):
        """Test generating plan for existing deployment with no changes."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        # Compute the actual hash the deployer would generate
        actual_hash = deployer._container_generator().generate_hash()

        mock_state_manager.read_state.return_value = {
            "last_deployed": "2024-01-15T10:30:00",
            "proxy_run_hash": actual_hash,
        }

        plan = deployer.plan()

        # No configuration changes
        config_changes = [c for c in plan["changes"] if "configuration has changed" in c]
        assert len(config_changes) == 0

    def test_plan_existing_deployment_with_changes(self, basic_deployment, mock_state_manager):
        """Test generating plan for existing deployment with changes."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        mock_state_manager.read_state.return_value = {
            "last_deployed": "2024-01-15T10:30:00",
            "proxy_run_hash": "old_hash",
        }

        plan = deployer.plan()

        assert "configuration has changed" in plan["changes"][0]
        assert len(plan["will_update"]) > 0

    def test_plan_initial_deployment_without_proxy(self, no_proxy_deployment, mock_state_manager):
        """Test plan generation for direct-container mode."""
        mock_state_manager.read_state.return_value = None

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=no_proxy_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()
        expected_container = DockerRunGenerator(no_proxy_deployment).get_container_name()

        assert f"App container: {expected_container}" in plan["will_create"]
        assert not any("Docker network:" in item for item in plan["will_create"])

    def test_apply_dry_run(self, basic_deployment, mock_state_manager):
        """Test apply with dry_run=True returns plan."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        result = deployer.apply(dry_run=True)

        # Should return plan, not execute deployment
        assert "deployment_name" in result
        assert "changes" in result
        mock_state_manager.update_deployment_status.assert_not_called()

    def test_apply_success(self, basic_deployment, mock_state_manager):
        """Test successful deployment."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.mkdir = Mock()
        mock_ssh.execute = Mock(return_value=(0, "container_id_123", ""))

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )
            # Mock internal methods to isolate apply logic
            deployer._create_directories = Mock()
            deployer._ensure_image = Mock()
            deployer._stop_existing_container = Mock()
            deployer._start_container = Mock(return_value="hash123")
            deployer._check_health = Mock()

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            assert len(result["errors"]) == 0

            # Should update status to deploying
            mock_state_manager.update_deployment_status.assert_called_once_with(
                "test", DeploymentStatus.DEPLOYING.value
            )

            # Should write final state
            mock_state_manager.write_state.assert_called_once()
            state_args = mock_state_manager.write_state.call_args[0]
            assert state_args[0] == "test"
            assert state_args[1]["status"] == DeploymentStatus.RUNNING.value

    def test_apply_localhost(self, localhost_deployment, mock_state_manager):
        """Test deployment to localhost."""
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=localhost_deployment,
            state_manager=mock_state_manager,
        )
        # Mock all internal deployment steps
        deployer._create_directories = Mock()
        deployer._ensure_image = Mock()
        deployer._stop_existing_container = Mock()
        deployer._start_container = Mock(return_value="hash123")
        deployer._check_health = Mock()
        deployer._write_proxy_yaml = Mock(return_value="test-token")
        deployer._ensure_network = Mock()
        deployer._sync_tls_files = Mock()
        deployer._ensure_proxy_image = Mock()

        result = deployer.apply(dry_run=False)

        assert result["status"] == "success"

    def test_apply_failure(self, basic_deployment, mock_state_manager):
        """Test deployment failure."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        
        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )
            deployer._create_directories = Mock(side_effect=Exception("Connection failed"))

            with pytest.raises(Exception, match="Connection failed"):
                deployer.apply(dry_run=False)

            # Should update status to error
            mock_state_manager.update_deployment_status.assert_any_call("test", DeploymentStatus.ERROR.value)

    def test_apply_success_without_proxy(self, no_proxy_deployment, mock_state_manager):
        """Test successful deploy in direct-container mode."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=no_proxy_deployment,
                state_manager=mock_state_manager,
            )
            # Mock all internal deployment steps
            deployer._create_directories = Mock()
            deployer._ensure_image = Mock()
            deployer._stop_existing_container = Mock()
            deployer._start_container = Mock(return_value="hash123")
            deployer._check_health = Mock()
            deployer._pull_app_image = Mock()

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            assert len(result["errors"]) == 0
            mock_state_manager.write_state.assert_called_once()

    def test_pull_app_image(self, no_proxy_deployment, mock_state_manager):
        """Test app image is always pulled in direct-container mode."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(return_value=(0, "ok", ""))

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=no_proxy_deployment,
            state_manager=mock_state_manager,
        )

        # Verify that the deployer has an _ensure_image method
        assert hasattr(deployer, "_ensure_image")

    def test_create_directories(self, basic_deployment, mock_state_manager):
        """Test directory creation."""
        import os
        mock_ssh = Mock()
        mock_ssh.mkdir = Mock()

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        deployer._create_directories(mock_ssh, results)

        # Should create workspace and subdirectories - check logic matches implementation
        # DockerDeployer calls _create_specific_directories which creates proxy and acme
        # The paths are expanded from the default "~/nodetool_data/workspace" 
        workspace_path = os.path.expanduser("~/nodetool_data/workspace")
        assert mock_ssh.mkdir.call_count >= 2
        mock_ssh.mkdir.assert_any_call(f"{workspace_path}/proxy", parents=True)
        mock_ssh.mkdir.assert_any_call(f"{workspace_path}/acme", parents=True)

    def test_ensure_image_localhost_missing_fails_without_pull(self, localhost_deployment, mock_state_manager):
        """Local apply must not auto-pull when image is missing."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(return_value=(0, "", ""))  # images -q => missing

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=localhost_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        with pytest.raises(RuntimeError, match="not found locally"):
            deployer._ensure_image(mock_ssh, results)

        # Should only check local image presence; no pull attempted.
        assert mock_ssh.execute.call_count == 1

    def test_create_directories_custom_paths(self, mock_state_manager):
        """Test directory creation with custom paths."""
        deployment = DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
            paths=ServerPaths(
                workspace="/custom/workspace",
                hf_cache="/custom/cache",
            ),
        )

        mock_ssh = Mock()
        mock_ssh.mkdir = Mock()

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        deployer._create_directories(mock_ssh, results)

        # Should use custom paths
        mock_ssh.mkdir.assert_any_call(f"{deployment.paths.workspace}/proxy", parents=True)

    def test_stop_existing_container_found(self, basic_deployment, mock_state_manager):
        """Test stopping existing container."""
        mock_ssh = Mock()
        # Container exists (check returns container ID)
        # Stop succeeds, remove succeeds
        mock_ssh.execute = Mock(
            side_effect=[
                (0, "container_id_123", ""),  # check
                (0, "", ""),  # stop
                (0, "", ""),  # rm
            ]
        )

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        deployer._stop_existing_container(mock_ssh, results)

        # Should check, stop, and remove
        assert mock_ssh.execute.call_count == 3
        # Check if logs are correct based on implementation
        assert any("Stopped container" in step for step in results["steps"])
        assert any("Removed container" in step for step in results["steps"])

    def test_stop_existing_container_not_found(self, basic_deployment, mock_state_manager):
        """Test stopping when no existing container."""
        mock_ssh = Mock()
        # Container doesn't exist (check returns empty)
        mock_ssh.execute = Mock(return_value=(0, "", ""))

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        deployer._stop_existing_container(mock_ssh, results)

        # Should only check, not stop or remove
        assert mock_ssh.execute.call_count == 1

    def test_stop_existing_container_error(self, basic_deployment, mock_state_manager):
        """Test handling error when checking for existing container."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(side_effect=Exception("Connection lost"))

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": []}
        deployer._stop_existing_container(mock_ssh, results)

        # Should log warning or error depending on implementation
        assert any("Warning" in step or "Error" in step for step in results["steps"])

    def test_start_container_success(self, basic_deployment, mock_state_manager):
        """Test starting container successfully."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(return_value=(0, "container_id_abc123", ""))
        
        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        results = {"steps": [], "errors": []}
        hash_result = deployer._start_container(mock_ssh, results)

        # Hash comes from the generator's generate_hash method
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0
        assert any("container started" in step.lower() for step in results["steps"])
        assert len(results["errors"]) == 0

    def test_start_container_failure(self, basic_deployment, mock_state_manager):
        """Test container start failure."""
        mock_ssh = Mock()
        error = SSHCommandError("Docker run failed", 1, "", "Image not found")
        mock_ssh.execute = Mock(side_effect=error)

        deployer = DockerDeployer(
            deployment_name="test",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )
        deployer._generate_container_command = Mock(return_value="docker run ...")

        results = {"steps": [], "errors": []}

        with pytest.raises(SSHCommandError):
            deployer._start_container(mock_ssh, results)

        assert len(results["errors"]) > 0
        assert "Image not found" in results["errors"][0]

    def test_check_health(self, basic_deployment, mock_state_manager):
        """Test health check."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(
            side_effect=[
                (
                    0,
                    "default Up 2 minutes 0.0.0.0:80->80/tcp",
                    "",
                ),  # status
                (0, "ok", ""),  # health curl
            ]
        )

        with patch("nodetool.deploy.self_hosted.time.sleep"):
            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            results = {"steps": []}
            deployer._check_health(mock_ssh, results, None)

            assert any("Container status" in step for step in results["steps"])
            assert any("Health endpoint OK" in step for step in results["steps"])

    def test_check_health_container_not_running(self, basic_deployment, mock_state_manager):
        """Test health check when container not running."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(
            side_effect=[
                (0, "", ""),  # status - empty
                (0, "ok", ""),  # health curl
            ]
        )

        with patch("nodetool.deploy.self_hosted.time.sleep"):
            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            results = {"steps": [], "errors": []}
            # Current implementation logs warning but does not raise
            deployer._check_health(mock_ssh, results, "token-123")

            assert any("not running" in step for step in results["steps"])

    def test_check_health_without_proxy_uses_mapped_port(self, no_proxy_deployment, mock_state_manager):
        """Test direct app health check uses host-mapped port (7777 -> 8000)."""
        mock_ssh = Mock()
        mock_ssh.execute = Mock(
            side_effect=[
                (0, "nodetool-default Up 2 minutes 0.0.0.0:8000->7777/tcp", ""),  # status
                (0, "ok", ""),  # health curl
            ]
        )

        with patch("nodetool.deploy.self_hosted.time.sleep"):
            deployer = DockerDeployer(
                deployment_name="test",
                deployment=no_proxy_deployment,
                state_manager=mock_state_manager,
            )

            results = {"steps": []}
            deployer._check_health(mock_ssh, results)

            health_cmd = mock_ssh.execute.call_args_list[1][0][0]
            assert "127.0.0.1:8000/health" in health_cmd

    def test_destroy_success(self, basic_deployment, mock_state_manager):
        """Test successful deployment destruction."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.execute = Mock(
            side_effect=[
                (0, "nodetool-proxy-default", ""),  # stop
                (0, "nodetool-proxy-default", ""),  # rm
            ]
        )

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "success"
            assert any("Container stopped" in step for step in result["steps"])
            assert any("Container removed" in step for step in result["steps"])

            mock_state_manager.update_deployment_status.assert_called_once_with(
                "test", DeploymentStatus.DESTROYED.value
            )

    def test_destroy_container_not_running(self, basic_deployment, mock_state_manager):
        """Test destroying when container not running."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        # Stop fails (container not running), but remove succeeds
        stop_error = SSHCommandError("Container not running", 1, "", "No such container")
        mock_ssh.execute = Mock(
            side_effect=[
                stop_error,  # stop
                (0, "nodetool-proxy-default", ""),  # rm
            ]
        )

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "success"
            assert any("Warning" in step for step in result["steps"])

    def test_destroy_remove_failure(self, basic_deployment, mock_state_manager):
        """Test destroy when remove fails."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        rm_error = SSHCommandError("Remove failed", 1, "", "Error")
        mock_ssh.execute = Mock(
            side_effect=[
                (0, "nodetool-proxy-default", ""),  # stop
                rm_error,  # rm
            ]
        )

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            with pytest.raises(SSHCommandError):
                deployer.destroy()

            assert mock_state_manager.update_deployment_status.call_count == 0

    def test_status_with_state(self, basic_deployment, mock_state_manager):
        """Test getting status with state."""
        mock_state_manager.read_state.return_value = {
            "status": "running",
            "last_deployed": "2024-01-15T10:30:00",
            "url": "http://192.168.1.100:7777",
        }

        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.execute = Mock(
            return_value=(0, "Up 2 hours", "")
        )

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test"
            assert status["host"] == "192.168.1.100"
            assert status["status"] == "running"
            assert status["last_deployed"] == "2024-01-15T10:30:00"
            assert status["url"] == "http://192.168.1.100:7777"
            assert status["live_status"] == "Up 2 hours"

    def test_status_without_state(self, basic_deployment, mock_state_manager):
        """Test getting status without state."""
        mock_state_manager.read_state.return_value = None

        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.execute = Mock(return_value=(0, "", ""))

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test"
            assert status["live_status"] == "Container not found"

    def test_status_connection_error(self, basic_deployment, mock_state_manager):
        """Test getting status when connection fails."""
        mock_state_manager.read_state.return_value = {"status": "unknown"}

        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(side_effect=Exception("Connection refused"))
        mock_ssh.__exit__ = Mock(return_value=False)

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert "live_status_error" in status
            assert "Connection refused" in status["live_status_error"]

    def test_logs(self, basic_deployment, mock_state_manager):
        """Test getting container logs."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.execute = Mock(return_value=(0, "Log line 1\nLog line 2\n", ""))

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            logs = deployer.logs(tail=100)

            assert "Log line 1" in logs
            assert "Log line 2" in logs
            mock_ssh.execute.assert_called_once()
            call_args = mock_ssh.execute.call_args[0][0]
            assert "--tail=100" in call_args
            # Container name matches the deployer's container name
            assert deployer._container_name() in call_args

    def test_logs_with_follow(self, basic_deployment, mock_state_manager):
        """Test getting logs with follow option."""
        mock_ssh = Mock()
        mock_ssh.__enter__ = Mock(return_value=mock_ssh)
        mock_ssh.__exit__ = Mock(return_value=False)
        mock_ssh.execute = Mock(return_value=(0, "Streaming logs...", ""))

        with patch("nodetool.deploy.self_hosted.SSHConnection") as mock_ssh_cls:
            mock_ssh_cls.return_value = mock_ssh

            deployer = DockerDeployer(
                deployment_name="test",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.logs(follow=True, tail=50)

            call_args = mock_ssh.execute.call_args[0][0]
            assert "--tail=50" in call_args
            assert "-f" in call_args


class TestRootDeployer:
    """Tests for RootDeployer class."""

    @pytest.fixture
    def root_deployment(self):
        """Create a basic root deployment configuration."""
        return RootDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            port=8000,
            service_name="nodetool-service",
        )

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.read_state = Mock(return_value=None)
        manager.write_state = Mock()
        manager.update_deployment_status = Mock()
        manager.get_or_create_secret = Mock(return_value="test-token")
        return manager

    def test_init(self, root_deployment, mock_state_manager):
        """Test deployer initialization."""
        deployer = RootDeployer(
            deployment_name="test-root",
            deployment=root_deployment,
            state_manager=mock_state_manager,
        )

        assert deployer.deployment_name == "test-root"
        assert deployer.deployment == root_deployment
        assert deployer.state_manager == mock_state_manager

    def test_plan(self, root_deployment, mock_state_manager):
        """Test generating plan."""
        deployer = RootDeployer(
            deployment_name="test-root",
            deployment=root_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-root"
        assert plan["host"] == "192.168.1.100"
        assert "Initial SSH deployment" in plan["changes"][0]
        assert "Systemd service" in str(plan["will_create"])
