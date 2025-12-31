"""
Unit tests for Fly.io deployment configuration and deployer.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nodetool.config.deployment import (
    DeploymentConfig,
    DeploymentStatus,
    DeploymentType,
    FlyBuildConfig,
    FlyDeployment,
    FlyImageConfig,
    FlyNetworkConfig,
    FlyResourceConfig,
    FlyState,
    FlyVolumeConfig,
)
from nodetool.deploy.fly import (
    FlyDeployer,
    check_flyctl_authenticated,
    check_flyctl_installed,
    generate_fly_toml,
    run_flyctl,
)


class TestFlyImageConfig:
    """Tests for FlyImageConfig model."""

    def test_basic_config(self):
        """Test basic image configuration."""
        config = FlyImageConfig(name="myapp", tag="v1.0.0")
        assert config.name == "myapp"
        assert config.tag == "v1.0.0"
        assert config.registry is None

    def test_full_name_with_registry(self):
        """Test full name with registry."""
        config = FlyImageConfig(
            name="myapp",
            tag="v1.0.0",
            registry="registry.fly.io",
        )
        assert config.full_name == "registry.fly.io/myapp:v1.0.0"

    def test_full_name_without_registry(self):
        """Test full name without registry."""
        config = FlyImageConfig(name="myapp", tag="latest")
        assert config.full_name == "myapp:latest"

    def test_full_name_no_name(self):
        """Test full name with no name set."""
        config = FlyImageConfig()
        assert config.full_name == "fly-app:latest"

    def test_build_config(self):
        """Test build configuration."""
        config = FlyImageConfig(
            build=FlyBuildConfig(platform="linux/arm64", dockerfile="Dockerfile.prod")
        )
        assert config.build.platform == "linux/arm64"
        assert config.build.dockerfile == "Dockerfile.prod"


class TestFlyResourceConfig:
    """Tests for FlyResourceConfig model."""

    def test_default_config(self):
        """Test default resource configuration."""
        config = FlyResourceConfig()
        assert config.vm_size == "shared-cpu-1x"
        assert config.memory == "256mb"
        assert config.cpu_kind == "shared"
        assert config.cpus == 1
        assert config.gpu_kind is None

    def test_custom_config(self):
        """Test custom resource configuration."""
        config = FlyResourceConfig(
            vm_size="performance-2x",
            memory="2gb",
            cpu_kind="performance",
            cpus=2,
            gpu_kind="a100-pcie-40gb",
        )
        assert config.vm_size == "performance-2x"
        assert config.memory == "2gb"
        assert config.cpu_kind == "performance"
        assert config.cpus == 2
        assert config.gpu_kind == "a100-pcie-40gb"


class TestFlyVolumeConfig:
    """Tests for FlyVolumeConfig model."""

    def test_volume_config(self):
        """Test volume configuration."""
        config = FlyVolumeConfig(
            name="data_vol",
            size_gb=10,
            mount_path="/data",
        )
        assert config.name == "data_vol"
        assert config.size_gb == 10
        assert config.mount_path == "/data"


class TestFlyNetworkConfig:
    """Tests for FlyNetworkConfig model."""

    def test_default_config(self):
        """Test default network configuration."""
        config = FlyNetworkConfig()
        assert config.internal_port == 8080
        assert config.force_https is True
        assert config.auto_stop_machines is True
        assert config.auto_start_machines is True
        assert config.min_machines_running == 0
        assert config.max_machines is None

    def test_custom_config(self):
        """Test custom network configuration."""
        config = FlyNetworkConfig(
            internal_port=7777,
            force_https=False,
            auto_stop_machines=False,
            min_machines_running=1,
            max_machines=5,
        )
        assert config.internal_port == 7777
        assert config.force_https is False
        assert config.auto_stop_machines is False
        assert config.min_machines_running == 1
        assert config.max_machines == 5


class TestFlyState:
    """Tests for FlyState model."""

    def test_default_state(self):
        """Test default state."""
        state = FlyState()
        assert state.app_name is None
        assert state.app_url is None
        assert state.last_deployed is None
        assert state.status == DeploymentStatus.UNKNOWN
        assert state.machine_ids == []


class TestFlyDeployment:
    """Tests for FlyDeployment model."""

    def test_basic_deployment(self):
        """Test basic Fly.io deployment."""
        deployment = FlyDeployment(
            app_name="my-nodetool-app",
            region="iad",
        )
        assert deployment.type == DeploymentType.FLY
        assert deployment.enabled is True
        assert deployment.app_name == "my-nodetool-app"
        assert deployment.region == "iad"
        assert deployment.organization is None

    def test_deployment_with_all_options(self):
        """Test deployment with all options."""
        deployment = FlyDeployment(
            app_name="my-nodetool-app",
            organization="my-org",
            region="lax",
            image=FlyImageConfig(name="nodetool", tag="v1.0"),
            resources=FlyResourceConfig(memory="1gb", cpus=2),
            network=FlyNetworkConfig(internal_port=7777),
            volumes=[FlyVolumeConfig(name="data", size_gb=10, mount_path="/data")],
            environment={"LOG_LEVEL": "DEBUG"},
            secrets=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        )
        assert deployment.organization == "my-org"
        assert deployment.region == "lax"
        assert deployment.resources.memory == "1gb"
        assert deployment.network.internal_port == 7777
        assert len(deployment.volumes) == 1
        assert deployment.environment["LOG_LEVEL"] == "DEBUG"
        assert "OPENAI_API_KEY" in deployment.secrets

    def test_get_server_url_with_state(self):
        """Test get_server_url with state."""
        deployment = FlyDeployment(
            app_name="my-app",
            state=FlyState(app_url="https://my-app.fly.dev"),
        )
        assert deployment.get_server_url() == "https://my-app.fly.dev"

    def test_get_server_url_default(self):
        """Test get_server_url default pattern."""
        deployment = FlyDeployment(app_name="my-app")
        assert deployment.get_server_url() == "https://my-app.fly.dev"


class TestDeploymentConfigWithFly:
    """Tests for DeploymentConfig with Fly.io deployments."""

    def test_config_with_fly_deployment(self):
        """Test configuration with Fly.io deployment."""
        config = DeploymentConfig(
            deployments={
                "fly-prod": FlyDeployment(
                    app_name="nodetool-prod",
                    region="iad",
                ),
            }
        )
        assert len(config.deployments) == 1
        assert "fly-prod" in config.deployments
        assert isinstance(config.deployments["fly-prod"], FlyDeployment)

    def test_yaml_serialization(self):
        """Test YAML serialization with Fly.io deployment."""
        config = DeploymentConfig(
            deployments={
                "fly-test": FlyDeployment(
                    app_name="nodetool-test",
                    region="iad",
                    environment={"LOG_LEVEL": "INFO"},
                ),
            }
        )

        # Serialize to dict
        data = config.model_dump(mode="json", exclude_none=True)

        # Should be serializable to YAML
        yaml_str = yaml.dump(data, default_flow_style=False)
        assert "fly-test" in yaml_str
        assert "nodetool-test" in yaml_str

        # Deserialize back
        loaded_data = yaml.safe_load(yaml_str)
        loaded_config = DeploymentConfig.model_validate(loaded_data)

        assert "fly-test" in loaded_config.deployments
        assert loaded_config.deployments["fly-test"].app_name == "nodetool-test"


class TestGenerateFlyToml:
    """Tests for fly.toml generation."""

    def test_basic_fly_toml(self):
        """Test basic fly.toml generation."""
        deployment = FlyDeployment(
            app_name="test-app",
            region="iad",
        )
        toml_content = generate_fly_toml(deployment)

        assert 'app = "test-app"' in toml_content
        assert 'primary_region = "iad"' in toml_content
        assert "[http_service]" in toml_content
        assert "internal_port = 8080" in toml_content
        assert "[[vm]]" in toml_content

    def test_fly_toml_with_env(self):
        """Test fly.toml generation with environment variables."""
        deployment = FlyDeployment(
            app_name="test-app",
            region="iad",
            environment={"LOG_LEVEL": "DEBUG", "WORKERS": "4"},
        )
        toml_content = generate_fly_toml(deployment)

        assert "[env]" in toml_content
        assert 'LOG_LEVEL = "DEBUG"' in toml_content
        assert 'WORKERS = "4"' in toml_content

    def test_fly_toml_with_volumes(self):
        """Test fly.toml generation with volumes."""
        deployment = FlyDeployment(
            app_name="test-app",
            region="iad",
            volumes=[
                FlyVolumeConfig(name="data", size_gb=10, mount_path="/data"),
            ],
        )
        toml_content = generate_fly_toml(deployment)

        assert "[[mounts]]" in toml_content
        assert 'source = "data"' in toml_content
        assert 'destination = "/data"' in toml_content

    def test_fly_toml_with_gpu(self):
        """Test fly.toml generation with GPU."""
        deployment = FlyDeployment(
            app_name="test-app",
            region="iad",
            resources=FlyResourceConfig(gpu_kind="a100-pcie-40gb"),
        )
        toml_content = generate_fly_toml(deployment)

        assert 'gpu_kind = "a100-pcie-40gb"' in toml_content


class TestRunFlyctl:
    """Tests for run_flyctl function."""

    @patch("subprocess.run")
    def test_run_flyctl_success(self, mock_run):
        """Test successful flyctl command."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "version"],
            returncode=0,
            stdout="flyctl v1.0.0",
            stderr="",
        )

        result = run_flyctl(["version"])
        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_flyctl_not_installed(self, mock_run):
        """Test flyctl not installed."""
        mock_run.side_effect = FileNotFoundError("flyctl not found")

        with pytest.raises(FileNotFoundError, match="flyctl is not installed"):
            run_flyctl(["version"])


class TestCheckFlyctl:
    """Tests for flyctl check functions."""

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_check_flyctl_installed_true(self, mock_run):
        """Test flyctl is installed."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "version"],
            returncode=0,
            stdout="flyctl v1.0.0",
            stderr="",
        )
        assert check_flyctl_installed() is True

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_check_flyctl_installed_false(self, mock_run):
        """Test flyctl is not installed."""
        mock_run.side_effect = FileNotFoundError("Not found")
        assert check_flyctl_installed() is False

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_check_flyctl_authenticated_true(self, mock_run):
        """Test flyctl is authenticated."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "auth", "whoami"],
            returncode=0,
            stdout="user@example.com",
            stderr="",
        )
        assert check_flyctl_authenticated() is True

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_check_flyctl_authenticated_false(self, mock_run):
        """Test flyctl is not authenticated."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "auth", "whoami"],
            returncode=1,
            stdout="",
            stderr="Not authenticated",
        )
        assert check_flyctl_authenticated() is False


class TestFlyDeployer:
    """Tests for FlyDeployer class."""

    def test_deployer_init(self):
        """Test deployer initialization."""
        deployment = FlyDeployment(app_name="test-app", region="iad")
        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
        )
        assert deployer.deployment_name == "test"
        assert deployer.deployment.app_name == "test-app"

    def test_plan_initial(self):
        """Test plan for initial deployment."""
        deployment = FlyDeployment(
            app_name="test-app",
            region="iad",
            volumes=[FlyVolumeConfig(name="data", size_gb=5, mount_path="/data")],
            secrets=["API_KEY"],
        )

        # Mock state manager
        mock_state_manager = MagicMock()
        mock_state_manager.read_state.return_value = None

        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        with patch("nodetool.deploy.fly.check_flyctl_installed", return_value=True):
            plan = deployer.plan()

        assert plan["deployment_name"] == "test"
        assert plan["type"] == "fly"
        assert plan["app_name"] == "test-app"
        assert "Initial deployment" in plan["changes"][0]
        assert any("test-app" in item for item in plan["will_create"])

    def test_plan_flyctl_not_installed(self):
        """Test plan when flyctl is not installed."""
        deployment = FlyDeployment(app_name="test-app", region="iad")

        mock_state_manager = MagicMock()
        mock_state_manager.read_state.return_value = None

        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        with patch("nodetool.deploy.fly.check_flyctl_installed", return_value=False):
            plan = deployer.plan()

        assert any("flyctl not installed" in change for change in plan["changes"])

    @patch("nodetool.deploy.fly.check_flyctl_installed", return_value=False)
    def test_apply_no_flyctl(self, mock_check):
        """Test apply when flyctl is not installed."""
        deployment = FlyDeployment(app_name="test-app", region="iad")

        mock_state_manager = MagicMock()
        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        with pytest.raises(RuntimeError, match="flyctl is not installed"):
            deployer.apply()

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_status(self, mock_run):
        """Test status retrieval."""
        deployment = FlyDeployment(app_name="test-app", region="iad")

        mock_state_manager = MagicMock()
        mock_state_manager.read_state.return_value = {
            "status": "running",
            "last_deployed": "2024-01-01T00:00:00Z",
            "app_url": "https://test-app.fly.dev",
        }

        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        # Mock flyctl status
        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "status"],
            returncode=0,
            stdout='{"Status": "running", "Hostname": "test-app.fly.dev"}',
            stderr="",
        )

        status = deployer.status()

        assert status["deployment_name"] == "test"
        assert status["type"] == "fly"
        assert status["app_name"] == "test-app"
        assert status["status"] == "running"

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_logs(self, mock_run):
        """Test logs retrieval."""
        deployment = FlyDeployment(app_name="test-app", region="iad")
        deployer = FlyDeployer(deployment_name="test", deployment=deployment)

        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "logs"],
            returncode=0,
            stdout="2024-01-01T00:00:00Z app[abc123] Starting...",
            stderr="",
        )

        logs = deployer.logs()

        assert "Starting" in logs
        mock_run.assert_called_once()

    def test_logs_follow_mode(self):
        """Test logs follow mode returns instruction."""
        deployment = FlyDeployment(app_name="test-app", region="iad")
        deployer = FlyDeployer(deployment_name="test", deployment=deployment)

        logs = deployer.logs(follow=True)

        assert "Follow mode" in logs
        assert "flyctl logs" in logs

    @patch("nodetool.deploy.fly.run_flyctl")
    def test_destroy(self, mock_run):
        """Test destroy operation."""
        deployment = FlyDeployment(app_name="test-app", region="iad")

        mock_state_manager = MagicMock()
        deployer = FlyDeployer(
            deployment_name="test",
            deployment=deployment,
            state_manager=mock_state_manager,
        )

        mock_run.return_value = subprocess.CompletedProcess(
            args=["flyctl", "apps", "destroy"],
            returncode=0,
            stdout="",
            stderr="",
        )

        results = deployer.destroy()

        assert results["status"] == "success"
        assert any("Deleted" in step or "test-app" in step for step in results["steps"])
        mock_state_manager.update_deployment_status.assert_called_once()
