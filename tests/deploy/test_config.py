"""
Unit tests for deployment configuration models and functions.
"""

import yaml

from nodetool.config.deployment import (
    ContainerConfig,
    DefaultsConfig,
    DeploymentConfig,
    DeploymentStatus,
    DeploymentType,
    DockerDeployment,
    ImageConfig,
    RootDeployment,
    SSHConfig,
)


class TestSSHConfig:
    """Tests for SSHConfig model."""

    def test_ssh_config_basic(self):
        """Test basic SSH configuration."""
        config = SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa", port=22)
        assert config.user == "ubuntu"
        assert config.port == 22
        # Path should be expanded
        assert "~" not in str(config.key_path)

    def test_ssh_config_with_password(self):
        """Test SSH configuration with password."""
        config = SSHConfig(user="admin", password="secret", port=2222)
        assert config.user == "admin"
        assert config.password == "secret"
        assert config.port == 2222

    def test_ssh_config_path_expansion(self):
        """Test that ~ is expanded in key path."""
        config = SSHConfig(user="user", key_path="~/keys/id_rsa")
        assert "~" not in str(config.key_path)
        assert str(config.key_path).startswith("/")


class TestImageConfig:
    """Tests for ImageConfig model."""

    def test_image_config_basic(self):
        """Test basic image configuration."""
        config = ImageConfig(name="nodetool/nodetool", tag="latest")
        assert config.name == "nodetool/nodetool"
        assert config.tag == "latest"
        assert config.registry == "docker.io"

    def test_image_full_name(self):
        """Test full image name property."""
        config = ImageConfig(name="myuser/myimage", tag="v1.0.0")
        assert config.full_name == "myuser/myimage:v1.0.0"

    def test_image_full_name_already_tagged(self):
        """Test full image name does not double-append tag."""
        config = ImageConfig(name="ghcr.io/nodetool-ai/nodetool:latest", tag="latest")
        assert config.full_name == "ghcr.io/nodetool-ai/nodetool:latest"


class TestContainerConfig:
    """Tests for ContainerConfig model."""

    def test_container_config_basic(self):
        """Test basic container configuration."""
        config = ContainerConfig(name="workflow1", port=8001, gpu="0", workflows=["wf1", "wf2"])
        assert config.name == "workflow1"
        assert config.port == 8001
        assert config.gpu == "0"
        assert len(config.workflows) == 2

    def test_container_config_no_gpu(self):
        """Test container without GPU."""
        config = ContainerConfig(name="cpu-only", port=8001)
        assert config.gpu is None
        assert config.workflows is None

    def test_container_config_multiple_gpus(self):
        """Test container with multiple GPUs."""
        config = ContainerConfig(name="multi-gpu", port=8001, gpu="0,1")
        assert config.gpu == "0,1"


class TestDockerDeployment:
    """Tests for DockerDeployment model."""

    def test_docker_deployment_basic(self):
        """Test basic Docker deployment."""
        deployment = DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
        )
        assert deployment.type == DeploymentType.DOCKER
        assert deployment.enabled is True
        assert deployment.host == "192.168.1.100"

    def test_docker_deployment_with_gpu(self):
        """Test Docker deployment with GPU."""
        deployment = DockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001, gpu="0", workflows=["abc123"]),
        )
        assert deployment.container.name == "wf1"
        assert deployment.container.port == 8001
        assert deployment.container.gpu == "0"

    def test_self_hosted_without_proxy(self):
        """Test self-hosted deployment configured without proxy."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
            use_proxy=False,
        )
        assert deployment.use_proxy is False
        assert deployment.proxy is None
        assert deployment.get_server_url() == "http://192.168.1.100:8001"

    def test_self_hosted_without_proxy_port_7777_maps_to_8000(self):
        """Test non-proxy URL follows host port remap used by docker run."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=7777),
            use_proxy=False,
        )
        assert deployment.get_server_url() == "http://192.168.1.100:8000"


class TestRootDeployment:
    """Tests for RootDeployment model."""

    def test_root_deployment_basic(self):
        """Test basic Root deployment."""
        deployment = RootDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            port=8000,
            service_name="nodetool-service",
        )
        assert deployment.type == DeploymentType.ROOT
        assert deployment.port == 8000
        assert deployment.service_name == "nodetool-service"


class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""

    def test_deployment_config_empty(self):
        """Test empty deployment configuration."""
        config = DeploymentConfig()
        assert config.version == "1.0"
        assert len(config.deployments) == 0
        assert config.defaults.chat_provider == "llama_cpp"

    def test_deployment_config_with_deployments(self):
        """Test deployment configuration with multiple deployments."""
        config = DeploymentConfig(
            deployments={
                "prod-docker": DockerDeployment(
                    host="192.168.1.100",
                    ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
                    image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                    container=ContainerConfig(name="default", port=7777),
                ),
                "prod-root": RootDeployment(
                    host="192.168.1.101",
                    ssh=SSHConfig(user="root", key_path="~/.ssh/id_rsa"),
                    port=8000,
                ),
            }
        )
        assert len(config.deployments) == 2
        assert "prod-docker" in config.deployments
        assert "prod-root" in config.deployments

    def test_deployment_config_custom_defaults(self):
        """Test deployment configuration with custom defaults."""
        config = DeploymentConfig(
            defaults=DefaultsConfig(
                chat_provider="openai",
                default_model="gpt-4",
                log_level="DEBUG",
                extra={"CUSTOM_VAR": "value"},
            )
        )
        assert config.defaults.chat_provider == "openai"
        assert config.defaults.default_model == "gpt-4"
        assert config.defaults.extra["CUSTOM_VAR"] == "value"


class TestConfigSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_yaml_serialization(self):
        """Test YAML serialization of deployment config."""
        config = DeploymentConfig(
            deployments={
                "test-docker": DockerDeployment(
                    host="localhost",
                    ssh=SSHConfig(user="test", key_path="/tmp/key"),
                    image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                    container=ContainerConfig(name="default", port=7777),
                ),
                "test-root": RootDeployment(
                    host="remote",
                    ssh=SSHConfig(user="root", key_path="/tmp/key"),
                    port=8000,
                )
            }
        )

        # Serialize to dict
        data = config.model_dump(mode="json", exclude_none=True)

        # Should be serializable to YAML
        yaml_str = yaml.dump(data, default_flow_style=False)
        assert "test-docker" in yaml_str
        assert "test-root" in yaml_str

        # Deserialize back
        loaded_data = yaml.safe_load(yaml_str)
        loaded_config = DeploymentConfig.model_validate(loaded_data)

        assert "test-docker" in loaded_config.deployments
        assert isinstance(loaded_config.deployments["test-docker"], DockerDeployment)
        assert "test-root" in loaded_config.deployments
        assert isinstance(loaded_config.deployments["test-root"], RootDeployment)


class TestConfigFileOperations:
    """Tests for config file loading and saving."""

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / "deployment.yaml"

        # Create config
        config = DeploymentConfig(
            deployments={
                "test-deployment": DockerDeployment(
                    host="192.168.1.100",
                    ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
                    image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                    container=ContainerConfig(name="default", port=7777),
                )
            }
        )

        # Save
        data = config.model_dump(mode="json", exclude_none=True)
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        # Load
        with open(config_path) as f:
            loaded_data = yaml.safe_load(f)

        loaded_config = DeploymentConfig.model_validate(loaded_data)

        assert "test-deployment" in loaded_config.deployments
        assert loaded_config.deployments["test-deployment"].host == "192.168.1.100"

    def test_atomic_write(self, tmp_path):
        """Test atomic write operation."""
        config_path = tmp_path / "deployment.yaml"
        temp_path = config_path.with_suffix(".tmp")

        config = DeploymentConfig()
        data = config.model_dump(mode="json", exclude_none=True)

        # Simulate atomic write
        with open(temp_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        temp_path.replace(config_path)

        assert config_path.exists()
        assert not temp_path.exists()


class TestDeploymentStatus:
    """Tests for DeploymentStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert DeploymentStatus.UNKNOWN.value == "unknown"
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.ERROR.value == "error"
        assert DeploymentStatus.DESTROYED.value == "destroyed"
