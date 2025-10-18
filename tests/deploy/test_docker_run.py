"""
Unit tests for Docker run command generation.
"""

import pytest

from nodetool.deploy.docker_run import (
    DockerRunGenerator,
    generate_docker_run_command,
    get_docker_run_hash,
    get_container_name,
)
from nodetool.config.deployment import (
    SelfHostedDeployment,
    SSHConfig,
    ImageConfig,
    ContainerConfig,
    SelfHostedPaths,
)


# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestDockerRunGenerator:
    """Tests for DockerRunGenerator class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic deployment configuration."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
        )

    @pytest.fixture
    def gpu_deployment(self):
        """Create a deployment with GPU configuration."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="gpu1", port=8001, gpu="0"),
        )

    @pytest.fixture
    def multi_gpu_deployment(self):
        """Create a deployment with multiple GPUs."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="gpu-multi", port=8002, gpu="0,1,2"),
        )

    @pytest.fixture
    def workflow_deployment(self):
        """Create a deployment with workflow IDs."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="wf1",
                port=8001,
                workflows=["workflow-abc", "workflow-def"],
            ),
        )

    @pytest.fixture
    def custom_env_deployment(self):
        """Create a deployment with custom environment variables."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="custom",
                port=8000,
                environment={"CUSTOM_VAR": "value", "DEBUG": "true"},
            ),
        )

    def test_init(self, basic_deployment):
        """Test generator initialization."""
        generator = DockerRunGenerator(basic_deployment)

        assert generator.deployment == basic_deployment
        assert generator.container == basic_deployment.container

    def test_generate_command_basic(self, basic_deployment):
        """Test basic command generation."""
        generator = DockerRunGenerator(basic_deployment)
        command = generator.generate_command()

        # Check essential parts
        assert "docker run" in command
        assert "-d" in command
        assert "--name nodetool-default" in command
        assert "--restart unless-stopped" in command
        assert "-p 8000:8000" in command
        assert "nodetool/nodetool:latest" in command

    def test_generate_command_volumes(self, basic_deployment):
        """Test volume mount generation."""
        generator = DockerRunGenerator(basic_deployment)
        command = generator.generate_command()

        # Check volume mounts
        assert "-v /data/workspace:/workspace" in command
        assert "-v /data/hf-cache:/hf-cache:ro" in command

    def test_generate_command_environment(self, basic_deployment):
        """Test environment variable generation."""
        generator = DockerRunGenerator(basic_deployment)
        command = generator.generate_command()

        # Check environment variables
        assert "-e PORT=8000" in command
        assert "-e NODETOOL_API_URL=http://localhost:8000" in command
        assert "-e DB_PATH=/workspace/nodetool.db" in command
        assert "-e HF_HOME=/hf-cache" in command

    def test_generate_command_healthcheck(self, basic_deployment):
        """Test health check configuration."""
        generator = DockerRunGenerator(basic_deployment)
        command = generator.generate_command()

        # Check health check
        assert "--health-cmd" in command
        assert "curl -f http://localhost:8000/health" in command
        assert "--health-interval=30s" in command
        assert "--health-timeout=10s" in command
        assert "--health-retries=3" in command
        assert "--health-start-period=40s" in command

    def test_generate_command_gpu_single(self, gpu_deployment):
        """Test GPU configuration for single GPU."""
        generator = DockerRunGenerator(gpu_deployment)
        command = generator.generate_command()

        # Check GPU configuration
        assert "--gpus" in command
        assert "device=0" in command

    def test_generate_command_gpu_multiple(self, multi_gpu_deployment):
        """Test GPU configuration for multiple GPUs."""
        generator = DockerRunGenerator(multi_gpu_deployment)
        command = generator.generate_command()

        # Check GPU configuration
        assert "--gpus" in command
        assert "device=0,1,2" in command

    def test_generate_command_workflows(self, workflow_deployment):
        """Test workflow IDs in environment."""
        generator = DockerRunGenerator(workflow_deployment)
        command = generator.generate_command()

        # Check workflow environment variable
        assert "-e NODETOOL_WORKFLOWS=workflow-abc,workflow-def" in command

    def test_generate_command_custom_environment(self, custom_env_deployment):
        """Test custom environment variables."""
        generator = DockerRunGenerator(custom_env_deployment)
        command = generator.generate_command()

        # Check custom environment variables
        assert "-e CUSTOM_VAR=value" in command
        assert "-e DEBUG=true" in command

    def test_generate_command_custom_port(self):
        """Test custom port mapping."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="custom", port=9000),
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        # Port should map host port to container port 8000
        assert "-p 9000:8000" in command
        assert "-e NODETOOL_API_URL=http://localhost:9000" in command

    def test_generate_command_custom_paths(self):
        """Test custom volume paths."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
            paths=SelfHostedPaths(
                workspace="/custom/workspace",
                hf_cache="/custom/cache",
            ),
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        # Check custom paths
        assert "-v /custom/workspace:/workspace" in command
        assert "-v /custom/cache:/hf-cache:ro" in command

    def test_generate_command_worker_auth_token(self):
        """Test worker authentication token."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
            worker_auth_token="secret-token-123",
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        # Check auth token
        assert "-e WORKER_AUTH_TOKEN=secret-token-123" in command

    def test_generate_command_format(self, basic_deployment):
        """Test command format with line continuations."""
        generator = DockerRunGenerator(basic_deployment)
        command = generator.generate_command()

        # Should use line continuation
        assert " \\\n  " in command

        # Each line should start with a flag or command
        lines = command.split(" \\\n  ")
        assert lines[0].startswith("docker run")

    def test_generate_hash_basic(self, basic_deployment):
        """Test hash generation."""
        generator = DockerRunGenerator(basic_deployment)
        hash_value = generator.generate_hash()

        # Should be a valid SHA256 hash
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_generate_hash_consistency(self, basic_deployment):
        """Test hash consistency for same configuration."""
        generator1 = DockerRunGenerator(basic_deployment)
        generator2 = DockerRunGenerator(basic_deployment)

        hash1 = generator1.generate_hash()
        hash2 = generator2.generate_hash()

        # Same configuration should produce same hash
        assert hash1 == hash2

    def test_generate_hash_different_configs(self):
        """Test hash changes with different configurations."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="v2.0"),  # Different tag
            container=ContainerConfig(name="default", port=8000),
        )

        hash1 = DockerRunGenerator(deployment1).generate_hash()
        hash2 = DockerRunGenerator(deployment2).generate_hash()

        # Different configs should produce different hashes
        assert hash1 != hash2

    def test_generate_hash_port_change(self):
        """Test hash changes when port changes."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=9000),  # Different port
        )

        hash1 = DockerRunGenerator(deployment1).generate_hash()
        hash2 = DockerRunGenerator(deployment2).generate_hash()

        assert hash1 != hash2

    def test_generate_hash_gpu_change(self):
        """Test hash changes when GPU config changes."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000, gpu="0"),
        )

        hash1 = DockerRunGenerator(deployment1).generate_hash()
        hash2 = DockerRunGenerator(deployment2).generate_hash()

        assert hash1 != hash2

    def test_generate_hash_environment_change(self):
        """Test hash changes when environment changes."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="default",
                port=8000,
                environment={"NEW_VAR": "value"},
            ),
        )

        hash1 = DockerRunGenerator(deployment1).generate_hash()
        hash2 = DockerRunGenerator(deployment2).generate_hash()

        assert hash1 != hash2

    def test_get_container_name(self, basic_deployment):
        """Test container name generation."""
        generator = DockerRunGenerator(basic_deployment)
        name = generator.get_container_name()

        assert name == "nodetool-default"

    def test_get_container_name_custom(self):
        """Test container name with custom name."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="my-workflow", port=8000),
        )

        generator = DockerRunGenerator(deployment)
        name = generator.get_container_name()

        assert name == "nodetool-my-workflow"

    def test_build_volumes(self, basic_deployment):
        """Test volume building."""
        generator = DockerRunGenerator(basic_deployment)
        volumes = generator._build_volumes()

        assert len(volumes) == 2
        assert "/data/workspace:/workspace" in volumes
        assert "/data/hf-cache:/hf-cache:ro" in volumes

    def test_build_volumes_custom_paths(self):
        """Test volume building with custom paths."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
            paths=SelfHostedPaths(
                workspace="/mnt/workspace",
                hf_cache="/mnt/cache",
            ),
        )

        generator = DockerRunGenerator(deployment)
        volumes = generator._build_volumes()

        assert "/mnt/workspace:/workspace" in volumes
        assert "/mnt/cache:/hf-cache:ro" in volumes

    def test_build_environment_basic(self, basic_deployment):
        """Test environment variable building."""
        generator = DockerRunGenerator(basic_deployment)
        env = generator._build_environment()

        # Check for required variables
        assert "PORT=8000" in env
        assert "NODETOOL_API_URL=http://localhost:8000" in env
        assert "DB_PATH=/workspace/nodetool.db" in env
        assert "HF_HOME=/hf-cache" in env

    def test_build_environment_with_custom(self, custom_env_deployment):
        """Test environment building with custom variables."""
        generator = DockerRunGenerator(custom_env_deployment)
        env = generator._build_environment()

        # Check custom variables
        assert "CUSTOM_VAR=value" in env
        assert "DEBUG=true" in env

        # Check default variables still present
        assert "PORT=8000" in env

    def test_build_environment_with_workflows(self, workflow_deployment):
        """Test environment building with workflows."""
        generator = DockerRunGenerator(workflow_deployment)
        env = generator._build_environment()

        # Check workflow variable
        assert "NODETOOL_WORKFLOWS=workflow-abc,workflow-def" in env

    def test_build_environment_without_workflows(self, basic_deployment):
        """Test environment building without workflows."""
        generator = DockerRunGenerator(basic_deployment)
        env = generator._build_environment()

        # Should not have NODETOOL_WORKFLOWS
        assert not any("NODETOOL_WORKFLOWS" in e for e in env)

    def test_build_environment_with_auth_token(self):
        """Test environment building with auth token."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000),
            worker_auth_token="token123",
        )

        generator = DockerRunGenerator(deployment)
        env = generator._build_environment()

        assert "WORKER_AUTH_TOKEN=token123" in env

    def test_build_gpu_args_none(self, basic_deployment):
        """Test GPU args when no GPU configured."""
        generator = DockerRunGenerator(basic_deployment)
        gpu_args = generator._build_gpu_args()

        assert gpu_args == []

    def test_build_gpu_args_single(self, gpu_deployment):
        """Test GPU args for single GPU."""
        generator = DockerRunGenerator(gpu_deployment)
        gpu_args = generator._build_gpu_args()

        assert len(gpu_args) == 1
        assert "--gpus" in gpu_args[0]
        assert "device=0" in gpu_args[0]

    def test_build_gpu_args_multiple(self, multi_gpu_deployment):
        """Test GPU args for multiple GPUs."""
        generator = DockerRunGenerator(multi_gpu_deployment)
        gpu_args = generator._build_gpu_args()

        assert len(gpu_args) == 1
        assert "--gpus" in gpu_args[0]
        assert "device=0,1,2" in gpu_args[0]

    def test_build_gpu_args_with_spaces(self):
        """Test GPU args with spaces in specification."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=8000, gpu=" 0,1 "),
        )

        generator = DockerRunGenerator(deployment)
        gpu_args = generator._build_gpu_args()

        # Should strip spaces
        assert "device=0,1" in gpu_args[0]


class TestDockerRunHelperFunctions:
    """Tests for helper functions."""

    def test_generate_docker_run_command(self):
        """Test generate_docker_run_command helper."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="test", port=8000),
        )

        command = generate_docker_run_command(deployment)

        assert "docker run" in command
        assert "nodetool-test" in command

    def test_get_docker_run_hash(self):
        """Test get_docker_run_hash helper."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="test", port=8000),
        )

        hash_value = get_docker_run_hash(deployment)

        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_get_container_name_helper(self):
        """Test get_container_name helper."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="my-container", port=8000),
        )

        name = get_container_name(deployment)

        assert name == "nodetool-my-container"


class TestDockerRunEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_environment(self):
        """Test with empty environment dict."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="test", port=8000, environment={}),
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        # Should still have default environment variables
        assert "-e PORT=8000" in command

    def test_empty_workflows(self):
        """Test with empty workflows list."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="test", port=8000, workflows=[]),
        )

        generator = DockerRunGenerator(deployment)
        env = generator._build_environment()

        # Should not have NODETOOL_WORKFLOWS
        assert not any("NODETOOL_WORKFLOWS" in e for e in env)

    def test_single_workflow(self):
        """Test with single workflow."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="test", port=8000, workflows=["wf1"]),
        )

        generator = DockerRunGenerator(deployment)
        env = generator._build_environment()

        assert "NODETOOL_WORKFLOWS=wf1" in env

    def test_special_characters_in_container_name(self):
        """Test container name with special characters."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="my_workflow-v2", port=8000),
        )

        generator = DockerRunGenerator(deployment)
        name = generator.get_container_name()

        # Should preserve underscores and dashes
        assert name == "nodetool-my_workflow-v2"

    def test_registry_in_image_name(self):
        """Test image name with registry."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(
                name="ghcr.io/user/nodetool",
                tag="latest",
            ),
            container=ContainerConfig(name="test", port=8000),
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        assert "ghcr.io/user/nodetool:latest" in command

    def test_image_tag_with_sha(self):
        """Test image tag with SHA."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(
                name="nodetool/nodetool",
                tag="sha256:abc123def456",
            ),
            container=ContainerConfig(name="test", port=8000),
        )

        generator = DockerRunGenerator(deployment)
        command = generator.generate_command()

        assert "nodetool/nodetool:sha256:abc123def456" in command

    def test_hash_determinism(self):
        """Test that hash is deterministic across multiple calls."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="test",
                port=8000,
                environment={"VAR1": "value1", "VAR2": "value2"},
            ),
        )

        generator = DockerRunGenerator(deployment)

        # Generate hash multiple times
        hashes = [generator.generate_hash() for _ in range(5)]

        # All hashes should be identical
        assert len(set(hashes)) == 1

    def test_environment_order_independence(self):
        """Test that environment variable order doesn't affect hash."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="test",
                port=8000,
                environment={"A": "1", "B": "2", "C": "3"},
            ),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(
                name="test",
                port=8000,
                environment={"C": "3", "A": "1", "B": "2"},  # Different order
            ),
        )

        hash1 = DockerRunGenerator(deployment1).generate_hash()
        hash2 = DockerRunGenerator(deployment2).generate_hash()

        # Hashes should be the same (environment is sorted)
        assert hash1 == hash2
