"""
Unit tests for Docker Compose generator.
"""

import pytest
import yaml

from nodetool.config.deployment import (
    ContainerConfig,
    ImageConfig,
    SelfHostedDeployment,
    SelfHostedPaths,
    SSHConfig,
)
from nodetool.deploy.compose import (
    ComposeGenerator,
    generate_compose_file,
    get_compose_hash,
)

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestComposeGenerator:
    """Tests for ComposeGenerator class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic deployment configuration."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
        )

    @pytest.fixture
    def multi_container_deployment(self):
        """Create a multi-container deployment configuration."""
        return SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001, workflows=["workflow-1"]),
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

    def test_init(self, basic_deployment):
        """Test ComposeGenerator initialization."""
        generator = ComposeGenerator(basic_deployment)

        assert generator.deployment == basic_deployment

    def test_generate_basic(self, basic_deployment):
        """Test generating basic docker-compose.yml."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()

        assert isinstance(content, str)
        assert "version:" in content
        assert "services:" in content
        assert "wf1:" in content

    def test_generate_valid_yaml(self, basic_deployment):
        """Test that generated content is valid YAML."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()

        # Should parse as valid YAML
        parsed = yaml.safe_load(content)

        assert isinstance(parsed, dict)
        assert "version" in parsed
        assert "services" in parsed

    def test_compose_version(self, basic_deployment):
        """Test docker-compose version."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        assert parsed["version"] == "3.8"

    def test_service_generation(self, basic_deployment):
        """Test service generation for a single container."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        assert "services" in parsed
        assert "wf1" in parsed["services"]

        service = parsed["services"]["wf1"]
        assert service["image"] == "nodetool/nodetool:latest"
        assert service["container_name"] == "nodetool-wf1"
        assert "8001:7777" in service["ports"]
        assert service["restart"] == "unless-stopped"

    def test_multi_container_services(self, multi_container_deployment):
        """Test generating service with workflow configuration."""
        generator = ComposeGenerator(multi_container_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        services = parsed["services"]
        assert len(services) == 1
        assert "wf1" in services

        # Check port is correctly mapped
        assert "8001:7777" in services["wf1"]["ports"]

    def test_volume_mounts(self, basic_deployment):
        """Test volume mount configuration."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        service = parsed["services"]["wf1"]
        volumes = service["volumes"]

        # Should have workspace and hf-cache volumes
        assert len(volumes) == 2

        # Check workspace volume (read-write)
        workspace_vol = next(v for v in volumes if "/workspace" in v)
        assert "/data/workspace:/workspace" in workspace_vol

        # Check HF cache volume (read-only)
        hf_vol = next(v for v in volumes if "/hf-cache:ro" in v)
        assert "/data/hf-cache:/hf-cache:ro" in hf_vol

    def test_environment_variables(self, basic_deployment):
        """Test environment variable configuration."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        service = parsed["services"]["wf1"]
        env = service["environment"]

        # Convert list of KEY=value to dict for easier testing
        env_dict = dict(e.split("=", 1) for e in env)

        assert "PORT" in env_dict
        assert env_dict["PORT"] == "8000"
        assert "NODETOOL_API_URL" in env_dict
        assert env_dict["NODETOOL_API_URL"] == "http://localhost:8001"

    def test_workflow_environment(self, multi_container_deployment):
        """Test workflow IDs in environment variables."""
        generator = ComposeGenerator(multi_container_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        wf1_env = parsed["services"]["wf1"]["environment"]
        wf1_env_dict = dict(e.split("=", 1) for e in wf1_env)

        # Container with workflows should have NODETOOL_WORKFLOWS
        assert "NODETOOL_WORKFLOWS" in wf1_env_dict
        assert wf1_env_dict["NODETOOL_WORKFLOWS"] == "workflow-1"

    def test_healthcheck_configuration(self, basic_deployment):
        """Test healthcheck configuration."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        service = parsed["services"]["wf1"]
        healthcheck = service["healthcheck"]

        assert healthcheck["test"] == [
            "CMD",
            "curl",
            "-f",
            "http://localhost:7777/health",
        ]
        assert healthcheck["interval"] == "30s"
        assert healthcheck["timeout"] == "10s"
        assert healthcheck["retries"] == 3
        assert healthcheck["start_period"] == "40s"

    def test_gpu_configuration(self, gpu_deployment):
        """Test GPU resource configuration."""
        generator = ComposeGenerator(gpu_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        # Check container with single GPU
        gpu1 = parsed["services"]["gpu1"]
        assert "deploy" in gpu1
        deploy = gpu1["deploy"]
        assert "resources" in deploy
        assert "reservations" in deploy["resources"]
        devices = deploy["resources"]["reservations"]["devices"]
        assert len(devices) == 1
        assert devices[0]["driver"] == "nvidia"
        assert devices[0]["device_ids"] == ["0"]
        assert "gpu" in devices[0]["capabilities"]

    def test_no_gpu_configuration(self, basic_deployment):
        """Test that deploy section is omitted when no GPU."""
        generator = ComposeGenerator(basic_deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        service = parsed["services"]["wf1"]
        assert "deploy" not in service

    def test_sanitize_service_name(self, basic_deployment):
        """Test service name sanitization."""
        generator = ComposeGenerator(basic_deployment)

        # Test valid names
        assert generator._sanitize_service_name("wf1") == "wf1"
        assert generator._sanitize_service_name("WF1") == "wf1"
        assert generator._sanitize_service_name("wf_1") == "wf_1"
        assert generator._sanitize_service_name("wf-1") == "wf-1"

        # Test invalid characters
        assert generator._sanitize_service_name("wf.1") == "wf-1"
        assert generator._sanitize_service_name("wf@1") == "wf-1"
        assert generator._sanitize_service_name("wf 1") == "wf-1"

        # Test starting with non-alphanumeric
        assert generator._sanitize_service_name("_wf1") == "c_wf1"
        assert generator._sanitize_service_name("-wf1") == "c-wf1"

    def test_generate_hash(self, basic_deployment):
        """Test hash generation for change detection."""
        generator = ComposeGenerator(basic_deployment)
        hash1 = generator.generate_hash()

        # Hash should be a valid SHA256 hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

        # Same config should produce same hash
        generator2 = ComposeGenerator(basic_deployment)
        hash2 = generator2.generate_hash()
        assert hash1 == hash2

    def test_hash_changes_with_config(self):
        """Test that hash changes when configuration changes."""
        deployment1 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
        )

        deployment2 = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8002),
        )

        hash1 = ComposeGenerator(deployment1).generate_hash()
        hash2 = ComposeGenerator(deployment2).generate_hash()

        assert hash1 != hash2

    def test_custom_paths(self):
        """Test custom path configuration."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
            paths=SelfHostedPaths(
                workspace="/custom/workspace",
                hf_cache="/custom/hf-cache",
            ),
        )

        generator = ComposeGenerator(deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        volumes = parsed["services"]["wf1"]["volumes"]

        # Check custom paths are used
        workspace_vol = next(v for v in volumes if "/workspace" in v)
        assert "/custom/workspace:/workspace" in workspace_vol

        hf_vol = next(v for v in volumes if "/hf-cache:ro" in v)
        assert "/custom/hf-cache:/hf-cache:ro" in hf_vol


class TestComposeHelperFunctions:
    """Tests for compose helper functions."""

    def test_generate_compose_file_without_output(self):
        """Test generating compose file without writing to disk."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
        )

        content = generate_compose_file(deployment)

        assert isinstance(content, str)
        assert "version:" in content
        assert "services:" in content

    def test_generate_compose_file_with_output(self, tmp_path):
        """Test generating compose file with output to disk."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
        )

        output_path = tmp_path / "docker-compose.yml"
        content = generate_compose_file(deployment, str(output_path))

        # File should be created
        assert output_path.exists()

        # Content should match
        with open(output_path) as f:
            file_content = f.read()

        assert file_content == content

        # Should be valid YAML
        with open(output_path) as f:
            parsed = yaml.safe_load(f)

        assert parsed["version"] == "3.8"
        assert "wf1" in parsed["services"]

    def test_get_compose_hash(self):
        """Test get_compose_hash helper function."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="wf1", port=8001),
        )

        hash_value = get_compose_hash(deployment)

        # Should be a valid SHA256 hex string
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

        # Should match direct generator hash
        generator = ComposeGenerator(deployment)
        assert hash_value == generator.generate_hash()


class TestComposeEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_container_with_special_characters(self):
        """Test container with special characters in name."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="workflow.1", port=8001),
        )

        generator = ComposeGenerator(deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        services = parsed["services"]

        # Name should be sanitized
        assert "workflow-1" in services
        assert len(services) == 1

    def test_multiple_workflows_per_container(self):
        """Test container with multiple workflow IDs."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="multi", port=8001, workflows=["wf-1", "wf-2", "wf-3"]),
        )

        generator = ComposeGenerator(deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        env = parsed["services"]["multi"]["environment"]
        env_dict = dict(e.split("=", 1) for e in env)

        # Workflows should be comma-separated
        assert "NODETOOL_WORKFLOWS" in env_dict
        assert env_dict["NODETOOL_WORKFLOWS"] == "wf-1,wf-2,wf-3"

    def test_container_with_custom_environment(self):
        """Test container with custom environment variables."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="custom", port=8001, environment={"CUSTOM_VAR": "custom_value"}),
        )

        generator = ComposeGenerator(deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        env = parsed["services"]["custom"]["environment"]
        env_dict = dict(e.split("=", 1) for e in env)

        # Custom environment should be included
        assert "CUSTOM_VAR" in env_dict
        assert env_dict["CUSTOM_VAR"] == "custom_value"

    def test_default_container(self):
        """Test deployment with default container."""
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
        )

        generator = ComposeGenerator(deployment)
        content = generator.generate()
        parsed = yaml.safe_load(content)

        # Should have valid structure with one service
        assert "services" in parsed
        assert len(parsed["services"]) == 1
        assert "default" in parsed["services"]
