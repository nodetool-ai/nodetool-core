"""
Tests for persistent_paths configuration in deployment.

Tests that persistent_paths are correctly applied to environment variables
and volume mounts for Docker, GCP, and RunPod deployments.
"""

import pytest

from nodetool.config.deployment import (
    ContainerConfig,
    ImageConfig,
    PersistentPaths,
    SelfHostedDockerDeployment,
    SelfHostedPaths,
    SSHConfig,
)
from nodetool.deploy.docker_run import DockerRunGenerator


class TestDockerPersistentPaths:
    """Tests for Docker deployment with persistent_paths."""

    @pytest.fixture
    def deployment_with_persistent_paths(self):
        """Create a deployment with persistent_paths configured."""
        return SelfHostedDockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
            persistent_paths=PersistentPaths(
                users_file="/data/users.yaml",
                db_path="/data/nodetool.db",
                chroma_path="/data/chroma",
                hf_cache="/data/hf-cache",
                asset_bucket="/data/assets",
            ),
            paths=SelfHostedPaths(workspace="/data/workspace", hf_cache="/data/hf-cache"),
        )

    @pytest.fixture
    def deployment_without_persistent_paths(self):
        """Create a deployment without persistent_paths."""
        return SelfHostedDockerDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            container=ContainerConfig(name="default", port=7777),
            paths=SelfHostedPaths(workspace="/data/workspace", hf_cache="/data/hf-cache"),
        )

    def test_environment_with_persistent_paths(self, deployment_with_persistent_paths):
        """Test that environment variables are correctly set from persistent_paths."""
        generator = DockerRunGenerator(deployment_with_persistent_paths)
        env_vars = generator._build_environment()
        env_dict = dict(item.split("=", 1) for item in env_vars)

        assert env_dict.get("USERS_FILE") == "/data/users.yaml"
        assert env_dict.get("DB_PATH") == "/data/nodetool.db"
        assert env_dict.get("CHROMA_PATH") == "/data/chroma"
        assert env_dict.get("HF_HOME") == "/data/hf-cache"
        assert env_dict.get("ASSET_BUCKET") == "/data/assets"
        assert env_dict.get("AUTH_PROVIDER") == "multi_user"

    def test_environment_without_persistent_paths(self, deployment_without_persistent_paths):
        """Test that fallback environment variables are used without persistent_paths."""
        generator = DockerRunGenerator(deployment_without_persistent_paths)
        env_vars = generator._build_environment()
        env_dict = dict(item.split("=", 1) for item in env_vars)

        # Fallback values
        assert env_dict.get("DB_PATH") == "/workspace/nodetool.db"
        assert env_dict.get("HF_HOME") == "/hf-cache"
        # AUTH_PROVIDER should not be set to multi_user
        assert env_dict.get("AUTH_PROVIDER") != "multi_user"

    def test_volumes_with_persistent_paths(self, deployment_with_persistent_paths):
        """Test that hf_cache volume is writable with persistent_paths."""
        generator = DockerRunGenerator(deployment_with_persistent_paths)
        volumes = generator._build_volumes()

        # Check workspace volume
        assert "/data/workspace:/workspace" in volumes

        # Check hf_cache volume is writable (no :ro)
        hf_volume = next((v for v in volumes if "/hf-cache" in v), None)
        assert hf_volume is not None
        assert hf_volume == "/data/hf-cache:/hf-cache"
        assert ":ro" not in hf_volume

    def test_volumes_without_persistent_paths(self, deployment_without_persistent_paths):
        """Test that hf_cache volume is read-only without persistent_paths."""
        generator = DockerRunGenerator(deployment_without_persistent_paths)
        volumes = generator._build_volumes()

        # Check hf_cache volume is read-only
        hf_volume = next((v for v in volumes if "/hf-cache" in v), None)
        assert hf_volume is not None
        assert ":ro" in hf_volume

    def test_persistent_paths_defaults(self):
        """Test that PersistentPaths has sensible defaults."""
        paths = PersistentPaths()

        assert paths.users_file == "/workspace/users.yaml"
        assert paths.db_path == "/workspace/nodetool.db"
        assert paths.chroma_path == "/workspace/chroma"
        assert paths.hf_cache == "/workspace/hf-cache"
        assert paths.asset_bucket == "/workspace/assets"

    def test_docker_run_command_includes_persistent_paths(self, deployment_with_persistent_paths):
        """Test that generated docker run command includes persistent_paths env vars."""
        generator = DockerRunGenerator(deployment_with_persistent_paths)
        command = generator.generate_command()

        assert "USERS_FILE=/data/users.yaml" in command
        assert "DB_PATH=/data/nodetool.db" in command
        assert "CHROMA_PATH=/data/chroma" in command
        assert "ASSET_BUCKET=/data/assets" in command
        assert "AUTH_PROVIDER=multi_user" in command
