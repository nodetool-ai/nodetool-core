"""
Unit tests for HuggingFace deployment functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from nodetool.config.deployment import (
    DeploymentConfig,
    DeploymentType,
    HuggingFaceBuildConfig,
    HuggingFaceDeployment,
    HuggingFaceImageConfig,
    HuggingFaceResourceConfig,
    HuggingFaceState,
)
from nodetool.deploy.deploy_to_hf import (
    sanitize_endpoint_name,
)


class TestHuggingFaceBuildConfig:
    """Tests for HuggingFaceBuildConfig model."""

    def test_default_values(self):
        """Test default build configuration values."""
        config = HuggingFaceBuildConfig()
        assert config.platform == "linux/amd64"
        assert config.dockerfile == "Dockerfile.hf"

    def test_custom_values(self):
        """Test custom build configuration values."""
        config = HuggingFaceBuildConfig(platform="linux/arm64", dockerfile="custom.Dockerfile")
        assert config.platform == "linux/arm64"
        assert config.dockerfile == "custom.Dockerfile"


class TestHuggingFaceImageConfig:
    """Tests for HuggingFaceImageConfig model."""

    def test_default_values(self):
        """Test default image configuration values."""
        config = HuggingFaceImageConfig(repository="user/image")
        assert config.registry == "docker.io"
        assert config.repository == "user/image"
        assert config.tag == "latest"

    def test_full_name_docker_hub(self):
        """Test full name for Docker Hub images."""
        config = HuggingFaceImageConfig(
            registry="docker.io", repository="myuser/myimage", tag="v1.0"
        )
        assert config.full_name == "myuser/myimage:v1.0"

    def test_full_name_custom_registry(self):
        """Test full name for custom registry images."""
        config = HuggingFaceImageConfig(
            registry="ghcr.io", repository="myuser/myimage", tag="v1.0"
        )
        assert config.full_name == "ghcr.io/myuser/myimage:v1.0"


class TestHuggingFaceResourceConfig:
    """Tests for HuggingFaceResourceConfig model."""

    def test_default_values(self):
        """Test default resource configuration values."""
        config = HuggingFaceResourceConfig()
        assert config.instance_size == "small"
        assert config.instance_type == "intel-icl"
        assert config.min_replica == 0
        assert config.max_replica == 1

    def test_custom_values(self):
        """Test custom resource configuration values."""
        config = HuggingFaceResourceConfig(
            instance_size="large",
            instance_type="nvidia-a10g",
            min_replica=1,
            max_replica=5,
        )
        assert config.instance_size == "large"
        assert config.instance_type == "nvidia-a10g"
        assert config.min_replica == 1
        assert config.max_replica == 5

    def test_min_replica_validation(self):
        """Test that min_replica cannot be negative."""
        with pytest.raises(ValueError):
            HuggingFaceResourceConfig(min_replica=-1)

    def test_max_replica_validation(self):
        """Test that max_replica must be at least 1."""
        with pytest.raises(ValueError):
            HuggingFaceResourceConfig(max_replica=0)


class TestHuggingFaceState:
    """Tests for HuggingFaceState model."""

    def test_default_values(self):
        """Test default state values."""
        state = HuggingFaceState()
        assert state.endpoint_url is None
        assert state.endpoint_name is None
        assert state.last_deployed is None
        assert state.status.value == "unknown"
        assert state.revision is None


class TestHuggingFaceDeployment:
    """Tests for HuggingFaceDeployment model."""

    def test_basic_deployment(self):
        """Test basic HuggingFace deployment configuration."""
        deployment = HuggingFaceDeployment(
            namespace="my-namespace",
            endpoint_name="my-endpoint",
            image=HuggingFaceImageConfig(repository="user/image"),
        )
        assert deployment.type == DeploymentType.HUGGINGFACE
        assert deployment.enabled is True
        assert deployment.namespace == "my-namespace"
        assert deployment.endpoint_name == "my-endpoint"
        assert deployment.region == "us-east-1"
        assert deployment.vendor == "aws"
        assert deployment.task == "custom"
        assert deployment.custom_image is True

    def test_deployment_with_environment(self):
        """Test deployment with environment variables."""
        deployment = HuggingFaceDeployment(
            namespace="my-namespace",
            endpoint_name="my-endpoint",
            image=HuggingFaceImageConfig(repository="user/image"),
            environment={"KEY1": "value1", "KEY2": "value2"},
        )
        assert deployment.environment == {"KEY1": "value1", "KEY2": "value2"}

    def test_get_server_url_without_url(self):
        """Test get_server_url when no URL is set."""
        deployment = HuggingFaceDeployment(
            namespace="my-namespace",
            endpoint_name="my-endpoint",
            image=HuggingFaceImageConfig(repository="user/image"),
        )
        assert deployment.get_server_url() is None

    def test_get_server_url_with_url(self):
        """Test get_server_url when URL is set."""
        deployment = HuggingFaceDeployment(
            namespace="my-namespace",
            endpoint_name="my-endpoint",
            image=HuggingFaceImageConfig(repository="user/image"),
            state=HuggingFaceState(endpoint_url="https://my-endpoint.endpoints.huggingface.cloud"),
        )
        assert deployment.get_server_url() == "https://my-endpoint.endpoints.huggingface.cloud"


class TestDeploymentConfigWithHuggingFace:
    """Tests for DeploymentConfig with HuggingFace deployments."""

    def test_deployment_config_with_huggingface(self):
        """Test deployment configuration with HuggingFace deployment."""
        config = DeploymentConfig(
            deployments={
                "my-hf-deployment": HuggingFaceDeployment(
                    namespace="my-namespace",
                    endpoint_name="my-endpoint",
                    image=HuggingFaceImageConfig(repository="user/image"),
                )
            }
        )
        assert len(config.deployments) == 1
        assert "my-hf-deployment" in config.deployments
        assert config.deployments["my-hf-deployment"].type == DeploymentType.HUGGINGFACE


class TestSanitizeEndpointName:
    """Tests for endpoint name sanitization."""

    def test_lowercase_conversion(self):
        """Test that names are converted to lowercase."""
        assert sanitize_endpoint_name("MyEndpoint") == "myendpoint"

    def test_special_character_replacement(self):
        """Test that special characters are replaced with hyphens."""
        assert sanitize_endpoint_name("my_endpoint@test") == "my-endpoint-test"

    def test_consecutive_hyphens_removal(self):
        """Test that consecutive hyphens are replaced with single hyphen."""
        assert sanitize_endpoint_name("my---endpoint") == "my-endpoint"

    def test_leading_number_handling(self):
        """Test that names starting with numbers get a prefix."""
        result = sanitize_endpoint_name("123endpoint")
        assert result.startswith("ep-")

    def test_leading_trailing_hyphen_removal(self):
        """Test that leading and trailing hyphens are removed."""
        assert sanitize_endpoint_name("-my-endpoint-") == "my-endpoint"

    def test_truncation(self):
        """Test that long names are truncated."""
        long_name = "a" * 50
        result = sanitize_endpoint_name(long_name)
        assert len(result) <= 32

    def test_empty_string_handling(self):
        """Test that empty strings get a default name."""
        assert sanitize_endpoint_name("---") == "nodetool-endpoint"

    def test_valid_name_unchanged(self):
        """Test that valid names are unchanged."""
        assert sanitize_endpoint_name("my-valid-endpoint") == "my-valid-endpoint"


class TestHuggingFaceDeploymentFunctions:
    """Tests for HuggingFace deployment functions."""

    @patch("httpx.get")
    @patch("httpx.post")
    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    def test_create_endpoint_new(self, mock_post, mock_get):
        """Test creating a new endpoint."""
        from nodetool.deploy.deploy_to_hf import create_huggingface_endpoint

        # Endpoint doesn't exist
        mock_get.return_value = MagicMock(status_code=404)

        # Create succeeds
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": {"state": "pending"}},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = create_huggingface_endpoint(
            namespace="test-namespace",
            endpoint_name="test-endpoint",
            image_url="user/image:latest",
        )

        assert mock_post.called
        assert "status" in result

    @patch("httpx.get")
    @patch("httpx.put")
    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    def test_update_existing_endpoint(self, mock_put, mock_get):
        """Test updating an existing endpoint."""
        from nodetool.deploy.deploy_to_hf import create_huggingface_endpoint

        # Endpoint exists
        mock_get.return_value = MagicMock(status_code=200)

        # Update succeeds
        mock_put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": {"state": "running"}},
        )
        mock_put.return_value.raise_for_status = MagicMock()

        result = create_huggingface_endpoint(
            namespace="test-namespace",
            endpoint_name="test-endpoint",
            image_url="user/image:latest",
        )

        assert mock_put.called
        assert "status" in result

    @patch.dict("os.environ", {}, clear=True)
    def test_create_endpoint_no_token(self):
        """Test that creating endpoint without token raises error."""
        from nodetool.deploy.deploy_to_hf import create_huggingface_endpoint

        with pytest.raises(ValueError, match="HF_TOKEN"):
            create_huggingface_endpoint(
                namespace="test-namespace",
                endpoint_name="test-endpoint",
                image_url="user/image:latest",
            )

    @patch("httpx.get")
    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    def test_get_endpoint_status(self, mock_get):
        """Test getting endpoint status."""
        from nodetool.deploy.deploy_to_hf import get_huggingface_endpoint_status

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"status": {"state": "running", "url": "https://test.endpoints.huggingface.cloud"}},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_huggingface_endpoint_status("test-namespace", "test-endpoint")

        assert result is not None
        assert "status" in result

    @patch("httpx.get")
    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    def test_get_endpoint_status_not_found(self, mock_get):
        """Test getting status for non-existent endpoint."""
        import httpx

        from nodetool.deploy.deploy_to_hf import get_huggingface_endpoint_status

        mock_response = MagicMock(status_code=404)
        mock_get.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=mock_response
        )
        mock_get.return_value.status_code = 404

        result = get_huggingface_endpoint_status("test-namespace", "nonexistent")

        assert result is None

    @patch("httpx.delete")
    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    def test_delete_endpoint(self, mock_delete):
        """Test deleting an endpoint."""
        from nodetool.deploy.deploy_to_hf import delete_huggingface_endpoint

        mock_delete.return_value = MagicMock(status_code=200)
        mock_delete.return_value.raise_for_status = MagicMock()

        result = delete_huggingface_endpoint("test-namespace", "test-endpoint")

        assert result is True
        assert mock_delete.called
