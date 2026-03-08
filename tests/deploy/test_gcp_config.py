"""
Unit tests for GCP configuration and deployment logic.
"""

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from nodetool.config.deployment import GCPDeployment, GCPImageConfig, GCPResourceConfig
from nodetool.deploy.deploy_to_gcp import deploy_to_gcp


class TestGCPResourceConfig:
    """Tests for GCPResourceConfig model."""

    def test_gcp_resource_config_defaults(self):
        """Test default GCP resource configuration."""
        config = GCPResourceConfig()
        assert config.cpu == "4"
        assert config.memory == "16Gi"
        assert config.gpu_type is None
        assert config.gpu_count is None

    def test_gcp_resource_config_with_gpu(self):
        """Test GCP resource configuration with GPU."""
        config = GCPResourceConfig(gpu_type="nvidia-l4", gpu_count=1)
        assert config.gpu_type == "nvidia-l4"
        assert config.gpu_count == 1


class TestGCPDeployment:
    """Tests for GCPDeployment model."""

    def test_gcp_deployment_with_gpu(self):
        """Test GCP deployment configuration with GPU."""
        deployment = GCPDeployment(
            project_id="my-project",
            service_name="my-service",
            image=GCPImageConfig(repository="my-repo", tag="latest"),
            resources=GCPResourceConfig(gpu_type="nvidia-t4", gpu_count=2),
        )
        assert deployment.resources.gpu_type == "nvidia-t4"
        assert deployment.resources.gpu_count == 2

    @patch("nodetool.deploy.deploy_to_gcp.deploy_to_cloud_run")
    @patch("nodetool.deploy.deploy_to_gcp.ensure_gcloud_auth")
    @patch("nodetool.deploy.deploy_to_gcp.ensure_project_set")
    @patch("nodetool.deploy.deploy_to_gcp.enable_required_apis")
    @patch("nodetool.deploy.deploy_to_gcp.push_to_gcr")
    def test_deploy_to_gcp_passes_gpu_args(
        self,
        mock_push_to_gcr,
        mock_enable_apis,
        mock_ensure_project,
        mock_ensure_auth,
        mock_deploy_to_cloud_run,
    ):
        """Test that deploy_to_gcp passes GPU arguments correctly to deploy_to_cloud_run."""

        # Setup mocks
        mock_ensure_project.return_value = "my-project"
        mock_push_to_gcr.return_value = "gcr.io/my-project/my-repo:latest"

        # Create deployment configuration
        deployment = GCPDeployment(
            project_id="my-project",
            service_name="my-service",
            image=GCPImageConfig(repository="my-repo", tag="latest"),
            resources=GCPResourceConfig(gpu_type="nvidia-l4", gpu_count=1),
        )

        # Call deploy_to_gcp
        deploy_to_gcp(deployment, skip_build=True, skip_push=False, skip_permission_setup=True)

        # Verify deploy_to_cloud_run was called with correct GPU arguments
        mock_deploy_to_cloud_run.assert_called_once()
        call_kwargs = mock_deploy_to_cloud_run.call_args[1]
        assert call_kwargs["gpu_type"] == "nvidia-l4"
        assert call_kwargs["gpu_count"] == 1

    @patch("nodetool.deploy.deploy_to_gcp.deploy_to_cloud_run")
    @patch("nodetool.deploy.deploy_to_gcp.ensure_gcloud_auth")
    @patch("nodetool.deploy.deploy_to_gcp.ensure_project_set")
    @patch("nodetool.deploy.deploy_to_gcp.enable_required_apis")
    @patch("nodetool.deploy.deploy_to_gcp.push_to_gcr")
    def test_deploy_to_gcp_defaults_gpu_count(
        self,
        mock_push_to_gcr,
        mock_enable_apis,
        mock_ensure_project,
        mock_ensure_auth,
        mock_deploy_to_cloud_run,
    ):
        """Test that deploy_to_gcp defaults gpu_count to 1 if gpu_type is set but gpu_count is not."""

        # Setup mocks
        mock_ensure_project.return_value = "my-project"
        mock_push_to_gcr.return_value = "gcr.io/my-project/my-repo:latest"

        # Create deployment configuration with gpu_type but no gpu_count
        deployment = GCPDeployment(
            project_id="my-project",
            service_name="my-service",
            image=GCPImageConfig(repository="my-repo", tag="latest"),
            resources=GCPResourceConfig(gpu_type="nvidia-l4"),
        )

        # Call deploy_to_gcp
        deploy_to_gcp(deployment, skip_build=True, skip_push=False, skip_permission_setup=True)

        # Verify deploy_to_cloud_run was called with correct GPU arguments
        mock_deploy_to_cloud_run.assert_called_once()
        call_kwargs = mock_deploy_to_cloud_run.call_args[1]
        assert call_kwargs["gpu_type"] == "nvidia-l4"
        assert call_kwargs["gpu_count"] == 1
