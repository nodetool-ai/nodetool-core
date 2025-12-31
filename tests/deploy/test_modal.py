"""
Unit tests for ModalDeployer.
"""

from unittest.mock import Mock, patch

import pytest

from nodetool.config.deployment import (
    DeploymentStatus,
    ModalDeployment,
    ModalGPUConfig,
    ModalImageConfig,
    ModalResourceConfig,
    ModalState,
)
from nodetool.deploy.modal import ModalDeployer

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestModalDeployer:
    """Tests for ModalDeployer class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic Modal deployment configuration."""
        return ModalDeployment(
            app_name="test-app",
            function_name="handler",
        )

    @pytest.fixture
    def advanced_deployment(self):
        """Create an advanced Modal deployment with GPU and custom settings."""
        return ModalDeployment(
            app_name="gpu-app",
            function_name="gpu_handler",
            image=ModalImageConfig(
                base_image="python:3.11-slim",
                pip_packages=["numpy", "pandas"],
                apt_packages=["ffmpeg"],
            ),
            resources=ModalResourceConfig(
                cpu=2.0,
                memory=4096,
                gpu=ModalGPUConfig(type="A10G", count=1),
                timeout=7200,
                container_idle_timeout=600,
                allow_concurrent_inputs=4,
            ),
            environment={"ENV": "production", "DEBUG": "false"},
            secrets=["my-secret"],
            region="us-east",
        )

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.read_state = Mock(return_value=None)
        manager.write_state = Mock()
        manager.update_deployment_status = Mock()
        return manager

    def test_init(self, basic_deployment, mock_state_manager):
        """Test deployer initialization."""
        deployer = ModalDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        assert deployer.deployment_name == "test-deployment"
        assert deployer.deployment == basic_deployment
        assert deployer.state_manager == mock_state_manager

    def test_init_without_state_manager(self, basic_deployment):
        """Test deployer creates default state manager if not provided."""
        with patch("nodetool.deploy.modal.StateManager") as mock_state_cls:
            mock_state_manager = Mock()
            mock_state_cls.return_value = mock_state_manager

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
            )

            assert deployer.state_manager == mock_state_manager
            mock_state_cls.assert_called_once()

    def test_plan_initial_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for initial deployment."""
        mock_state_manager.read_state.return_value = None

        deployer = ModalDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert plan["type"] == "modal"
        assert plan["app_name"] == "test-app"
        assert "Initial deployment" in plan["changes"][0]
        assert any("Modal app" in item for item in plan["will_create"])
        assert any("Modal function" in item for item in plan["will_create"])

    def test_plan_existing_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for existing deployment."""
        mock_state_manager.read_state.return_value = {
            "last_deployed": "2024-01-15T10:30:00",
            "status": "active",
        }

        deployer = ModalDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert "Configuration may have changed" in plan["changes"][0]
        assert any("Modal app" in item for item in plan["will_update"])

    def test_plan_includes_resources(self, advanced_deployment, mock_state_manager):
        """Test that plan includes resource summary."""
        mock_state_manager.read_state.return_value = None

        deployer = ModalDeployer(
            deployment_name="gpu-deployment",
            deployment=advanced_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert "resources" in plan
        assert plan["resources"]["cpu"] == 2.0
        assert plan["resources"]["memory"] == "4096MB"
        assert "A10G" in plan["resources"]["gpu"]

    def test_apply_dry_run(self, basic_deployment, mock_state_manager):
        """Test apply with dry_run=True returns plan."""
        deployer = ModalDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        result = deployer.apply(dry_run=True)

        # Should return plan, not execute deployment
        assert "deployment_name" in result
        assert "changes" in result
        mock_state_manager.update_deployment_status.assert_not_called()

    def test_apply_success_basic(self, basic_deployment, mock_state_manager):
        """Test successful basic deployment."""
        with patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy:
            mock_deploy.return_value = {
                "app_name": "test-app",
                "app_id": "app-12345",
                "function_url": "https://test-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            assert len(result["errors"]) == 0
            assert "Modal deployment completed" in result["steps"]
            assert result["app_id"] == "app-12345"
            assert result["function_url"] == "https://test-app.modal.run"

            # Should call deploy_modal_app
            mock_deploy.assert_called_once()
            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["app_name"] == "test-app"
            assert call_kwargs["function_name"] == "handler"

            # Should update state
            mock_state_manager.update_deployment_status.assert_called_with(
                "test-deployment", DeploymentStatus.DEPLOYING.value
            )
            mock_state_manager.write_state.assert_called_once()
            state_args = mock_state_manager.write_state.call_args[0]
            assert state_args[0] == "test-deployment"
            assert state_args[1]["status"] == DeploymentStatus.ACTIVE.value

    def test_apply_success_advanced(self, advanced_deployment, mock_state_manager):
        """Test successful advanced deployment with GPU and scaling."""
        with patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy:
            mock_deploy.return_value = {
                "app_name": "gpu-app",
                "app_id": "app-gpu-12345",
                "function_url": "https://gpu-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="gpu-deployment",
                deployment=advanced_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"

            # Verify all advanced options were passed
            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["app_name"] == "gpu-app"
            assert call_kwargs["function_name"] == "gpu_handler"
            assert call_kwargs["cpu"] == 2.0
            assert call_kwargs["memory"] == 4096
            assert call_kwargs["gpu_type"] == "A10G"
            assert call_kwargs["gpu_count"] == 1
            assert call_kwargs["timeout"] == 7200
            assert call_kwargs["container_idle_timeout"] == 600
            assert call_kwargs["allow_concurrent_inputs"] == 4
            assert call_kwargs["environment"] == {"ENV": "production", "DEBUG": "false"}
            assert call_kwargs["secrets"] == ["my-secret"]
            assert call_kwargs["region"] == "us-east"

    def test_apply_with_environment(self, basic_deployment, mock_state_manager):
        """Test deployment with environment variables."""
        basic_deployment.environment = {
            "API_KEY": "secret123",
            "LOG_LEVEL": "debug",
        }

        with patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy:
            mock_deploy.return_value = {
                "app_name": "test-app",
                "app_id": "app-12345",
                "function_url": "https://test-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["environment"] == {
                "API_KEY": "secret123",
                "LOG_LEVEL": "debug",
            }

    def test_apply_without_environment(self, basic_deployment, mock_state_manager):
        """Test deployment without environment variables."""
        basic_deployment.environment = None

        with patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy:
            mock_deploy.return_value = {
                "app_name": "test-app",
                "app_id": "app-12345",
                "function_url": "https://test-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["environment"] is None

    def test_apply_failure(self, basic_deployment, mock_state_manager):
        """Test deployment failure."""
        with patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy:
            mock_deploy.side_effect = Exception("Modal API error")

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            with pytest.raises(Exception, match="Modal API error"):
                deployer.apply(dry_run=False)

            # Should update state to error
            mock_state_manager.update_deployment_status.assert_any_call(
                "test-deployment", DeploymentStatus.ERROR.value
            )

    def test_status_with_state(self, basic_deployment, mock_state_manager):
        """Test getting status with saved state."""
        mock_state_manager.read_state.return_value = {
            "status": "active",
            "last_deployed": "2024-01-15T10:30:00",
            "app_id": "app-12345",
            "function_url": "https://test-app.modal.run",
        }

        with patch("nodetool.deploy.modal.get_modal_app_status") as mock_status:
            mock_status.return_value = {"status": "active"}

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test-deployment"
            assert status["type"] == "modal"
            assert status["status"] == "active"
            assert status["last_deployed"] == "2024-01-15T10:30:00"
            assert status["app_id"] == "app-12345"
            assert status["function_url"] == "https://test-app.modal.run"

    def test_status_without_state(self, basic_deployment, mock_state_manager):
        """Test getting status without saved state."""
        mock_state_manager.read_state.return_value = None

        with patch("nodetool.deploy.modal.get_modal_app_status") as mock_status:
            mock_status.return_value = {"status": "not_found"}

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test-deployment"
            assert status["type"] == "modal"
            # Should not have state fields
            assert "status" not in status or status.get("status") is None

    def test_logs_returns_instructions(self, basic_deployment, mock_state_manager):
        """Test that logs returns instructions for accessing logs."""
        with patch("nodetool.deploy.modal.get_modal_logs") as mock_logs:
            mock_logs.return_value = "Modal logs for 'test-app' are available via..."

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            logs = deployer.logs()

            assert "test-app" in logs or mock_logs.called
            mock_logs.assert_called_once_with("test-app", tail=100)

    def test_logs_with_parameters(self, basic_deployment, mock_state_manager):
        """Test logs with parameters."""
        with patch("nodetool.deploy.modal.get_modal_logs") as mock_logs:
            mock_logs.return_value = "Logs..."

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.logs(service="api", follow=True, tail=50)

            mock_logs.assert_called_once_with("test-app", tail=50)

    def test_destroy(self, basic_deployment, mock_state_manager):
        """Test deployment destruction."""
        with patch("nodetool.deploy.modal.delete_modal_app") as mock_delete:
            mock_delete.return_value = {"status": "deleted"}

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "success"
            assert result["deployment_name"] == "test-deployment"
            assert any("stopped" in step.lower() for step in result["steps"])

            # Should update state
            mock_state_manager.update_deployment_status.assert_called_once_with(
                "test-deployment", DeploymentStatus.DESTROYED.value
            )

    def test_destroy_not_found(self, basic_deployment, mock_state_manager):
        """Test destroy when app not found."""
        with patch("nodetool.deploy.modal.delete_modal_app") as mock_delete:
            mock_delete.return_value = {"status": "not_found"}

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "success"
            assert any("not found" in step.lower() for step in result["steps"])

    def test_destroy_failure(self, basic_deployment, mock_state_manager):
        """Test destroy with state manager failure."""
        mock_state_manager.update_deployment_status.side_effect = Exception("State update failed")

        with patch("nodetool.deploy.modal.delete_modal_app") as mock_delete:
            mock_delete.return_value = {"status": "deleted"}

            deployer = ModalDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            with pytest.raises(Exception, match="State update failed"):
                deployer.destroy()


class TestModalDeployerEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def minimal_deployment(self):
        """Create minimal deployment with only required fields."""
        return ModalDeployment(
            app_name="minimal-app",
        )

    def test_minimal_deployment_config(self, minimal_deployment):
        """Test deployment with minimal configuration."""
        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager") as mock_state_cls,
        ):
            mock_deploy.return_value = {
                "app_name": "minimal-app",
                "app_id": "app-min",
                "function_url": "https://minimal-app.modal.run",
                "status": "deployed",
            }
            mock_state_manager = Mock()
            mock_state_manager.read_state = Mock(return_value=None)
            mock_state_manager.write_state = Mock()
            mock_state_manager.update_deployment_status = Mock()
            mock_state_cls.return_value = mock_state_manager

            deployer = ModalDeployer(
                deployment_name="minimal",
                deployment=minimal_deployment,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            mock_deploy.assert_called_once()

    def test_empty_secrets_list(self):
        """Test deployment with empty secrets list."""
        deployment = ModalDeployment(
            app_name="test-app",
            secrets=[],
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "test-app",
                "app_id": "app-123",
                "function_url": "https://test-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="test",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["secrets"] is None or call_kwargs["secrets"] == []

    def test_no_gpu_config(self):
        """Test deployment without GPU configuration."""
        deployment = ModalDeployment(
            app_name="cpu-only-app",
            resources=ModalResourceConfig(
                cpu=1.0,
                memory=2048,
                gpu=None,
            ),
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "cpu-only-app",
                "app_id": "app-cpu",
                "function_url": "https://cpu-only-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="cpu-only",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["gpu_type"] is None

    def test_multiple_gpu_types(self):
        """Test deployment with GPU configuration."""
        deployment = ModalDeployment(
            app_name="gpu-app",
            resources=ModalResourceConfig(
                gpu=ModalGPUConfig(type="H100", count=2),
            ),
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "gpu-app",
                "app_id": "app-gpu",
                "function_url": "https://gpu-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="gpu",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["gpu_type"] == "H100"
            assert call_kwargs["gpu_count"] == 2

    def test_custom_image_with_dockerfile(self):
        """Test deployment with custom Dockerfile."""
        deployment = ModalDeployment(
            app_name="custom-image-app",
            image=ModalImageConfig(
                dockerfile="/path/to/Dockerfile",
                context_dir="/path/to/context",
            ),
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "custom-image-app",
                "app_id": "app-custom",
                "function_url": "https://custom-image-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="custom",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["image_config"]["dockerfile"] == "/path/to/Dockerfile"
            assert call_kwargs["image_config"]["context_dir"] == "/path/to/context"

    def test_custom_region(self):
        """Test deployment with custom region."""
        deployment = ModalDeployment(
            app_name="regional-app",
            region="eu-west",
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "regional-app",
                "app_id": "app-eu",
                "function_url": "https://regional-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="regional",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["region"] == "eu-west"

    def test_timeout_configs(self):
        """Test deployment with timeout configurations."""
        deployment = ModalDeployment(
            app_name="timeout-app",
            resources=ModalResourceConfig(
                timeout=14400,  # 4 hours
                container_idle_timeout=1800,  # 30 minutes
            ),
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "timeout-app",
                "app_id": "app-timeout",
                "function_url": "https://timeout-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="timeout",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["timeout"] == 14400
            assert call_kwargs["container_idle_timeout"] == 1800

    def test_concurrency_config(self):
        """Test deployment with concurrency configuration."""
        deployment = ModalDeployment(
            app_name="concurrent-app",
            resources=ModalResourceConfig(
                allow_concurrent_inputs=10,
            ),
        )

        with (
            patch("nodetool.deploy.modal.deploy_modal_app") as mock_deploy,
            patch("nodetool.deploy.modal.StateManager"),
        ):
            mock_deploy.return_value = {
                "app_name": "concurrent-app",
                "app_id": "app-concurrent",
                "function_url": "https://concurrent-app.modal.run",
                "status": "deployed",
            }

            deployer = ModalDeployer(
                deployment_name="concurrent",
                deployment=deployment,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["allow_concurrent_inputs"] == 10
