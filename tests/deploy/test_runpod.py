"""
Unit tests for RunPodDeployer.
"""

import pytest
from unittest.mock import Mock, patch

from nodetool.deploy.runpod import RunPodDeployer
from nodetool.config.deployment import (
    RunPodDeployment,
    RunPodImageConfig,
    RunPodDockerConfig,
    RunPodTemplateConfig,
    RunPodEndpointConfig,
    DeploymentStatus,
)


# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestRunPodDeployer:
    """Tests for RunPodDeployer class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic RunPod deployment configuration."""
        return RunPodDeployment(
            docker=RunPodDockerConfig(
                username="myuser",
                registry="docker.io",
            ),
            image=RunPodImageConfig(
                name="nodetool-workflow",
                tag="latest",
            ),
            template_name="my-template",
            compute_type="serverless",
        )

    @pytest.fixture
    def advanced_deployment(self):
        """Create an advanced RunPod deployment with GPU and scaling."""
        return RunPodDeployment(
            docker=RunPodDockerConfig(
                username="myuser",
                registry="docker.io",
            ),
            image=RunPodImageConfig(
                name="nodetool-workflow",
                tag="v1.0.0",
            ),
            template_name="gpu-template",
            compute_type="serverless",
            gpu_types=["NVIDIA RTX A5000", "NVIDIA A40"],
            gpu_count=1,
            data_centers=["US-CA-1", "US-TX-1"],
            workers_min=0,
            workers_max=3,
            idle_timeout=5,
            execution_timeout=300,
            flashboot=True,
            environment={"ENV": "production", "DEBUG": "false"},
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
        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        assert deployer.deployment_name == "test-deployment"
        assert deployer.deployment == basic_deployment
        assert deployer.state_manager == mock_state_manager

    def test_init_without_state_manager(self, basic_deployment):
        """Test deployer creates default state manager if not provided."""
        with patch("nodetool.deploy.runpod.StateManager") as mock_state_cls:
            mock_state_manager = Mock()
            mock_state_cls.return_value = mock_state_manager

            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
            )

            assert deployer.state_manager == mock_state_manager
            mock_state_cls.assert_called_once()

    def test_plan_initial_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for initial deployment."""
        mock_state_manager.read_state.return_value = None

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert plan["type"] == "runpod"
        assert "Initial deployment" in plan["changes"][0]
        assert "Docker image" in plan["will_create"]
        assert "RunPod template" in plan["will_create"]
        assert "RunPod serverless endpoint" in plan["will_create"]

    def test_plan_existing_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for existing deployment."""
        mock_state_manager.read_state.return_value = {
            "last_deployed": "2024-01-15T10:30:00",
            "status": "active",
        }

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert "Configuration may have changed" in plan["changes"][0]
        assert "RunPod endpoint configuration" in plan["will_update"]

    def test_apply_dry_run(self, basic_deployment, mock_state_manager):
        """Test apply with dry_run=True returns plan."""
        deployer = RunPodDeployer(
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
        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            assert len(result["errors"]) == 0
            assert "RunPod deployment completed" in result["steps"]

            # Should call deploy_to_runpod
            mock_deploy.assert_called_once()
            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["docker_username"] == "myuser"
            assert call_kwargs["docker_registry"] == "docker.io"
            assert call_kwargs["image_name"] == "nodetool-workflow"
            assert call_kwargs["tag"] == "latest"
            assert call_kwargs["template_name"] == "my-template"
            assert call_kwargs["platform"] == "linux/amd64"

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
        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="gpu-deployment",
                deployment=advanced_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"

            # Verify all advanced options were passed
            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["gpu_types"] == ("NVIDIA RTX A5000", "NVIDIA A40")
            assert call_kwargs["gpu_count"] == 1
            assert call_kwargs["data_centers"] == ("US-CA-1", "US-TX-1")
            assert call_kwargs["workers_min"] == 0
            assert call_kwargs["workers_max"] == 3
            assert call_kwargs["idle_timeout"] == 5
            assert call_kwargs["execution_timeout"] == 300
            assert call_kwargs["flashboot"] is True
            assert call_kwargs["env"] == {"ENV": "production", "DEBUG": "false"}

    def test_apply_with_environment(self, basic_deployment, mock_state_manager):
        """Test deployment with environment variables."""
        basic_deployment.environment = {
            "API_KEY": "secret123",
            "LOG_LEVEL": "debug",
        }

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["env"] == {
                "API_KEY": "secret123",
                "LOG_LEVEL": "debug",
            }

    def test_apply_without_environment(self, basic_deployment, mock_state_manager):
        """Test deployment without environment variables."""
        basic_deployment.environment = None

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["env"] == {}

    def test_apply_with_default_template_name(
        self, basic_deployment, mock_state_manager
    ):
        """Test deployment uses deployment_name when template_name is None."""
        basic_deployment.template_name = None

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="my-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["template_name"] == "my-deployment"

    def test_apply_failure(self, basic_deployment, mock_state_manager):
        """Test deployment failure."""
        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            mock_deploy.side_effect = Exception("RunPod API error")

            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            with pytest.raises(Exception, match="RunPod API error"):
                deployer.apply(dry_run=False)

            # Should update state to error
            mock_state_manager.update_deployment_status.assert_any_call(
                "test-deployment", DeploymentStatus.ERROR.value
            )

    def test_apply_skip_flags(self, basic_deployment, mock_state_manager):
        """Test that apply passes correct skip flags."""
        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            deployer = RunPodDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            # Should not skip anything by default
            assert call_kwargs["skip_build"] is False
            assert call_kwargs["skip_push"] is False
            assert call_kwargs["skip_template"] is False
            assert call_kwargs["skip_endpoint"] is False

    def test_status_with_state(self, basic_deployment, mock_state_manager):
        """Test getting status with saved state."""
        mock_state_manager.read_state.return_value = {
            "status": "active",
            "last_deployed": "2024-01-15T10:30:00",
            "template_name": "my-template",
            "pod_id": "pod-12345",
        }

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        status = deployer.status()

        assert status["deployment_name"] == "test-deployment"
        assert status["type"] == "runpod"
        assert status["status"] == "active"
        assert status["last_deployed"] == "2024-01-15T10:30:00"
        assert status["template_name"] == "my-template"
        assert status["pod_id"] == "pod-12345"

    def test_status_without_state(self, basic_deployment, mock_state_manager):
        """Test getting status without saved state."""
        mock_state_manager.read_state.return_value = None

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        status = deployer.status()

        assert status["deployment_name"] == "test-deployment"
        assert status["type"] == "runpod"
        # Should not have state fields
        assert "status" not in status or status.get("status") is None

    def test_status_partial_state(self, basic_deployment, mock_state_manager):
        """Test status with partial state information."""
        mock_state_manager.read_state.return_value = {
            "status": "active",
            # Missing other fields
        }

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        status = deployer.status()

        assert status["status"] == "active"
        assert status["last_deployed"] == "unknown"
        assert status["template_name"] == "unknown"
        assert status["pod_id"] == "unknown"

    def test_logs_not_implemented(self, basic_deployment, mock_state_manager):
        """Test that logs raises NotImplementedError."""
        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        with pytest.raises(NotImplementedError) as exc_info:
            deployer.logs()

        assert "RunPod serverless" in str(exc_info.value)
        assert "log access" in str(exc_info.value)

    def test_logs_with_parameters(self, basic_deployment, mock_state_manager):
        """Test logs with parameters (should still raise NotImplementedError)."""
        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        with pytest.raises(NotImplementedError):
            deployer.logs(service="api", follow=True, tail=50)

    def test_destroy(self, basic_deployment, mock_state_manager):
        """Test deployment destruction."""
        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        result = deployer.destroy()

        assert result["status"] == "success"
        assert result["deployment_name"] == "test-deployment"
        assert any("manually" in step for step in result["steps"])
        assert any("console" in step for step in result["steps"])

        # Should update state
        mock_state_manager.update_deployment_status.assert_called_once_with(
            "test-deployment", DeploymentStatus.DESTROYED.value
        )

    def test_destroy_failure(self, basic_deployment, mock_state_manager):
        """Test destroy with state manager failure."""
        mock_state_manager.update_deployment_status.side_effect = Exception(
            "State update failed"
        )

        deployer = RunPodDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        with pytest.raises(Exception, match="State update failed"):
            deployer.destroy()


class TestRunPodDeployerEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def minimal_deployment(self):
        """Create minimal deployment with only required fields."""
        return RunPodDeployment(
            docker=RunPodDockerConfig(
                username="user",
                registry="docker.io",
            ),
            image=RunPodImageConfig(
                name="image",
                tag="tag",
            ),
        )

    def test_minimal_deployment_config(self, minimal_deployment):
        """Test deployment with minimal configuration."""
        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager") as mock_state_cls:
                mock_state_manager = Mock()
                mock_state_manager.read_state = Mock(return_value=None)
                mock_state_manager.write_state = Mock()
                mock_state_manager.update_deployment_status = Mock()
                mock_state_cls.return_value = mock_state_manager

                deployer = RunPodDeployer(
                    deployment_name="minimal",
                    deployment=minimal_deployment,
                )

                result = deployer.apply(dry_run=False)

                assert result["status"] == "success"
                mock_deploy.assert_called_once()

    def test_empty_gpu_types_list(self):
        """Test deployment with empty GPU types list."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            gpu_types=[],
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.gpu_types == []

    def test_none_gpu_types(self):
        """Test deployment with default GPU types (not specified)."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            # gpu_types not specified, will use default empty list
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.gpu_types == []

    def test_empty_data_centers_list(self):
        """Test deployment with empty data centers list."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            data_centers=[],
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.data_centers == []

    def test_custom_compute_type(self):
        """Test deployment with custom compute type."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            compute_type="spot",
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.compute_type == "spot"

    def test_network_volume_id(self):
        """Test deployment with network volume ID."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            network_volume_id="volume-12345",
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.network_volume_id == "volume-12345"

    def test_workers_scaling_config(self):
        """Test deployment with worker scaling configuration."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            workers_min=1,
            workers_max=10,
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.workers_min == 1
                assert deployment.workers_max == 10

    def test_timeout_configs(self):
        """Test deployment with timeout configurations."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            idle_timeout=10,
            execution_timeout=600,
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.idle_timeout == 10
                assert deployment.execution_timeout == 600

    def test_flashboot_enabled(self):
        """Test deployment with flashboot enabled."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            flashboot=True,
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.flashboot is True

    def test_flashboot_disabled(self):
        """Test deployment with flashboot disabled."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            flashboot=False,
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.flashboot is False

    def test_multiple_gpu_types(self):
        """Test deployment with multiple GPU type preferences."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            gpu_types=["NVIDIA A100", "NVIDIA A40", "NVIDIA RTX A6000"],
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.gpu_types == [
                    "NVIDIA A100",
                    "NVIDIA A40",
                    "NVIDIA RTX A6000",
                ]

    def test_multiple_data_centers(self):
        """Test deployment with multiple data center preferences."""
        deployment = RunPodDeployment(
            docker=RunPodDockerConfig(username="user", registry="docker.io"),
            image=RunPodImageConfig(name="image", tag="tag"),
            data_centers=["US-CA-1", "US-TX-1", "EU-NL-1"],
        )

        with patch("nodetool.deploy.runpod.deploy_to_runpod") as mock_deploy:
            with patch("nodetool.deploy.runpod.StateManager"):
                deployer = RunPodDeployer(
                    deployment_name="test",
                    deployment=deployment,
                )

                deployer.apply(dry_run=False)

                call_kwargs = mock_deploy.call_args[1]
                deployment = call_kwargs["deployment"]
                assert deployment.data_centers == ["US-CA-1", "US-TX-1", "EU-NL-1"]
