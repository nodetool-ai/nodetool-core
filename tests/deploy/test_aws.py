"""
Unit tests for AWSDeployer.
"""
# ruff: noqa: SIM117

from unittest.mock import Mock, patch

import pytest

from nodetool.config.deployment import (
    AWSDeployment,
    AWSImageConfig,
    AWSResourceConfig,
    AWSHealthCheckConfig,
    AWSNetworkConfig,
    DeploymentStatus,
)
from nodetool.deploy.aws import AWSDeployer

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestAWSDeployer:
    """Tests for AWSDeployer class."""

    @pytest.fixture
    def basic_deployment(self):
        """Create a basic AWS deployment configuration."""
        return AWSDeployment(
            region="us-east-1",
            service_name="test-service",
            image=AWSImageConfig(
                repository="nodetool-workflow",
                tag="latest",
            ),
        )

    @pytest.fixture
    def advanced_deployment(self):
        """Create an advanced AWS deployment with custom resources."""
        return AWSDeployment(
            region="us-west-2",
            service_name="advanced-service",
            image=AWSImageConfig(
                repository="nodetool-workflow",
                tag="v1.0.0",
            ),
            resources=AWSResourceConfig(
                cpu="2 vCPU",
                memory="4 GB",
                min_instances=1,
                max_instances=5,
                max_concurrency=50,
            ),
            health_check=AWSHealthCheckConfig(
                path="/api/health",
                interval=15,
                timeout=10,
            ),
            network=AWSNetworkConfig(
                is_publicly_accessible=True,
            ),
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
        deployer = AWSDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        assert deployer.deployment_name == "test-deployment"
        assert deployer.deployment == basic_deployment
        assert deployer.state_manager == mock_state_manager

    def test_init_without_state_manager(self, basic_deployment):
        """Test deployer creates default state manager if not provided."""
        with patch("nodetool.deploy.aws.StateManager") as mock_state_cls:
            mock_state_manager = Mock()
            mock_state_cls.return_value = mock_state_manager

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
            )

            assert deployer.state_manager == mock_state_manager
            mock_state_cls.assert_called_once()

    def test_plan_initial_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for initial deployment."""
        mock_state_manager.read_state.return_value = None

        deployer = AWSDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert plan["type"] == "aws"
        assert plan["region"] == "us-east-1"
        assert "Initial deployment" in plan["changes"][0]
        assert "Docker image" in plan["will_create"]
        assert "ECR repository" in plan["will_create"]
        assert "App Runner service: test-service" in plan["will_create"]

    def test_plan_existing_deployment(self, basic_deployment, mock_state_manager):
        """Test generating plan for existing deployment."""
        mock_state_manager.read_state.return_value = {
            "last_deployed": "2024-01-15T10:30:00",
            "status": "serving",
        }

        deployer = AWSDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        plan = deployer.plan()

        assert plan["deployment_name"] == "test-deployment"
        assert "Configuration may have changed" in plan["changes"][0]
        assert "App Runner service: test-service" in plan["will_update"]

    def test_apply_dry_run(self, basic_deployment, mock_state_manager):
        """Test apply with dry_run=True returns plan."""
        deployer = AWSDeployer(
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
        with patch("nodetool.deploy.aws.deploy_to_aws") as mock_deploy:
            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            assert len(result["errors"]) == 0
            assert "AWS App Runner deployment completed" in result["steps"]

            # Should call deploy_to_aws
            mock_deploy.assert_called_once()
            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["deployment"] == basic_deployment

            # Should update state
            mock_state_manager.update_deployment_status.assert_called_with(
                "test-deployment", DeploymentStatus.DEPLOYING.value
            )
            mock_state_manager.write_state.assert_called_once()
            state_args = mock_state_manager.write_state.call_args[0]
            assert state_args[0] == "test-deployment"
            assert state_args[1]["status"] == DeploymentStatus.SERVING.value

    def test_apply_with_environment(self, basic_deployment, mock_state_manager):
        """Test deployment with environment variables."""
        basic_deployment.environment = {
            "API_KEY": "secret123",
            "LOG_LEVEL": "debug",
        }

        with patch("nodetool.deploy.aws.deploy_to_aws") as mock_deploy:
            deployer = AWSDeployer(
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

        with patch("nodetool.deploy.aws.deploy_to_aws") as mock_deploy:
            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            deployer.apply(dry_run=False)

            call_kwargs = mock_deploy.call_args[1]
            assert call_kwargs["env"] == {}

    def test_apply_failure(self, basic_deployment, mock_state_manager):
        """Test deployment failure."""
        with patch("nodetool.deploy.aws.deploy_to_aws") as mock_deploy:
            mock_deploy.side_effect = Exception("AWS API error")

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            with pytest.raises(Exception, match="AWS API error"):
                deployer.apply(dry_run=False)

            # Should update state to error
            mock_state_manager.update_deployment_status.assert_any_call("test-deployment", DeploymentStatus.ERROR.value)

    def test_status_with_state(self, basic_deployment, mock_state_manager):
        """Test getting status with saved state."""
        mock_state_manager.read_state.return_value = {
            "status": "serving",
            "last_deployed": "2024-01-15T10:30:00",
        }

        with patch("nodetool.deploy.aws.list_aws_services") as mock_list:
            mock_list.return_value = [
                {
                    "ServiceName": "test-service",
                    "Status": "RUNNING",
                    "ServiceUrl": "abc123.us-east-1.awsapprunner.com",
                }
            ]

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test-deployment"
            assert status["type"] == "aws"
            assert status["region"] == "us-east-1"
            assert status["status"] == "serving"
            assert status["last_deployed"] == "2024-01-15T10:30:00"
            assert status["live_status"] == "RUNNING"
            assert status["url"] == "https://abc123.us-east-1.awsapprunner.com"

    def test_status_without_state(self, basic_deployment, mock_state_manager):
        """Test getting status without saved state."""
        mock_state_manager.read_state.return_value = None

        with patch("nodetool.deploy.aws.list_aws_services") as mock_list:
            mock_list.return_value = []

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            status = deployer.status()

            assert status["deployment_name"] == "test-deployment"
            assert status["type"] == "aws"
            # Should not have state fields
            assert "status" not in status or status.get("status") is None

    def test_logs(self, basic_deployment, mock_state_manager):
        """Test getting logs from CloudWatch."""
        deployer = AWSDeployer(
            deployment_name="test-deployment",
            deployment=basic_deployment,
            state_manager=mock_state_manager,
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="Log line 1\nLog line 2\n", returncode=0)

            logs = deployer.logs()

            assert "Log line 1" in logs
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "aws" in call_args
            assert "logs" in call_args
            assert "tail" in call_args
            assert "/aws/apprunner/test-service" in call_args

    def test_destroy(self, basic_deployment, mock_state_manager):
        """Test deployment destruction."""
        with patch("nodetool.deploy.aws.delete_aws_service") as mock_delete:
            mock_delete.return_value = True

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "success"
            assert result["deployment_name"] == "test-deployment"

            # Should call delete
            mock_delete.assert_called_once_with(
                service_name="test-service",
                region="us-east-1",
            )

            # Should update state
            mock_state_manager.update_deployment_status.assert_called_once_with(
                "test-deployment", DeploymentStatus.DESTROYED.value
            )

    def test_destroy_failure(self, basic_deployment, mock_state_manager):
        """Test destroy when deletion fails."""
        with patch("nodetool.deploy.aws.delete_aws_service") as mock_delete:
            mock_delete.return_value = False

            deployer = AWSDeployer(
                deployment_name="test-deployment",
                deployment=basic_deployment,
                state_manager=mock_state_manager,
            )

            result = deployer.destroy()

            assert result["status"] == "error"
            assert "Failed to delete service" in result["errors"]


class TestAWSDeployerEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def minimal_deployment(self):
        """Create minimal deployment with only required fields."""
        return AWSDeployment(
            service_name="minimal-svc",
            image=AWSImageConfig(
                repository="image",
                tag="tag",
            ),
        )

    def test_minimal_deployment_config(self, minimal_deployment):
        """Test deployment with minimal configuration."""
        with (
            patch("nodetool.deploy.aws.deploy_to_aws") as mock_deploy,
            patch("nodetool.deploy.aws.StateManager") as mock_state_cls,
        ):
            mock_state_manager = Mock()
            mock_state_manager.read_state = Mock(return_value=None)
            mock_state_manager.write_state = Mock()
            mock_state_manager.update_deployment_status = Mock()
            mock_state_cls.return_value = mock_state_manager

            deployer = AWSDeployer(
                deployment_name="minimal",
                deployment=minimal_deployment,
            )

            result = deployer.apply(dry_run=False)

            assert result["status"] == "success"
            mock_deploy.assert_called_once()

    def test_default_region(self, minimal_deployment):
        """Test deployment uses default region."""
        assert minimal_deployment.region == "us-east-1"

    def test_default_resources(self, minimal_deployment):
        """Test deployment uses default resources."""
        assert minimal_deployment.resources.cpu == "2 vCPU"
        assert minimal_deployment.resources.memory == "4 GB"
        assert minimal_deployment.resources.min_instances == 1
        assert minimal_deployment.resources.max_instances == 3
        assert minimal_deployment.resources.max_concurrency == 100

    def test_default_health_check(self, minimal_deployment):
        """Test deployment uses default health check settings."""
        assert minimal_deployment.health_check.path == "/health"
        assert minimal_deployment.health_check.interval == 10
        assert minimal_deployment.health_check.timeout == 5

    def test_default_network(self, minimal_deployment):
        """Test deployment uses default network settings."""
        assert minimal_deployment.network.is_publicly_accessible is True
        assert minimal_deployment.network.vpc_connector_arn is None
