"""
Unit tests for DeploymentManager orchestrator.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from nodetool.deploy.manager import DeploymentManager
from nodetool.config.deployment import (
    DeploymentConfig,
    SelfHostedDeployment,
    RunPodDeployment,
    GCPDeployment,
    SSHConfig,
    ImageConfig,
    ContainerConfig,
    RunPodImageConfig,
    RunPodTemplateConfig,
    RunPodEndpointConfig,
    GCPImageConfig,
    DeploymentType,
)


# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestDeploymentManager:
    """Tests for DeploymentManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock deployment config."""
        # Create self-hosted deployment
        self_hosted = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            containers=[
                ContainerConfig(name="wf1", port=8001),
            ],
        )

        # Create RunPod deployment
        runpod = RunPodDeployment(
            image=RunPodImageConfig(name="nodetool/nodetool", tag="latest"),
            template=RunPodTemplateConfig(name="nodetool-template"),
            endpoint=RunPodEndpointConfig(name="nodetool-endpoint"),
        )

        # Create GCP deployment
        gcp = GCPDeployment(
            project_id="my-project",
            region="us-central1",
            service_name="nodetool-service",
            image=GCPImageConfig(
                repository="my-project/nodetool/app",
                tag="latest",
            ),
        )

        # Create config with all deployments
        config = DeploymentConfig(
            deployments={
                "production": self_hosted,
                "serverless": runpod,
                "cloud": gcp,
            }
        )

        return config

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        manager = Mock()
        manager.read_state = Mock(return_value=None)
        manager.write_state = Mock()
        manager.get_all_states = Mock(return_value={})
        return manager

    @pytest.fixture
    def manager(self, mock_config, mock_state_manager):
        """Create a DeploymentManager with mocked dependencies."""
        with patch("nodetool.deploy.manager.load_deployment_config") as mock_load:
            with patch("nodetool.deploy.manager.StateManager") as mock_state_cls:
                mock_load.return_value = mock_config
                mock_state_cls.return_value = mock_state_manager

                manager = DeploymentManager()
                manager.config = mock_config
                manager.state_manager = mock_state_manager

                return manager

    def test_init(self, mock_config, mock_state_manager):
        """Test DeploymentManager initialization."""
        with patch("nodetool.deploy.manager.load_deployment_config") as mock_load:
            with patch("nodetool.deploy.manager.StateManager") as mock_state_cls:
                mock_load.return_value = mock_config
                mock_state_cls.return_value = mock_state_manager

                manager = DeploymentManager()

                assert manager.config == mock_config
                assert manager.state_manager == mock_state_manager
                mock_load.assert_called_once()
                mock_state_cls.assert_called_once()

    def test_init_with_config_path(self, mock_config, mock_state_manager):
        """Test DeploymentManager initialization with custom config path."""
        config_path = Path("/custom/path/deployment.yaml")

        with patch("nodetool.deploy.manager.load_deployment_config") as mock_load:
            with patch("nodetool.deploy.manager.StateManager") as mock_state_cls:
                mock_load.return_value = mock_config
                mock_state_cls.return_value = mock_state_manager

                _ = DeploymentManager(config_path=config_path)

                mock_state_cls.assert_called_once_with(config_path=config_path)

    def test_list_deployments_no_state(self, manager):
        """Test listing deployments without state."""
        manager.state_manager.read_state.return_value = None

        deployments = manager.list_deployments()

        assert len(deployments) == 3
        assert deployments[0]["name"] == "production"
        assert deployments[0]["type"] == DeploymentType.SELF_HOSTED
        assert deployments[0]["status"] == "unknown"
        assert deployments[0]["host"] == "192.168.1.100"
        assert deployments[0]["containers"] == 1

        assert deployments[1]["name"] == "serverless"
        assert deployments[1]["type"] == DeploymentType.RUNPOD
        assert deployments[1]["pod_id"] is None

        assert deployments[2]["name"] == "cloud"
        assert deployments[2]["type"] == DeploymentType.GCP
        assert deployments[2]["project"] == "my-project"
        assert deployments[2]["region"] == "us-central1"

    def test_list_deployments_with_state(self, manager):
        """Test listing deployments with state."""

        def mock_read_state(name):
            if name == "production":
                return {
                    "status": "running",
                    "last_deployed": "2024-01-15T10:30:00",
                }
            elif name == "serverless":
                return {
                    "status": "active",
                    "pod_id": "pod-123",
                }
            return None

        manager.state_manager.read_state.side_effect = mock_read_state

        deployments = manager.list_deployments()

        assert deployments[0]["status"] == "running"
        assert deployments[0]["last_deployed"] == "2024-01-15T10:30:00"

        assert deployments[1]["status"] == "active"
        assert deployments[1]["pod_id"] == "pod-123"

    def test_get_deployment(self, manager):
        """Test getting deployment by name."""
        deployment = manager.get_deployment("production")

        assert isinstance(deployment, SelfHostedDeployment)
        assert deployment.host == "192.168.1.100"

    def test_get_deployment_not_found(self, manager):
        """Test getting non-existent deployment."""
        with pytest.raises(KeyError, match="Deployment 'nonexistent' not found"):
            manager.get_deployment("nonexistent")

    def test_plan_self_hosted(self, manager):
        """Test generating plan for self-hosted deployment."""
        mock_deployer = Mock()
        mock_deployer.plan.return_value = {"changes": ["deploy compose file"]}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.plan("production")

            assert result == {"changes": ["deploy compose file"]}
            mock_deployer_cls.assert_called_once_with(
                deployment_name="production",
                deployment=manager.config.deployments["production"],
                state_manager=manager.state_manager,
            )
            mock_deployer.plan.assert_called_once()

    def test_plan_runpod(self, manager):
        """Test generating plan for RunPod deployment."""
        mock_deployer = Mock()
        mock_deployer.plan.return_value = {"changes": ["create template"]}

        with patch("nodetool.deploy.manager.RunPodDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.plan("serverless")

            assert result == {"changes": ["create template"]}
            mock_deployer_cls.assert_called_once()
            mock_deployer.plan.assert_called_once()

    def test_plan_gcp(self, manager):
        """Test generating plan for GCP deployment."""
        mock_deployer = Mock()
        mock_deployer.plan.return_value = {"changes": ["deploy service"]}

        with patch("nodetool.deploy.manager.GCPDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.plan("cloud")

            assert result == {"changes": ["deploy service"]}
            mock_deployer_cls.assert_called_once()
            mock_deployer.plan.assert_called_once()

    def test_plan_deployment_not_found(self, manager):
        """Test plan with non-existent deployment."""
        with pytest.raises(KeyError):
            manager.plan("nonexistent")

    def test_apply_self_hosted(self, manager):
        """Test applying self-hosted deployment."""
        mock_deployer = Mock()
        mock_deployer.apply.return_value = {"success": True}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.apply("production")

            assert result == {"success": True}
            mock_deployer_cls.assert_called_once()
            mock_deployer.apply.assert_called_once_with(dry_run=False)

    def test_apply_with_dry_run(self, manager):
        """Test applying deployment with dry_run flag."""
        mock_deployer = Mock()
        mock_deployer.apply.return_value = {"dry_run": True}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.apply("production", dry_run=True)

            assert result == {"dry_run": True}
            mock_deployer.apply.assert_called_once_with(dry_run=True)

    def test_apply_runpod(self, manager):
        """Test applying RunPod deployment."""
        mock_deployer = Mock()
        mock_deployer.apply.return_value = {"endpoint_id": "ep-123"}

        with patch("nodetool.deploy.manager.RunPodDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.apply("serverless")

            assert result == {"endpoint_id": "ep-123"}
            mock_deployer_cls.assert_called_once()
            mock_deployer.apply.assert_called_once()

    def test_apply_gcp(self, manager):
        """Test applying GCP deployment."""
        mock_deployer = Mock()
        mock_deployer.apply.return_value = {"service_url": "https://example.run.app"}

        with patch("nodetool.deploy.manager.GCPDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.apply("cloud")

            assert result == {"service_url": "https://example.run.app"}
            mock_deployer_cls.assert_called_once()
            mock_deployer.apply.assert_called_once()

    def test_status_self_hosted(self, manager):
        """Test getting status of self-hosted deployment."""
        mock_deployer = Mock()
        mock_deployer.status.return_value = {"status": "running", "containers": 1}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.status("production")

            assert result == {"status": "running", "containers": 1}
            mock_deployer_cls.assert_called_once()
            mock_deployer.status.assert_called_once()

    def test_status_runpod(self, manager):
        """Test getting status of RunPod deployment."""
        mock_deployer = Mock()
        mock_deployer.status.return_value = {"status": "active", "workers": 2}

        with patch("nodetool.deploy.manager.RunPodDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.status("serverless")

            assert result == {"status": "active", "workers": 2}
            mock_deployer.status.assert_called_once()

    def test_status_gcp(self, manager):
        """Test getting status of GCP deployment."""
        mock_deployer = Mock()
        mock_deployer.status.return_value = {"status": "serving", "instances": 1}

        with patch("nodetool.deploy.manager.GCPDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.status("cloud")

            assert result == {"status": "serving", "instances": 1}
            mock_deployer.status.assert_called_once()

    def test_logs_self_hosted(self, manager):
        """Test getting logs from self-hosted deployment."""
        mock_deployer = Mock()
        mock_deployer.logs.return_value = "container logs here"

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.logs("production", service="wf1", tail=50)

            assert result == "container logs here"
            mock_deployer_cls.assert_called_once()
            mock_deployer.logs.assert_called_once_with(
                service="wf1", follow=False, tail=50
            )

    def test_logs_with_follow(self, manager):
        """Test getting logs with follow option."""
        mock_deployer = Mock()
        mock_deployer.logs.return_value = "streaming logs..."

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.logs("production", follow=True)

            assert result == "streaming logs..."
            mock_deployer.logs.assert_called_once_with(
                service=None, follow=True, tail=100
            )

    def test_logs_runpod(self, manager):
        """Test getting logs from RunPod deployment."""
        mock_deployer = Mock()
        mock_deployer.logs.return_value = "pod logs here"

        with patch("nodetool.deploy.manager.RunPodDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.logs("serverless")

            assert result == "pod logs here"
            mock_deployer.logs.assert_called_once()

    def test_logs_gcp(self, manager):
        """Test getting logs from GCP deployment."""
        mock_deployer = Mock()
        mock_deployer.logs.return_value = "cloud run logs here"

        with patch("nodetool.deploy.manager.GCPDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.logs("cloud")

            assert result == "cloud run logs here"
            mock_deployer.logs.assert_called_once()

    def test_destroy_self_hosted(self, manager):
        """Test destroying self-hosted deployment."""
        mock_deployer = Mock()
        mock_deployer.destroy.return_value = {"destroyed": True}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.destroy("production")

            assert result == {"destroyed": True}
            mock_deployer_cls.assert_called_once()
            mock_deployer.destroy.assert_called_once()

    def test_destroy_runpod(self, manager):
        """Test destroying RunPod deployment."""
        mock_deployer = Mock()
        mock_deployer.destroy.return_value = {"deleted": True}

        with patch("nodetool.deploy.manager.RunPodDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.destroy("serverless")

            assert result == {"deleted": True}
            mock_deployer.destroy.assert_called_once()

    def test_destroy_gcp(self, manager):
        """Test destroying GCP deployment."""
        mock_deployer = Mock()
        mock_deployer.destroy.return_value = {"service_deleted": True}

        with patch("nodetool.deploy.manager.GCPDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.destroy("cloud")

            assert result == {"service_deleted": True}
            mock_deployer.destroy.assert_called_once()

    def test_validate_all_deployments(self, manager):
        """Test validating all deployments."""
        result = manager.validate()

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        # May have warnings about SSH authentication
        assert isinstance(result["warnings"], list)

    def test_validate_specific_deployment(self, manager):
        """Test validating a specific deployment."""
        result = manager.validate("production")

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_deployment_not_found(self, manager):
        """Test validating non-existent deployment."""
        result = manager.validate("nonexistent")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "nonexistent" in result["errors"][0]

    def test_validate_self_hosted_no_ssh_auth(self, manager):
        """Test validation warns about missing SSH authentication."""
        # Create deployment without SSH key or password
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            containers=[ContainerConfig(name="wf1", port=8001)],
        )

        manager.config.deployments["no-auth"] = deployment

        result = manager.validate("no-auth")

        assert result["valid"] is True
        assert len(result["warnings"]) > 0
        assert "no-auth" in result["warnings"][0]
        assert "authentication" in result["warnings"][0].lower()

    def test_validate_self_hosted_no_containers(self, manager):
        """Test validation fails for deployment without containers."""
        # Create deployment without containers
        deployment = SelfHostedDeployment(
            host="192.168.1.100",
            ssh=SSHConfig(user="ubuntu", key_path="~/.ssh/id_rsa"),
            image=ImageConfig(name="nodetool/nodetool", tag="latest"),
            containers=[],
        )

        manager.config.deployments["no-containers"] = deployment

        result = manager.validate("no-containers")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "no-containers" in result["errors"][0]
        assert "containers" in result["errors"][0].lower()

    def test_has_changes_true(self, manager):
        """Test has_changes returns True when there are changes."""
        mock_deployer = Mock()
        mock_deployer.plan.return_value = {"changes": ["deploy compose file"]}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.has_changes("production")

            assert result is True

    def test_has_changes_false(self, manager):
        """Test has_changes returns False when there are no changes."""
        mock_deployer = Mock()
        mock_deployer.plan.return_value = {"changes": []}

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.has_changes("production")

            assert result is False

    def test_has_changes_error(self, manager):
        """Test has_changes returns False on error."""
        mock_deployer = Mock()
        mock_deployer.plan.side_effect = Exception("Connection failed")

        with patch("nodetool.deploy.manager.SelfHostedDeployer") as mock_deployer_cls:
            mock_deployer_cls.return_value = mock_deployer

            result = manager.has_changes("production")

            assert result is False

    def test_get_all_states(self, manager):
        """Test getting all deployment states."""
        mock_states = {
            "production": {"status": "running"},
            "serverless": {"status": "active"},
        }
        manager.state_manager.get_all_states.return_value = mock_states

        result = manager.get_all_states()

        assert result == mock_states
        manager.state_manager.get_all_states.assert_called_once()


class TestDeploymentManagerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def empty_config(self):
        """Create an empty deployment config."""
        return DeploymentConfig(deployments={})

    @pytest.fixture
    def manager_empty(self, empty_config):
        """Create a manager with empty config."""
        mock_state_manager = Mock()

        with patch("nodetool.deploy.manager.load_deployment_config") as mock_load:
            with patch("nodetool.deploy.manager.StateManager") as mock_state_cls:
                mock_load.return_value = empty_config
                mock_state_cls.return_value = mock_state_manager

                manager = DeploymentManager()
                manager.config = empty_config
                manager.state_manager = mock_state_manager

                return manager

    def test_list_deployments_empty(self, manager_empty):
        """Test listing deployments when none are configured."""
        deployments = manager_empty.list_deployments()

        assert len(deployments) == 0
        assert isinstance(deployments, list)

    def test_validate_all_empty(self, manager_empty):
        """Test validating when no deployments exist."""
        result = manager_empty.validate()

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0

    def test_unknown_deployment_type(self):
        """Test handling of unknown deployment type."""

        # Create a deployment with invalid type
        class UnknownDeployment:
            def __init__(self):
                self.type = "unknown"

        mock_config = Mock()
        mock_config.deployments = {"unknown": UnknownDeployment()}

        manager = Mock()
        manager.config = mock_config
        manager.get_deployment = lambda name: mock_config.deployments[name]

        # Test that unknown type raises ValueError
        with pytest.raises(ValueError, match="Unknown deployment type"):
            DeploymentManager.plan(manager, "unknown")
