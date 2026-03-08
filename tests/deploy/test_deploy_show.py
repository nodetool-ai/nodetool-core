"""
Tests for `nodetool deploy show` command.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from nodetool.cli import cli
from nodetool.config.deployment import ContainerConfig, DockerDeployment, ImageConfig


@pytest.fixture
def mock_deployment_manager():
    with patch("nodetool.deploy.manager.DeploymentManager") as MockManager:
        manager = MockManager.return_value

        # Mock deployment
        deployment = DockerDeployment(
            host="127.0.0.1",
            image=ImageConfig(name="test/image", tag="latest"),
            container=ContainerConfig(name="test-container", port=8000),
        )
        manager.get_deployment.return_value = deployment

        # Mock state
        manager.state_manager.read_state.return_value = {"status": "running", "last_deployed": "2024-01-01T12:00:00"}

        yield manager


@patch("nodetool.utils.display.show_deployment_details")
def test_deploy_show_calls_display_function(mock_show_details, mock_deployment_manager):
    """Test that deploy show command calls show_deployment_details with correct arguments."""
    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", "show", "test-deployment"])

    assert result.exit_code == 0

    # Verify manager calls
    mock_deployment_manager.get_deployment.assert_called_with("test-deployment")
    mock_deployment_manager.state_manager.read_state.assert_called_with("test-deployment")

    # Verify display function called with correct args
    mock_show_details.assert_called_once()
    args = mock_show_details.call_args[0]
    assert args[0] == "test-deployment"
    assert isinstance(args[1], DockerDeployment)
    assert args[2] == {"status": "running", "last_deployed": "2024-01-01T12:00:00"}


@patch("nodetool.utils.display.show_deployment_details")
def test_deploy_show_deployment_not_found(mock_show_details, mock_deployment_manager):
    """Test behavior when deployment is not found."""
    mock_deployment_manager.get_deployment.side_effect = KeyError("Deployment not found")

    runner = CliRunner()
    result = runner.invoke(cli, ["deploy", "show", "non-existent"])

    # CLI exits with 1 when deployment is not found, which is expected behavior
    assert result.exit_code == 1
    assert "Deployment 'non-existent' not found" in result.output
    mock_show_details.assert_not_called()
