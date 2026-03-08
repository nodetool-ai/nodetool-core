"""
Tests for deployment display utility.
"""

from unittest.mock import Mock, patch

import pytest

from nodetool.config.deployment import ContainerConfig, DockerDeployment, ImageConfig
from nodetool.utils.display import show_deployment_details


@patch("nodetool.utils.display.console")
def test_show_deployment_details_executes_without_error(mock_console):
    """Verify that show_deployment_details runs without error and calls console.print."""
    deployment = DockerDeployment(
        host="127.0.0.1",
        image=ImageConfig(name="test/image", tag="latest"),
        container=ContainerConfig(name="test-container", port=8000),
    )
    state = {"status": "running", "last_deployed": "2024-01-01T12:00:00"}

    show_deployment_details("test-deployment", deployment, state)

    mock_console.print.assert_called_once()
