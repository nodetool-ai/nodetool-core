from typing import Any
from unittest.mock import Mock
import httpx
import pytest
from nodetool.types.graph import Node
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def http_client():
    return Mock(httpx.AsyncClient)


@pytest.fixture()
def context(http_client):
    return ProcessingContext(
        user_id="1",
        workflow_id="1",
        auth_token="local_token",
        http_client=http_client,
    )


def make_node(id, type: str, data: dict[str, Any]):
    """
    Create a node for workflow testing.

    Args:
        id: The node ID.
        type (str): The node type identifier.
        data (dict[str, Any]): Node configuration data.

    Returns:
        Node: The created node instance.
    """
    return Node(id=id, type=type, data=data)
