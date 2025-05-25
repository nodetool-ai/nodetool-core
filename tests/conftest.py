import asyncio
from typing import Any
from unittest.mock import Mock
from pydantic import Field
from fastapi.testclient import TestClient
import httpx
import pytest
from nodetool.api.server import create_app
from nodetool.storage.memory_storage import MemoryStorage
from nodetool.types.graph import Node, Edge
from nodetool.common.environment import Environment
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow
from nodetool.models.job import Job
from nodetool.models.asset import Asset
from nodetool.models.schema import create_all_tables, drop_all_tables
import PIL.ImageChops
from nodetool.workflows.base_node import BaseNode, InputNode
from nodetool.workflows.processing_context import ProcessingContext
import io
import uuid
import PIL.Image


@pytest.fixture(autouse=True, scope="function")
def setup_and_teardown():
    Environment.set_remote_auth(False)

    create_all_tables()

    yield

    drop_all_tables()
    Environment.set_remote_auth(True)


def pil_to_bytes(image: PIL.Image.Image, format="PNG") -> bytes:
    """
    Convert a PIL.Image.Image to bytes.

    Args:
        image (PIL.Image.Image): The image to convert.
        format (str, optional): The format to use. Defaults to "PNG".

    Returns:
        bytes: The image as bytes.
    """
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return buffer.getvalue()


def upload_test_image(image: Asset, width: int = 512, height: int = 512):
    """
    Upload a test image to the memory storage.

    Args:
        image (Asset): The asset to upload the image for.
        width (int, optional): Width of the test image. Defaults to 512.
        height (int, optional): Height of the test image. Defaults to 512.
    """
    storage = Environment.get_asset_storage()
    assert isinstance(storage, MemoryStorage)
    img = PIL.Image.new("RGB", (width, height))
    storage.storage[image.file_name] = pil_to_bytes(img)


def make_image(
    user_id: str,
    workflow_id: str | None = None,
    parent_id: str | None = None,
    width: int = 512,
    height: int = 512,
) -> Asset:
    """
    Create and upload a test image asset.

    Args:
        user_id (str): The user ID who owns the image.
        workflow_id (str | None, optional): Associated workflow ID. Defaults to None.
        parent_id (str | None, optional): Parent asset ID. Defaults to None.
        width (int, optional): Width of the test image. Defaults to 512.
        height (int, optional): Height of the test image. Defaults to 512.

    Returns:
        Asset: The created image asset.
    """
    image = Asset.create(
        user_id=user_id,
        name="test_image",
        parent_id=parent_id,
        content_type="image/jpeg",
        workflow_id=workflow_id,
    )
    upload_test_image(image, width, height)
    return image


def make_text(
    user_id: str,
    content: str,
    workflow_id: str | None = None,
    parent_id: str | None = None,
):
    """
    Create and upload a test text asset.

    Args:
        user_id (str): The user ID who owns the text asset.
        content (str): The text content to upload.
        workflow_id (str | None, optional): Associated workflow ID. Defaults to None.
        parent_id (str | None, optional): Parent asset ID. Defaults to None.

    Returns:
        Asset: The created text asset.
    """
    asset = Asset.create(
        user_id=user_id,
        name="test_text",
        parent_id=parent_id,
        content_type="text/plain",
        workflow_id=workflow_id,
    )
    storage = Environment.get_asset_storage()
    asyncio.run(storage.upload(asset.file_name, io.BytesIO(content.encode())))
    return asset


def make_job(user_id: str, **kwargs):
    """
    Create a test job.

    Args:
        user_id (str): The user ID who owns the job.
        **kwargs: Additional job attributes.

    Returns:
        Job: The created job instance.
    """
    return Job.create(
        workflow_id=str(uuid.uuid4()),
        user_id=user_id,
        **kwargs,
    )


@pytest.fixture()
def image(user_id: str):
    return make_image(user_id)


@pytest.fixture()
def text_asset(user_id: str):
    return make_text(user_id, "test content")


@pytest.fixture()
def user_id() -> str:
    return "1"


@pytest.fixture
def http_client():
    return Mock(httpx.AsyncClient)


@pytest.fixture()
def context(user_id: str, http_client):
    test_auth_token = "test_token"
    return ProcessingContext(
        user_id=user_id,
        workflow_id="1",
        auth_token=test_auth_token,
        http_client=http_client,
    )


@pytest.fixture()
def client():
    """
    Create a test client for the FastAPI app.

    This fixture is scoped to the module, so it will only be created once for the entire test run.
    """
    return TestClient(create_app())


@pytest.fixture()
def headers(user_id: str):
    """
    Create headers for a http request that requires authentication.

    This fixture is scoped to the function, so it will be created once for each test function.
    """
    test_auth_token = "test_token"
    return {"Authorization": f"Bearer {test_auth_token}"}


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


@pytest.fixture()
def thread(user_id: str):
    th = Thread.create(user_id=user_id)
    return th


@pytest.fixture()
def message(user_id: str, thread: Thread):
    msg = Message.create(
        user_id=user_id, thread_id=thread.id, role="user", content="Hello"
    )
    return msg


class FloatInput(InputNode):
    value: float = Field(default=0)

    @classmethod
    def get_node_type(cls) -> str:
        return "nodetool.input.FloatInput"

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Add(BaseNode):
    a: int = 0
    b: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.a + self.b


@pytest.fixture()
def workflow(user_id: str):
    # Restore graph definition from previous version
    nodes = [
        make_node("1", FloatInput.get_node_type(), {"name": "in1", "value": 10}),
        make_node("2", Add.get_node_type(), {"b": 1, "a": 1}),
    ]
    edges = [
        Edge(
            source=nodes[0].id,
            target=nodes[1].id,
            sourceHandle="output",
            targetHandle="a",
        ),
    ]
    wf = Workflow.create(
        user_id=user_id,  # Use the string user_id
        name="test_workflow",
        graph={
            "nodes": [node.model_dump() for node in nodes],
            "edges": [edge.model_dump() for edge in edges],
        },
    )
    return wf
