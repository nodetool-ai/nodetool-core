from typing import Any
from unittest.mock import Mock
from nodetool.models.workflow import Workflow
from pydantic import Field
from fastapi.testclient import TestClient
import httpx
import pytest
import os
import pytest_asyncio
from nodetool.api.server import create_app
from nodetool.storage.memory_storage import MemoryStorage
from nodetool.types.graph import Node, Edge
from nodetool.config.environment import Environment
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.models.job import Job
from nodetool.models.asset import Asset
from nodetool.workflows.base_node import BaseNode, InputNode
from nodetool.workflows.processing_context import ProcessingContext
import io
import uuid
import PIL.Image
import asyncio
from nodetool.models.base_model import close_all_database_adapters


@pytest.fixture(scope="session", autouse=True)
def _silence_aiosqlite_logging():
    """Reduce noisy aiosqlite logs during tests."""
    import logging

    for name in (
        "aiosqlite",
        "aiosqlite.core",
        "aiosqlite.cursor",
        "aiosqlite.connection",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False


@pytest_asyncio.fixture(autouse=True, scope="function")
async def setup_and_teardown(request):
    if request.node.get_closest_marker("no_setup"):
        yield
        return

    Environment.set_remote_auth(False)
    Environment.clear_test_storage()

    # Reset JobExecutionManager singleton for test isolation
    # This prevents tests from interfering with each other
    from nodetool.workflows.job_execution_manager import JobExecutionManager

    if JobExecutionManager._instance is not None:
        manager = JobExecutionManager.get_instance()
        # Cancel all jobs and clean up their resources
        for job_id in list(manager._jobs.keys()):
            try:
                job = manager._jobs.get(job_id)
                if job:
                    if not job.is_completed():
                        cancel_result = job.cancel()
                        if asyncio.iscoroutine(cancel_result):
                            await cancel_result
                    cleanup_result = job.cleanup_resources()
                    if asyncio.iscoroutine(cleanup_result):
                        await cleanup_result
            except Exception:
                pass
        # Clear jobs dict
        manager._jobs.clear()
        # Cancel cleanup task if running
        if manager._cleanup_task and not manager._cleanup_task.done():
            manager._cleanup_task.cancel()
            try:
                await manager._cleanup_task
            except asyncio.CancelledError:
                pass
        manager._cleanup_task = None

    yield

    # Clean up JobExecutionManager after test
    try:
        from nodetool.workflows.job_execution_manager import JobExecutionManager

        if JobExecutionManager._instance is not None:
            manager = JobExecutionManager.get_instance()
            # Cancel all jobs and clean up their resources
            for job_id in list(manager._jobs.keys()):
                try:
                    job = manager._jobs.get(job_id)
                    if job:
                        if not job.is_completed():
                            cancel_result = job.cancel()
                            if asyncio.iscoroutine(cancel_result):
                                await cancel_result
                        cleanup_result = job.cleanup_resources()
                        if asyncio.iscoroutine(cleanup_result):
                            await cleanup_result
                except Exception as e:
                    # Log but don't fail on cleanup errors
                    import logging

                    logging.debug(f"Error cleaning up job {job_id}: {e}")
            # Clear jobs dict
            manager._jobs.clear()
            # Cancel cleanup task if running
            if manager._cleanup_task and not manager._cleanup_task.done():
                manager._cleanup_task.cancel()
                try:
                    await manager._cleanup_task
                except asyncio.CancelledError:
                    pass
            manager._cleanup_task = None
    except Exception:
        pass

    # Clear all database tables for test isolation
    try:
        from nodetool.models.asset import Asset
        from nodetool.models.job import Job
        from nodetool.models.thread import Thread
        from nodetool.models.message import Message
        from nodetool.models.workflow import Workflow
        from nodetool.models.prediction import Prediction

        # Clear tables in order (respecting foreign key constraints if any)
        for model_class in [Message, Job, Prediction, Asset, Workflow, Thread]:
            try:
                adapter = await model_class.adapter()
                # Delete all rows from the table
                await adapter.connection.execute(f"DELETE FROM {adapter.table_name}")
                await adapter.connection.commit()
            except Exception:
                # Ignore errors if table doesn't exist or other issues
                pass
    except Exception:
        # Ignore any errors during cleanup
        pass

    # Close all database connections to prevent SQLite lock issues and leaks
    try:
        await close_all_database_adapters()
    except Exception:
        # Ignore errors during adapter cleanup
        pass

    # Clear thread-local caches
    Environment.clear_thread_caches()

    Environment.set_remote_auth(True)

    # Avoid manual cancellation of event‑loop tasks here; pytest-asyncio/anyio
    # manages loop lifecycle. Force-cancelling unknown tasks can corrupt the
    # loop state and trigger errors like missing _ssock on loop close.


@pytest.fixture(autouse=True)
def _set_dummy_api_keys(monkeypatch):
    """Provide dummy API keys so provider constructors don't fail in unit tests.

    These tests mock network calls; real keys are not required. Setting the env vars
    prevents providers from raising ApiKeyMissingError during initialization.
    """
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "test-openai-key"))
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    )
    monkeypatch.setenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "test-gemini-key"))
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN", "test-hf-token"))


@pytest.fixture(scope="function")
def event_loop():
    """Provide a fresh asyncio event loop per test and close it safely.

    Ensures the loop runs at least one cycle before close so the internal
    self-pipe is initialized, preventing AttributeError on close in CPython 3.11.
    """
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        try:
            loop.call_soon(lambda: None)
            loop.run_until_complete(asyncio.sleep(0))
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


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


async def upload_test_image(image: Asset, width: int = 512, height: int = 512):
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
    content = io.BytesIO(pil_to_bytes(img))
    await storage.upload(image.file_name, content)


async def make_image(
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
    image = await Asset.create(
        user_id=user_id,
        name="test_image",
        parent_id=parent_id,
        content_type="image/jpeg",
        workflow_id=workflow_id,
    )
    await upload_test_image(image, width, height)
    return image


async def make_text(
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
    asset = await Asset.create(
        user_id=user_id,
        name="test_text",
        parent_id=parent_id,
        content_type="text/plain",
        workflow_id=workflow_id,
    )
    storage = Environment.get_asset_storage()
    await storage.upload(asset.file_name, io.BytesIO(content.encode()))
    return asset


async def make_job(user_id: str, **kwargs):
    """
    Create a test job.

    Args:
        user_id (str): The user ID who owns the job.
        **kwargs: Additional job attributes.

    Returns:
        Job: The created job instance.
    """
    return await Job.create(
        workflow_id=str(uuid.uuid4()),
        user_id=user_id,
        **kwargs,
    )


@pytest_asyncio.fixture()
async def image(user_id: str):
    return await make_image(user_id)


@pytest_asyncio.fixture()
async def text_asset(user_id: str):
    return await make_text(user_id, "test content")


@pytest.fixture(scope="session")
def user_id() -> str:
    """User ID for tests - session scoped since it never changes."""
    return "1"


@pytest.fixture(scope="session")
def http_client():
    """Mock HTTP client - session scoped since it's stateless."""
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
    Create a test client for the FastAPI app and ensure it closes.

    Use a context-managed TestClient to ensure proper startup/shutdown
    and avoid event loop cleanup issues across tests.
    """
    with TestClient(create_app()) as c:
        yield c


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


@pytest_asyncio.fixture()
async def thread(user_id: str):
    th = await Thread.create(user_id=user_id)
    return th


@pytest_asyncio.fixture()
async def message(user_id: str, thread: Thread):
    msg = await Message.create(
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

    @classmethod
    def get_node_type(cls) -> str:
        return "tests.conftest.Add"

    async def process(self, context: ProcessingContext) -> int:
        return self.a + self.b


@pytest_asyncio.fixture()
async def workflow(user_id: str):
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
    wf = await Workflow.create(
        user_id=user_id,  # Use the string user_id
        name="test_workflow",
        graph={
            "nodes": [node.model_dump() for node in nodes],
            "edges": [edge.model_dump() for edge in edges],
        },
    )
    return wf
