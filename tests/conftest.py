import asyncio
import gc
import io
import os
import threading
import uuid
from typing import Any
from unittest.mock import Mock

import httpx
import PIL.Image
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import Field

from nodetool.api.server import create_app
from nodetool.config.environment import Environment
from nodetool.config.logging_config import configure_logging
from nodetool.deploy.auth import get_worker_auth_token
from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.message import Message
from nodetool.models.run_state import RunState
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import ResourceScope, require_scope
from nodetool.storage.memory_storage import MemoryStorage
from nodetool.types.graph import Edge, Node
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext


async def get_job_status(job_id: str) -> str:
    """Get the authoritative status for a job from RunState."""
    try:
        run_state = await RunState.get(job_id)
        if run_state:
            return run_state.status
    except Exception:
        pass
    return "unknown"


configure_logging("DEBUG")


@pytest.fixture(scope="session")
def worker_id(request):
    """Provide worker_id for pytest-xdist compatibility.

    Returns 'master' when not using xdist (sequential execution),
    or the actual worker id when using xdist (parallel execution).
    """
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


# @pytest.fixture(scope="session", autouse=True)
# def _silence_aiosqlite_logging():
#     """Reduce noisy aiosqlite logs during tests."""
#     import logging

#     for name in (
#         "aiosqlite",
#         "aiosqlite.core",
#         "aiosqlite.cursor",
#         "aiosqlite.connection",
#     ):
#         logger = logging.getLogger(name)
#         logger.setLevel(logging.ERROR)
#         logger.propagate = False


@pytest_asyncio.fixture(scope="session")
async def test_db_pool(worker_id):
    """Create test database once for entire session with connection pool.

    This fixture:
    - Creates a single temporary database file per worker (for pytest-xdist compatibility)
    - Runs migrations once at session start
    - Creates a persistent connection pool
    - Cleans up pool at session end

    Args:
        worker_id: Provided by pytest-xdist to identify the worker process

    Yields:
        tuple: (pool, db_path) for use in tests
    """
    import os
    import tempfile

    from nodetool.models.migrations import run_startup_migrations
    from nodetool.runtime.db_sqlite import SQLiteConnectionPool

    # Create a temporary database file for the entire test session
    # Use worker_id to ensure each xdist worker gets a unique database
    worker_suffix = f"_{worker_id}" if worker_id != "master" else ""
    with tempfile.NamedTemporaryFile(
        suffix=f"{worker_suffix}.sqlite3", prefix="nodetool_test_session_", delete=False
    ) as temp_db:
        db_path = temp_db.name

    pool = None
    try:
        # Create connection pool (will be reused across all tests in this worker)
        pool = await SQLiteConnectionPool.get_shared(db_path)
        # Run migrations once for the session
        await run_startup_migrations(pool)

        yield pool

    finally:
        # Clean up pool at session end
        if pool is not None:
            try:
                await pool.close_all()
                import asyncio

                loop_id = id(asyncio.get_running_loop())
                SQLiteConnectionPool._pools.pop((loop_id, db_path), None)
            except Exception as e:
                import logging

                logging.warning(f"Error cleaning up session connection pool: {e}")

        # Remove temporary database file
        try:
            os.unlink(db_path)
        except Exception:
            pass


async def _truncate_all_tables(pool):
    """Truncate all tables to reset database state between tests."""
    connection = None
    try:
        connection = await pool.acquire()
        cursor = await connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'nodetool_%'"
        )
        tables = await cursor.fetchall()

        # Use a single transaction for all deletes
        for table in tables:
            table_name = table[0]
            try:
                await connection.execute(f"DELETE FROM {table_name}")
            except Exception as e:
                import logging

                logging.debug(f"Failed to truncate table {table_name}: {e}")
                pass

        # Commit all deletes in a single transaction
        await connection.commit()
    except Exception as e:
        # Rollback on error
        import logging

        logging.debug(f"Error during table truncation, rolling back: {e}")
        if connection is not None:
            try:
                await connection.rollback()
            except Exception:
                pass
        raise
    finally:
        if connection is not None:
            await pool.release(connection)


@pytest_asyncio.fixture(autouse=True, scope="function")
async def setup_and_teardown(request, test_db_pool):
    """Set up ResourceScope with table truncation for test isolation.

    This fixture:
    - Uses shared database and connection pool across all tests
    - Provides ResourceScope for each test
    - Truncates all tables after test to ensure clean state
    - Much faster than creating new database per test
    """
    if request.node.get_closest_marker("no_setup"):
        yield
        return

    from nodetool.workflows.job_execution_manager import JobExecutionManager

    # Use ResourceScope with the shared test database
    async with ResourceScope(pool=test_db_pool):
        try:
            yield
        finally:
            # Clean up JobExecutionManager after test
            try:
                if JobExecutionManager._instance is not None:
                    await JobExecutionManager.get_instance().shutdown()
            except Exception:
                pass

    # Truncate all tables to reset state for next test
    # Retry truncation if it fails due to lock (can happen during parallel execution)
    max_truncate_retries = 3
    for attempt in range(max_truncate_retries):
        try:
            await _truncate_all_tables(test_db_pool)
            break
        except Exception as e:
            import logging

            if attempt < max_truncate_retries - 1:
                logging.debug(f"Error truncating tables (attempt {attempt + 1}/{max_truncate_retries}), retrying: {e}")
                await asyncio.sleep(0.1 * (attempt + 1))  # Brief delay before retry
            else:
                logging.warning(f"Error truncating tables after {max_truncate_retries} attempts: {e}")


@pytest.fixture(autouse=True)
def _set_dummy_api_keys(monkeypatch):
    """Provide dummy API keys so provider constructors don't fail in unit tests.

    This fixture provides environment variables for backward compatibility and for
    code paths that check environment variables directly.

    RECOMMENDED: For provider tests, use BaseProviderTest.create_provider() instead,
    which dynamically fetches secrets using provider_class.required_secrets() and
    builds the secrets dict. This is cleaner and doesn't rely on env vars.

    These tests mock network calls; real keys are not required. Setting the env vars
    prevents providers from raising ApiKeyMissingError during initialization.
    """
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "test-openai-key"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", "test-anthropic-key"))
    monkeypatch.setenv("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "test-gemini-key"))
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN", "test-hf-token"))
    monkeypatch.setenv("OLLAMA_API_URL", os.getenv("OLLAMA_API_URL", "http://localhost:11434"))
    monkeypatch.setenv("REPLICATE_API_TOKEN", os.getenv("REPLICATE_API_TOKEN", "test-replicate-token"))
    monkeypatch.setenv("ELEVENLABS_API_KEY", os.getenv("ELEVENLABS_API_KEY", "test-elevenlabs-key"))
    monkeypatch.setenv("FAL_API_KEY", os.getenv("FAL_API_KEY", "test-fal-key"))


@pytest.fixture(autouse=True)
def mock_keyring(monkeypatch):
    """Provide an in-memory keyring implementation for tests.

    The default keyring backend used in CI raises NoKeyringError because no
    secure storage backend is available. Tests that interact with
    ``MasterKeyManager`` need ``keyring`` to behave like a functional backend,
    so we replace ``get_password``/``set_password``/``delete_password`` with a
    simple dictionary-backed store.
    """

    import keyring

    from nodetool.security.master_key import MasterKeyManager

    store: dict[tuple[str, str], str] = {}

    def get_password(service: str, username: str) -> str | None:
        return store.get((service, username))

    def set_password(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def delete_password(service: str, username: str) -> None:
        store.pop((service, username), None)

    monkeypatch.setattr(keyring, "get_password", get_password)
    monkeypatch.setattr(keyring, "set_password", set_password)
    monkeypatch.setattr(keyring, "delete_password", delete_password)

    # Ensure tests don't reuse a cached master key from previous runs
    MasterKeyManager.clear_cache()

    try:
        yield
    finally:
        store.clear()
        MasterKeyManager.clear_cache()


@pytest.fixture(scope="session")
def event_loop():
    """Provide a shared asyncio event loop for the entire test session.

    Session-scoped to support session-scoped async fixtures (test_db_pool).
    Ensures the loop runs at least one cycle before close so the internal
    self-pipe is initialized, preventing AttributeError on close in CPython 3.11.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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
    storage = require_scope().get_asset_storage()
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
    storage = require_scope().get_asset_storage()
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
    token = get_worker_auth_token()
    return {"Authorization": f"Bearer {token}"}


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
    msg = await Message.create(user_id=user_id, thread_id=thread.id, role="user", content="Hello")
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


def pytest_sessionfinish(session, exitstatus):
    """Clean up resources after all tests complete to prevent hanging."""
    import logging
    import os
    import time

    # Force garbage collection
    gc.collect()

    # Close any lingering event loops
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass  # No event loop in current thread

    # Log any non-daemon threads that might prevent exit
    main_thread = threading.main_thread()
    non_daemon_threads = [t for t in threading.enumerate() if t != main_thread and t.is_alive() and not t.daemon]

    if non_daemon_threads:
        logging.warning(
            f"Found {len(non_daemon_threads)} non-daemon threads that may prevent exit: "
            f"{[t.name for t in non_daemon_threads]}"
        )

        # Force exit if there are hanging threads
        # Give threads a brief moment to clean up, then force exit
        def force_exit_thread():
            time.sleep(1)
            os._exit(exitstatus)

        exit_thread = threading.Thread(target=force_exit_thread, daemon=True)
        exit_thread.start()

    # Shutdown any thread pools or executors
    gc.collect()
