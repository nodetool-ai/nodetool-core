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
from pydantic import Field

from nodetool.config.environment import Environment
from nodetool.config.logging_config import configure_logging
from nodetool.runtime.resources import ResourceScope, require_scope
from nodetool.storage.memory_storage import MemoryStorage
from nodetool.types.api_graph import Edge, Node
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext

configure_logging("DEBUG")


@pytest.fixture(scope="session")
def worker_id(request):
    if hasattr(request.config, "workerinput"):
        return request.config.workerinput["workerid"]
    return "master"


@pytest_asyncio.fixture(autouse=True, scope="function")
async def setup_and_teardown(request):
    if request.node.get_closest_marker("no_setup"):
        yield
        return

    scope = ResourceScope()
    # Use memory storage for tests
    scope._asset_storage = MemoryStorage(base_url="http://localhost:7777/api/storage")
    scope._temp_storage = MemoryStorage(base_url="http://localhost:7777/api/temp")
    async with scope:
        yield


@pytest.fixture(autouse=True)
def _set_dummy_api_keys(monkeypatch):
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

    MasterKeyManager.clear_cache()
    try:
        yield
    finally:
        store.clear()
        MasterKeyManager.clear_cache()


@pytest.fixture(scope="session")
def event_loop():
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
    with io.BytesIO() as buffer:
        image.save(buffer, format=format)
        return buffer.getvalue()


async def upload_test_image(asset_id: str, ext: str = "jpeg", width: int = 512, height: int = 512):
    storage = require_scope().get_asset_storage()
    assert isinstance(storage, MemoryStorage)
    img = PIL.Image.new("RGB", (width, height))
    content = io.BytesIO(pil_to_bytes(img))
    await storage.upload(f"{asset_id}.{ext}", content)


async def make_image(
    user_id: str,
    workflow_id: str | None = None,
    parent_id: str | None = None,
    width: int = 512,
    height: int = 512,
) -> dict[str, Any]:
    asset_id = str(uuid.uuid4())
    await upload_test_image(asset_id, "jpeg", width, height)
    return {
        "id": asset_id,
        "name": "test_image",
        "content_type": "image/jpeg",
        "file_name": f"{asset_id}.jpeg",
        "user_id": user_id,
    }


async def make_text(
    user_id: str,
    content: str,
    workflow_id: str | None = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    asset_id = str(uuid.uuid4())
    storage = require_scope().get_asset_storage()
    await storage.upload(f"{asset_id}.txt", io.BytesIO(content.encode()))
    return {
        "id": asset_id,
        "name": "test_text",
        "content_type": "text/plain",
        "file_name": f"{asset_id}.txt",
        "user_id": user_id,
    }


@pytest_asyncio.fixture()
async def image(user_id: str):
    return await make_image(user_id)


@pytest_asyncio.fixture()
async def text_asset(user_id: str):
    return await make_text(user_id, "test content")


@pytest.fixture(scope="session")
def user_id() -> str:
    return "1"


@pytest.fixture(scope="session")
def http_client():
    return Mock(httpx.AsyncClient)


@pytest.fixture()
def context(user_id: str, http_client):
    return ProcessingContext(
        user_id=user_id,
        workflow_id="1",
        auth_token="test_token",
        http_client=http_client,
    )


def make_node(id, type: str, data: dict[str, Any]):
    return Node(id=id, type=type, data=data)


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


def pytest_sessionfinish(session, exitstatus):
    gc.collect()

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass

    main_thread = threading.main_thread()
    non_daemon_threads = [t for t in threading.enumerate() if t != main_thread and t.is_alive() and not t.daemon]

    if non_daemon_threads:
        import logging
        import time

        logging.warning(
            f"Found {len(non_daemon_threads)} non-daemon threads that may prevent exit: "
            f"{[t.name for t in non_daemon_threads]}"
        )

        def force_exit_thread():
            time.sleep(1)
            os._exit(exitstatus)

        exit_thread = threading.Thread(target=force_exit_thread, daemon=True)
        exit_thread.start()

    gc.collect()
