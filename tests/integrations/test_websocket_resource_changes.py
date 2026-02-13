"""Tests for WebSocket resource change updates."""

import asyncio
import json

import pytest
import pytest_asyncio

from nodetool.config.logging_config import get_logger
from nodetool.integrations.websocket.websocket_updates import (
    ResourceChangeUpdate,
    WebSocketUpdates,
)
from nodetool.models.base_model import (
    DBField,
    DBModel,
    ModelChangeEvent,
    ModelObserver,
)

log = get_logger(__name__)


class WsTestModel(DBModel):
    @classmethod
    def get_table_schema(cls):
        return {"table_name": "test_ws_table"}

    id: str = DBField(hash_key=True)
    name: str = DBField(default="")


@pytest_asyncio.fixture(scope="function")
async def ws_model():
    """Create and tear down the test table."""
    try:
        await WsTestModel.create_table()
    except Exception as e:
        log.info(f"create test table: {e}")
    yield
    ModelObserver.clear()
    await WsTestModel.drop_table()


# ------------------------------------------------------------------
# ResourceChangeUpdate serialisation tests
# ------------------------------------------------------------------


def test_resource_change_update_serialisation():
    """ResourceChangeUpdate serialises to the expected JSON structure."""
    update = ResourceChangeUpdate(
        event="created",
        resource_type="workflow",
        resource={"id": "abc", "etag": "deadbeef"},
    )
    data = json.loads(update.model_dump_json())
    assert data["type"] == "resource_change"
    assert data["event"] == "created"
    assert data["resource_type"] == "workflow"
    assert data["resource"]["id"] == "abc"
    assert data["resource"]["etag"] == "deadbeef"


def test_resource_change_update_type_literal():
    """The 'type' field is always 'resource_change'."""
    update = ResourceChangeUpdate(
        event="deleted",
        resource_type="asset",
        resource={"id": "x"},
    )
    assert update.type == "resource_change"


# ------------------------------------------------------------------
# WebSocketUpdates observer integration tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_updates_registers_observer():
    """Connecting a client registers a model observer."""
    ws_updates = WebSocketUpdates()
    assert ws_updates._observer_registered is False

    # Simulate registration without a real WebSocket
    ws_updates._loop = asyncio.get_running_loop()
    ws_updates._register_observer()
    assert ws_updates._observer_registered is True

    ws_updates._unregister_observer()
    assert ws_updates._observer_registered is False


@pytest.mark.asyncio
async def test_websocket_updates_on_model_change_builds_update(ws_model):
    """_on_model_change creates a ResourceChangeUpdate and schedules broadcast."""
    ws_updates = WebSocketUpdates()
    ws_updates._loop = asyncio.get_running_loop()

    # Track broadcast calls
    broadcast_calls: list[ResourceChangeUpdate] = []

    async def mock_broadcast(update):
        if isinstance(update, ResourceChangeUpdate):
            broadcast_calls.append(update)

    ws_updates.broadcast_update = mock_broadcast  # type: ignore[assignment]

    # Simulate an active connection so the callback doesn't short-circuit
    class FakeWS:
        pass

    ws_updates.active_connections.add(FakeWS())  # type: ignore[arg-type]
    ws_updates._register_observer()

    # Trigger a model change
    await WsTestModel.create(id="ws1", name="hello")

    # Allow the scheduled coroutine to run
    await asyncio.sleep(0.1)

    # Verify at least one resource_change was broadcast
    assert len(broadcast_calls) > 0
    assert broadcast_calls[0].resource_type == "wstestmodel"
    assert broadcast_calls[0].resource["id"] == "ws1"
    assert "etag" in broadcast_calls[0].resource

    ws_updates._unregister_observer()
    ws_updates.active_connections.clear()


# ------------------------------------------------------------------
# ETag in API types tests
# ------------------------------------------------------------------


def test_workflow_type_has_etag():
    """Workflow API type includes an etag field."""
    from nodetool.types.workflow import Workflow

    assert "etag" in Workflow.model_fields


def test_asset_type_has_etag():
    """Asset API type includes an etag field."""
    from nodetool.types.asset import Asset

    assert "etag" in Asset.model_fields


def test_thread_type_has_etag():
    """Thread API type includes an etag field."""
    from nodetool.types.thread import Thread

    assert "etag" in Thread.model_fields


def test_job_type_has_etag():
    """Job API type includes an etag field."""
    from nodetool.types.job import Job

    assert "etag" in Job.model_fields
