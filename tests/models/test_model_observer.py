"""Tests for DBModel observer pattern, ETags, and resource change notifications."""

import pytest
import pytest_asyncio

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import (
    DBField,
    DBModel,
    ModelChangeEvent,
    ModelObserver,
    compute_etag,
)
from nodetool.models.condition_builder import Field

log = get_logger(__name__)


class ObserverTestModel(DBModel):
    @classmethod
    def get_table_schema(cls):
        return {"table_name": "test_observer_table"}

    id: str = DBField(hash_key=True)
    name: str = DBField(default="")


@pytest_asyncio.fixture(scope="function")
async def observer_model():
    """Create and tear down the test table."""
    try:
        await ObserverTestModel.create_table()
    except Exception as e:
        log.info(f"create test table: {e}")
    yield
    ModelObserver.clear()
    await ObserverTestModel.drop_table()


# ------------------------------------------------------------------
# ETag tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compute_etag_deterministic():
    """compute_etag produces the same hash for the same data."""
    data = {"id": "1", "name": "hello"}
    assert compute_etag(data) == compute_etag(data)


@pytest.mark.asyncio
async def test_compute_etag_changes_with_data():
    """compute_etag produces a different hash when data changes."""
    tag1 = compute_etag({"id": "1", "name": "hello"})
    tag2 = compute_etag({"id": "1", "name": "world"})
    assert tag1 != tag2


@pytest.mark.asyncio
async def test_model_get_etag(observer_model):
    """DBModel.get_etag() returns a non-empty string."""
    m = ObserverTestModel(id="1", name="test")
    etag = m.get_etag()
    assert isinstance(etag, str)
    assert len(etag) > 0


@pytest.mark.asyncio
async def test_model_etag_changes_on_update(observer_model):
    """ETag changes when the model is mutated."""
    m = ObserverTestModel(id="1", name="before")
    tag_before = m.get_etag()

    m.name = "after"
    tag_after = m.get_etag()

    assert tag_before != tag_after


# ------------------------------------------------------------------
# Observer pattern tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_create_notification(observer_model):
    """Creating a model fires a CREATED event."""
    events: list[tuple[DBModel, ModelChangeEvent]] = []

    def on_change(instance, event):
        events.append((instance, event))

    ModelObserver.subscribe(on_change, ObserverTestModel)
    await ObserverTestModel.create(id="1", name="test")

    # create() calls save() internally which fires UPDATED, then create() fires CREATED
    created = [e for e in events if e[1] == ModelChangeEvent.CREATED]
    assert len(created) == 1
    assert created[0][0].id == "1"  # type: ignore[attr-defined]

    updated = [e for e in events if e[1] == ModelChangeEvent.UPDATED]
    assert len(updated) >= 1


@pytest.mark.asyncio
async def test_observer_save_notification(observer_model):
    """Saving a model fires an UPDATED event."""
    events: list[tuple[DBModel, ModelChangeEvent]] = []

    def on_change(instance, event):
        events.append((instance, event))

    ModelObserver.subscribe(on_change, ObserverTestModel)

    m = ObserverTestModel(id="2", name="original")
    await m.save()

    updated = [e for e in events if e[1] == ModelChangeEvent.UPDATED]
    assert len(updated) >= 1
    assert updated[0][0].id == "2"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_observer_delete_notification(observer_model):
    """Deleting a model fires a DELETED event."""
    events: list[tuple[DBModel, ModelChangeEvent]] = []

    def on_change(instance, event):
        events.append((instance, event))

    ModelObserver.subscribe(on_change, ObserverTestModel)

    m = await ObserverTestModel.create(id="3", name="to_delete")
    events.clear()

    await m.delete()

    deleted = [e for e in events if e[1] == ModelChangeEvent.DELETED]
    assert len(deleted) == 1
    assert deleted[0][0].id == "3"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_observer_global_subscription(observer_model):
    """A global observer (model_class=None) receives all events."""
    events: list[tuple[DBModel, ModelChangeEvent]] = []

    def on_change(instance, event):
        events.append((instance, event))

    ModelObserver.subscribe(on_change)  # No model_class → global

    await ObserverTestModel.create(id="4", name="global_test")
    assert len(events) > 0


@pytest.mark.asyncio
async def test_observer_unsubscribe(observer_model):
    """After unsubscribing, the callback is no longer called."""
    events: list[tuple[DBModel, ModelChangeEvent]] = []

    def on_change(instance, event):
        events.append((instance, event))

    ModelObserver.subscribe(on_change, ObserverTestModel)
    ModelObserver.unsubscribe(on_change, ObserverTestModel)

    await ObserverTestModel.create(id="5", name="no_notify")
    assert len(events) == 0


@pytest.mark.asyncio
async def test_observer_clear(observer_model):
    """ModelObserver.clear() removes all observers."""

    def on_change(instance, event):
        pass

    ModelObserver.subscribe(on_change, ObserverTestModel)
    ModelObserver.subscribe(on_change)
    ModelObserver.clear()

    assert ModelObserver._observers == {}
