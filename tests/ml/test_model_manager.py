from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodetool.config.environment import Environment
from nodetool.ml.core import model_manager
from nodetool.ml.core.model_manager import ModelManager


class FakePsutil:
    """Small stand-in for psutil so tests can deterministically control memory stats."""

    def __init__(self):
        self.percent = 42.0
        self.available = 8 * 1024**3
        self.total = 16 * 1024**3
        self.rss = 2 * 1024**3

    def virtual_memory(self):
        return SimpleNamespace(
            percent=self.percent,
            available=self.available,
            total=self.total,
        )

    def Process(self):
        rss = self.rss

        class _Proc:
            def __init__(self, rss_value: float):
                self._rss_value = rss_value

            def memory_info(self):
                return SimpleNamespace(rss=self._rss_value)

        return _Proc(rss)


@pytest.fixture(autouse=True)
def ensure_non_production_env(monkeypatch):
    """Ensure tests run with a non-production ENV and clean ModelManager state."""
    Environment.set_env("development")
    ModelManager.clear()
    ModelManager._last_memory_cleanup = 0.0
    yield
    ModelManager.clear()
    ModelManager._last_memory_cleanup = 0.0


def test_model_manager_clears_cache_when_memory_pressure_detected(monkeypatch):
    """set_model should trigger cleanup when memory usage exceeds thresholds."""
    fake_psutil = FakePsutil()
    monkeypatch.setattr(model_manager, "psutil", fake_psutil, raising=False)

    ModelManager.set_model("node-1", "modelA", "task", object())
    assert len(ModelManager._models) == 1
    assert "node-1" in ModelManager._models_by_node

    fake_psutil.percent = 99.0
    fake_psutil.available = 0.1 * 1024**3

    ModelManager.set_model("node-2", "modelB", "task", object())

    assert set(ModelManager._models_by_node.keys()) == {"node-2"}
    assert len(ModelManager._models) == 1
    assert ModelManager._last_memory_cleanup > 0


def test_free_memory_if_snapshot_unavailable_triggers_cleanup(monkeypatch):
    """free_memory_if_needed should clear cache when telemetry capture fails."""
    monkeypatch.setattr(
        ModelManager,
        "_capture_memory_snapshot",
        classmethod(lambda cls: None),
    )

    ModelManager._models["modelA_task_None"] = object()
    ModelManager._models_by_node["node-1"] = "modelA_task_None"

    ModelManager.free_memory_if_needed(reason="test cleanup")

    assert ModelManager._models == {}
    assert ModelManager._models_by_node == {}


class TestLockEviction:
    """Evicting a model must not hand out a second lock for the same key.

    Dropping a held lock lets the next ``get_model_lock`` mint a fresh
    ``asyncio.Lock``, so two coroutines could enter the same critical section
    (e.g. both loading the same model into VRAM) at once.
    """

    def setup_method(self):
        ModelManager.clear()

    def teardown_method(self):
        ModelManager.clear()

    @pytest.mark.asyncio
    async def test_unload_keeps_a_held_lock(self):
        ModelManager.set_model("node1", "m", "task", object())
        key = ModelManager._make_cache_key("m", "task")

        async with ModelManager.lock_model(key):
            assert ModelManager.unload_model("m", "task") is True
            # Same lock object, still held — a concurrent waiter blocks.
            same_lock = await ModelManager.get_model_lock(key)
            assert same_lock.locked()

        # Released; a later eviction can now collect it.
        assert ModelManager._discard_lock(key) is True
        assert key not in ModelManager._locks

    @pytest.mark.asyncio
    async def test_unload_drops_an_idle_lock(self):
        ModelManager.set_model("node1", "m", "task", object())
        key = ModelManager._make_cache_key("m", "task")
        await ModelManager.get_model_lock(key)

        assert ModelManager.unload_model("m", "task") is True
        assert key not in ModelManager._locks

    @pytest.mark.asyncio
    async def test_clear_keeps_a_held_lock(self):
        ModelManager.set_model("node1", "m", "task", object())
        key = ModelManager._make_cache_key("m", "task")

        async with ModelManager.lock_model(key):
            ModelManager.clear()
            assert key in ModelManager._locks

    @pytest.mark.asyncio
    async def test_clear_unused_keeps_a_held_lock(self):
        ModelManager.set_model("node1", "m", "task", object())
        key = ModelManager._make_cache_key("m", "task")

        async with ModelManager.lock_model(key):
            ModelManager.clear_unused(["node1"])
            assert key in ModelManager._locks
            assert ModelManager._models == {}
