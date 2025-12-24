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
