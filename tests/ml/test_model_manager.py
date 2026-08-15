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


class FakeParam:
    """Stand-in for a torch parameter."""

    def __init__(self, device: str, numel: int = 1000, element_size: int = 4):
        self.device = device
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class FakeModule:
    def __init__(self, device: str, numel: int = 1000):
        self._params = [FakeParam(device, numel)]

    def parameters(self):
        return list(self._params)


class FakePipeline:
    """Diffusers-pipeline shape: no .parameters(), modules under .components."""

    def __init__(self):
        self.components = {
            "unet": FakeModule("cuda:0", numel=2000),
            "vae": FakeModule("cpu", numel=500),
            "scheduler": object(),  # non-module component
            "feature_extractor": None,
        }


class TestDeviceAndSizeDetection:
    def test_torch_module_detected(self):
        device, size = ModelManager._detect_torch_model_device_and_size(FakeModule("cuda:0", numel=10))
        assert device == "cuda:0"
        assert size == 40

    def test_diffusers_pipeline_detected_via_components(self):
        """A pipeline exposes neither .parameters() nor tensor attrs — its
        components must still be seen, or VRAM eviction walks past exactly
        the objects holding the memory."""
        device, size = ModelManager._detect_torch_model_device_and_size(FakePipeline())
        assert device == "cuda:0"  # hottest device wins over cpu components
        assert size == (2000 + 500) * 4

    def test_unknown_object_stays_unknown(self):
        device, size = ModelManager._detect_torch_model_device_and_size(object())
        assert device == "unknown"
        assert size is None


class TestEvictAllExcept:
    def test_evicts_everything_but_the_kept_key(self):
        ModelManager.set_model("node-1", "modelA_task", FakeModule("cuda:0"))
        ModelManager.set_model("node-2", "modelB_task", FakeModule("cuda:0"))

        evicted = ModelManager.evict_all_except("modelB_task")

        assert evicted == ["modelA_task"]
        assert set(ModelManager._models.keys()) == {"modelB_task"}
        assert "node-1" not in ModelManager._models_by_node
        assert "modelA_task" not in ModelManager._model_last_used

    def test_none_evicts_all(self):
        ModelManager.set_model("node-1", "modelA_task", FakeModule("cpu"))
        evicted = ModelManager.evict_all_except(None)
        assert evicted == ["modelA_task"]
        assert ModelManager._models == {}


def test_vram_snapshot_usable_credits_reclaimable():
    from nodetool.ml.core.model_manager import VramSnapshot

    snap = VramSnapshot(
        percent=90.0,
        available_gb=1.0,
        total_gb=24.0,
        process_allocated_gb=15.0,
        reclaimable_gb=6.0,
    )
    assert snap.usable_gb == 7.0
    # 24 - 7 usable = 17/24 ≈ 70.8% used — under the 92% threshold, so a
    # steady state with big reserved-but-free blocks is not treated as pressure.
    assert ModelManager._needs_vram_cleanup(snap, required_free_gb=None) is False
    # But a genuine requirement larger than usable still triggers cleanup.
    assert ModelManager._needs_vram_cleanup(snap, required_free_gb=10.0) is True
