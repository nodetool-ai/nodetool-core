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


class MovableModule(FakeModule):
    """Torch-module shape that records ``.to()`` calls and moves its params."""

    def __init__(self, device: str, numel: int = 1000, params: list | None = None):
        if params is not None:
            self._params = params
        else:
            super().__init__(device, numel)
        self.moved_to: list[str] = []

    def to(self, device: str):
        self.moved_to.append(device)
        for param in self._params:
            param.device = device
        return self


class MovablePipeline:
    """Diffusers-pipeline shape whose components may be shared with other keys."""

    def __init__(self, components: dict):
        self.components = components
        self.moved_to: list[str] = []

    def to(self, device: str):
        self.moved_to.append(device)
        for component in self.components.values():
            if hasattr(component, "to"):
                component.to(device)
        return self


def _gb(n: float) -> int:
    """Bytes for ``n`` GB expressed as a FakeParam element count (4 bytes each)."""
    return int(n * 1024**3 / 4)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Make ModelManager believe CUDA is present without installing torch."""
    import sys
    from types import ModuleType

    torch_stub = ModuleType("torch")
    cuda_stub = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        empty_cache=lambda: None,
    )
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    return torch_stub


def _snapshot(available_gb: float, total_gb: float = 24.0):
    from nodetool.ml.core.model_manager import VramSnapshot

    return VramSnapshot(
        percent=100.0 * (1 - available_gb / total_gb),
        available_gb=available_gb,
        total_gb=total_gb,
        process_allocated_gb=total_gb - available_gb,
    )


class TestOffloadAccounting:
    """``_offload_gpu_models_until_free`` must not credit shared weights twice."""

    def test_shared_weights_are_counted_once(self, monkeypatch, fake_cuda):
        # A transformer cached on its own AND as a component of a pipeline is
        # the normal case on the HF side, not a bug: the same 4 GB of weights
        # legitimately live under two cache keys. Crediting them per key makes
        # the loop believe it freed 8 GB and stop with VRAM still occupied.
        shared_params = [FakeParam("cuda:0", numel=_gb(4.0))]
        transformer = MovableModule("cuda:0", params=shared_params)
        pipeline = MovablePipeline(
            {
                "transformer": transformer,
                "vae": MovableModule("cuda:0", numel=_gb(1.0)),
            }
        )
        other = MovableModule("cuda:0", numel=_gb(3.0))

        ModelManager.set_model("node-t", "transformer", transformer)
        ModelManager.set_model("node-p", "pipeline", pipeline)
        ModelManager.set_model("node-o", "other", other)
        # LRU order: transformer, then pipeline, then other.
        ModelManager._model_last_used.update({"transformer": 1.0, "pipeline": 2.0, "other": 3.0})

        # Freeze telemetry so the loop's running tally is the only thing that
        # decides when to stop.
        monkeypatch.setattr(ModelManager, "_capture_vram_snapshot", classmethod(lambda cls: None))

        ModelManager._offload_gpu_models_until_free(
            target_free_gb=6.0,
            snapshot=_snapshot(available_gb=0.0),
            reason="test",
        )

        # Real bytes freed: 4 (transformer) + 1 (vae) = 5 GB, short of the 6 GB
        # target, so `other` must also be offloaded. Double-counting the shared
        # 4 GB would reach a phantom 9 GB and leave `other` on the GPU.
        # The transformer is moved twice — once on its own key, once through the
        # pipeline that shares it. That is the aliasing, and it is fine.
        assert transformer.moved_to == ["cpu", "cpu"]
        assert pipeline.moved_to == ["cpu"]
        assert other.moved_to == ["cpu"], "loop stopped early — shared weights were counted twice"

    def test_shared_weights_counted_once_when_the_pipeline_goes_first(self, monkeypatch, fake_cuda):
        """Reverse order: the aliased key is already on CPU by the time it comes up.

        Its recorded ``_model_size_bytes`` still says 4 GB. Falling back to that
        number would re-credit weights the pipeline offload already moved.
        """
        shared_params = [FakeParam("cuda:0", numel=_gb(4.0))]
        transformer = MovableModule("cuda:0", params=shared_params)
        pipeline = MovablePipeline(
            {
                "transformer": transformer,
                "vae": MovableModule("cuda:0", numel=_gb(1.0)),
            }
        )
        other = MovableModule("cuda:0", numel=_gb(3.0))

        ModelManager.set_model("node-p", "pipeline", pipeline)
        ModelManager.set_model("node-t", "transformer", transformer)
        ModelManager.set_model("node-o", "other", other)
        ModelManager._model_last_used.update({"pipeline": 1.0, "transformer": 2.0, "other": 3.0})

        monkeypatch.setattr(ModelManager, "_capture_vram_snapshot", classmethod(lambda cls: None))

        ModelManager._offload_gpu_models_until_free(
            target_free_gb=6.0,
            snapshot=_snapshot(available_gb=0.0),
            reason="test",
        )

        assert other.moved_to == ["cpu"], "loop stopped early — shared weights were counted twice"

    def test_stops_once_the_target_is_genuinely_reached(self, monkeypatch, fake_cuda):
        first = MovableModule("cuda:0", numel=_gb(5.0))
        second = MovableModule("cuda:0", numel=_gb(5.0))
        ModelManager.set_model("node-1", "first", first)
        ModelManager.set_model("node-2", "second", second)
        ModelManager._model_last_used.update({"first": 1.0, "second": 2.0})

        monkeypatch.setattr(ModelManager, "_capture_vram_snapshot", classmethod(lambda cls: None))

        ModelManager._offload_gpu_models_until_free(
            target_free_gb=4.0,
            snapshot=_snapshot(available_gb=0.0),
            reason="test",
        )

        assert first.moved_to == ["cpu"]
        assert second.moved_to == []

    def test_models_pinned_by_an_execution_scope_are_never_offloaded(self, monkeypatch, fake_cuda):
        in_use = MovableModule("cuda:0", numel=_gb(5.0))
        idle = MovableModule("cuda:0", numel=_gb(5.0))
        ModelManager.set_model("node-1", "in_use", in_use)
        ModelManager.set_model("node-2", "idle", idle)
        ModelManager._model_last_used.update({"in_use": 1.0, "idle": 2.0})

        monkeypatch.setattr(ModelManager, "_capture_vram_snapshot", classmethod(lambda cls: None))

        with ModelManager.execution_scope():
            # The executing node fetches its model; that pins it.
            assert ModelManager.get_model("in_use") is in_use
            ModelManager._offload_gpu_models_until_free(
                target_free_gb=100.0,
                snapshot=_snapshot(available_gb=0.0),
                reason="test",
            )

        assert in_use.moved_to == [], "offloaded a model the running node is using"
        assert idle.moved_to == ["cpu"]


class TestExecutionScope:
    def test_scope_pins_keys_only_while_open(self):
        ModelManager.set_model("node-1", "m", FakeModule("cuda:0"))

        assert ModelManager.is_model_in_use("m") is False
        with ModelManager.execution_scope():
            ModelManager.get_model("m")
            assert ModelManager.is_model_in_use("m") is True
        assert ModelManager.is_model_in_use("m") is False

    def test_scopes_nest(self):
        with ModelManager.execution_scope():
            ModelManager.get_model("outer")
            with ModelManager.execution_scope():
                ModelManager.get_model("inner")
                assert ModelManager.is_model_in_use("outer") is True
                assert ModelManager.is_model_in_use("inner") is True
            assert ModelManager.is_model_in_use("inner") is False
            assert ModelManager.is_model_in_use("outer") is True
        assert ModelManager._active_scopes == {}

    @pytest.mark.asyncio
    async def test_concurrent_tasks_get_independent_scopes(self):
        import asyncio

        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            with ModelManager.execution_scope():
                ModelManager.get_model("held")
                started.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await started.wait()
        try:
            with ModelManager.execution_scope():
                # The other task's key is in use, but not "ours".
                assert ModelManager.is_model_in_use("held") is True
                assert ModelManager.is_model_in_use("held", include_current_scope=False) is True
                ModelManager.get_model("mine")
                assert ModelManager.is_model_in_use("mine", include_current_scope=False) is False
        finally:
            release.set()
            await task

        assert ModelManager.is_model_in_use("held") is False


class TestReleaseNodes:
    def test_releases_models_owned_by_retired_nodes(self):
        ModelManager.set_model("node-1", "m1", FakeModule("cpu"))
        ModelManager.set_model("node-2", "m2", FakeModule("cpu"))

        ModelManager.release_nodes(["node-1"])

        assert set(ModelManager._models) == {"m2"}
        assert "node-1" not in ModelManager._models_by_node
        assert "m1" not in ModelManager._model_size_bytes

    def test_keeps_keys_another_node_still_references(self):
        shared = FakeModule("cpu")
        ModelManager.set_model("node-1", "shared", shared)
        ModelManager.set_model("node-2", "shared", shared)

        ModelManager.release_nodes(["node-1"])

        assert "shared" in ModelManager._models
        assert ModelManager._models_by_node["node-2"] == {"shared"}

    def test_keeps_models_pinned_by_a_running_node(self):
        ModelManager.set_model("node-1", "m1", FakeModule("cpu"))

        with ModelManager.execution_scope():
            ModelManager.get_model("m1")
            ModelManager.release_nodes(["node-1"])
            assert "m1" in ModelManager._models
            # The association survives so the key is not orphaned.
            assert ModelManager._models_by_node["node-1"] == {"m1"}

        # Once the scope closes the release actually takes effect.
        ModelManager.release_nodes(["node-1"])
        assert ModelManager._models == {}

    def test_clear_unused_is_an_alias_for_release_nodes(self):
        ModelManager.set_model("node-1", "m1", FakeModule("cpu"))
        ModelManager.clear_unused(["node-1"])
        assert ModelManager._models == {}


class TestEvictAllExceptRespectsScopes:
    def test_skips_models_used_by_a_concurrent_scope(self):
        import asyncio

        async def scenario():
            started = asyncio.Event()
            release = asyncio.Event()

            async def holder():
                with ModelManager.execution_scope():
                    ModelManager.get_model("busy")
                    started.set()
                    await release.wait()

            ModelManager.set_model("node-1", "busy", FakeModule("cuda:0"))
            ModelManager.set_model("node-2", "idle", FakeModule("cuda:0"))

            task = asyncio.create_task(holder())
            await started.wait()
            try:
                evicted = ModelManager.evict_all_except("keep-me")
            finally:
                release.set()
                await task
            return evicted

        evicted = asyncio.run(scenario())

        assert evicted == ["idle"]
        assert "busy" in ModelManager._models
