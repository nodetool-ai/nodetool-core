"""NODETOOL_TORCH_DEVICE selects the torch device, and GPU OOM detection."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import nodetool.workflows.torch_support as ts


def _fake_torch(*, mps: bool, cuda_count: int):
    class FakeOutOfMemoryError(RuntimeError):
        pass

    return SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        cuda=SimpleNamespace(
            is_available=lambda: cuda_count > 0,
            device_count=lambda: cuda_count,
            OutOfMemoryError=FakeOutOfMemoryError,
        ),
        OutOfMemoryError=FakeOutOfMemoryError,
    )


@pytest.fixture
def machine(monkeypatch):
    def configure(*, mps: bool = False, cuda_count: int = 0, override: str | None = None):
        monkeypatch.setattr(ts, "TORCH_AVAILABLE", True)
        monkeypatch.setattr(ts, "torch", _fake_torch(mps=mps, cuda_count=cuda_count))
        monkeypatch.setattr(ts, "_warned_device_overrides", set())
        if override is None:
            monkeypatch.delenv(ts.TORCH_DEVICE_ENV, raising=False)
        else:
            monkeypatch.setenv(ts.TORCH_DEVICE_ENV, override)

    return configure


def test_auto_selection_prefers_mps_then_cuda_then_cpu(machine):
    machine(mps=True, cuda_count=1)
    assert ts.resolve_torch_device() == "mps"
    machine(cuda_count=1)
    assert ts.resolve_torch_device() == "cuda"
    machine()
    assert ts.resolve_torch_device() == "cpu"


@pytest.mark.parametrize(
    ("override", "expected"),
    [("cpu", "cpu"), ("CUDA", "cuda"), ("cuda:1", "cuda:1"), ("mps", "mps"), (" cpu ", "cpu")],
)
def test_valid_override_wins_over_auto_selection(machine, override, expected):
    machine(mps=True, cuda_count=2, override=override)
    assert ts.resolve_torch_device() == expected


@pytest.mark.parametrize(
    "override",
    ["cuda:2", "gpu", "cuda:x", "mps:0", "cpu:1"],
)
def test_invalid_override_warns_once_and_falls_back(machine, override, caplog):
    machine(cuda_count=2, override=override)
    with caplog.at_level("WARNING", logger=ts.log.name):
        assert ts.resolve_torch_device() == "cuda"
        assert ts.resolve_torch_device() == "cuda"
    warnings = [r for r in caplog.records if ts.TORCH_DEVICE_ENV in r.getMessage()]
    assert len(warnings) == 1


def test_override_naming_missing_backend_falls_back(machine):
    machine(override="cuda")
    assert ts.resolve_torch_device() == "cpu"
    machine(cuda_count=1, override="mps")
    assert ts.resolve_torch_device() == "cuda"


def test_explicit_device_wins_over_override(machine):
    machine(cuda_count=1, override="cpu")
    assert ts.resolve_torch_device("cuda") == "cuda"


def test_processing_context_uses_override(machine):
    from nodetool.workflows.processing_context import ProcessingContext

    machine(mps=True, override="cpu")
    assert ProcessingContext().device == "cpu"


def test_gpu_oom_detection_covers_cuda_and_mps(machine):
    machine(cuda_count=1)
    assert ts.is_gpu_oom_exception(ts.torch.OutOfMemoryError("CUDA out of memory"))
    assert ts.is_gpu_oom_exception(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert not ts.is_gpu_oom_exception(RuntimeError("shape mismatch"))
    assert not ts.is_gpu_oom_exception(ValueError("MPS backend out of memory"))


@pytest.mark.parametrize(("preset", "expected"), [(None, "1"), ("0", "0")])
def test_worker_import_sets_mps_fallback_without_overriding_user_value(preset, expected):
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    if preset is not None:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = preset
    code = (
        "import os, sys\n"
        "import nodetool.worker\n"
        "assert 'torch' not in sys.modules\n"
        "print(os.environ['PYTORCH_ENABLE_MPS_FALLBACK'])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == expected


def _fake_cuda_telemetry(calls: list[tuple[str, object]]) -> SimpleNamespace:
    def record(name, result=None):
        def call(device=None):
            calls.append((name, device))
            return result

        return call

    return SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        current_device=lambda: 0,
        synchronize=record("synchronize"),
        mem_get_info=record("mem_get_info", (8 * 1024**3, 24 * 1024**3)),
        memory_allocated=record("memory_allocated", 0),
        memory_reserved=record("memory_reserved", 0),
        get_device_properties=record("get_device_properties", SimpleNamespace(total_memory=24 * 1024**3)),
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
    )


@pytest.mark.parametrize(("override", "expected"), [("cuda:1", 1), ("cuda", 0), (None, 0)])
def test_cuda_device_index_follows_override(machine, monkeypatch, override, expected):
    machine(cuda_count=2, override=override)
    monkeypatch.setattr(ts.torch.cuda, "current_device", lambda: 0, raising=False)
    assert ts.cuda_device_index() == expected


def test_cuda_device_index_is_none_without_cuda(machine):
    machine(mps=True)
    assert ts.cuda_device_index() is None


def test_vram_telemetry_measures_the_override_device(machine, monkeypatch):
    """With cuda:1, VRAM snapshots and the OOM path must not read GPU 0."""
    from types import ModuleType

    from nodetool.ml.core.model_manager import ModelManager

    calls: list[tuple[str, object]] = []
    cuda = _fake_cuda_telemetry(calls)
    machine(cuda_count=2, override="cuda:1")
    monkeypatch.setattr(ts, "torch", SimpleNamespace(backends=ts.torch.backends, cuda=cuda))
    torch_stub = ModuleType("torch")
    torch_stub.cuda = cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    snapshot = ModelManager.get_vram_snapshot()
    support = ts.TorchWorkflowSupport(base_delay=0, max_delay=0, max_retries=1)
    support.get_available_vram()
    support.log_vram_usage(None)

    assert snapshot is not None
    assert calls, "no CUDA telemetry was read"
    assert {device for _, device in calls} == {1}, calls
