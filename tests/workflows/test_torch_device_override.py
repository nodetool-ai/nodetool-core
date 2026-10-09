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
