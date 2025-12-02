import importlib
import sys
import types

import pytest

from nodetool.system import system_stats


class DummyNvml:
    class NVMLError(Exception):
        pass

    @classmethod
    def nvmlInit(cls):
        return None

    @classmethod
    def nvmlShutdown(cls):
        return None

    @classmethod
    def nvmlDeviceGetHandleByIndex(cls, index):
        return f"handle-{index}"

    @classmethod
    def nvmlDeviceGetMemoryInfo(cls, handle):
        return DummyInfo


class DummyMem:
    total = 16 * 1024**3
    used = 8 * 1024**3
    percent = 50.0


class DummyInfo:
    total = 8 * 1024**3
    used = 4 * 1024**3


@pytest.mark.parametrize("has_gpu", [True, False])
def test_get_system_stats(monkeypatch, has_gpu):
    monkeypatch.setattr(system_stats.psutil, "cpu_percent", lambda interval=0.0: 42.0)
    monkeypatch.setattr(system_stats.psutil, "virtual_memory", lambda: DummyMem)

    if has_gpu:
        nvml_impl = DummyNvml
    else:

        class DummyNvmlNoGpu(DummyNvml):
            @classmethod
            def nvmlDeviceGetMemoryInfo(cls, handle):
                raise cls.NVMLError(1)

        nvml_impl = DummyNvmlNoGpu

    dummy_module = types.ModuleType("nvidia")
    dummy_module.nvml = nvml_impl
    monkeypatch.setitem(sys.modules, "nvidia", dummy_module)

    importlib.reload(system_stats)
    stats = system_stats.get_system_stats()
    assert stats.cpu_percent == 42.0
    assert stats.memory_total_gb == 16.0
    assert stats.memory_used_gb == 8.0
    if has_gpu:
        assert stats.vram_total_gb == 8.0
        assert stats.vram_used_gb == 4.0
    else:
        assert stats.vram_total_gb is None
        assert stats.vram_used_gb is None
