import importlib
import pytest
from nodetool.system import system_stats

class DummyMem:
    total = 16 * 1024 ** 3
    used = 8 * 1024 ** 3
    percent = 50.0

class DummyInfo:
    total = 8 * 1024 ** 3
    used = 4 * 1024 ** 3

@pytest.mark.parametrize('has_gpu', [True, False])
def test_get_system_stats(monkeypatch, has_gpu):
    monkeypatch.setattr(system_stats.psutil, 'cpu_percent', lambda interval=1: 42.0)
    monkeypatch.setattr(system_stats.psutil, 'virtual_memory', lambda: DummyMem)

    monkeypatch.setattr(system_stats.pynvml, 'nvmlInit', lambda: None)
    monkeypatch.setattr(system_stats.pynvml, 'nvmlDeviceGetHandleByIndex', lambda idx: 'h')
    if has_gpu:
        monkeypatch.setattr(system_stats.pynvml, 'nvmlDeviceGetMemoryInfo', lambda h: DummyInfo)
    else:
        def raise_err(h):
            raise system_stats.pynvml.NVMLError(1)
        monkeypatch.setattr(system_stats.pynvml, 'nvmlDeviceGetMemoryInfo', raise_err)
    monkeypatch.setattr(system_stats.pynvml, 'nvmlShutdown', lambda: None)

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

