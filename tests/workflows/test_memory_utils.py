import sys
import time
from unittest.mock import MagicMock, call, patch

import pytest

from nodetool.workflows import memory_utils


class TestMemoryUtils:
    @patch("nodetool.workflows.memory_utils.psutil.Process")
    @patch("nodetool.workflows.memory_utils.os.getpid")
    def test_get_memory_usage_mb(self, mock_getpid, mock_process):
        mock_getpid.return_value = 12345
        mock_process_instance = mock_process.return_value
        mock_process_instance.memory_info.return_value.rss = 104857600  # 100 MB

        usage = memory_utils.get_memory_usage_mb()
        assert usage == 100.0
        mock_process.assert_called_with(12345)

    def test_get_gpu_memory_usage_mb_available(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 209715200  # 200 MB
            mock_torch.cuda.memory_reserved.return_value = 314572800  # 300 MB

            usage = memory_utils.get_gpu_memory_usage_mb()
            assert usage == (200.0, 300.0)

    def test_get_gpu_memory_usage_mb_unavailable(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = False
            usage = memory_utils.get_gpu_memory_usage_mb()
            assert usage is None

    @patch("nodetool.workflows.memory_utils.log")
    @patch("nodetool.workflows.memory_utils.get_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    def test_log_memory(self, mock_get_gpu, mock_get_mem, mock_log):
        mock_get_mem.return_value = 100.0
        mock_get_gpu.return_value = (200.0, 300.0)

        memory_utils.log_memory("test_label", include_gpu=True)

        mock_log.info.assert_any_call("[MEMORY] test_label: RAM=100.0MB")
        mock_log.info.assert_any_call("[MEMORY] test_label: GPU allocated=200.0MB, reserved=300.0MB")

    @patch("nodetool.workflows.memory_utils.log")
    @patch("nodetool.workflows.memory_utils.get_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    @patch("gc.collect")
    def test_run_gc(self, mock_gc_collect, mock_get_gpu, mock_get_mem, mock_log):
        mock_get_mem.side_effect = [100.0, 50.0]  # before, after
        mock_get_gpu.return_value = (200.0, 300.0)

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            freed = memory_utils.run_gc("test_gc")

            assert freed == 50.0
            mock_gc_collect.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.synchronize.assert_called_once()

            mock_log.info.assert_any_call("[GC] test_gc - Before GC: RAM=100.0MB")
            mock_log.info.assert_any_call("[GC] test_gc - After GC: RAM=50.0MB (freed 50.0MB)")
            mock_log.info.assert_any_call("[GC] test_gc - GPU after: allocated=200.0MB, reserved=300.0MB")

    @patch("nodetool.workflows.memory_utils.log")
    @patch("nodetool.workflows.memory_utils.get_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.run_gc")
    def test_memory_tracker(self, mock_run_gc, mock_get_gpu, mock_get_mem, mock_log):
        mock_get_mem.side_effect = [100.0, 120.0]
        mock_get_gpu.side_effect = [(200.0, 300.0), (220.0, 320.0)]

        with memory_utils.MemoryTracker("test_tracker", run_gc_after=True):
            pass

        mock_log.info.assert_any_call("[MEMORY TRACK] test_tracker - START: RAM=100.0MB")
        mock_log.info.assert_any_call("[MEMORY TRACK] test_tracker - END: RAM=120.0MB (delta: +20.0MB)")
        mock_run_gc.assert_called_with("test_tracker cleanup", log_before_after=True)

    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.time.time")
    def test_gpu_iteration_tracer(self, mock_time, mock_get_gpu):
        mock_time.side_effect = [1000.0, 1001.0]  # start, end
        mock_get_gpu.side_effect = [(100.0, 200.0), (110.0, 210.0)]  # start, end

        tracer = memory_utils.GPUIterationTracer(report_interval=1)
        tracer.start_iteration(0)
        stats = tracer.end_iteration(0)

        assert stats["iteration"] == 0
        assert stats["duration_ms"] == 1000.0
        assert stats["gpu_allocated_delta_mb"] == 10.0
        assert tracer.should_report(0)

        report = tracer.get_iteration_report(0)
        assert "Iter 0" in report
        assert "GPU=110.0MB" in report

    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.time.time")
    def test_gpu_trace_session(self, mock_time, mock_get_gpu):
        # We need to return values for every call to time.time() and get_gpu_memory_usage_mb()
        # session.start(): time(), _take_snapshot -> get_mem, get_gpu
        # iteration(0) enter: get_gpu, time(), should_log -> _take_snapshot -> get_mem, get_gpu
        # iteration(0) exit: time(), get_gpu, should_log -> _take_snapshot -> get_mem, get_gpu
        # session.finish(): _take_snapshot -> get_mem, get_gpu

        # Simplified: let's just use side_effect or return_value that changes over time if we can, or just lists
        # But wait, `get_memory_usage_mb` is also called in `_take_snapshot`.

        with patch("nodetool.workflows.memory_utils.get_memory_usage_mb", return_value=100.0):
            mock_time.side_effect = list(range(1000, 1100))  # plenty of timestamps
            mock_get_gpu.return_value = (100.0, 200.0)

            session = memory_utils.GPUTraceSession("test_session", log_interval=1)
            session.start()

            with session.iteration(0):
                pass

            session.finish()

            assert len(session.iterations) == 1
            summary = session.summary()
            assert "Iterations: 1" in summary

    @patch("gc.collect")
    def test_cleanup_gpu_memory(self, mock_gc_collect):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.side_effect = [209715200, 104857600]  # 200MB, 100MB

            stats = memory_utils.cleanup_gpu_memory(force=True)

            assert stats["freed_mb"] == 100.0
            mock_gc_collect.assert_called()
            mock_torch.cuda.empty_cache.assert_called()
            mock_torch.cuda.ipc_collect.assert_called()

    def test_get_gpu_memory_breakdown(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_properties.return_value.name = "Test GPU"
            mock_torch.cuda.get_device_properties.return_value.total_memory = 1024 * 1024 * 1024  # 1GB

            breakdown = memory_utils.get_gpu_memory_breakdown()

            assert breakdown["available"] is True
            assert len(breakdown["devices"]) == 1
            assert breakdown["devices"][0]["name"] == "Test GPU"

    @patch("nodetool.workflows.memory_utils.log")
    @patch("nodetool.workflows.memory_utils.get_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.get_gpu_memory_usage_mb")
    @patch("nodetool.workflows.memory_utils.get_memory_uri_cache_stats")
    def test_log_memory_summary(self, mock_get_cache, mock_get_gpu, mock_get_mem, mock_log):
        mock_get_mem.return_value = 100.0
        mock_get_gpu.return_value = (200.0, 300.0)
        mock_get_cache.return_value = {"count": 10}

        stats = memory_utils.log_memory_summary("Test Summary")

        assert stats["ram_mb"] == 100.0
        assert stats["gpu_allocated_mb"] == 200.0
        assert stats["memory_cache_count"] == 10
        mock_log.info.assert_any_call("[MEMORY SUMMARY] Test Summary")

    def test_get_memory_uri_cache_stats(self):
        mock_scope = MagicMock()
        mock_cache = MagicMock()
        mock_cache._cache = {1: 1, 2: 2}  # len 2
        mock_scope.get_memory_uri_cache.return_value = mock_cache

        with patch("nodetool.runtime.resources.maybe_scope", return_value=mock_scope):
            stats = memory_utils.get_memory_uri_cache_stats()
            assert stats["count"] == 2

    def test_clear_memory_uri_cache(self):
        mock_scope = MagicMock()
        mock_cache = MagicMock()
        mock_cache._cache = {1: 1, 2: 2}
        mock_scope.get_memory_uri_cache.return_value = mock_cache

        with patch("nodetool.runtime.resources.maybe_scope", return_value=mock_scope):
            count = memory_utils.clear_memory_uri_cache()
            assert count == 2
            mock_cache.clear.assert_called_once()
