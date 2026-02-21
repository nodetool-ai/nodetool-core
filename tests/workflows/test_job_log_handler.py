"""
Tests for the JobLogHandler (workflows/job_log_handler.py).

Tests job-specific log capture and handler lifecycle management.
"""

import logging

import pytest

from nodetool.workflows.job_log_handler import JobLogHandler


class TestJobLogHandler:
    """Test JobLogHandler class."""

    def test_handler_initialization(self):
        """Test JobLogHandler initialization with default parameters."""
        handler = JobLogHandler(job_id="test-job")
        assert handler.job_id == "test-job"
        assert handler.max_logs == 1000
        assert handler.level == logging.INFO
        assert len(handler.logs) == 0

    def test_handler_initialization_custom_params(self):
        """Test JobLogHandler initialization with custom parameters."""
        handler = JobLogHandler(job_id="test-job", max_logs=100, level=logging.DEBUG)
        assert handler.job_id == "test-job"
        assert handler.max_logs == 100
        assert handler.level == logging.DEBUG

    def test_emit_log_record(self):
        """Test emitting a log record."""
        handler = JobLogHandler(job_id="test-job")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        assert len(handler.logs) == 1
        log_entry = handler.logs[0]
        assert log_entry["level"] == "INFO"
        assert log_entry["logger"] == "test.logger"
        assert "Test message" in log_entry["message"]
        assert log_entry["module"] == "test"
        assert log_entry["line"] == 42

    def test_max_logs_limit(self):
        """Test that handler respects max_logs limit."""
        handler = JobLogHandler(job_id="test-job", max_logs=5)

        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Should only keep the last 5 logs
        assert len(handler.logs) == 5
        assert handler.logs[0]["line"] == 5  # First kept log
        assert handler.logs[4]["line"] == 9  # Last kept log

    def test_get_logs_no_limit(self):
        """Test getting all logs without limit."""
        handler = JobLogHandler(job_id="test-job", max_logs=10)

        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        logs = handler.get_logs()
        assert len(logs) == 5

    def test_get_logs_with_limit(self):
        """Test getting logs with a limit."""
        handler = JobLogHandler(job_id="test-job", max_logs=10)

        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=i,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        logs = handler.get_logs(limit=3)
        assert len(logs) == 3
        # Should return the most recent 3 logs
        assert logs[0]["line"] == 7
        assert logs[2]["line"] == 9

    def test_clear_logs(self):
        """Test clearing all logs."""
        handler = JobLogHandler(job_id="test-job")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        assert len(handler.logs) == 1
        handler.clear_logs()
        assert len(handler.logs) == 0

    def test_get_handler(self):
        """Test getting a handler by job ID."""
        handler1 = JobLogHandler(job_id="job1")
        handler2 = JobLogHandler(job_id="job2")

        retrieved = JobLogHandler.get_handler("job1")
        assert retrieved is handler1

        retrieved2 = JobLogHandler.get_handler("job2")
        assert retrieved2 is handler2

        non_existent = JobLogHandler.get_handler("nonexistent")
        assert non_existent is None

    def test_close_handler(self):
        """Test closing a handler removes it from registry."""
        handler = JobLogHandler(job_id="test-job")
        assert JobLogHandler.get_handler("test-job") is handler

        handler.close()
        assert JobLogHandler.get_handler("test-job") is None

    def test_uninstall_nonexistent_handler(self):
        """Test uninstalling a handler that doesn't exist (should not error)."""
        # Should not raise an exception
        JobLogHandler.uninstall_handler("nonexistent-job")

    def test_multiple_handlers_different_jobs(self):
        """Test that multiple handlers can coexist for different jobs."""
        handler1 = JobLogHandler(job_id="job1")
        handler2 = JobLogHandler(job_id="job2")

        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Job 1 message",
            args=(),
            exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=2,
            msg="Job 2 message",
            args=(),
            exc_info=None,
        )

        handler1.emit(record1)
        handler2.emit(record2)

        assert len(handler1.logs) == 1
        assert len(handler2.logs) == 1
        assert handler1.logs[0]["message"] != handler2.logs[0]["message"]
