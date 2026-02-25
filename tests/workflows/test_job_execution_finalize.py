from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.models.job import Job
from nodetool.workflows.job_execution import JobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest


class MockJobExecution(JobExecution):
    def is_running(self) -> bool:
        return self._status == "running"

    def is_completed(self) -> bool:
        return self._status in ("completed", "error", "cancelled")

    async def cancel(self) -> bool:
        return True

    async def cleanup_resources(self) -> None:
        pass

    def push_input_value(self, input_name: str, value: object, source_handle: str) -> None:
        pass

@pytest.fixture
def mock_job_model():
    job = MagicMock(spec=Job)
    job.id = "test_job_id"
    job.status = "running"
    job.finished_at = None
    job.mark_completed = AsyncMock()
    job.mark_failed = AsyncMock()
    job.mark_cancelled = AsyncMock()
    job.update = AsyncMock()
    return job

@pytest.fixture
def mock_execution(mock_job_model):
    context = MagicMock(spec=ProcessingContext)
    request = MagicMock(spec=RunJobRequest)

    # We need to mock _install_log_handler during initialization to avoid side effects
    with patch("nodetool.workflows.job_execution.JobLogHandler.install_handler"):
        execution = MockJobExecution(
            job_id="test_job_id",
            context=context,
            request=request,
            job_model=mock_job_model
        )
    return execution

@pytest.mark.asyncio
async def test_finalize_state_handles_db_exception(mock_execution):
    """Test that finalize_state handles exceptions during DB operations gracefully."""

    # Setup execution state
    mock_execution._status = "completed"

    # Mock Job.get to raise an exception
    with patch("nodetool.models.job.Job.get", side_effect=Exception("DB Connection Error")):
        # Mock logger to verify exception is logged
        with patch("nodetool.workflows.job_execution.log") as mock_log:
            await mock_execution.finalize_state()

            # Verify exception was logged
            mock_log.exception.assert_called_once()
            args, _ = mock_log.exception.call_args
            assert "JobExecution.finalize_state: failed to persist state" in args[0]

@pytest.mark.asyncio
async def test_finalize_state_success(mock_execution, mock_job_model):
    """Test successful finalization of job state."""

    # Setup execution state
    mock_execution._status = "completed"

    # Mock Job.get to return our mock job
    with patch("nodetool.models.job.Job.get", return_value=mock_job_model):
        # Mock _uninstall_log_handler to return some logs
        with patch.object(mock_execution, '_uninstall_log_handler', return_value=[{"msg": "test log"}]):
            await mock_execution.finalize_state()

            # Verify job status updated
            mock_job_model.mark_completed.assert_called_once()

            # Verify job updated with finished_at and logs
            mock_job_model.update.assert_called_once()
            kwargs = mock_job_model.update.call_args.kwargs
            assert "finished_at" in kwargs
            assert kwargs["logs"] == [{"msg": "test log"}]

@pytest.mark.asyncio
async def test_finalize_state_failed_job(mock_execution, mock_job_model):
    """Test finalization of failed job."""

    # Setup execution state
    mock_execution._status = "failed"
    mock_execution._error = "Test Error"

    # Mock Job.get to return our mock job
    with patch("nodetool.models.job.Job.get", return_value=mock_job_model):
        await mock_execution.finalize_state()

        # Verify job marked failed
        mock_job_model.mark_failed.assert_called_once_with(error="Test Error")

        # Verify update called
        mock_job_model.update.assert_called_once()

@pytest.mark.asyncio
async def test_finalize_state_cancelled_job(mock_execution, mock_job_model):
    """Test finalization of cancelled job."""

    # Setup execution state
    mock_execution._status = "cancelled"

    # Mock Job.get to return our mock job
    with patch("nodetool.models.job.Job.get", return_value=mock_job_model):
        await mock_execution.finalize_state()

        # Verify job marked cancelled
        mock_job_model.mark_cancelled.assert_called_once()

        # Verify update called
        mock_job_model.update.assert_called_once()
