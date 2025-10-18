"""
Unit tests for progress management module.
"""

import pytest
from unittest.mock import Mock, patch
from rich.console import Console

from nodetool.deploy.progress import ProgressManager


# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestProgressManager:
    """Tests for ProgressManager class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock Console."""
        return Mock(spec=Console)

    @pytest.fixture
    def manager(self, mock_console):
        """Create a ProgressManager with mocked console."""
        return ProgressManager(console=mock_console)

    def test_init(self, mock_console):
        """Test manager initialization."""
        manager = ProgressManager(console=mock_console)

        assert manager.console == mock_console
        assert manager.progress is None
        assert manager.tasks == {}
        assert manager.current_operations == {}

    def test_init_without_console(self):
        """Test manager initialization without console."""
        with patch("nodetool.deploy.progress.Console") as mock_console_cls:
            mock_console = Mock()
            mock_console_cls.return_value = mock_console

            manager = ProgressManager()

            assert manager.console == mock_console
            mock_console_cls.assert_called_once()

    def test_start(self, manager):
        """Test starting progress display."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value = mock_progress

            manager.start()

            assert manager.progress == mock_progress
            mock_progress.start.assert_called_once()

    def test_start_idempotent(self, manager):
        """Test that starting multiple times doesn't create multiple progress instances."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value = mock_progress

            manager.start()
            first_progress = manager.progress

            manager.start()
            second_progress = manager.progress

            # Should be the same instance
            assert first_progress == second_progress
            assert mock_progress_cls.call_count == 1

    def test_stop(self, manager):
        """Test stopping progress display."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value = mock_progress

            manager.start()
            manager.stop()

            assert manager.progress is None
            assert manager.tasks == {}
            assert manager.current_operations == {}
            mock_progress.stop.assert_called_once()

    def test_stop_when_not_started(self, manager):
        """Test stopping when progress not started."""
        # Should not raise error
        manager.stop()

        assert manager.progress is None

    def test_add_task(self, manager):
        """Test adding a progress task."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            task_id = manager.add_task("op1", "Test operation", total=100)

            assert task_id == 1
            assert "op1" in manager.tasks
            assert manager.tasks["op1"] == 1
            assert "op1" in manager.current_operations
            assert manager.current_operations["op1"]["description"] == "Test operation"
            assert manager.current_operations["op1"]["total"] == 100
            assert manager.current_operations["op1"]["completed"] == 0

    def test_add_task_starts_progress(self, manager):
        """Test that adding task starts progress if not started."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            assert manager.progress is None

            manager.add_task("op1", "Test")

            assert manager.progress is not None
            mock_progress.start.assert_called_once()

    def test_add_task_without_total(self, manager):
        """Test adding task without total."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test operation")

            assert manager.current_operations["op1"]["total"] is None

    def test_add_task_duplicate_operation_id(self, manager):
        """Test adding task with duplicate operation ID."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            task_id1 = manager.add_task("op1", "First")
            task_id2 = manager.add_task("op1", "Second")

            # Should return same task ID
            assert task_id1 == task_id2
            # Should only call add_task once
            assert mock_progress.add_task.call_count == 1

    def test_update_task(self, manager):
        """Test updating a progress task."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.update_task("op1", completed=50)

            # Should advance by 50
            mock_progress.update.assert_called_with(1, advance=50)
            assert manager.current_operations["op1"]["completed"] == 50

    def test_update_task_multiple_times(self, manager):
        """Test updating task multiple times with incremental progress."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.update_task("op1", completed=30)
            manager.update_task("op1", completed=70)

            # First update: advance by 30
            # Second update: advance by 40 (70 - 30)
            calls = mock_progress.update.call_args_list
            assert calls[0][1]["advance"] == 30
            assert calls[1][1]["advance"] == 40

    def test_update_task_description(self, manager):
        """Test updating task description."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Original")
            manager.update_task("op1", description="Updated")

            mock_progress.update.assert_called_with(1, description="Updated")
            assert manager.current_operations["op1"]["description"] == "Updated"

    def test_update_task_both_completed_and_description(self, manager):
        """Test updating both completed and description."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.update_task("op1", completed=50, description="Half done")

            mock_progress.update.assert_called_with(
                1, advance=50, description="Half done"
            )

    def test_update_task_nonexistent(self, manager):
        """Test updating nonexistent task doesn't raise error."""
        # Should not raise error
        manager.update_task("nonexistent", completed=50)

    def test_update_task_no_advance_when_completed_lower(self, manager):
        """Test that no advance happens when completed value is lower."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.update_task("op1", completed=50)
            manager.update_task("op1", completed=30)  # Lower value

            # Should not call update for the lower value
            # Only the first update should have advance
            calls = [
                call
                for call in mock_progress.update.call_args_list
                if "advance" in call[1]
            ]
            assert len(calls) == 1

    def test_complete_task(self, manager):
        """Test completing a task."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.complete_task("op1")

            # Should set completed to total
            mock_progress.update.assert_called_with(1, completed=100)
            # Should remove task
            mock_progress.remove_task.assert_called_with(1)
            assert "op1" not in manager.tasks
            assert "op1" not in manager.current_operations

    def test_complete_task_without_total(self, manager):
        """Test completing task without total."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test")  # No total
            manager.complete_task("op1")

            # Should not try to update with completed
            update_calls = [
                call
                for call in mock_progress.update.call_args_list
                if "completed" in call[1]
            ]
            assert len(update_calls) == 0
            # Should still remove task
            mock_progress.remove_task.assert_called_with(1)

    def test_complete_task_stops_progress_when_no_tasks(self, manager):
        """Test that completing last task stops progress."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.complete_task("op1")

            # Should stop progress when no tasks remain
            mock_progress.stop.assert_called_once()
            assert manager.progress is None

    def test_complete_task_doesnt_stop_progress_with_remaining_tasks(self, manager):
        """Test that completing task doesn't stop progress if other tasks exist."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.side_effect = [1, 2]
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test 1", total=100)
            manager.add_task("op2", "Test 2", total=100)
            manager.complete_task("op1")

            # Should not stop progress
            mock_progress.stop.assert_not_called()
            assert manager.progress is not None

    def test_remove_task(self, manager):
        """Test removing a task."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.remove_task("op1")

            # Should remove task without completing
            mock_progress.remove_task.assert_called_with(1)
            assert "op1" not in manager.tasks
            assert "op1" not in manager.current_operations

    def test_remove_task_stops_progress_when_no_tasks(self, manager):
        """Test that removing last task stops progress."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.remove_task("op1")

            # Should stop progress
            mock_progress.stop.assert_called_once()

    def test_remove_task_nonexistent(self, manager):
        """Test removing nonexistent task doesn't raise error."""
        # Should not raise error
        manager.remove_task("nonexistent")

    def test_display_progress_update_starting(self, manager):
        """Test displaying starting status."""
        progress_update = {
            "status": "starting",
            "message": "Starting deployment",
        }

        manager._display_progress_update(progress_update)

        manager.console.print.assert_called_once()
        call_args = manager.console.print.call_args[0][0]
        assert "Starting deployment" in call_args

    def test_display_progress_update_completed(self, manager):
        """Test displaying completed status."""
        progress_update = {
            "status": "completed",
            "message": "Deployment complete",
            "downloaded_files": 5,
        }

        manager._display_progress_update(progress_update)

        # Should print completion message and file count
        assert manager.console.print.call_count >= 2

    def test_display_progress_update_error(self, manager):
        """Test displaying error status."""
        progress_update = {
            "status": "error",
            "error": "Connection failed",
        }

        with patch("sys.exit") as mock_exit:
            manager._display_progress_update(progress_update)

            # Should print error and exit
            manager.console.print.assert_called()
            mock_exit.assert_called_with(1)

    def test_display_progress_update_file_progress(self, manager):
        """Test displaying file download progress."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            progress_update = {
                "status": "progress",
                "current_file": "model.bin",
                "file_progress": 3,
                "total_files": 10,
            }

            manager._display_progress_update(progress_update)

            # Should add/update task for file progress
            assert len(manager.tasks) > 0

    def test_display_progress_update_download_size(self, manager):
        """Test displaying download with size information."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            progress_update = {
                "status": "progress",
                "downloaded_size": 50 * 1024 * 1024,  # 50 MB
                "total_size": 100 * 1024 * 1024,  # 100 MB
                "operation_id": "download_model",
                "current_file": "model.safetensors",
            }

            manager._display_progress_update(progress_update)

            # Should add task for download
            assert "download_model" in manager.tasks

    def test_display_progress_update_healthy_status(self, manager):
        """Test displaying healthy system status."""
        progress_update = {
            "status": "healthy",
            "platform": "Linux",
            "python_version": "3.11.0",
            "hostname": "worker-1",
            "memory": {
                "available_gb": 8.5,
                "total_gb": 16.0,
                "used_percent": 47,
            },
            "disk": {
                "free_gb": 250.0,
                "total_gb": 500.0,
                "used_percent": 50,
            },
            "gpus": [
                {
                    "name": "NVIDIA RTX 3090",
                    "memory_used_mb": 2048,
                    "memory_total_mb": 24576,
                }
            ],
        }

        manager._display_progress_update(progress_update)

        # Should print multiple system info lines
        assert manager.console.print.call_count >= 5

    def test_display_progress_update_pulling_layer(self, manager):
        """Test displaying Docker layer pulling status."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            progress_update = {
                "status": "pulling abc123",
                "digest": "sha256:abc123def456",
                "total": 100 * 1024 * 1024,  # 100 MB
                "completed": 50 * 1024 * 1024,  # 50 MB
            }

            manager._display_progress_update(progress_update)

            # Should add task for pulling
            assert len(manager.tasks) > 0


class TestProgressManagerEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.fixture
    def manager(self):
        """Create a manager with mocked console."""
        mock_console = Mock(spec=Console)
        return ProgressManager(console=mock_console)

    def test_multiple_tasks_concurrent(self, manager):
        """Test managing multiple tasks concurrently."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.side_effect = [1, 2, 3]
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Task 1", total=100)
            manager.add_task("op2", "Task 2", total=200)
            manager.add_task("op3", "Task 3", total=300)

            assert len(manager.tasks) == 3
            assert len(manager.current_operations) == 3

    def test_complete_tasks_in_different_order(self, manager):
        """Test completing tasks in different order than creation."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.side_effect = [1, 2, 3]
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Task 1", total=100)
            manager.add_task("op2", "Task 2", total=200)
            manager.add_task("op3", "Task 3", total=300)

            # Complete in different order
            manager.complete_task("op2")
            assert "op2" not in manager.tasks
            assert "op1" in manager.tasks
            assert "op3" in manager.tasks

            manager.complete_task("op1")
            assert "op1" not in manager.tasks
            assert "op3" in manager.tasks

    def test_update_task_with_zero_completed(self, manager):
        """Test updating task with zero completed."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "Test", total=100)
            manager.update_task("op1", completed=0)

            # Should not try to advance
            update_calls = [
                call
                for call in mock_progress.update.call_args_list
                if "advance" in call[1]
            ]
            assert len(update_calls) == 0

    def test_display_progress_update_with_missing_fields(self, manager):
        """Test displaying progress update with missing optional fields."""
        progress_update = {
            "status": "progress",
            "message": "Processing...",
        }

        # Should not raise error
        manager._display_progress_update(progress_update)

    def test_display_progress_update_healthy_no_gpus(self, manager):
        """Test healthy status with no GPUs."""
        progress_update = {
            "status": "healthy",
            "platform": "Linux",
            "python_version": "3.11.0",
            "hostname": "worker-1",
            "gpus": "unavailable",
        }

        manager._display_progress_update(progress_update)

        # Should handle unavailable GPUs
        manager.console.print.assert_called()

    def test_start_stop_start_cycle(self, manager):
        """Test starting, stopping, and starting again."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress_cls.return_value = mock_progress

            manager.start()

            manager.stop()
            assert manager.progress is None

            manager.start()
            second_progress = manager.progress

            # Should be different instances
            assert second_progress is not None

    def test_task_description_with_unicode(self, manager):
        """Test task with unicode characters in description."""
        with patch("nodetool.deploy.progress.Progress") as mock_progress_cls:
            mock_progress = Mock()
            mock_progress.add_task.return_value = 1
            mock_progress_cls.return_value = mock_progress

            manager.add_task("op1", "📦 Downloading model 🚀", total=100)

            assert (
                manager.current_operations["op1"]["description"]
                == "📦 Downloading model 🚀"
            )
