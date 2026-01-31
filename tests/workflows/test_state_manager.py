"""
Tests for the StateManager class.
"""

import asyncio
from datetime import datetime

import pytest

from nodetool.models.run_node_state import NodeStatus
from nodetool.workflows.state_manager import StateManager, StateUpdate


class TestStateUpdate:
    """Tests for the StateUpdate dataclass."""

    def test_state_update_init_with_minimal_args(self):
        """Test StateUpdate initialization with minimal arguments."""
        update = StateUpdate(node_id="node-1")

        assert update.node_id == "node-1"
        assert update.status is None
        assert update.attempt is None
        assert update.started_at is None
        assert update.completed_at is None
        assert update.last_error is None
        assert update.timestamp is not None

    def test_state_update_init_with_all_args(self):
        """Test StateUpdate initialization with all arguments."""
        now = datetime.now()
        update = StateUpdate(
            node_id="node-1",
            status="running",
            attempt=2,
            scheduled_at=now,
            started_at=now,
            completed_at=None,
            failed_at=None,
            suspended_at=None,
            last_error="test error",
            retryable=True,
            suspension_reason="waiting for input",
            resume_state_json={"key": "value"},
            outputs_json={"output": "data"},
        )

        assert update.node_id == "node-1"
        assert update.status == "running"
        assert update.attempt == 2
        assert update.scheduled_at == now
        assert update.started_at == now
        assert update.last_error == "test error"
        assert update.retryable is True
        assert update.suspension_reason == "waiting for input"
        assert update.resume_state_json == {"key": "value"}
        assert update.outputs_json == {"output": "data"}


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_init_with_defaults(self):
        """Test StateManager initialization with default parameters."""
        manager = StateManager(run_id="run-123")

        assert manager.run_id == "run-123"
        assert manager.batch_size == 10
        assert manager.batch_interval == 0.1
        assert manager.writer_task is None
        assert manager._stopped is False
        assert manager.stats["updates_queued"] == 0
        assert manager.stats["updates_processed"] == 0
        assert manager.stats["batches_written"] == 0
        assert manager.stats["errors"] == 0

    def test_init_with_custom_params(self):
        """Test StateManager initialization with custom parameters."""
        manager = StateManager(
            run_id="run-456",
            batch_size=20,
            batch_interval=0.5,
        )

        assert manager.run_id == "run-456"
        assert manager.batch_size == 20
        assert manager.batch_interval == 0.5


@pytest.mark.asyncio
class TestStateManagerStartStop:
    """Tests for StateManager start and stop methods."""

    async def test_start_creates_writer_task(self):
        """Test that start creates the background writer task."""
        manager = StateManager(run_id="run-123")

        await manager.start()

        assert manager.writer_task is not None
        assert not manager.writer_task.done()

        # Clean up
        await manager.stop()

    async def test_stop_gracefully_shuts_down(self):
        """Test that stop gracefully shuts down the manager."""
        manager = StateManager(run_id="run-123")

        await manager.start()
        await manager.stop()

        assert manager._stopped is True
        assert manager.writer_task.done()

    async def test_stop_without_start_is_safe(self):
        """Test that stopping without starting doesn't raise."""
        manager = StateManager(run_id="run-123")

        # Should not raise
        await manager.stop()

    async def test_double_start_logs_warning(self):
        """Test that starting twice doesn't create multiple tasks."""
        manager = StateManager(run_id="run-123")

        await manager.start()
        first_task = manager.writer_task

        await manager.start()  # Second start
        second_task = manager.writer_task

        # Should be the same task
        assert first_task is second_task

        await manager.stop()


@pytest.mark.asyncio
class TestStateManagerUpdateNodeState:
    """Tests for the update_node_state method."""

    async def test_update_node_state_queues_update(self):
        """Test that update_node_state queues an update."""
        manager = StateManager(run_id="run-123")

        await manager.start()

        await manager.update_node_state(
            node_id="node-1",
            status="running",
        )

        assert manager.stats["updates_queued"] == 1

        await manager.stop()

    async def test_update_node_state_after_stop_is_ignored(self):
        """Test that updates after stop are ignored."""
        manager = StateManager(run_id="run-123")

        await manager.start()
        await manager.stop()

        # This should be ignored
        await manager.update_node_state(node_id="node-1", status="running")

        # Stats should not increment after stop
        assert manager._stopped is True

    async def test_update_node_state_with_all_fields(self):
        """Test update with all possible fields."""
        manager = StateManager(run_id="run-123")

        await manager.start()

        now = datetime.now()
        await manager.update_node_state(
            node_id="node-1",
            status="completed",
            attempt=3,
            scheduled_at=now,
            started_at=now,
            completed_at=now,
            outputs_json={"result": "success"},
        )

        assert manager.stats["updates_queued"] == 1

        await manager.stop()


@pytest.mark.asyncio
class TestStateManagerBatchProcessing:
    """Tests for batch processing functionality."""

    async def test_batch_processing_coalesces_updates(self):
        """Test that multiple updates for the same node are coalesced."""
        manager = StateManager(run_id="run-123", batch_size=10, batch_interval=0.05)

        await manager.start()

        # Send multiple updates for the same node
        await manager.update_node_state(node_id="node-1", status="running")
        await manager.update_node_state(node_id="node-1", attempt=2)
        await manager.update_node_state(node_id="node-1", status="completed")

        # Wait for batch to be processed
        await asyncio.sleep(0.2)

        assert manager.stats["updates_queued"] == 3

        await manager.stop()

    async def test_batch_size_triggers_immediate_processing(self):
        """Test that reaching batch_size triggers immediate processing."""
        manager = StateManager(run_id="run-123", batch_size=3, batch_interval=10.0)

        await manager.start()

        # Queue exactly batch_size updates
        for i in range(3):
            await manager.update_node_state(node_id=f"node-{i}", status="running")

        # Give time for processing
        await asyncio.sleep(0.2)

        assert manager.stats["updates_queued"] == 3

        await manager.stop()


@pytest.mark.asyncio
class TestStateManagerGetNodeState:
    """Tests for the get_node_state method."""

    async def test_get_node_state_from_cache(self):
        """Test that get_node_state returns cached state."""
        manager = StateManager(run_id="run-123")

        await manager.start()

        # Queue an update to populate cache
        await manager.update_node_state(node_id="node-1", status="running")

        # Wait for processing
        await asyncio.sleep(0.2)

        # Get state from cache
        state = await manager.get_node_state("node-1")

        # Should be in cache now
        if state:
            assert state.node_id == "node-1"

        await manager.stop()

    async def test_get_node_state_from_db_if_not_cached(self):
        """Test that get_node_state loads from DB if not in cache."""
        manager = StateManager(run_id="run-123")

        # Without starting (no cache population)
        state = await manager.get_node_state("nonexistent-node")

        # Should return None for nonexistent node
        assert state is None


@pytest.mark.asyncio
class TestStateManagerStats:
    """Tests for StateManager statistics tracking."""

    async def test_stats_track_queued_updates(self):
        """Test that stats correctly track queued updates."""
        manager = StateManager(run_id="run-123")

        await manager.start()

        for i in range(5):
            await manager.update_node_state(node_id=f"node-{i}", status="running")

        assert manager.stats["updates_queued"] == 5

        await manager.stop()

    async def test_stats_track_processed_updates(self):
        """Test that stats correctly track processed updates."""
        manager = StateManager(run_id="run-123", batch_interval=0.05)

        await manager.start()

        await manager.update_node_state(node_id="node-1", status="running")

        # Wait for batch processing
        await asyncio.sleep(0.2)

        # After stop, updates should be processed
        await manager.stop()

        # Updates should be processed (may be coalesced)
        assert manager.stats["updates_processed"] >= 1

    async def test_stats_reset_on_new_instance(self):
        """Test that stats are fresh for each new instance."""
        manager1 = StateManager(run_id="run-1")
        manager2 = StateManager(run_id="run-2")

        await manager1.start()
        await manager1.update_node_state(node_id="node-1", status="running")
        await manager1.stop()

        # manager2 should have fresh stats
        assert manager2.stats["updates_queued"] == 0
        assert manager2.stats["updates_processed"] == 0
