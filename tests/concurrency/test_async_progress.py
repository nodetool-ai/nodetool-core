import asyncio

import pytest

from nodetool.concurrency.async_progress import (
    AsyncProgressTracker,
    ProgressCallback,
    ProgressUpdate,
)


class TestProgressUpdate:
    """Tests for ProgressUpdate class."""

    def test_init(self):
        """Test ProgressUpdate initialization."""
        update = ProgressUpdate(current=5, total=10)
        assert update.current == 5
        assert update.total == 10
        assert update.message is None
        assert update.metadata == {}

    def test_init_with_message_and_metadata(self):
        """Test initialization with message and metadata."""
        metadata = {"item": "test", "status": "processing"}
        update = ProgressUpdate(
            current=3, total=10, message="Processing", metadata=metadata
        )
        assert update.current == 3
        assert update.total == 10
        assert update.message == "Processing"
        assert update.metadata == metadata

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        update = ProgressUpdate(current=5, total=10)
        assert update.percentage == 0.5

        update = ProgressUpdate(current=0, total=10)
        assert update.percentage == 0.0

        update = ProgressUpdate(current=10, total=10)
        assert update.percentage == 1.0

    def test_percentage_clamping(self):
        """Test percentage is clamped to valid range."""
        update = ProgressUpdate(current=-5, total=10)
        assert update.percentage == 0.0

        update = ProgressUpdate(current=15, total=10)
        assert update.percentage == 1.0

    def test_percentage_zero_total(self):
        """Test percentage with zero total."""
        update = ProgressUpdate(current=5, total=0)
        assert update.percentage == 1.0

    def test_percentage_str(self):
        """Test percentage string formatting."""
        update = ProgressUpdate(current=7.5, total=10)
        assert update.percentage_str == "75.0%"

        update = ProgressUpdate(current=1, total=3)
        assert update.percentage_str == "33.3%"

    def test_repr(self):
        """Test string representation."""
        update = ProgressUpdate(current=5, total=10)
        assert repr(update) == "ProgressUpdate(50.0%, 5/10)"


class TestAsyncProgressTracker:
    """Tests for AsyncProgressTracker class."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = AsyncProgressTracker(total=100)
        assert tracker.total == 100
        assert tracker.current == 0.0
        assert tracker.percentage == 0.0
        assert tracker.message is None
        assert not tracker.is_started
        assert not tracker.is_completed

    def test_init_with_message(self):
        """Test initialization with message."""
        tracker = AsyncProgressTracker(total=100, message="Processing")
        assert tracker.message == "Processing"

    def test_init_with_initial(self):
        """Test initialization with initial progress."""
        tracker = AsyncProgressTracker(total=100, initial=25)
        assert tracker.current == 25
        assert tracker.percentage == 0.25

    def test_init_validation_total_positive(self):
        """Test validation that total must be positive."""
        with pytest.raises(ValueError, match="total must be positive"):
            AsyncProgressTracker(total=0)

        with pytest.raises(ValueError, match="total must be positive"):
            AsyncProgressTracker(total=-10)

    def test_init_validation_initial_non_negative(self):
        """Test validation that initial must be non-negative."""
        with pytest.raises(ValueError, match="initial must be non-negative"):
            AsyncProgressTracker(total=100, initial=-5)

    def test_init_validation_initial_not_exceed_total(self):
        """Test validation that initial cannot exceed total."""
        with pytest.raises(ValueError, match=r"initial .* cannot exceed total"):
            AsyncProgressTracker(total=100, initial=150)

    def test_properties(self):
        """Test tracker properties."""
        tracker = AsyncProgressTracker(total=100)
        assert tracker.total == 100
        assert tracker.current == 0.0

    @pytest.mark.asyncio
    async def test_update(self):
        """Test updating progress."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.update(50)
        assert tracker.current == 50
        assert tracker.percentage == 0.5
        assert tracker.is_started

    @pytest.mark.asyncio
    async def test_update_with_message(self):
        """Test updating with message override."""
        tracker = AsyncProgressTracker(total=100, message="Initial")
        await tracker.update(50, message="Halfway")
        assert tracker.message == "Halfway"

    @pytest.mark.asyncio
    async def test_update_clamping(self):
        """Test update values are clamped."""
        tracker = AsyncProgressTracker(total=100)

        await tracker.update(-10)
        assert tracker.current == 0.0

        await tracker.update(150)
        assert tracker.current == 100

    @pytest.mark.asyncio
    async def test_update_completed_tracker_fails(self):
        """Test that updating completed tracker raises error."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.complete()

        with pytest.raises(RuntimeError, match="Cannot update a completed tracker"):
            await tracker.update(50)

    @pytest.mark.asyncio
    async def test_increment(self):
        """Test incrementing progress."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.increment(10)
        assert tracker.current == 10

        await tracker.increment(5)
        assert tracker.current == 15

    @pytest.mark.asyncio
    async def test_increment_default(self):
        """Test increment with default amount."""
        tracker = AsyncProgressTracker(total=10)
        await tracker.increment()
        assert tracker.current == 1

    @pytest.mark.asyncio
    async def test_complete(self):
        """Test marking tracker as complete."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.update(50)
        assert not tracker.is_completed

        await tracker.complete()
        assert tracker.is_completed
        assert tracker.current == 100
        assert tracker.percentage == 1.0

    @pytest.mark.asyncio
    async def test_complete_with_message(self):
        """Test completing with custom message."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.complete(message="Done!")
        assert tracker.message == "Done!"
        assert tracker.is_completed

    @pytest.mark.asyncio
    async def test_complete_idempotent(self):
        """Test that completing multiple times is safe."""
        tracker = AsyncProgressTracker(total=100)
        await tracker.complete()
        await tracker.complete()  # Should not raise
        assert tracker.is_completed

    def test_set_message(self):
        """Test setting message."""
        tracker = AsyncProgressTracker(total=100)
        tracker.set_message("New message")
        assert tracker.message == "New message"

    def test_set_metadata(self):
        """Test setting metadata."""
        tracker = AsyncProgressTracker(total=100)
        tracker.set_metadata("key1", "value1")
        tracker.set_metadata("key2", 42)

        progress = tracker.get_progress()
        assert progress.metadata == {"key1": "value1", "key2": 42}

    @pytest.mark.asyncio
    async def test_callback_notification(self):
        """Test that callbacks are notified on update."""
        tracker = AsyncProgressTracker(total=100)
        updates = []

        def callback(update: ProgressUpdate):
            updates.append(update)

        tracker.add_callback(callback)
        await tracker.update(50)

        assert len(updates) == 1
        assert updates[0].current == 50
        assert updates[0].total == 100

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self):
        """Test that multiple callbacks are all notified."""
        tracker = AsyncProgressTracker(total=100)
        updates1 = []
        updates2 = []

        tracker.add_callback(lambda u: updates1.append(u))
        tracker.add_callback(lambda u: updates2.append(u))

        await tracker.update(50)

        assert len(updates1) == 1
        assert len(updates2) == 1

    @pytest.mark.asyncio
    async def test_remove_callback(self):
        """Test removing callbacks."""
        tracker = AsyncProgressTracker(total=100)
        updates = []

        def callback(u):
            updates.append(u)

        tracker.add_callback(callback)
        await tracker.update(25)
        assert len(updates) == 1

        tracker.remove_callback(callback)
        await tracker.update(50)
        assert len(updates) == 1  # No new update

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test that async callbacks work."""
        tracker = AsyncProgressTracker(total=100)
        updates = []

        async def async_callback(update: ProgressUpdate):
            await asyncio.sleep(0.01)
            updates.append(update)

        tracker.add_callback(async_callback)
        await tracker.update(50)

        assert len(updates) == 1

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_break_others(self):
        """Test that callback exceptions don't break other callbacks."""
        tracker = AsyncProgressTracker(total=100)
        updates = []

        def failing_callback(_):
            raise ValueError("Test error")

        def working_callback(u):
            updates.append(u)

        tracker.add_callback(failing_callback)
        tracker.add_callback(working_callback)

        await tracker.update(50)  # Should not raise

        assert len(updates) == 1

    @pytest.mark.asyncio
    async def test_get_progress(self):
        """Test getting progress without updating."""
        tracker = AsyncProgressTracker(total=100, message="Test")
        tracker.set_metadata("key", "value")

        progress = tracker.get_progress()
        assert progress.current == 0
        assert progress.total == 100
        assert progress.message == "Test"
        assert progress.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using tracker as context manager."""
        updates = []

        async def work():
            tracker = AsyncProgressTracker(total=100)
            tracker.add_callback(lambda u: updates.append(u))

            async with tracker:
                await tracker.update(50)
                assert tracker.is_started
                assert not tracker.is_completed

            # After exiting, should be complete
            assert tracker.is_completed
            assert tracker.current == 100

        await work()
        # Should have final update at 100%
        assert any(u.current == 100 for u in updates)

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test context manager doesn't complete on exception."""
        tracker = AsyncProgressTracker(total=100)

        try:
            async with tracker:
                await tracker.update(50)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert tracker.current == 50
        assert not tracker.is_completed

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test thread-safety with concurrent updates."""
        tracker = AsyncProgressTracker(total=100)
        updates = []

        tracker.add_callback(lambda u: updates.append(u))

        async def update_task(value):
            await tracker.update(value)

        # Run multiple updates concurrently
        tasks = [update_task(i * 10) for i in range(1, 11)]
        await asyncio.gather(*tasks)

        # Should handle all updates
        assert len(updates) > 0
        assert tracker.current == 100

    def test_repr_pending(self):
        """Test repr for pending tracker."""
        tracker = AsyncProgressTracker(total=100)
        assert "pending" in repr(tracker)

    def test_repr_active(self):
        """Test repr for active tracker."""
        tracker = AsyncProgressTracker(total=100)

        async def mark_active():
            await tracker.update(50)

        asyncio.run(mark_active())
        assert "active" in repr(tracker)

    def test_repr_completed(self):
        """Test repr for completed tracker."""
        tracker = AsyncProgressTracker(total=100)

        async def mark_complete():
            await tracker.complete()

        asyncio.run(mark_complete())
        assert "completed" in repr(tracker)

    @pytest.mark.asyncio
    async def test_workflow_example(self):
        """Test realistic workflow usage."""
        results = []

        async def process_item(item: int):
            await asyncio.sleep(0.01)
            return item * 2

        async def process_all():
            items = list(range(10))
            tracker = AsyncProgressTracker(
                total=len(items), message="Processing items"
            )

            def on_progress(update: ProgressUpdate):
                results.append(f"Progress: {update.percentage_str}")

            tracker.add_callback(on_progress)

            processed = []
            for i, item in enumerate(items):
                result = await process_item(item)
                processed.append(result)
                await tracker.update(i + 1)

            await tracker.complete()
            return processed

        processed = await process_all()
        assert len(processed) == 10
        assert processed[0] == 0
        assert processed[9] == 18
        assert len(results) > 0
