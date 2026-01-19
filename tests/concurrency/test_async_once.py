import asyncio

import pytest

from nodetool.concurrency.async_once import AsyncOnce


class TestAsyncOnce:
    """Tests for AsyncOnce class."""

    def test_init(self):
        """Test that AsyncOnce starts in uninitialized state."""
        once = AsyncOnce()
        assert not once.done
        assert once.result is None
        assert once.exception is None

    @pytest.mark.asyncio
    async def test_runs_once(self):
        """Test that the coroutine is only executed once."""
        once = AsyncOnce()
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return call_count

        result1 = await once.run(increment())
        result2 = await once.run(increment())
        result3 = await once.run(increment())

        assert call_count == 1
        assert result1 == 1
        assert result2 == 1
        assert result3 == 1

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test that concurrent calls all wait for the same execution."""
        once = AsyncOnce()
        execution_order = []

        async def track_execution():
            execution_order.append("start")
            await asyncio.sleep(0.05)
            execution_order.append("end")
            return "done"

        results = await asyncio.gather(
            once.run(track_execution()),
            once.run(track_execution()),
            once.run(track_execution()),
        )

        assert execution_order == ["start", "end"]
        assert all(r == "done" for r in results)

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are cached and re-raised."""
        once = AsyncOnce()

        async def raise_error():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await once.run(raise_error())

        with pytest.raises(ValueError, match="test error"):
            await once.run(raise_error())

        assert once.done
        assert once.exception is not None
        assert isinstance(once.exception, ValueError)

    @pytest.mark.asyncio
    async def test_done_property(self):
        """Test the done property reflects completion status."""
        once = AsyncOnce()

        assert not once.done

        async def slow_task():
            await asyncio.sleep(0.1)
            return "done"

        await once.run(slow_task())
        assert once.done

    @pytest.mark.asyncio
    async def test_result_property(self):
        """Test the result property returns the cached result."""
        once = AsyncOnce()

        assert once.result is None

        result = await once.run(asyncio.sleep(0.01, result="success"))
        assert result == "success"
        assert once.result == "success"

    @pytest.mark.asyncio
    async def test_exception_property(self):
        """Test the exception property returns the cached exception."""
        once = AsyncOnce()

        assert once.exception is None

        async def raise_error():
            await asyncio.sleep(0.01)
            raise RuntimeError("error")

        with pytest.raises(RuntimeError, match="error"):
            await once.run(raise_error())

        assert once.exception is not None
        assert isinstance(once.exception, RuntimeError)

    @pytest.mark.asyncio
    async def test_with_return_value(self):
        """Test AsyncOnce with a coroutine that returns a value."""
        once = AsyncOnce()

        result = await once.run(asyncio.sleep(0.01, result=42))
        assert result == 42
        assert once.result == 42

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self):
        """Test multiple sequential calls to run()."""
        once = AsyncOnce()

        results = []
        for _ in range(5):
            result = await once.run(asyncio.sleep(0.01, result="value"))
            results.append(result)

        assert all(r == "value" for r in results)
        assert len(results) == 5
