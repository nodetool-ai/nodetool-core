import asyncio

import pytest

from nodetool.concurrency.timeout import (
    TimeoutContext,
    TimeoutError,
    TimeoutPolicy,
    timeout,
    with_timeout,
)


class TestTimeoutError:
    """Tests for TimeoutError exception class."""

    def test_timeout_error_default_message(self):
        """Test default error message includes timeout duration."""
        error = TimeoutError(5.0)
        assert "5.0s" in str(error)
        assert error.timeout_seconds == 5.0

    def test_timeout_error_custom_message(self):
        """Test custom error message."""
        error = TimeoutError(10.0, "Custom timeout message")
        assert "Custom timeout message" in str(error)
        assert error.timeout_seconds == 10.0

    def test_timeout_error_inherits_from_exception(self):
        """Test that TimeoutError inherits from Exception."""
        error = TimeoutError(1.0)
        assert isinstance(error, Exception)


class TestTimeoutDecorator:
    """Tests for timeout decorator function."""

    @pytest.mark.asyncio
    async def test_timeout_decorator_success(self):
        """Test that successful operations complete normally."""

        @timeout(5.0)
        async def quick_function():
            return "success"

        result = await quick_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_decorator_raises_on_timeout(self):
        """Test that timeout raises TimeoutError."""

        @timeout(0.1)
        async def slow_function():
            await asyncio.sleep(1.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await slow_function()

    @pytest.mark.asyncio
    async def test_timeout_decorator_custom_message(self):
        """Test that custom message is included in TimeoutError."""

        @timeout(0.1, "Custom timeout")
        async def slow_function():
            await asyncio.sleep(1.0)

        with pytest.raises(TimeoutError) as exc_info:
            await slow_function()

        assert "Custom timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_decorator_wrapper_name(self):
        """Test that decorator wrapper has expected name."""

        @timeout(5.0)
        async def my_function():
            """My docstring."""
            pass

        assert "wrapper" in my_function.__name__

    @pytest.mark.asyncio
    async def test_timeout_decorator_passes_args_and_kwargs(self):
        """Test that decorator passes arguments correctly."""

        @timeout(5.0)
        async def func_with_args(x, y, z=10):
            return x + y + z

        result = await func_with_args(1, 2, z=3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_timeout_decorator_propagates_other_exceptions(self):
        """Test that non-timeout exceptions propagate."""

        @timeout(5.0)
        async def raising_function():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            await raising_function()


class TestWithTimeoutFunction:
    """Tests for with_timeout function."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self):
        """Test successful operation completes normally."""

        async def quick_coro():
            return "success"

        result = await with_timeout(quick_coro, timeout_seconds=5.0)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_with_timeout_raises_on_timeout(self):
        """Test that timeout raises TimeoutError."""

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await with_timeout(slow_coro, timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_with_timeout_custom_exception(self):
        """Test that custom exception type can be used."""

        class CustomTimeoutError(Exception):
            pass

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(CustomTimeoutError):
            await with_timeout(
                slow_coro,
                timeout_seconds=0.1,
                timeout_exception=CustomTimeoutError,
            )

    @pytest.mark.asyncio
    async def test_with_timeout_custom_message(self):
        """Test custom error message."""

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await with_timeout(
                slow_coro,
                timeout_seconds=0.1,
                exception_message="Custom message",
            )

        assert "Custom message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_timeout_propagates_other_exceptions(self):
        """Test that non-timeout exceptions propagate."""

        async def raising_coro():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            await with_timeout(raising_coro, timeout_seconds=5.0)


class TestTimeoutPolicy:
    """Tests for TimeoutPolicy class."""

    def test_default_values(self):
        """Test default configuration values."""
        policy = TimeoutPolicy()
        assert policy.default_timeout == 30.0
        assert policy.timeout_exception is TimeoutError
        assert policy.default_message is None

    def test_custom_values(self):
        """Test custom configuration values."""

        class CustomError(Exception):
            pass

        policy = TimeoutPolicy(
            default_timeout=60.0,
            timeout_exception=CustomError,
            default_message="Custom default",
        )

        assert policy.default_timeout == 60.0
        assert policy.timeout_exception is CustomError
        assert policy.default_message == "Custom default"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with successful operation."""
        policy = TimeoutPolicy(default_timeout=5.0)

        async def quick_coro():
            return "success"

        result = await policy.execute(quick_coro)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execute with timeout."""
        policy = TimeoutPolicy(default_timeout=0.1)

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await policy.execute(slow_coro)

    @pytest.mark.asyncio
    async def test_execute_override_timeout(self):
        """Test execute with overridden timeout."""
        policy = TimeoutPolicy(default_timeout=60.0)

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError):
            await policy.execute(slow_coro, timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_execute_custom_message(self):
        """Test execute with custom message."""
        policy = TimeoutPolicy(default_timeout=0.1)

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await policy.execute(slow_coro, exception_message="Custom message")

        assert "Custom message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_propagates_other_exceptions(self):
        """Test that non-timeout exceptions propagate."""
        policy = TimeoutPolicy(default_timeout=5.0)

        async def raising_coro():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            await policy.execute(raising_coro)

    @pytest.mark.asyncio
    async def test_execute_custom_exception_type(self):
        """Test execute with custom exception type."""

        class CustomTimeoutError(Exception):
            pass

        policy = TimeoutPolicy(
            default_timeout=0.1,
            timeout_exception=CustomTimeoutError,
        )

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(CustomTimeoutError):
            await policy.execute(slow_coro)

    def test_timeout_context_manager(self):
        """Test creating timeout context manager."""
        policy = TimeoutPolicy(default_timeout=30.0)

        ctx = policy.timeout(5.0)
        assert isinstance(ctx, TimeoutContext)
        assert ctx.timeout_seconds == 5.0
        assert ctx.timeout_exception is TimeoutError

    def test_timeout_with_custom_message(self):
        """Test creating timeout context manager with custom message."""
        policy = TimeoutPolicy(default_timeout=30.0)

        ctx = policy.timeout(5.0, exception_message="Custom message")
        assert ctx.exception_message == "Custom message"

    def test_decorator_with_default_timeout(self):
        """Test using policy as decorator with default timeout."""
        policy = TimeoutPolicy(default_timeout=5.0)

        @policy
        async def quick_function():
            return "success"

        async def run():
            return await quick_function()

        result = asyncio.run(run())
        assert result == "success"

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        policy = TimeoutPolicy(default_timeout=5.0)

        @policy
        async def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestTimeoutContext:
    """Tests for TimeoutContext class."""

    def test_init_default_values(self):
        """Test default initialization values."""
        ctx = TimeoutContext(timeout_seconds=5.0)
        assert ctx.timeout_seconds == 5.0
        assert ctx.timeout_exception is TimeoutError
        assert ctx.exception_message is None

    def test_init_custom_values(self):
        """Test custom initialization values."""

        class CustomError(Exception):
            pass

        ctx = TimeoutContext(
            timeout_seconds=10.0,
            timeout_exception=CustomError,
            exception_message="Custom message",
        )

        assert ctx.timeout_seconds == 10.0
        assert ctx.timeout_exception is CustomError
        assert ctx.exception_message == "Custom message"

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful execution with context."""
        ctx = TimeoutContext(timeout_seconds=5.0)

        async def quick_coro():
            return "success"

        result = await ctx.run(quick_coro)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_run_timeout(self):
        """Test timeout in context."""
        ctx = TimeoutContext(timeout_seconds=0.1)

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await ctx.run(slow_coro)

    @pytest.mark.asyncio
    async def test_run_custom_exception(self):
        """Test custom exception type in context."""

        class CustomError(Exception):
            pass

        ctx = TimeoutContext(
            timeout_seconds=0.1,
            timeout_exception=CustomError,
        )

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(CustomError):
            await ctx.run(slow_coro)

    @pytest.mark.asyncio
    async def test_run_custom_message(self):
        """Test custom message in context."""
        ctx = TimeoutContext(
            timeout_seconds=0.1,
            exception_message="Custom timeout message",
        )

        async def slow_coro():
            await asyncio.sleep(10.0)

        with pytest.raises(TimeoutError) as exc_info:
            await ctx.run(slow_coro)

        assert "Custom timeout message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_run_propagates_other_exceptions(self):
        """Test that non-timeout exceptions propagate."""
        ctx = TimeoutContext(timeout_seconds=5.0)

        async def raising_coro():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            await ctx.run(raising_coro)

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test using context as async context manager."""
        ctx = TimeoutContext(timeout_seconds=5.0)

        async with ctx:
            result = await asyncio.sleep(0)
            assert result is None

    @pytest.mark.asyncio
    async def test_context_manager_run_with_timeout(self):
        """Test that run() method enforces timeout."""
        ctx = TimeoutContext(timeout_seconds=0.1)

        async def slow_coro():
            await asyncio.sleep(10.0)
            return "should not reach here"

        with pytest.raises(TimeoutError):
            await ctx.run(slow_coro)


class TestTimeoutEdgeCases:
    """Tests for edge cases in timeout functionality."""

    @pytest.mark.asyncio
    async def test_zero_timeout(self):
        """Test timeout of zero immediately times out."""

        async def quick_coro():
            await asyncio.sleep(0)
            return "success"

        with pytest.raises(TimeoutError):
            await with_timeout(quick_coro, timeout_seconds=0)

    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        """Test very short timeout still works correctly."""

        async def very_quick_coro():
            return "success"

        result = await with_timeout(very_quick_coro, timeout_seconds=0.001)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_nested_timeouts(self):
        """Test nested timeout operations."""
        inner_timeout = TimeoutPolicy(default_timeout=0.1)
        outer_timeout = TimeoutPolicy(default_timeout=1.0)

        async def quick_inner():
            return "inner"

        async def slow_middle():
            await asyncio.sleep(10.0)
            return "middle"

        outer_result = await outer_timeout.execute(quick_inner)
        assert outer_result == "inner"

        with pytest.raises(TimeoutError):
            await inner_timeout.execute(slow_middle)

    @pytest.mark.asyncio
    async def test_timeout_with_concurrent_operations(self):
        """Test timeout behavior with concurrent operations."""
        policy = TimeoutPolicy(default_timeout=0.5)

        async def slow_operation():
            await asyncio.sleep(10.0)
            return "slow"

        async def quick_operation():
            await asyncio.sleep(0)
            return "quick"

        with pytest.raises(TimeoutError):
            await policy.execute(slow_operation)

        result = await policy.execute(quick_operation)
        assert result == "quick"
