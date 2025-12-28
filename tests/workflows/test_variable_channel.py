"""Tests for variable channel functionality.

This module tests the VariableChannel and VariableChannelManager classes,
as well as their integration with ProcessingContext.
"""

import asyncio
import pytest
from nodetool.workflows.variable_channel import VariableChannel, VariableChannelManager
from nodetool.workflows.processing_context import ProcessingContext


class TestVariableChannel:
    """Tests for the VariableChannel class."""

    def test_channel_creation(self):
        """Test basic channel creation."""
        channel = VariableChannel("test_var")
        assert channel.name == "test_var"
        assert not channel.has_value
        assert channel.latest_value is None

    @pytest.mark.asyncio
    async def test_scalar_mode_put_and_get(self):
        """Test putting and getting values in scalar mode."""
        channel = VariableChannel("test_var", scalar_mode=True)

        await channel.put("value1")
        assert channel.has_value
        assert channel.latest_value == "value1"

        # Scalar mode replaces the value
        await channel.put("value2")
        assert channel.latest_value == "value2"

    @pytest.mark.asyncio
    async def test_streaming_mode_put_and_iter(self):
        """Test putting and iterating values in streaming mode."""
        channel = VariableChannel("test_var", scalar_mode=False)

        # Put multiple values
        await channel.put("value1")
        await channel.put("value2")
        await channel.put("value3")

        # Close the channel so iteration terminates
        await channel.close()

        # Iterate and collect all values
        values = []
        async for value in channel.iter_values():
            values.append(value)

        assert values == ["value1", "value2", "value3"]

    @pytest.mark.asyncio
    async def test_get_with_timeout(self):
        """Test getting a value with timeout."""
        channel = VariableChannel("test_var")

        # No value yet, should return default after timeout
        result = await channel.get(default="default", timeout=0.01)
        assert result == "default"

        # Put a value and get it
        await channel.put("actual_value")
        result = await channel.get(default="default", timeout=0.1)
        assert result == "actual_value"

    def test_get_nowait(self):
        """Test non-blocking get."""
        channel = VariableChannel("test_var")

        # No value yet
        assert channel.get_nowait(default="default") == "default"

        # Use sync put
        channel.put_sync("value")
        assert channel.get_nowait() == "value"

    @pytest.mark.asyncio
    async def test_producer_tracking(self):
        """Test producer registration and completion."""
        channel = VariableChannel("test_var", scalar_mode=False)

        channel.add_producer(2)
        assert channel.is_open()

        channel.mark_producer_done()
        assert channel.is_open()  # Still one producer left

        channel.mark_producer_done()
        assert not channel.is_open()  # All producers done

    @pytest.mark.asyncio
    async def test_buffer_limit(self):
        """Test buffer limit enforcement."""
        channel = VariableChannel("test_var", buffer_limit=2, scalar_mode=False)

        await channel.put("value1")
        await channel.put("value2")
        await channel.put("value3")  # Should drop value1

        await channel.close()

        values = []
        async for value in channel.iter_values():
            values.append(value)

        assert values == ["value2", "value3"]

    @pytest.mark.asyncio
    async def test_close_channel(self):
        """Test closing a channel."""
        channel = VariableChannel("test_var")

        await channel.put("value")
        await channel.close()

        # Iteration should stop immediately after draining
        values = []
        async for value in channel.iter_values():
            values.append(value)

        assert values == ["value"]


class TestVariableChannelManager:
    """Tests for the VariableChannelManager class."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = VariableChannelManager()
        assert manager.list_variables() == []

    def test_get_or_create_channel(self):
        """Test getting and creating channels."""
        manager = VariableChannelManager()

        # Create channel
        channel = manager.get_channel("test_var", create=True)
        assert channel is not None
        assert channel.name == "test_var"

        # Get same channel
        channel2 = manager.get_channel("test_var", create=False)
        assert channel2 is channel

        # Non-existent channel without create
        channel3 = manager.get_channel("other_var", create=False)
        assert channel3 is None

    @pytest.mark.asyncio
    async def test_set_and_get_variable(self):
        """Test setting and getting variables through manager."""
        manager = VariableChannelManager()

        await manager.set_variable("test_var", "test_value")
        assert await manager.get_variable("test_var") == "test_value"

        # Non-existent variable
        assert await manager.get_variable("other_var", default="default") == "default"

    def test_sync_set_variable(self):
        """Test synchronous variable setting."""
        manager = VariableChannelManager()

        manager.set_variable_sync("test_var", "test_value")
        assert manager.get_variable_nowait("test_var") == "test_value"

    @pytest.mark.asyncio
    async def test_iter_variable(self):
        """Test iterating a streaming variable."""
        manager = VariableChannelManager()

        # Create streaming channel
        channel = manager.get_channel("stream_var", create=True, scalar_mode=False)
        assert channel is not None
        await channel.put("value1")
        await channel.put("value2")
        await channel.close()

        values = []
        async for value in manager.iter_variable("stream_var"):
            values.append(value)

        assert values == ["value1", "value2"]

    def test_list_variables(self):
        """Test listing variables."""
        manager = VariableChannelManager()

        manager.set_variable_sync("var1", "value1")
        manager.set_variable_sync("var2", "value2")

        vars = manager.list_variables()
        assert set(vars) == {"var1", "var2"}

    def test_get_all_values(self):
        """Test getting all variable values."""
        manager = VariableChannelManager()

        manager.set_variable_sync("var1", "value1")
        manager.set_variable_sync("var2", "value2")

        values = manager.get_all_values()
        assert values == {"var1": "value1", "var2": "value2"}

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all channels."""
        manager = VariableChannelManager()

        manager.set_variable_sync("var1", "value1")
        manager.set_variable_sync("var2", "value2")

        await manager.close_all()
        assert manager.list_variables() == []


class TestProcessingContextVariables:
    """Tests for variable channel integration with ProcessingContext."""

    def test_legacy_get_set(self):
        """Test that legacy get/set methods still work."""
        ctx = ProcessingContext()

        ctx.set("test_key", "test_value")
        assert ctx.get("test_key") == "test_value"
        assert ctx.get("nonexistent", "default") == "default"

    def test_set_updates_channel(self):
        """Test that set() updates both variables dict and channel."""
        ctx = ProcessingContext()

        ctx.set("test_key", "test_value")

        # Check legacy dict
        assert ctx.variables["test_key"] == "test_value"

        # Check channel manager
        assert ctx._variable_channels.get_variable_nowait("test_key") == "test_value"

    def test_get_checks_channel_first(self):
        """Test that get() checks channel manager first."""
        ctx = ProcessingContext()

        # Set via channel manager only
        ctx._variable_channels.set_variable_sync("channel_var", "channel_value")

        # get() should find it
        assert ctx.get("channel_var") == "channel_value"

    @pytest.mark.asyncio
    async def test_async_set_get_variable(self):
        """Test async set_variable and get_variable methods."""
        ctx = ProcessingContext()

        await ctx.set_variable("async_var", "async_value")
        assert await ctx.get_variable("async_var") == "async_value"

    @pytest.mark.asyncio
    async def test_streaming_variable(self):
        """Test streaming variable with iter_variable."""
        ctx = ProcessingContext()

        # Get channel and add values
        channel = ctx.get_variable_channel("stream_var", streaming=True)
        await channel.put("value1")
        await channel.put("value2")
        await channel.close()

        values = []
        async for value in ctx.iter_variable("stream_var"):
            values.append(value)

        assert values == ["value1", "value2"]

    @pytest.mark.asyncio
    async def test_get_variable_with_timeout(self):
        """Test get_variable with timeout."""
        ctx = ProcessingContext()

        # Should return default after timeout
        result = await ctx.get_variable("nonexistent", default="default", timeout=0.01)
        assert result == "default"

    def test_context_copy_shares_channels(self):
        """Test that copied context shares variable channels."""
        ctx1 = ProcessingContext()
        ctx1.set("shared_var", "shared_value")

        ctx2 = ctx1.copy()

        # Should share the same channel manager
        assert ctx2._variable_channels is ctx1._variable_channels

        # Setting in one should be visible in the other
        ctx2.set("new_var", "new_value")
        assert ctx1.get("new_var") == "new_value"

    def test_list_variable_channels(self):
        """Test listing variable channels."""
        ctx = ProcessingContext()

        ctx.set("var1", "value1")
        ctx.set("var2", "value2")

        vars = ctx.list_variable_channels()
        assert set(vars) == {"var1", "var2"}

    def test_get_all_variable_values(self):
        """Test getting all variable values."""
        ctx = ProcessingContext()

        ctx.set("var1", "value1")
        ctx.set("var2", "value2")

        values = ctx.get_all_variable_values()
        assert values == {"var1": "value1", "var2": "value2"}

    @pytest.mark.asyncio
    async def test_cleanup_closes_channels(self):
        """Test that cleanup closes variable channels."""
        ctx = ProcessingContext()

        ctx.set("var1", "value1")
        channel = ctx.get_variable_channel("var1")
        assert channel.has_value

        await ctx.cleanup()

        # Channels should be closed
        assert ctx.list_variable_channels() == []


class TestVariableChannelConcurrency:
    """Tests for concurrent access to variable channels."""

    @pytest.mark.asyncio
    async def test_concurrent_producers(self):
        """Test multiple producers writing to same channel."""
        channel = VariableChannel("test_var", scalar_mode=False)
        channel.add_producer(3)

        async def producer(id: int, count: int):
            for i in range(count):
                await channel.put(f"producer_{id}_value_{i}")
                await asyncio.sleep(0.001)
            channel.mark_producer_done()

        # Start multiple producers
        await asyncio.gather(
            producer(1, 3),
            producer(2, 3),
            producer(3, 3),
        )

        # Collect all values
        values = []
        async for value in channel.iter_values():
            values.append(value)

        # Should have all 9 values
        assert len(values) == 9
        assert all(v.startswith("producer_") for v in values)

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self):
        """Test producer-consumer pattern with variable channel."""
        channel = VariableChannel("test_var", scalar_mode=False)
        channel.add_producer(1)
        consumed = []

        async def producer():
            for i in range(5):
                await channel.put(f"value_{i}")
                await asyncio.sleep(0.001)
            channel.mark_producer_done()

        async def consumer():
            async for value in channel.iter_values():
                consumed.append(value)

        await asyncio.gather(producer(), consumer())

        assert consumed == [f"value_{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_late_subscriber(self):
        """Test that late subscriber gets latest value in scalar mode."""
        channel = VariableChannel("test_var", scalar_mode=True)

        # Producer writes values
        await channel.put("value1")
        await channel.put("value2")
        await channel.put("value3")

        # Late subscriber should get latest value
        assert channel.get_nowait() == "value3"
