import pytest

from nodetool.concurrency.async_iterators import (
    AsyncByteStream,
    async_first,
    async_list,
    async_merge,
    async_slice,
    async_take,
)


class TestAsyncByteStream:
    """Tests for AsyncByteStream class."""

    def test_init_default_chunk_size(self):
        """Test that default chunk_size is 1024."""
        stream = AsyncByteStream(b"hello world")
        assert stream.chunk_size == 1024
        assert stream.data == b"hello world"
        assert stream.index == 0

    def test_init_custom_chunk_size(self):
        """Test that custom chunk_size is preserved."""
        stream = AsyncByteStream(b"test data", chunk_size=256)
        assert stream.chunk_size == 256
        assert stream.data == b"test data"
        assert stream.index == 0

    def test_init_chunk_size_one(self):
        """Test that chunk_size=1 works correctly."""
        stream = AsyncByteStream(b"abc", chunk_size=1)
        assert stream.chunk_size == 1

    def test_aiter_returns_self(self):
        """Test that __aiter__ returns the stream instance."""
        stream = AsyncByteStream(b"test")
        assert stream.__aiter__() is stream

    @pytest.mark.asyncio
    async def test_iterate_single_chunk(self):
        """Test that data fitting in one chunk yields single item."""
        stream = AsyncByteStream(b"hi", chunk_size=1024)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"hi"]

    @pytest.mark.asyncio
    async def test_iterate_exact_multiple_chunks(self):
        """Test that data exactly matching chunk boundaries works."""
        stream = AsyncByteStream(b"abcdefgh", chunk_size=4)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"abcd", b"efgh"]

    @pytest.mark.asyncio
    async def test_iterate_multiple_chunks_with_remainder(self):
        """Test iteration with data that has a remainder chunk."""
        stream = AsyncByteStream(b"hello world", chunk_size=4)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"hell", b"o wo", b"rld"]

    @pytest.mark.asyncio
    async def test_iterate_small_data(self):
        """Test iteration with small data smaller than chunk_size."""
        stream = AsyncByteStream(b"a", chunk_size=1024)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"a"]

    @pytest.mark.asyncio
    async def test_iterate_empty_data(self):
        """Test iteration with empty data."""
        stream = AsyncByteStream(b"", chunk_size=1024)
        chunks = [chunk async for chunk in stream]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_iterate_chunk_size_one(self):
        """Test iteration with chunk_size=1."""
        stream = AsyncByteStream(b"abc", chunk_size=1)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"a", b"b", b"c"]

    @pytest.mark.asyncio
    async def test_iterate_large_data(self):
        """Test iteration with large data."""
        data = b"x" * 10000
        stream = AsyncByteStream(data, chunk_size=1000)
        chunks = [chunk async for chunk in stream]
        assert len(chunks) == 10
        assert all(len(chunk) == 1000 for chunk in chunks[:-1])
        assert chunks == [b"x" * 1000 for _ in range(9)] + [b"x" * 1000]

    @pytest.mark.asyncio
    async def test_iterate_binary_data(self):
        """Test iteration with binary data containing null bytes."""
        data = b"\x00\x01\x02\xff\xfe\xfd"
        stream = AsyncByteStream(data, chunk_size=3)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"\x00\x01\x02", b"\xff\xfe\xfd"]

    @pytest.mark.asyncio
    async def test_iteration_maintains_index(self):
        """Test that iteration updates index correctly."""
        stream = AsyncByteStream(b"0123456789", chunk_size=3)
        assert stream.index == 0
        chunk1 = await stream.__anext__()
        assert chunk1 == b"012"
        assert stream.index == 3
        chunk2 = await stream.__anext__()
        assert chunk2 == b"345"
        assert stream.index == 6

    @pytest.mark.asyncio
    async def test_raise_stop_async_iteration_at_end(self):
        """Test that StopAsyncIteration is raised at end of data."""
        stream = AsyncByteStream(b"a", chunk_size=1024)
        await stream.__anext__()
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    @pytest.mark.asyncio
    async def test_raise_stop_async_iteration_empty_data(self):
        """Test that StopAsyncIteration is raised immediately for empty data."""
        stream = AsyncByteStream(b"", chunk_size=1024)
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    @pytest.mark.asyncio
    async def test_multiple_iterations(self):
        """Test that new iteration starts from beginning."""
        stream = AsyncByteStream(b"test", chunk_size=2)
        chunks1 = [chunk async for chunk in stream]
        assert chunks1 == [b"te", b"st"]
        stream2 = AsyncByteStream(b"test", chunk_size=2)
        chunks2 = [chunk async for chunk in stream2]
        assert chunks2 == [b"te", b"st"]

    @pytest.mark.asyncio
    async def test_unicode_data(self):
        """Test iteration with UTF-8 encoded unicode data."""
        data = "こんにちは世界".encode()
        stream = AsyncByteStream(data, chunk_size=10)
        chunks = [chunk async for chunk in stream]
        assert b"".join(chunks) == data

    @pytest.mark.asyncio
    async def test_chunk_size_larger_than_data(self):
        """Test when chunk_size is larger than data length."""
        stream = AsyncByteStream(b"abc", chunk_size=100)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"abc"]

    @pytest.mark.asyncio
    async def test_chunk_size_equals_data_length(self):
        """Test when chunk_size equals data length."""
        stream = AsyncByteStream(b"abc", chunk_size=3)
        chunks = [chunk async for chunk in stream]
        assert chunks == [b"abc"]


class TestAsyncTake:
    """Tests for async_take function."""

    @pytest.mark.asyncio
    async def test_take_less_than_available(self):
        """Test taking fewer items than available."""
        async def gen():
            for i in range(10):
                yield i

        result = await async_take(gen(), 3)
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_take_exact_count(self):
        """Test taking exact number of available items."""
        async def gen():
            for i in range(3):
                yield i

        result = await async_take(gen(), 3)
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_take_more_than_available(self):
        """Test taking more items than available."""
        async def gen():
            for i in range(3):
                yield i

        result = await async_take(gen(), 10)
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_take_zero(self):
        """Test taking zero items."""
        async def gen():
            for i in range(5):
                yield i

        result = await async_take(gen(), 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_take_from_empty(self):
        """Test taking from empty iterable."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_take(gen(), 5)
        assert result == []

    @pytest.mark.asyncio
    async def test_take_strings(self):
        """Test taking string items."""
        async def gen():
            for s in ["a", "b", "c", "d"]:
                yield s

        result = await async_take(gen(), 2)
        assert result == ["a", "b"]


class TestAsyncSlice:
    """Tests for async_slice function."""

    @pytest.mark.asyncio
    async def test_slice_with_stop(self):
        """Test slicing with start and stop."""
        async def gen():
            for i in range(10):
                yield i

        result = await async_slice(gen(), 2, 5)
        assert result == [2, 3, 4]

    @pytest.mark.asyncio
    async def test_slice_without_stop(self):
        """Test slicing with only start (no stop)."""
        async def gen():
            for i in range(10):
                yield i

        result = await async_slice(gen(), 5)
        assert result == [5, 6, 7, 8, 9]

    @pytest.mark.asyncio
    async def test_slice_start_zero(self):
        """Test slicing from start."""
        async def gen():
            for i in range(10):
                yield i

        result = await async_slice(gen(), 0, 3)
        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_slice_beyond_end(self):
        """Test slicing beyond end of iterable."""
        async def gen():
            for i in range(5):
                yield i

        result = await async_slice(gen(), 3, 10)
        assert result == [3, 4]

    @pytest.mark.asyncio
    async def test_slice_empty_result(self):
        """Test slicing with start beyond end."""
        async def gen():
            for i in range(5):
                yield i

        result = await async_slice(gen(), 10, 15)
        assert result == []

    @pytest.mark.asyncio
    async def test_slice_from_empty(self):
        """Test slicing from empty iterable."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_slice(gen(), 0, 5)
        assert result == []

    @pytest.mark.asyncio
    async def test_slice_strings(self):
        """Test slicing string items."""
        async def gen():
            for s in ["a", "b", "c", "d", "e"]:
                yield s

        result = await async_slice(gen(), 1, 4)
        assert result == ["b", "c", "d"]


class TestAsyncFirst:
    """Tests for async_first function."""

    @pytest.mark.asyncio
    async def test_first_from_nonempty(self):
        """Test getting first item from nonempty iterable."""
        async def gen():
            for i in range(5):
                yield i

        result = await async_first(gen())
        assert result == 0

    @pytest.mark.asyncio
    async def test_first_from_empty_without_default(self):
        """Test getting first from empty iterable without default."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_first(gen())
        assert result is None

    @pytest.mark.asyncio
    async def test_first_from_empty_with_default(self):
        """Test getting first from empty iterable with default."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_first(gen(), default=42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_first_strings(self):
        """Test getting first string item."""
        async def gen():
            for s in ["a", "b", "c"]:
                yield s

        result = await async_first(gen())
        assert result == "a"

    @pytest.mark.asyncio
    async def test_first_with_falsey_default(self):
        """Test that Falsey default values work correctly."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_first(gen(), default=0)
        assert result == 0

        result = await async_first(gen(), default="")
        assert result == ""

        result = await async_first(gen(), default=False)
        assert result is False


class TestAsyncList:
    """Tests for async_list function."""

    @pytest.mark.asyncio
    async def test_list_from_nonempty(self):
        """Test consuming nonempty iterable into list."""
        async def gen():
            for i in range(5):
                yield i

        result = await async_list(gen())
        assert result == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_list_from_empty(self):
        """Test consuming empty iterable into list."""
        async def gen():
            return
            yield  # pragma: no cover

        result = await async_list(gen())
        assert result == []

    @pytest.mark.asyncio
    async def test_list_strings(self):
        """Test consuming string iterable into list."""
        async def gen():
            for s in ["a", "b", "c"]:
                yield s

        result = await async_list(gen())
        assert result == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_single_item(self):
        """Test consuming single item iterable."""
        async def gen():
            yield 42

        result = await async_list(gen())
        assert result == [42]


class TestAsyncMerge:
    """Tests for async_merge function."""

    @pytest.mark.asyncio
    async def test_merge_two_iterables(self):
        """Test merging two async iterables."""
        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield 3
            yield 4

        result = await async_list(async_merge(gen1(), gen2()))
        assert result == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_merge_three_iterables(self):
        """Test merging three async iterables."""
        async def gen1():
            yield 1

        async def gen2():
            yield 2

        async def gen3():
            yield 3

        result = await async_list(async_merge(gen1(), gen2(), gen3()))
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_merge_empty_iterables(self):
        """Test merging empty iterables."""
        async def gen1():
            return
            yield  # pragma: no cover

        async def gen2():
            return
            yield  # pragma: no cover

        result = await async_list(async_merge(gen1(), gen2()))
        assert result == []

    @pytest.mark.asyncio
    async def test_merge_with_empty_first(self):
        """Test merging when first iterable is empty."""
        async def gen1():
            return
            yield  # pragma: no cover

        async def gen2():
            yield 1
            yield 2

        result = await async_list(async_merge(gen1(), gen2()))
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_merge_with_empty_last(self):
        """Test merging when last iterable is empty."""
        async def gen1():
            yield 1
            yield 2

        async def gen2():
            return
            yield  # pragma: no cover

        result = await async_list(async_merge(gen1(), gen2()))
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_merge_strings(self):
        """Test merging string iterables."""
        async def gen1():
            yield "a"
            yield "b"

        async def gen2():
            yield "c"
            yield "d"

        result = await async_list(async_merge(gen1(), gen2()))
        assert result == ["a", "b", "c", "d"]

    @pytest.mark.asyncio
    async def test_merge_single_iterable(self):
        """Test merging single iterable."""
        async def gen1():
            yield 1
            yield 2

        result = await async_list(async_merge(gen1()))
        assert result == [1, 2]
