import asyncio

import pytest

from nodetool.concurrency.async_iterators import (
    AsyncByteStream,
    async_filter,
    async_first,
    async_flat_map,
    async_list,
    async_map,
    async_merge,
    async_reduce,
    async_slice,
    async_take,
    async_zip,
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


class TestAsyncFilter:
    """Tests for async_filter function."""

    @pytest.mark.asyncio
    async def test_filter_with_sync_predicate(self):
        """Test filtering with a sync predicate function."""

        async def gen():
            for i in range(10):
                yield i

        result = await async_list(async_filter(lambda x: x % 2 == 0, gen()))
        assert result == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_filter_with_async_predicate(self):
        """Test filtering with an async predicate function."""

        async def gen():
            for i in range(10):
                yield i

        async def is_even(x):
            return x % 2 == 0

        result = await async_list(async_filter(is_even, gen()))
        assert result == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_filter_all_pass(self):
        """Test filtering when all items pass the predicate."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_filter(lambda x: True, gen()))
        assert result == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_filter_none_pass(self):
        """Test filtering when no items pass the predicate."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_filter(lambda x: False, gen()))
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_empty_iterable(self):
        """Test filtering an empty iterable."""

        async def gen():
            return
            yield  # pragma: no cover

        result = await async_list(async_filter(lambda x: True, gen()))
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_strings(self):
        """Test filtering string items."""

        async def gen():
            for s in ["apple", "banana", "cherry", "date"]:
                yield s

        result = await async_list(async_filter(lambda x: len(x) > 5, gen()))
        assert result == ["banana", "cherry"]

    @pytest.mark.asyncio
    async def test_filter_with_complex_predicate(self):
        """Test filtering with a more complex predicate."""

        async def gen():
            for i in range(20):
                yield i

        # Filter numbers that are divisible by 2 or 3
        result = await async_list(async_filter(lambda x: x % 2 == 0 or x % 3 == 0, gen()))
        assert result == [0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

    @pytest.mark.asyncio
    async def test_filter_preserves_order(self):
        """Test that filtering preserves the original order."""

        async def gen():
            for i in [5, 2, 8, 1, 9, 3, 7, 4, 6]:
                yield i

        # Filter even numbers
        result = await async_list(async_filter(lambda x: x % 2 == 0, gen()))
        assert result == [2, 8, 4, 6]

    @pytest.mark.asyncio
    async def test_filter_with_async_predicate_that_awaits(self):
        """Test async predicate that performs async operations."""

        async def gen():
            for i in range(5):
                yield i

        async def is_greater_than_two(x):
            # Simulate async operation
            await asyncio.sleep(0)
            return x > 2

        result = await async_list(async_filter(is_greater_than_two, gen()))
        assert result == [3, 4]

    @pytest.mark.asyncio
    async def test_filter_single_item_passes(self):
        """Test filtering when only one item passes."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_filter(lambda x: x == 2, gen()))
        assert result == [2]

    @pytest.mark.asyncio
    async def test_filter_first_item_fails(self):
        """Test filtering when first item fails predicate."""

        async def gen():
            for i in range(1, 6):
                yield i

        result = await async_list(async_filter(lambda x: x > 1, gen()))
        assert result == [2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_filter_last_item_fails(self):
        """Test filtering when last item fails predicate."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_filter(lambda x: x < 4, gen()))
        assert result == [0, 1, 2, 3]


class TestAsyncMap:
    """Tests for async_map function."""

    @pytest.mark.asyncio
    async def test_map_with_sync_function(self):
        """Test mapping with a sync function."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_map(lambda x: x * 2, gen()))
        assert result == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_map_with_async_function(self):
        """Test mapping with an async function."""

        async def gen():
            for i in range(5):
                yield i

        async def double(x):
            return x * 2

        result = await async_list(async_map(double, gen()))
        assert result == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_map_empty_iterable(self):
        """Test mapping over empty iterable."""

        async def gen():
            return
            yield  # pragma: no cover

        result = await async_list(async_map(lambda x: x * 2, gen()))
        assert result == []

    @pytest.mark.asyncio
    async def test_map_single_item(self):
        """Test mapping single item."""

        async def gen():
            yield 5

        result = await async_list(async_map(lambda x: x * 3, gen()))
        assert result == [15]

    @pytest.mark.asyncio
    async def test_map_type_transformation(self):
        """Test mapping that transforms types."""

        async def gen():
            for i in range(3):
                yield i

        result = await async_list(async_map(lambda x: str(x), gen()))
        assert result == ["0", "1", "2"]

    @pytest.mark.asyncio
    async def test_map_complex_transformation(self):
        """Test mapping with complex transformation."""

        async def gen():
            for i in range(3):
                yield i

        result = await async_list(async_map(lambda x: x**2 + 1, gen()))
        assert result == [1, 2, 5]

    @pytest.mark.asyncio
    async def test_map_strings(self):
        """Test mapping string items."""

        async def gen():
            for s in ["a", "b", "c"]:
                yield s

        result = await async_list(async_map(lambda x: x.upper(), gen()))
        assert result == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_map_with_async_function_that_awaits(self):
        """Test async mapper that performs async operations."""

        async def gen():
            for i in range(3):
                yield i

        async def add_with_delay(x):
            await asyncio.sleep(0)
            return x + 10

        result = await async_list(async_map(add_with_delay, gen()))
        assert result == [10, 11, 12]

    @pytest.mark.asyncio
    async def test_map_preserves_order(self):
        """Test that mapping preserves original order."""

        async def gen():
            for i in [5, 2, 8, 1, 9]:
                yield i

        result = await async_list(async_map(lambda x: x * 2, gen()))
        assert result == [10, 4, 16, 2, 18]

    @pytest.mark.asyncio
    async def test_map_with_lambda_returning_none(self):
        """Test mapping when function returns None."""

        async def gen():
            for i in range(3):
                yield i

        result = await async_list(async_map(lambda x: None, gen()))
        assert result == [None, None, None]

    @pytest.mark.asyncio
    async def test_map_with_falsey_values(self):
        """Test mapping when function returns Falsey values."""

        async def gen():
            for i in range(3):
                yield i

        result = await async_list(async_map(lambda x: 0 if x % 2 == 0 else "", gen()))
        assert result == [0, "", 0]

    @pytest.mark.asyncio
    async def test_map_identity(self):
        """Test mapping with identity function."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_list(async_map(lambda x: x, gen()))
        assert result == [0, 1, 2, 3, 4]


class TestAsyncReduce:
    """Tests for async_reduce function."""

    @pytest.mark.asyncio
    async def test_reduce_sum(self):
        """Test reducing with sum operation."""

        async def gen():
            for i in range(5):
                yield i

        result = await async_reduce(lambda acc, x: acc + x, gen(), 0)
        assert result == 10

    @pytest.mark.asyncio
    async def test_reduce_multiply(self):
        """Test reducing with multiplication."""

        async def gen():
            for i in range(1, 6):
                yield i

        result = await async_reduce(lambda acc, x: acc * x, gen(), 1)
        assert result == 120

    @pytest.mark.asyncio
    async def test_reduce_with_async_function(self):
        """Test reducing with an async reduction function."""

        async def gen():
            for i in range(5):
                yield i

        async def async_add(acc, x):
            return acc + x

        result = await async_reduce(async_add, gen(), 0)
        assert result == 10

    @pytest.mark.asyncio
    async def test_reduce_empty_iterable(self):
        """Test reducing empty iterable returns initial value."""

        async def gen():
            return
            yield  # pragma: no cover

        result = await async_reduce(lambda acc, x: acc + x, gen(), 42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_reduce_single_item(self):
        """Test reducing single item."""

        async def gen():
            yield 5

        result = await async_reduce(lambda acc, x: acc + x, gen(), 10)
        assert result == 15

    @pytest.mark.asyncio
    async def test_reduce_build_list(self):
        """Test reducing to build a list."""

        async def gen():
            for i in range(4):
                yield i

        async def append(acc, x):
            acc.append(x)
            return acc

        result = await async_reduce(append, gen(), [])
        assert result == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_reduce_build_dict(self):
        """Test reducing to build a dict."""

        async def gen():
            for s in ["a", "b", "c"]:
                yield s

        def add_to_dict(acc, x):
            acc[x] = len(x)
            return acc

        result = await async_reduce(add_to_dict, gen(), {})
        assert result == {"a": 1, "b": 1, "c": 1}

    @pytest.mark.asyncio
    async def test_reduce_with_string_initial(self):
        """Test reducing with string initial value."""

        async def gen():
            for s in ["hello", " ", "world"]:
                yield s

        result = await async_reduce(lambda acc, x: acc + x, gen(), "")
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_reduce_max_value(self):
        """Test reducing to find maximum value."""

        async def gen():
            for i in [3, 7, 2, 9, 4]:
                yield i

        result = await async_reduce(lambda acc, x: acc if acc > x else x, gen(), 0)
        assert result == 9

    @pytest.mark.asyncio
    async def test_reduce_with_complex_transformation(self):
        """Test reducing with complex transformation."""

        async def gen():
            for i in range(4):
                yield i

        def complex_reduce(acc, x):
            return acc + (x * 2)

        result = await async_reduce(complex_reduce, gen(), 0)
        assert result == 12  # (0*2) + (1*2) + (2*2) + (3*2) = 0 + 2 + 4 + 6 = 12

    @pytest.mark.asyncio
    async def test_reduce_preserves_order(self):
        """Test that reduce processes items in order."""

        async def gen():
            for i in [1, 2, 3, 4, 5]:
                yield i

        result = await async_reduce(lambda acc, x: acc * 10 + x, gen(), 0)
        assert result == 12345

    @pytest.mark.asyncio
    async def test_reduce_with_tuple_initial(self):
        """Test reducing with tuple as accumulator."""

        async def gen():
            for i in range(3):
                yield i

        def tuple_append(acc, x):
            return (*acc, x)

        result = await async_reduce(tuple_append, gen(), ())
        assert result == (0, 1, 2)

    @pytest.mark.asyncio
    async def test_reduce_with_set_initial(self):
        """Test reducing with set as accumulator."""

        async def gen():
            for i in [1, 2, 2, 3, 3, 3]:
                yield i

        def set_add(acc, x):
            acc.add(x)
            return acc

        result = await async_reduce(set_add, gen(), set())
        assert result == {1, 2, 3}

    @pytest.mark.asyncio
    async def test_reduce_count_matching(self):
        """Test reducing to count matching items."""

        async def gen():
            for i in range(10):
                yield i

        def count_even(acc, x):
            return acc + (1 if x % 2 == 0 else 0)

        result = await async_reduce(count_even, gen(), 0)
        assert result == 5


class TestAsyncFlatMap:
    """Tests for async_flat_map function."""

    @pytest.mark.asyncio
    async def test_flat_map_with_sync_function(self):
        """Test flat mapping with a sync function."""

        async def gen():
            for i in range(3):
                yield i

        async def split(x):
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(split, gen()))
        # gen() yields 0, 1, 2
        # split(0) -> [], split(1) -> [0], split(2) -> [0, 1]
        assert result == [0, 0, 1]

    @pytest.mark.asyncio
    async def test_flat_map_with_async_function(self):
        """Test flat mapping with an async function."""

        async def gen():
            for i in range(3):
                yield i

        async def async_split(x):
            # Simulate async operation
            await asyncio.sleep(0)
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(async_split, gen()))
        # gen() yields 0, 1, 2
        # async_split(0) -> [], async_split(1) -> [0], async_split(2) -> [0, 1]
        assert result == [0, 0, 1]

    @pytest.mark.asyncio
    async def test_flat_map_empty_iterable(self):
        """Test flat mapping over empty iterable."""

        async def gen():
            return
            yield  # pragma: no cover

        async def split(x):
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(split, gen()))
        assert result == []

    @pytest.mark.asyncio
    async def test_flat_map_empty_inner_iterables(self):
        """Test flat mapping when all inner iterables are empty."""

        async def gen():
            for _i in range(3):
                yield 0

        async def split(x):
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(split, gen()))
        assert result == []

    @pytest.mark.asyncio
    async def test_flat_map_single_item(self):
        """Test flat mapping single item."""

        async def gen():
            yield 2

        async def split(x):
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(split, gen()))
        assert result == [0, 1]

    @pytest.mark.asyncio
    async def test_flat_map_type_transformation(self):
        """Test flat mapping that transforms types."""

        async def gen():
            for s in ["hello", "world"]:
                yield s

        async def split_chars(s):
            for c in s:
                yield c

        result = await async_list(async_flat_map(split_chars, gen()))
        assert result == ["h", "e", "l", "l", "o", "w", "o", "r", "l", "d"]

    @pytest.mark.asyncio
    async def test_flat_map_preserves_order(self):
        """Test that flat mapping preserves original order."""

        async def gen():
            for i in [2, 1, 3]:
                yield i

        async def split(x):
            for j in range(x):
                yield j

        result = await async_list(async_flat_map(split, gen()))
        # gen() yields 2, 1, 3
        # split(2) -> [0, 1], split(1) -> [0], split(3) -> [0, 1, 2]
        assert result == [0, 1, 0, 0, 1, 2]

    @pytest.mark.asyncio
    async def test_flat_map_with_complex_transformation(self):
        """Test flat mapping with complex transformation."""

        async def gen():
            for i in [1, 2, 3]:
                yield i

        async def replicate_and_add(x):
            for offset in range(x):
                yield x + offset

        result = await async_list(async_flat_map(replicate_and_add, gen()))
        # gen() yields 1, 2, 3
        # replicate_and_add(1) -> [1], replicate_and_add(2) -> [2, 3], replicate_and_add(3) -> [3, 4, 5]
        assert result == [1, 2, 3, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_flat_map_with_filter(self):
        """Test flat mapping combined with filtering behavior."""

        async def gen():
            for i in range(3):
                yield i

        async def only_even_splits(x):
            for j in range(x):
                if j % 2 == 0:
                    yield j

        result = await async_list(async_flat_map(only_even_splits, gen()))
        # gen() yields 0, 1, 2
        # only_even_splits(0) -> [], only_even_splits(1) -> [0], only_even_splits(2) -> [0]
        assert result == [0, 0]

    @pytest.mark.asyncio
    async def test_flat_map_nested_lists(self):
        """Test flat mapping nested list-like structures."""

        async def gen():
            for lst in [[1, 2], [3, 4, 5], [6]]:
                yield lst

        async def flatten(lst):
            for item in lst:
                yield item

        result = await async_list(async_flat_map(flatten, gen()))
        assert result == [1, 2, 3, 4, 5, 6]

    @pytest.mark.asyncio
    async def test_flat_map_with_single_element_inner(self):
        """Test flat mapping when inner iterables have single element."""

        async def gen():
            for i in range(5):
                yield i

        async def wrap(x):
            yield x

        result = await async_list(async_flat_map(wrap, gen()))
        assert result == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_flat_map_with_varying_lengths(self):
        """Test flat mapping with varying inner iterable lengths."""

        async def gen():
            for i in range(5):
                yield i

        async def variable_split(x):
            # Create iterables of different lengths
            for j in range(x % 3):
                yield j

        result = await async_list(async_flat_map(variable_split, gen()))
        # gen() yields 0, 1, 2, 3, 4
        # variable_split(0) -> [], variable_split(1) -> [0], variable_split(2) -> [0, 1]
        # variable_split(3) -> [], variable_split(4) -> [0]
        assert result == [0, 0, 1, 0]

    @pytest.mark.asyncio
    async def test_flat_map_identity(self):
        """Test flat mapping that yields each input as-is."""

        async def gen():
            for i in [1, 2, 3]:
                yield i

        async def identity(x):
            yield x

        result = await async_list(async_flat_map(identity, gen()))
        assert result == [1, 2, 3]


class TestAsyncZip:
    """Tests for async_zip function."""

    @pytest.mark.asyncio
    async def test_zip_two_iterables_equal_length(self):
        """Test zipping two iterables of equal length."""

        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield 'a'
            yield 'b'

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(1, 'a'), (2, 'b')]

    @pytest.mark.asyncio
    async def test_zip_two_iterables_unequal_length(self):
        """Test zipping two iterables where one is longer."""

        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield 'a'
            yield 'b'
            yield 'c'

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(1, 'a'), (2, 'b')]

    @pytest.mark.asyncio
    async def test_zip_three_iterables(self):
        """Test zipping three iterables."""

        async def gen1():
            yield 1
            yield 2

        async def gen2():
            yield 'a'
            yield 'b'

        async def gen3():
            yield True
            yield False

        result = await async_list(async_zip(gen1(), gen2(), gen3()))
        assert result == [(1, 'a', True), (2, 'b', False)]

    @pytest.mark.asyncio
    async def test_zip_single_iterable(self):
        """Test zipping a single iterable returns 1-tuples."""

        async def gen1():
            yield 1
            yield 2

        result = await async_list(async_zip(gen1()))
        assert result == [(1,), (2,)]

    @pytest.mark.asyncio
    async def test_zip_empty_iterables(self):
        """Test zipping empty iterables."""

        async def gen1():
            return
            yield  # pragma: no cover

        async def gen2():
            return
            yield  # pragma: no cover

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == []

    @pytest.mark.asyncio
    async def test_zip_one_empty_iterable(self):
        """Test zipping when one iterable is empty."""

        async def gen1():
            return
            yield  # pragma: no cover

        async def gen2():
            yield 1
            yield 2

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == []

    @pytest.mark.asyncio
    async def test_zip_first_exhausted(self):
        """Test zipping when first iterable is shortest."""

        async def gen1():
            yield 1

        async def gen2():
            yield 'a'
            yield 'b'
            yield 'c'

        async def gen3():
            yield True
            yield False

        result = await async_list(async_zip(gen1(), gen2(), gen3()))
        assert result == [(1, 'a', True)]

    @pytest.mark.asyncio
    async def test_zip_middle_exhausted(self):
        """Test zipping when middle iterable is shortest."""

        async def gen1():
            yield 1
            yield 2
            yield 3

        async def gen2():
            yield 'a'

        async def gen3():
            yield True
            yield False

        result = await async_list(async_zip(gen1(), gen2(), gen3()))
        assert result == [(1, 'a', True)]

    @pytest.mark.asyncio
    async def test_zip_different_types(self):
        """Test zipping iterables with different types."""

        async def gen_int():
            yield 1
            yield 2

        async def gen_str():
            yield 'a'
            yield 'b'

        async def gen_bool():
            yield True
            yield False

        async def gen_float():
            yield 1.5
            yield 2.5

        result = await async_list(async_zip(gen_int(), gen_str(), gen_bool(), gen_float()))
        assert result == [(1, 'a', True, 1.5), (2, 'b', False, 2.5)]

    @pytest.mark.asyncio
    async def test_zip_preserves_order(self):
        """Test that zipping preserves order within each iterable."""

        async def gen1():
            for i in [5, 2, 8]:
                yield i

        async def gen2():
            for s in ['a', 'b', 'c']:
                yield s

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(5, 'a'), (2, 'b'), (8, 'c')]

    @pytest.mark.asyncio
    async def test_zip_no_iterables(self):
        """Test zipping with no iterables."""

        result = await async_list(async_zip())
        assert result == []

    @pytest.mark.asyncio
    async def test_zip_single_item_iterables(self):
        """Test zipping iterables with single items."""

        async def gen1():
            yield 42

        async def gen2():
            yield 'hello'

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(42, 'hello')]

    @pytest.mark.asyncio
    async def test_zip_with_none_values(self):
        """Test zipping iterables containing None values."""

        async def gen1():
            yield 1
            yield None
            yield 3

        async def gen2():
            yield None
            yield 2
            yield None

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(1, None), (None, 2), (3, None)]

    @pytest.mark.asyncio
    async def test_zip_with_falsey_values(self):
        """Test zipping iterables containing Falsey values."""

        async def gen1():
            yield 0
            yield ''

        async def gen2():
            yield False
            yield []

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [(0, False), ('', [])]

    @pytest.mark.asyncio
    async def test_zip_complex_objects(self):
        """Test zipping iterables with complex objects."""

        async def gen1():
            yield [1, 2]
            yield [3, 4]

        async def gen2():
            yield {'a': 1}
            yield {'b': 2}

        result = await async_list(async_zip(gen1(), gen2()))
        assert result == [([1, 2], {'a': 1}), ([3, 4], {'b': 2})]
