import pytest

from nodetool.concurrency.async_iterators import AsyncByteStream


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
