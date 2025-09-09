import asyncio
import io

import pytest

from nodetool.config.environment import Environment
from nodetool.storage.memory_uri_cache import MemoryUriCache
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_download_file_uses_uri_cache_for_http_url():
    # Ensure a clean, short-lived cache
    Environment.set_memory_uri_cache(MemoryUriCache(default_ttl=60))

    ctx = ProcessingContext(user_id="u", auth_token="t")

    url = "https://example.com/test.bin"
    payload = b"cached-payload"

    # Pre-populate cache with the URL content
    Environment.get_memory_uri_cache().set(url, payload)

    # Should return the cached bytes without performing a network request
    f = await ctx.download_file(url)
    assert isinstance(f, io.BytesIO)
    assert f.read() == payload

