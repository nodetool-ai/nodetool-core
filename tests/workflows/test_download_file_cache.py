import io

import pytest

from nodetool.runtime.resources import ResourceScope
from nodetool.storage.memory_uri_cache import MemoryUriCache
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_download_file_uses_uri_cache_for_http_url():
    # Use ResourceScope to manage resources
    async with ResourceScope() as scope:
        ctx = ProcessingContext(user_id="u", auth_token="t")

        url = "https://example.com/test.bin"
        payload = b"cached-payload"

        # Pre-populate cache with the URL content
        scope.get_memory_uri_cache().set(url, payload)

        # Should return the cached bytes without performing a network request
        f = await ctx.download_file(url)
        assert isinstance(f, io.BytesIO)
        assert f.read() == payload
