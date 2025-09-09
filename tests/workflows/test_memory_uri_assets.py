import base64
import io

import PIL.Image
import pytest

from nodetool.config.environment import Environment
from nodetool.storage.memory_uri_cache import MemoryUriCache
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.media.image.image_utils import image_ref_to_base64_jpeg


@pytest.mark.asyncio
async def test_image_memory_uri_roundtrip_png_and_jpeg():
    # Use a short TTL to avoid leakage across tests
    Environment.set_memory_uri_cache(MemoryUriCache(default_ttl=60))

    ctx = ProcessingContext(user_id="u", auth_token="t")

    # Create a small red image and store as memory:// ImageRef
    img = PIL.Image.new("RGB", (10, 6), color=(255, 0, 0))
    image_ref = await ctx.image_from_pil(img)
    assert image_ref.uri.startswith("memory://")

    # Convert via ProcessingContext (PNG path)
    b64_png = await ctx.image_to_base64(image_ref)
    png_bytes = base64.b64decode(b64_png)
    with PIL.Image.open(io.BytesIO(png_bytes)) as out_img:
        assert out_img.size == (10, 6)
        assert out_img.format == "PNG"

    # Convert via image_utils (JPEG path) and ensure it loads
    b64_jpeg = image_ref_to_base64_jpeg(image_ref, max_size=(64, 64))
    jpeg_bytes = base64.b64decode(b64_jpeg)
    with PIL.Image.open(io.BytesIO(jpeg_bytes)) as out_jpeg:
        assert out_jpeg.size[0] > 0 and out_jpeg.size[1] > 0
        assert out_jpeg.format == "JPEG"

