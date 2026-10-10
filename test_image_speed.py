import asyncio
import time
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.runtime.resources import require_scope, ResourceScope
from PIL import Image
import numpy as np

async def main():
    async with ResourceScope():
        ctx = ProcessingContext()
        pil_img = Image.new('RGB', (1024, 1024))
        img_ref = await ctx.image_from_pil(pil_img)
        print("Testing image_to_numpy speed...")

        start = time.time()
        for _ in range(100):
            arr = await ctx.image_to_numpy(img_ref)
        end = time.time()
        print(f"Original image_to_numpy 100 times: {end-start:.4f}s")

        # Test my new idea: Use np.asarray(image) instead of np.array(image)
        # to avoid unnecessary byte-copying.
        async def fast_image_to_numpy(image_ref):
            image = await ctx.image_to_pil(image_ref)
            from nodetool.workflows.processing_offload import _in_thread
            return await _in_thread(np.asarray, image)

        start = time.time()
        for _ in range(100):
            arr = await fast_image_to_numpy(img_ref)
        end = time.time()
        print(f"Fast image_to_numpy 100 times: {end-start:.4f}s")

asyncio.run(main())
