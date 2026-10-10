import asyncio
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import ImageRef
from nodetool.runtime.resources import require_scope, ResourceScope
from PIL import Image
import numpy as np

async def main():
    async with ResourceScope():
        ctx = ProcessingContext()
        pil_img = Image.new('RGB', (10, 10))
        img_ref = await ctx.image_from_pil(pil_img)
        print("Testing image_to_numpy...")
        try:
            arr = await ctx.image_to_numpy(img_ref)
            print("image_to_numpy returned shape:", arr.shape)
        except Exception as e:
            print("image_to_numpy failed:", e)

asyncio.run(main())
