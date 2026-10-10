import asyncio
import time
import numpy as np
from PIL import Image
from nodetool.workflows.torch_support import tensor_from_array

def test_tensor_from_pil():
    import torch

    # Original way
    def original_tensor_from_pil(image):
        return tensor_from_array(np.array(image))

    # Fast way
    def fast_tensor_from_pil(image):
        return tensor_from_array(np.asarray(image))

    pil_img = Image.new('RGB', (1024, 1024))

    print("Testing tensor_from_pil speed...")

    start = time.time()
    for _ in range(100):
        t1 = original_tensor_from_pil(pil_img)
    end = time.time()
    print(f"Original tensor_from_pil 100 times: {end-start:.4f}s")

    start = time.time()
    for _ in range(100):
        t2 = fast_tensor_from_pil(pil_img)
    end = time.time()
    print(f"Fast tensor_from_pil 100 times: {end-start:.4f}s")

test_tensor_from_pil()
