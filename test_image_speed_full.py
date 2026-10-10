import asyncio
import time
import numpy as np
from PIL import Image

def run_tests():
    from nodetool.workflows.torch_support import tensor_from_array
    from nodetool.workflows.processing_context import ProcessingContext

    pil_img = Image.new('RGB', (1024, 1024))

    # test 1: ProcessingContext.image_to_numpy
    def original_image_to_numpy(image):
        return np.array(image)

    def fast_image_to_numpy(image):
        return np.asarray(image)

    start = time.time()
    for _ in range(100):
        t1 = original_image_to_numpy(pil_img)
    end = time.time()
    print(f"Original image_to_numpy 100 times: {end-start:.4f}s")

    start = time.time()
    for _ in range(100):
        t2 = fast_image_to_numpy(pil_img)
    end = time.time()
    print(f"Fast image_to_numpy 100 times: {end-start:.4f}s")


    # test 2: torch_support.tensor_from_pil
    def original_tensor_from_pil(image):
        return tensor_from_array(np.array(image))

    def fast_tensor_from_pil(image):
        return tensor_from_array(np.asarray(image))

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

if __name__ == '__main__':
    run_tests()
