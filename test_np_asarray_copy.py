import numpy as np
from PIL import Image
import time

def test_performance():
    img = Image.new('RGB', (1024, 1024))

    t0 = time.perf_counter()
    for _ in range(100):
        arr = np.array(img)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    for _ in range(100):
        arr = np.asarray(img)
    t3 = time.perf_counter()

    print(f"np.array: {t1 - t0:.4f}s")
    print(f"np.asarray: {t3 - t2:.4f}s")

test_performance()
