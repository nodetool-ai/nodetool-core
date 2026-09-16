import numpy as np
from PIL import Image

img = Image.new('RGB', (10, 10))
arr1 = np.array(img)
arr2 = np.asarray(img)

print("np.array:", arr1.flags.c_contiguous, arr1.flags.writeable)
print("np.asarray:", arr2.flags.c_contiguous, arr2.flags.writeable)
