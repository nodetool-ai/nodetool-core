import numpy as np

arr = np.array([1, 2, 3])
arr2 = np.array(arr)
arr3 = np.asarray(arr)

print(arr is arr2)
print(arr is arr3)
