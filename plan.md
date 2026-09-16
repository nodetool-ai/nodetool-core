1. **Goal:** Improve performance in `tensor_from_pil` by using `np.asarray()` instead of `np.array()`.
2. **Context:** `tensor_from_pil` is called repeatedly for processing images, especially in batch/video workflows. Using `np.array(image)` deep copies the PIL image buffer into a new NumPy array.
3. **Change:** Modify `src/nodetool/workflows/torch_support.py` to use `np.asarray(image)`.
4. **Impact:** Avoids deep copies and speeds up inference pipelines converting PIL images to tensors. Since `tensor_from_array` will do a `.copy()` if `c_contiguous` or `writeable` is false anyway, we are effectively just avoiding the initial `np.array` copy, or PyTorch handles read-only views safely. Wait, `np.asarray().copy()` is faster than `np.array()` on PIL images because `np.array()` uses python-level iteration or inefficient paths inside PIL when deep copying, whereas `np.asarray()` gets the raw buffer view instantly, and then NumPy's optimized C backend copies it.
5. **Code change:**
```python
<<<<<<< SEARCH
def tensor_from_pil(image: Image.Image) -> Any:
    """Create a tensor from a PIL image."""
    return tensor_from_array(np.array(image))
=======
def tensor_from_pil(image: Image.Image) -> Any:
    """Create a tensor from a PIL image."""
    # ⚡ Bolt Optimization: Use np.asarray() instead of np.array() to avoid
    # unnecessary memory allocations and inefficient byte-copying from PIL Images.
    return tensor_from_array(np.asarray(image))
>>>>>>> REPLACE
```
6. **Pre-commit:** run linting/testing and create PR.
