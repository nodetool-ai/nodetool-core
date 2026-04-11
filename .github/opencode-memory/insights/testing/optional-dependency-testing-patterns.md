# Optional Dependency Testing Patterns

**Insight**: Test optional dependencies by checking availability flags and testing both code paths

**Rationale**: Many modules have optional dependencies (torch, etc.). Tests should handle both cases gracefully without requiring the dependency to be installed.

**Example**:
```python
def test_is_cuda_available_when_torch_not_available(self):
    """Test is_cuda_available returns False when torch is not available."""
    if not TORCH_AVAILABLE:
        assert is_cuda_available() is False

def test_tensor_from_array_raises_when_torch_not_available(self):
    """Test tensor_from_array raises ImportError when torch is not available."""
    if not TORCH_AVAILABLE:
        array = np.array([1, 2, 3])
        with pytest.raises(ImportError, match="torch is required"):
            tensor_from_array(array)
```

**Impact**: Created 34 tests for torch_support.py that all pass regardless of whether torch is installed, improving CI reliability.

**Files**: `tests/workflows/test_torch_support.py`

**Date**: 2026-04-11
