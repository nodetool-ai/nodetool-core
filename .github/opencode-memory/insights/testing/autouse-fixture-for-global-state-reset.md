# Autouse Fixture for Global State Reset

**Insight**: When testing modules with global state, use `@pytest.fixture(autouse=True)` to automatically reset state between tests, preventing cross-test pollution and flaky failures.

**Rationale**: Python modules with global variables (like configuration flags, caches, or initialized state) can cause tests to interfere with each other when the global state is modified by one test and not reset before the next test runs. This is especially problematic with parallel test execution using pytest-xdist.

**Example**: Adding a fixture to reset tracing module state between tests:
```python
@pytest.fixture(autouse=True)
def reset_tracing_state():
    """Reset tracing module state between tests."""
    yield

    # Reset to defaults after each test
    import nodetool.observability.tracing as tracing_module
    tracing_module._tracing_config = TracingConfig()
    tracing_module._tracing_initialized = False
    tracing_module._global_tracers.clear()
```

**Pattern**: 
1. The `autouse=True` parameter makes the fixture run automatically for every test in the module
2. Use `yield` before the reset code to run cleanup after each test completes
3. Import the module inside the fixture to avoid circular imports at module level

**Impact**: Prevents flaky test failures caused by global state pollution, especially when running tests in parallel with pytest-xdist.

**Files**: `tests/observability/test_tracing.py`

**Date**: 2026-02-21
