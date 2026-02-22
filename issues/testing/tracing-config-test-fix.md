# TracingConfig Test Fix

**Problem**: Tests for `TracingConfig` expected incorrect default values (`enabled=True`, `exporter="console"`) but the actual defaults are `enabled=False` and `exporter="none"`.

**Solution**: Updated tests to match the actual implementation:
- `TestTracingConfig.test_default_config`: Changed expectations to `enabled=False` and `exporter="none"`
- `TestInitTracing.test_init_tracing`: Added `configure_tracing(TracingConfig(enabled=True))` before calling `init_tracing()` to properly enable tracing for the test
- `TestInitTracing.test_init_tracing_with_exporter`: Added `configure_tracing(TracingConfig(enabled=True, exporter="console"))` before calling `init_tracing()` with the console exporter

**Why**: The tests were written before the defaults were finalized, causing 3 test failures.

**Files**:
- `tests/observability/test_tracing.py`

**Date**: 2026-02-21
