# Tracing Configuration Defaults Test Fix

**Problem**: Tests expected incorrect default values for `TracingConfig` and incorrect behavior for `init_tracing()`.

**Solution**:
1. Fixed `test_default_config` to expect correct defaults: `enabled=False`, `exporter="none"`
2. Updated `init_tracing()` to enable tracing when explicitly called (instead of returning early if config is disabled)

**Why**:
- Tracing should be disabled by default for security and performance reasons
- Calling `init_tracing()` should enable tracing, otherwise the function has no useful purpose
- The tests were incorrectly assuming the old defaults

**Files**:
- `src/nodetool/observability/tracing.py`
- `tests/observability/test_tracing.py`

**Date**: 2026-02-21
