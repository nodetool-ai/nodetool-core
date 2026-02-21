# Tracing Test Global State Isolation Issue

**Problem**: Three tests in `tests/observability/test_tracing.py` failed because they expected tracing to be enabled by default, but the `TracingConfig` dataclass has `enabled=False` as the default. Additionally, the tracing module uses global state (`_tracing_initialized`, `_tracing_config`, `_global_tracers`) that persists between tests, causing flaky behavior.

**Solution**:
1. Fixed `test_default_config` to expect the actual default values (`enabled=False`, `exporter="none"`)
2. Updated `test_init_tracing` and `test_init_tracing_with_exporter` to explicitly configure tracing as enabled before calling `init_tracing()`
3. Added an `autouse=True` fixture `reset_tracing_state` to reset global state between tests

**Why**: The tracing system is designed to be opt-in (disabled by default) to avoid performance overhead in production. Tests need to explicitly enable tracing when testing tracing functionality. Global state must be reset between tests to prevent cross-test pollution.

**Files**:
- `tests/observability/test_tracing.py`
- `src/nodetool/observability/tracing.py` (for context)

**Date**: 2026-02-21
