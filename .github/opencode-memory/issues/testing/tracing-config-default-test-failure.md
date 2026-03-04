# TracingConfig Defaults Test Failure

**Problem**: Tests expected `TracingConfig` to have `enabled=True` and `exporter="console"` by default, but the implementation had `enabled=False` and `exporter="none"`.

**Solution**: Restored defaults to `enabled: bool = True` and `exporter: str = "console"` in `TracingConfig` dataclass.

**Why**:
- The `init_tracing()` function checks `_tracing_config.enabled` before initializing
- With `enabled=False` by default, `init_tracing()` would just log "Tracing disabled by configuration" and return
- Tests expect `init_tracing()` to actually enable tracing
- Original implementation (commit d3b54c5c) had these defaults
- A previous QA fix (commit b37a5e9b) had also restored these defaults

**Files**: `src/nodetool/observability/tracing.py:769-770`, `tests/observability/test_tracing.py:324`

**Date**: 2026-02-21
