### Span Auto-Started Events

**Date Discovered**: 2026-01-12

**Context**: The tracing Span automatically adds a "span_started" event when created

**Solution**: Tests should check for expected event names rather than exact count

**Related Files**: `src/nodetool/observability/tracing.py`, `tests/observability/test_tracing.py`

**Prevention**: Document auto-generated events in span lifecycle
