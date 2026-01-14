# Dead Import Removal and Missing Export Fixes

**Problem**: Dead imports and non-existent exports were causing ImportError during test collection, preventing 40+ tests from running.

**Issues Fixed**:
1. `src/nodetool/providers/base.py` imported `record_cost` and `trace_provider_call` from `nodetool.observability.tracing` but never used them
2. `src/nodetool/observability/__init__.py` exported `record_cost`, `trace_api_call`, `trace_provider_call`, and `set_response_attributes` which don't exist in `tracing.py`
3. `tests/observability/test_tracing.py` imported and tested non-existent functions

**Solution**: 
- Removed dead imports from `providers/base.py`
- Removed non-existent exports from `observability/__init__.py`
- Updated test file to remove references to missing functions and test classes

**Files Modified**:
- `src/nodetool/providers/base.py`
- `src/nodetool/observability/__init__.py`
- `tests/observability/test_tracing.py`

**Impact**: Test collection errors reduced from 40 to 0, 2200+ tests now run successfully.

**Date**: 2026-01-14
