# Improved Exception Handling with Logging in base_node.py

**Problem**: Bare `except Exception:` clauses were swallowing exceptions without any logging, making debugging difficult when type hint resolution or introspection failed silently.

**Solution**: Added debug-level logging to all bare except clauses in `src/nodetool/workflows/base_node.py`:

1. Line 590: `get_type_hints()` fallback in `__init_subclass__`
2. Line 733: `fetch_model_info()` fallback in async model building
3. Line 835: `field_types()` fallback in property access
4. Line 1387: `get_type_hints()` fallback for return type annotations
5. Line 1425: `type_metadata()` fallback in `add_output()`

Each exception handler now logs the exception with `log.debug()` to aid troubleshooting while maintaining graceful fallback behavior.

**Why**: These exception handlers are intentional fallbacks for type introspection edge cases, but logging them helps:
- Debug type resolution issues in node classes
- Identify classes with problematic type annotations
- Maintain observability while handling edge cases gracefully

**Files Modified**:
- `src/nodetool/workflows/base_node.py`

**Date**: 2026-01-20
