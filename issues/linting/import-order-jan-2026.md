# Import Order Issues - January 2026

**Problem**: Module-level imports not at top of file (E402 violations) in multiple files.

**Solution**: Fixed import order in the following files:

1. `src/nodetool/agents/tools/filesystem_tools.py`:
   - Moved docstring after imports
   - Fixed import ordering

2. `src/nodetool/agents/tools/tool_registry.py`:
   - Moved `log = logging.getLogger(__name__)` after imports
   - Fixed import ordering

3. `src/nodetool/workflows/processing_context.py`:
   - Moved `import builtins` into TYPE_CHECKING block (TC003)
   - Fixed import ordering

**Why**: E402 violations indicate poor code organization and can cause subtle import-time side effects. Moving imports to the top of files ensures consistent and predictable module loading.

**Files**:
- `src/nodetool/agents/tools/filesystem_tools.py`
- `src/nodetool/agents/tools/tool_registry.py`
- `src/nodetool/workflows/processing_context.py`

**Date**: 2026-01-19
