# Missing ClassVar Annotations on Mutable Class Attributes

**Problem**: RUF012 linting errors - mutable class attributes (dicts, lists) in Tool classes were not annotated with `typing.ClassVar`, causing ruff to warn about potential instance attribute confusion.

**Solution**: Added `typing.ClassVar` annotation import and annotated all `input_schema` class attributes with `ClassVar[dict[str, Any]]` in `src/nodetool/agents/tools/mcp_tools.py`. Used sed for bulk replacement: `sed -i 's/^    input_schema = {$/    input_schema: ClassVar[dict[str, Any]] = {/g'`.

**Why**: ClassVar helps type checkers distinguish between class-level and instance-level attributes, preventing accidental instance shadowing and improving type safety for the MCP tool wrappers.

**Files**:
- `src/nodetool/agents/tools/mcp_tools.py`

**Date**: 2026-02-07
