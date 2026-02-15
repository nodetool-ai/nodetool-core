# Bulk Linting Fixes with sed Pattern

**Insight**: sed can efficiently fix repetitive linting errors across multiple lines with a single command, avoiding manual edits.

**Rationale**: When dealing with mechanical linting errors like missing type annotations on repeated patterns, sed's search-and-replace is faster and less error-prone than manual edits.

**Example**: Adding ClassVar annotations to all `input_schema` attributes:
```bash
sed -i 's/^    input_schema = {$/    input_schema: ClassVar[dict[str, Any]] = {/g' src/nodetool/agents/tools/mcp_tools.py
```

**Pattern**: Anchor the pattern to the start of lines (`^`) and include unique context (ending `{` for dicts, `=` for assignments) to avoid false matches.

**Impact**: Fixed 20 RUF012 errors in one command, reducing manual effort and ensuring consistency.

**Files**: Applicable to any repetitive linting fixes where the pattern is consistent.

**Date**: 2026-02-07
