# Simplified Conditional Returns

**Problem**: SIM103 linting errors - functions had unnecessary `if` statements that could be simplified by returning the negated condition directly.

**Solution**: Simplified two validation functions in both `src/nodetool/agents/agent.py` and `src/nodetool/api/skills.py`:
- `_is_valid_skill_name`: Changed from `if any(...): return False; return True` to `return not any(...)`
- `_is_valid_skill_description`: Changed from `if _XML_TAG_RE.search(...): return False; return True` to `return not _XML_TAG_RE.search(...)`

**Why**: Simplifying conditional returns makes code more concise, easier to read, and follows Python best practices.

**Files**:
- `src/nodetool/agents/agent.py`
- `src/nodetool/api/skills.py`

**Date**: 2026-02-07
