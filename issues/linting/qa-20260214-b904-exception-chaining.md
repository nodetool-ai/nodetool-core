# B904: Exception Chaining in Except Blocks

**Problem**: Several except blocks raised exceptions without `from err` or `from None`, making it hard to distinguish from errors in exception handling itself.

**Solution**: Use `raise ... from None` in except blocks when intentionally wrapping exceptions without preserving the original traceback.

**Why**: Python's B904 rule requires explicit exception chaining in except blocks to make error origins clear.

**Files**:
- `src/nodetool/api/users.py` (3 instances)
- `src/nodetool/deploy/self_hosted.py` (4 instances)

**Date**: 2026-02-14
