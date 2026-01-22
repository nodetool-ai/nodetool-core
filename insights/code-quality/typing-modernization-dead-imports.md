# Typing Modernization - Dead Import Removal

**Insight**: Removed unused `List`, `Dict` imports from multiple files in the nodetool codebase. These imports were from the deprecated `typing` module and were not being used in the code.

**Rationale**: Python 3.9+ allows using built-in collection types (`list`, `dict`, `set`, `tuple`) directly as type hints. The old-style imports from `typing` module (`List`, `Dict`, etc.) are deprecated (UP006 violations) and should be replaced. Removing unused imports also reduces module load time and improves code clarity.

**Example**: Before:
```python
from typing import List, Dict, Optional
def process(items: List[str]) -> Dict[str, int]:
    ...
```

After:
```python
from typing import Optional
def process(items: list[str]) -> dict[str, int]:
    ...
```

**Files Modified**:
- `src/nodetool/api/utils.py`
- `src/nodetool/api/server.py`
- `src/nodetool/api/collection.py`
- `src/nodetool/api/openai.py`
- `src/nodetool/api/file.py`
- `src/nodetool/api/settings.py`
- `src/nodetool/api/job.py`
- `src/nodetool/api/font.py`
- `src/nodetool/api/workspace.py`
- `src/nodetool/models/database_adapter.py`
- `src/nodetool/models/supabase_adapter.py`
- `src/nodetool/models/postgres_adapter.py`
- `src/nodetool/models/sqlite_adapter.py`
- `src/nodetool/models/asset.py`
- `src/nodetool/models/condition_builder.py`
- `src/nodetool/types/prediction.py`

**Impact**:
- Removed 17 unused/obsolete typing imports
- Code now uses modern Python 3.9+ type hints
- All tests pass (2389 passed, 69 skipped)
- Lint and typecheck pass

**Date**: 2026-01-20
