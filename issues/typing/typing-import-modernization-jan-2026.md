# Typing Import Modernization - January 2026

**Problem**: Several API files were importing deprecated typing module types (`List`, `Dict`) that show warnings in ruff (UP035).

**Solution**: Removed deprecated typing imports from 10 API files:
- `src/nodetool/api/file.py` - Removed `from typing import List`
- `src/nodetool/api/font.py` - Removed `from typing import List`
- `src/nodetool/api/collection.py` - Removed `from typing import Any, List, Optional`
- `src/nodetool/api/job.py` - Removed `from typing import List, Optional`
- `src/nodetool/api/openai.py` - Removed `from typing import List`
- `src/nodetool/api/server.py` - Removed `from typing import Any, ClassVar, List`
- `src/nodetool/api/settings.py` - Removed `from typing import Any, Dict, List, Optional`
- `src/nodetool/api/middleware.py` - Changed `from typing import Callable` to `from collections.abc import Callable`
- `src/nodetool/api/utils.py` - Removed `from typing import List, Optional`
- `src/nodetool/api/workspace.py` - Removed `from typing import List, Optional`

**Why**: The `typing.List`, `typing.Dict`, and `typing.Callable` are deprecated in favor of:
- `list`, `dict`, `set` for concrete types (Python 3.9+)
- `collections.abc.Callable` for ABCs (Python 3.10+)

**Impact**:
- 10 files modernized
- All lint checks pass
- All type checks pass
- All API tests pass

**Files**: `src/nodetool/api/*.py`

**Date**: 2026-01-20
