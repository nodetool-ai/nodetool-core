# Unused Typing Imports - January 2026

**Problem**: Multiple files had unused `typing` imports (List, Dict, Tuple, Set, Type) cluttering the codebase and causing F401 lint errors.

**Solution**: Removed unused imports from commonly-used API and chat modules:
- `src/nodetool/api/collection.py`: Removed unused `List`
- `src/nodetool/api/file.py`: Removed unused `List`
- `src/nodetool/api/font.py`: Removed unused `List`
- `src/nodetool/api/job.py`: Removed unused `asyncio`, `List`
- `src/nodetool/api/openai.py`: Removed unused `List` (kept `json`)
- `src/nodetool/api/server.py`: Removed unused `List`
- `src/nodetool/api/settings.py`: Removed unused `Dict`, `List`
- `src/nodetool/api/utils.py`: Removed unused `List`
- `src/nodetool/api/workspace.py`: Removed unused `List`
- `src/nodetool/chat/base_chat_runner.py`: Removed unused `List`
- `src/nodetool/chat/chat_cli.py`: Removed unused `Dict`, `List`

**Note**: Many more unused typing imports exist across the codebase. Consider running `ruff check --select=F401` to identify remaining files.

**Date**: 2026-01-21
