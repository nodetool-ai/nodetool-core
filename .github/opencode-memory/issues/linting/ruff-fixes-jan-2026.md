# Ruff Lint Fixes - January 2026

**Problem**: Multiple ruff lint violations found during scheduled QA:
- B007: Unused loop variables
- F541: f-string without placeholders
- W293: Blank lines with whitespace
- TC006: Missing quotes in typing.cast() type expressions
- SIM108: Use ternary operator instead of if-else
- B010: Use direct assignment instead of setattr()

**Solution**: Fixed 42 lint violations across 10 files:
- `src/nodetool/api/mock_data.py`: Renamed unused loop vars to `_`, removed f-string prefix
- `src/nodetool/api/server.py`: Removed trailing whitespace from blank lines
- `src/nodetool/cli.py`: Removed trailing whitespace from blank lines
- `src/nodetool/io/media_fetch.py`: Added quotes to cast() type expression
- `src/nodetool/media/video/video_utils.py`: Converted if-else to ternary operator
- `src/nodetool/nodes/apple/__init__.py`: Replaced setattr() with direct assignment
- `src/nodetool/providers/anthropic_provider.py`: Added quotes to cast() type expressions
- `src/nodetool/providers/openai_compat.py`: Added quotes to cast() type expressions
- `src/nodetool/providers/openai_provider.py`: Added quotes to cast() type expressions
- `tests/api/test_mock_data.py`: Removed trailing whitespace from blank lines

**Files**: Multiple files in src/nodetool/api, src/nodetool/providers, src/nodetool/nodes/apple, tests/api

**Date**: 2026-01-17
