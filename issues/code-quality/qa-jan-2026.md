# QA Check January 2026

**Status**: All validation commands pass

**Typecheck**: 8 warnings for dynamic module attributes in `src/nodetool/nodes/apple/__init__.py` - these are false positives from `ModuleType` dynamic attribute assignment, not actual issues.

**Lint**: No issues

**Tests**: 2342 passed, 69 skipped (job execution tests skipped due to known flakiness with threaded event loops)

**Date**: 2026-01-20
