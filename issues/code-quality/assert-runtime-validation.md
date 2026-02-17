# Assert Statements Used for Runtime Validation

**Problem**: Multiple files used `assert` statements for runtime validation of user input and configuration. Assert statements can be disabled with `python -O`, making them unsuitable for runtime checks.

**Solution**: Replaced all assert statements used for runtime validation with proper exception handling (ValueError, TypeError, RuntimeError).

**Why**: Assert statements are for debugging and can be disabled in production. Runtime validation should always use explicit exception raising to ensure checks are always performed.

**Files**:
- `src/nodetool/deploy/deploy_to_runpod.py` (2 instances)
- `src/nodetool/deploy/runpod_api.py` (3 instances)
- `src/nodetool/messaging/agent_message_processor.py` (1 instance)
- `src/nodetool/messaging/workflow_message_processor.py` (1 instance)
- `src/nodetool/cli_migrations.py` (1 instance)
- `src/nodetool/api/font.py` (1 instance)
- `src/nodetool/runtime/db_sqlite.py` (1 instance)
- `src/nodetool/runtime/resources.py` (1 instance)
- `src/nodetool/proxy/docker_manager.py` (1 instance - refactored to use local variable)

**Date**: 2026-02-17
