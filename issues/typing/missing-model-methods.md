# Missing Model Methods for MongoDB-Style Queries

**Problem**: The `durable_inbox.py` module used `RunInboxMessage.find_one()` and `RunInboxMessage.find()` methods with MongoDB-style queries that didn't exist on the model.

**Solution**: Added `find_one()` and `find()` class methods to `RunInboxMessage` in `src/nodetool/models/run_inbox_message.py` that use the existing `ConditionBuilder` API to implement MongoDB-style query support.

**Files**:
- `src/nodetool/models/run_inbox_message.py`
- `src/nodetool/workflows/durable_inbox.py`

**Date**: 2026-01-16
