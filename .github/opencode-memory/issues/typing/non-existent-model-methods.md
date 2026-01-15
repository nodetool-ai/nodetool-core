# Non-existent Model Methods in durable_inbox.py

**Problem**: `durable_inbox.py` called `RunInboxMessage.find_one()` and `RunInboxMessage.find()` methods that don't exist on the model.

**Solution**: Replaced with existing methods:
- `find_one({"message_id": id})` → `get_by_message_id(id)`
- `find(query, sort, limit)` → Direct adapter queries with `ConditionBuilder`

**Why**: The `RunInboxMessage` model (and `DBModel` base class) has `query()`, `get()`, and `get_by_message_id()` methods, but not `find()` or `find_one()`. The code was using MongoDB-like syntax that doesn't match the actual ORM.

**Files**:
- `src/nodetool/workflows/durable_inbox.py`

**Date**: 2026-01-15
