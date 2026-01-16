# Type Safety Fixes for Durable Inbox and Recovery

**Problem**: `durable_inbox.py` was using non-existent model methods (`find_one`, `find`) and had a type mismatch for `payload_json` field (expecting `dict[str, Any]` but receiving `str | None` from `json.dumps()`).

**Solution**: 
1. Replaced `find_one` with `get_by_message_id` 
2. Added `_find_messages` method to `RunInboxMessage` model for flexible queries
3. Fixed `payload_json` type by passing dict directly instead of JSON string
4. Added `cast` with proper type guard for `_set_resuming_state` call in `recovery.py`

**Files**:
- `src/nodetool/workflows/durable_inbox.py`
- `src/nodetool/models/run_inbox_message.py`
- `src/nodetool/workflows/recovery.py`

**Date**: 2026-01-16
