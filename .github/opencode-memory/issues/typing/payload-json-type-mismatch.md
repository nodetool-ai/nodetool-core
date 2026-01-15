# Payload JSON Type Mismatch

**Problem**: `durable_inbox.py` passed a JSON string to `RunInboxMessage.payload_json` field, but the field is typed as `dict[str, Any]`.

**Solution**: Changed the code to pass `payload_dict` (a dict) instead of `payload_json` (a JSON string). Large payload detection now calculates JSON string length separately.

**Why**: The model defines `payload_json: dict[str, Any] = DBField(default_factory=dict)`, but the code was serializing the payload to JSON string format. Fixed to pass the actual dict, consistent with the model's `append_message` method.

**Files**:
- `src/nodetool/workflows/durable_inbox.py`
- `src/nodetool/models/run_inbox_message.py`

**Date**: 2026-01-15
