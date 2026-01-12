### Insecure Deserialization in Model Cache

**Date Discovered**: 2026-01-12

**Context**: `ModelCache` class used `pickle.load()` to deserialize cached data from disk. Pickle is insecure by design and can execute arbitrary code during deserialization if the cache file is tampered with.

**Solution**: Replaced `pickle.load()`/`pickle.dump()` with JSON serialization using a custom `CacheJSONEncoder` that handles bytes, datetime, and set types.

**Related Files**:
- `src/nodetool/ml/models/model_cache.py`

**Prevention**: Never use pickle for untrusted data. Use JSON or other safe serialization formats.
