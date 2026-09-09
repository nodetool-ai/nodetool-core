# Memory URI cache

`ProcessingContext.get_memory_stats()` reports the number of live objects in
the current resource scope's memory URI cache and a breakdown by Python type
name. Expired entries are excluded.

`ProcessingContext.clear_memory()` clears that cache and returns the number of
live entries removed. Pass a shell-style pattern to select complete cache keys,
for example `context.clear_memory("memory://image-*")`. The same operations are
available through `get_memory_uri_cache_stats()` and `clear_memory_uri_cache()`
in `nodetool.workflows.memory_utils`.

Nested `ResourceScope` instances share their parent's cache. An explicit clear,
including one performed by `ProcessingContext.cleanup()`, affects that shared
cache. Exiting a nested scope does not clear its parent's cache. Use a pattern
when clearing only a subset of objects during an ongoing execution.

Without an active resource scope, the context methods report zero objects and
remove zero entries.
