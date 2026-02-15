# File Metadata Batching in Storage Layer

**Insight**: File metadata operations (exists, size, mtime) were making separate system calls even when called together, resulting in multiple stat() calls for the same file.

**Files**: 
- `src/nodetool/storage/abstract_storage.py` (new FileMetadata dataclass and get_file_metadata method)
- `src/nodetool/storage/file_storage.py` (optimized with single stat() call)
- `src/nodetool/storage/memory_storage.py` (in-memory optimization)
- `src/nodetool/storage/supabase_storage.py` (single API call optimization)

**Original Pattern** (3 separate calls):
```python
exists = await storage.file_exists(key)
if exists:
    size = await storage.get_size(key)
    mtime = await storage.get_mtime(key)
```

**Solution**: Added FileMetadata dataclass and batch method:
```python
@dataclass
class FileMetadata:
    """Batch file metadata to reduce system calls."""
    exists: bool
    size: int | None = None
    mtime: datetime | None = None

async def get_file_metadata(self, key: str) -> FileMetadata:
    """Get file metadata in a single system call for better performance."""
    full_path = os.path.join(self.base_path, key)

    def _stat():
        try:
            stat_result = os.stat(full_path)
            return FileMetadata(
                exists=True,
                size=stat_result.st_size,
                mtime=datetime.fromtimestamp(stat_result.st_mtime),
            )
        except FileNotFoundError:
            return FileMetadata(exists=False)

    return await asyncio.to_thread(_stat)
```

**Impact**: Reduces system calls from 3 to 1 when getting file metadata. For FileStorage, this uses a single stat() instead of separate isfile(), getsize(), and getmtime() calls. For SupabaseStorage, this uses a single info() API call instead of separate exists() and info() calls.

**Date**: 2026-02-06
