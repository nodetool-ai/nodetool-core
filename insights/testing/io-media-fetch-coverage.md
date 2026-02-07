# IO Media Fetch Module Coverage

**Insight**: Added comprehensive test coverage for the universal URI fetching system in `src/nodetool/io/media_fetch.py`.

**Rationale**: The media_fetch module is a critical utility used throughout the system for fetching media from various sources (HTTP, files, assets, data URIs, memory URIs). It handles complex error handling, multiple URI schemes, and image format conversions.

**Tests Added**:
- `tests/io/test_media_fetch.py` (39 tests across 7 test classes)

**Coverage**:
- `_parse_data_uri()`: base64 data URI parsing with/without MIME types, charset parameters, error handling
- `_fetch_file_uri()`: file:// URI reading, MIME type detection
- `_normalize_image_like_to_png_bytes()`: PIL Image, numpy arrays, bytes, file-like objects, error cases
- `_is_local_storage_url()`: localhost/127.0.0.1 detection with/without ports, path validation
- `_extract_storage_key_from_url()`: UUID and simple key extraction from storage URLs
- `_parse_asset_id_from_uri()`: asset:// URI parsing with/without extensions
- `fetch_uri_bytes_and_mime_async()`: data:, memory://, file:// URIs, unsupported scheme errors

**Files**:
- `tests/io/test_media_fetch.py`
- `src/nodetool/io/media_fetch.py`

**Impact**: 39 new tests covering multi-scheme URI fetching with comprehensive edge case and error handling.
