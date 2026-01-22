# Path Traversal Protection Added to File Download

**Problem**: The `/api/files/download/{path:path}` endpoint in `api/file.py` allowed downloading any file on the system that the server process had access to, without path validation. This could allow authenticated users to access sensitive system files.

**Solution**: Added a `_is_safe_download_path()` function that checks if the requested path is a sensitive system path and blocks access to paths like `/etc/passwd`, `/etc/shadow`, `/root`, `/var/log`, `/proc`, `/sys`, etc.

**Files**:
- `src/nodetool/api/file.py:56-72`
- `src/nodetool/api/file.py:164-201`

**Date**: 2026-01-21

**Severity**: Medium - Could allow access to sensitive system files if server has broad file permissions.

**Changes Made**:
1. Added `SENSITIVE_PATHS` constant listing protected system paths
2. Added `_is_safe_download_path()` function to validate paths
3. Added check at the start of `download_file()` endpoint to reject sensitive paths

**Protected Paths**:
- `/etc/passwd`, `/etc/shadow` - System configuration files
- `/root` - Root user home directory
- `/home` - User home directories
- `/var/log` - Log files that may contain secrets
- `/proc` - Process information (can leak system info)
- `/sys` - System information (can leak system info)

**Note**: This is a defense-in-depth measure. The endpoint requires authentication, and in production environments, files should typically be accessed through a workspace-relative path system rather than absolute paths.
