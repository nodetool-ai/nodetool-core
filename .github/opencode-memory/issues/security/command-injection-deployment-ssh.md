# Command Injection in Deployment SSH Commands

**Problem**: SSH commands in `remote_users.py` and `self_hosted.py` were constructed with user-supplied paths without proper shell escaping, allowing command injection.

**Solution**: Added `shlex.quote()` to all user-controlled path variables before interpolating them into shell commands. Added security documentation to `LocalExecutor.execute()` warning about `shell=True` usage.

**Why**: An attacker who can control the `users_file` path or `remote_path` could execute arbitrary commands on the remote/localhost system by injecting shell metacharacters like `; && rm -rf /`.

**Files**:
- `src/nodetool/deploy/remote_users.py` (lines 54, 71, 78, 81)
- `src/nodetool/deploy/self_hosted.py` (line 182, LocalExecutor.execute)

**Date**: 2026-02-11
