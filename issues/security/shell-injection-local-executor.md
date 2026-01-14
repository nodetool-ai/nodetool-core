# Shell Injection Vulnerability in LocalExecutor

**Problem**: The `LocalExecutor.execute()` method in `self_hosted.py` used `subprocess.run()` with `shell=True`, which is vulnerable to command injection attacks.

**Solution**: Changed to use `shlex.split()` with `shell=False` to properly escape and validate command arguments.

**Files**:
- `src/nodetool/deploy/self_hosted.py:44-72`

**Date**: 2026-01-14

**Severity**: High - Command injection could allow arbitrary code execution.
