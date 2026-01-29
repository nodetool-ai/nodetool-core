# Shell Injection in docker.py Fixed

**Problem**: The `run_command()` function in `deploy/docker.py` used `subprocess.run()` and `subprocess.Popen()` with `shell=True`, which is vulnerable to command injection attacks if command arguments contain untrusted input.

**Solution**: Changed to use `shlex.split()` with `shell=False` to properly parse the command string into a list of arguments without shell expansion.

**Files**:
- `src/nodetool/deploy/docker.py:29-63`

**Date**: 2026-01-21

**Severity**: High - Command injection could allow arbitrary code execution.

**Changes Made**:
1. Replaced `subprocess.run(command, shell=True, ...)` with `subprocess.run(shlex.split(command), shell=False, ...)`
2. Replaced `subprocess.Popen(command, shell=True, ...)` with `subprocess.Popen(shlex.split(command), shell=False, ...)`

**Why This Fix Works**:
- `shlex.split()` parses the command string into a list of arguments while properly handling quoting and escaping
- Using `shell=False` prevents shell expansion of special characters like `$(...)`, `|`, `&&`, etc.
- The existing `_shell_escape()` function was already available but not used in `run_command()`
