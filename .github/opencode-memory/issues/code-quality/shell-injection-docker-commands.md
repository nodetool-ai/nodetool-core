### Shell Injection Risk in Docker Commands

**Date Discovered**: 2026-01-12

**Context**: Docker build and push commands used string interpolation with user-controlled variables (`image_name`, `tag`, `platform`) without proper escaping when calling `subprocess.run()` with `shell=True`.

**Solution**: Added `_shell_escape()` helper function using `shlex.quote()` to properly escape all variables interpolated into shell commands.

**Related Files**:
- `src/nodetool/deploy/docker.py`

**Prevention**: Always use `shlex.quote()` when interpolating variables into shell commands with `shell=True`, or prefer list-based subprocess calls.
