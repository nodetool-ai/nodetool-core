# Test Failure: Shell Builtin Commands

**Problem**: Tests `test_execute_failure_with_check` and `test_execute_failure_without_check` in `tests/deploy/test_self_hosted.py` were failing with `FileNotFoundError: [Errno 2] No such file or directory: 'exit'`.

**Solution**: Changed test commands from `exit 1` to `sh -c 'exit 1'`. The `exit` command is a shell builtin, not an executable, so it cannot be run directly via `subprocess.run` without `shell=True`. Using `sh -c 'exit 1'` properly invokes a shell to execute the builtin command.

**Files**: `tests/deploy/test_self_hosted.py:65-79`

**Date**: 2026-01-14
