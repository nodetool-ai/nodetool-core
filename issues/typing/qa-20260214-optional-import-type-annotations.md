# Optional Imports Need Type Annotations

**Problem**: When optional dependencies are imported in try/except blocks, assigning None to module/class names causes type errors.

**Solution**: Use explicit `Any` type annotations with `type: ignore[assignment]` for None assignments in ImportError handlers.

**Why**: Type checkers see `module_name = None` as incompatible with the module's type. Explicit Any annotations with ignore comments properly type this pattern.

**Example**:
```python
try:
    import paramiko
    from paramiko import AutoAddPolicy, SSHClient
    from paramiko.sftp_client import SFTPClient
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko: Any = None  # type: ignore[assignment]
    SSHClient: Any = None  # type: ignore[assignment]
    AutoAddPolicy: Any = None  # type: ignore[assignment]
    SFTPClient: Any = None  # type: ignore[assignment]
```

**Files**:
- `src/nodetool/deploy/ssh.py`

**Date**: 2026-02-14
