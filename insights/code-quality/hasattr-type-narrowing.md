# Type Narrowing with hasattr() is Limited

**Insight**: Using `hasattr()` to check for attributes doesn't properly narrow types in all type checkers when dealing with union types and optional attributes.

**Rationale**: Even after `hasattr(deployment, "ssh") and deployment.ssh`, type checkers may still see the attribute as unavailable on some union variants.

**Example**:
```python
# Problem: Type checker still warns about key_path and password
if isinstance(deployment, SelfHostedDeployment):
    if hasattr(deployment, "ssh") and deployment.ssh and not deployment.ssh.key_path and not deployment.ssh.password:
        results["warnings"].append("...")

# Solution: Use getattr() with explicit None check
if isinstance(deployment, SelfHostedDeployment):
    ssh_config = getattr(deployment, "ssh", None)
    if ssh_config and not ssh_config.key_path and not ssh_config.password:
        results["warnings"].append("...")
```

**Impact**: Changed SSH validation logic to use `getattr()` pattern in `src/nodetool/deploy/manager.py`.

**Files**:
- `src/nodetool/deploy/manager.py` (line 397)

**Date**: 2026-02-14
