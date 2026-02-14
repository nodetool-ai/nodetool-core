# Handling click.prompt() Nullable Return Type

**Insight**: `click.prompt()` returns `str | None` (None when user cancels with Ctrl+C), but most APIs expect `str`.

**Rationale**: Type checkers don't understand that `sys.exit()` calls make subsequent code unreachable, so explicit null handling is needed.

**Example**:
```python
admin_token = click.prompt("Enter admin bearer token", hide_input=True)
if admin_token is None:
    console.print("[red]❌ Admin token is required[/]")
    sys.exit(1)
# Type checker still thinks admin_token might be None here
manager = APIUserManager(server_url, admin_token)  # type: ignore[arg-type]
```

**Impact**: Required 4 `type: ignore[arg-type]` comments in CLI commands for user management.

**Alternative**: Could use `assert admin_token is not None` but type checkers may not respect asserts depending on configuration.

**Files**:
- `src/nodetool/cli.py` (lines 5351, 5384, 5433, 5464)

**Date**: 2026-02-14
