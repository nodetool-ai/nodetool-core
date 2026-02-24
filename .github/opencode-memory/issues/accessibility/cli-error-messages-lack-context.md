# CLI Error Messages Lack Context

**Problem**: CLI error messages for JSON validation and option conflicts were minimal and did not provide actionable guidance to users.

**Solution**: Enhanced error messages with:
- Clear explanation of why the error occurred
- Examples of correct usage
- Type information when validation fails
- User-friendly suggestions

**Example**:
```python
# Before:
raise click.UsageError(f"Use only one of {json_option} or {json_file_option}.")

# After:
raise click.UsageError(
    f"Cannot specify both {json_option} and {json_file_option}. "
    f"Provide JSON data via only one option: use {json_option} for inline JSON "
    f"or {json_file_option} to read from a file."
)
```

**Files**:
- `src/nodetool/cli.py:220` - Option conflict error
- `src/nodetool/cli.py:471,722` - JSON validation errors

**Date**: 2026-02-24
