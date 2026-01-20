# Environment Variable Documentation Best Practices

**Insight**: Environment variable descriptions should follow a consistent pattern including: purpose, default value, valid values, and example usage.

**Rationale**: Developers often discover environment variables through grep or documentation. Consistent formatting makes it easier to understand configuration options at a glance.

**Example**:
```python
# Instead of:
description="Enable debug mode"

# Use:
description="Enable debug mode for verbose logging. Set to 'true' or 'false'. Default: 'false'. Example: DEBUG=true nodetool serve"
```

**Impact**: Reduces configuration errors and support queries.

**Files**: `src/nodetool/config/settings.py`

**Date**: 2026-01-20
