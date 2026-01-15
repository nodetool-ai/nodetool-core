# TODO Debt Inventory

**Insight**: The codebase contains 22 TODO comments across multiple files, indicating incomplete implementations or known limitations.

**Rationale**: While TODOs are useful for tracking future work, a large number of TODOs in production code can indicate technical debt that may cause issues:
- TODOs in error handling paths may indicate incomplete error recovery
- TODOs in core functionality may indicate missing features
- Stale TODOs may become technical debt that never gets addressed

**High-Priority TODOs Found**:
- `src/nodetool/workflows/actor.py` - Multiple TODOs about tracking attempts and outputs properly (lines 795, 810, 853, 865, 947, 959, 961)
- `src/nodetool/workflows/output_serialization.py` - TODO about implementing temp storage (lines 125, 322)
- `src/nodetool/workflows/base_node.py` - TODO about handling more comfy special nodes (line 1971)

**Recommendation**: Audit TODOs quarterly and either implement them or convert to tracked issues.

**Date**: 2026-01-15
