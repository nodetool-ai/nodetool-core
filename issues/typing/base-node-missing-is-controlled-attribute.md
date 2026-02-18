# BaseNode Missing _is_controlled Attribute

**Problem**: The `BaseNode` class was missing the `_is_controlled` attribute definition, causing a type checker warning when `workflow_runner.py` tried to set it dynamically.

**Solution**: Added `_is_controlled: bool = PrivateAttr(default=False)` to the `BaseNode` class attribute definitions.

**Why**: The `_is_controlled` attribute is set dynamically by `WorkflowRunner` when a node has incoming control edges. The type checker couldn't see this attribute being defined, causing an "unresolved attribute" warning. By adding it as a private attribute with a default value, we satisfy the type checker while maintaining the dynamic behavior.

**Files**:
- `src/nodetool/workflows/base_node.py:405`
- `src/nodetool/workflows/workflow_runner.py:1463`

**Date**: 2026-02-18
