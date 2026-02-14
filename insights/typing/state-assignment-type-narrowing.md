# State Assignment Type Narrowing

**Insight**: When assigning to Pydantic model state attributes using `model_validate()`, type checkers can't verify that the returned type matches the specific union variant of the deployment.

**Rationale**: The state is a union type like `SelfHostedState | RunPodState | GCPState`, and `deployment.state.__class__` doesn't narrow the type properly for type checkers.

**Example**:
```python
# Type checker sees this as assigning union to union - can't verify correctness
deployment.state = deployment.state.__class__.model_validate(current_state)

# Solution: Use type: ignore[assignment] comment
deployment.state = deployment.state.__class__.model_validate(current_state)  # type: ignore[assignment]
```

**Impact**: Required 4 type ignore comments in `src/nodetool/deploy/state.py` for state revalidation operations.

**Files**:
- `src/nodetool/deploy/state.py` (lines 208, 287, 330, 453)

**Date**: 2026-02-14
