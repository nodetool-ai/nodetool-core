# Type Errors in Deployment and State Code

**Problem**: 153 type errors found when running basedpyright type checking.

**Categories**:
- Deployment configuration mismatches (RunPodDeployment, GCPDeployment)
- State assignment type mismatches (SelfHostedState, RunPodState, GCPState)
- Function argument type mismatches (filter_repo_paths, WorkflowRunner)

**Severity**: Errors (exits with code 1)

**Impact**: Type safety not fully enforced; potential runtime errors.

**Suggested fixes**:
1. Review deployment model definitions for correct attribute types
2. Fix state union type assignments
3. Update function signatures to match actual usage

**Date**: 2026-01-12
