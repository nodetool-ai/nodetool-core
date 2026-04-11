# Workflow Utilities Test Coverage

**Insight**: Focus testing on core workflow utilities rather than removed TypeScript components

**Rationale**: After the refactor that moved workflow orchestration, API routes, and agent/LLM orchestration to TypeScript, the Python codebase primarily contains:
- Node execution core (BaseNode, ProcessingContext)
- Workflow utilities (NodeInputs, NodeOutputs, inbox, channel)
- Torch support and device management
- Storage adapters and persistence
- Provider infrastructure

**Example**: When adding tests, prioritize:
1. `NodeInputs` - functional inbox wrapper for node execution
2. `TorchWorkflowSupport` - GPU memory management and OOM retry logic
3. `ProcessingContext` helpers - caching, messaging, environment access

Avoid testing dead code like `NodeOutputs` which depends on the removed `workflow_runner`.

**Impact**: Created 45 new tests covering critical workflow utilities, improving coverage for node execution core.

**Files**: 
- `tests/workflows/test_io.py`
- `tests/workflows/test_torch_support.py`
- `tests/workflows/test_processing_context_helpers.py`

**Date**: 2026-04-11
