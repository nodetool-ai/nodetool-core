# Pre-existing Type Check Errors

**Problem**: The codebase has 145 pre-existing type check errors when running `uv run ty check src --ignore unresolved-import`.

**Categories of errors**:
- `invalid-override` (method overrides): ~40 errors
- `invalid-argument-type`: ~30 errors
- `unresolved-attribute`: ~20 errors
- `call-non-callable`: ~7 errors
- `unsupported-operator`: ~4 errors
- `invalid-assignment`: ~4 errors
- Other miscellaneous errors

**Why**: These are baseline issues that existed before the current branch. They appear to be related to:
1. Pydantic v2 type stubs mismatch with usage
2. Third-party library type stubs (cv2, huggingface_hub, imageio, tqdm)
3. Complex generic type handling in workflow/engine code
4. Optional type handling in async code

**Files with most errors**:
- `src/nodetool/cli.py`: Deployment-related type issues
- `src/nodetool/workflows/`: Workflow state and execution type issues
- `src/nodetool/integrations/`: Third-party integration type issues

**Date**: 2026-01-16
