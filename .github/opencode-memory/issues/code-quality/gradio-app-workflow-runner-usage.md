# Incorrect WorkflowRunner Usage in gradio_app.py

**Problem**: The `_execute_workflow()` function in `gradio_app.py` was incorrectly using `WorkflowRunner`:
1. Passed `workflow` and `context` as constructor arguments that don't exist
2. Called non-existent `run_stream()` method
3. Called `run()` without required `RunJobRequest` and `ProcessingContext` parameters

**Solution**: Refactored to use the existing `run_graph()` function from `nodetool.dsl.graph`, which properly handles Graph objects and workflow execution. Added type checking to support both `Graph` objects and objects with a `Graph` attribute.

**Files**:
- `src/nodetool/dsl/gradio_app.py:51-70`

**Date**: 2026-01-14
