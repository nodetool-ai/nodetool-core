# Test Helper SIM108 Ternary Operator

**Problem**: The `ThresholdProcessor.process()` method used an if-else block to set `exceeds` variable, triggering a SIM108 linting rule violation.

**Solution**: Replaced the if-else block with a ternary operator: `exceeds = self.value > self.threshold if self.mode == "strict" else self.value >= self.threshold`

**Why**: The logic was simple enough to be expressed as a ternary operator, which is more concise and readable for this case.

**Files**:
- `src/nodetool/workflows/test_helper.py:100-103`

**Date**: 2026-02-18
