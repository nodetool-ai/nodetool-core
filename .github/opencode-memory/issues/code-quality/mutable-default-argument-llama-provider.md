# Mutable Default Argument in LlamaProvider

**Problem**: The `_normalize_messages_for_llama` method in `LlamaProvider` had a mutable default argument `tools: Sequence[Tool] = []`.

**Solution**: Changed to `tools: Sequence[Tool] | None = None` with a check `if tools is None: tools = ()` at the start of the function.

**Why**: Mutable default arguments in Python are created once at function definition time, not each call. This causes the default list/dict to retain values from previous calls, leading to subtle bugs where state leaks between function invocations.

**Files**: `src/nodetool/providers/llama_provider.py:107`

**Detection**: Static analysis with AST scanning found functions with empty list/dict default arguments.

**Fix Date**: 2026-03-27

**Verification**:
- All 46 llama provider tests pass
- Type checking passes
- Linting passes
