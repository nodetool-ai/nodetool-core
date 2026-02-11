# Environment Variable Leak in Subprocess Execution

**Problem**: When executing user code via subprocess mode in `runtime_base.py`, the entire parent environment (`os.environ.copy()`) was passed to the subprocess, potentially exposing API keys, tokens, and other secrets to user code.

**Solution**: Created `_filter_sensitive_env_vars()` function that filters out environment variables matching secret patterns (`*_SECRET`, `*_PASSWORD`, `*_TOKEN`, `*_API_KEY`, etc.) before passing the environment to subprocesses.

**Why**: User code running in workflows could inspect `os.environ` and extract sensitive credentials like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. This secrets leak could be exploited to exfiltrate credentials.

**Files**:
- `src/nodetool/code_runners/runtime_base.py` (line 446, added filter function)

**Date**: 2026-02-11
