# Missing Encoding Arguments on File Operations

**Problem**: 47 file operations (`open()` and `Path.open()`) in text mode did not specify explicit encoding. This causes inconsistent behavior across platforms since Python's default encoding varies (UTF-8 on modern Linux, but can be different on Windows or older systems).

**Solution**: Added explicit `encoding="utf-8"` parameter to all 47 instances of file operations in text mode across the codebase. The fix was applied using `ruff --preview --select=PLW1514 --fix --unsafe-fixes`.

**Why**: Explicit UTF-8 encoding ensures:
- Consistent cross-platform behavior (Linux, Windows, macOS)
- Predictable text file handling regardless of system locale
- Best practice compliance (PEP 597 recommends explicit encoding)
- Prevents potential encoding-related bugs when code runs on different platforms

**Files Affected**: 47 files across multiple modules including:
- `src/nodetool/agents/agent.py`
- `src/nodetool/agents/docker_runner.py`
- `src/nodetool/agents/serp_providers/serp_types.py`
- `src/nodetool/chat/chat_cli.py`
- `src/nodetool/cli.py`
- `src/nodetool/config/deployment.py`
- `src/nodetool/config/settings.py`
- `src/nodetool/deploy/auth.py`
- `src/nodetool/deploy/deploy_to_runpod.py`
- `src/nodetool/packages/gen_docs.py`
- `src/nodetool/packages/gen_node_docs.py`
- `src/nodetool/packages/gen_workflow_docs.py`
- `src/nodetool/packages/registry.py`
- `src/nodetool/providers/base.py`
- `src/nodetool/proxy/config.py`
- And others...

**Date**: 2026-02-10
