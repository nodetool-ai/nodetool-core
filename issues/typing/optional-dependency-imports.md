# Optional Dependency Import Resolution

**Problem**: The type checker (ty/basedpyright) reports unresolved imports for `torch` and `sklearn` in files that conditionally import these optional dependencies. These are not runtime errors - the imports are wrapped in try/except blocks - but the type checker cannot resolve them in the current environment.

**Files affected**:
- `src/nodetool/api/memory.py:51` - `import torch`
- `src/nodetool/api/memory.py:66` - `import torch`
- `src/nodetool/api/mcp_server.py` - similar torch imports
- `src/nodetool/cli.py:1270` - `get_supported_hf_types` import
- `src/nodetool/cli.py:3320` - `close_all_database_adapters` import
- `src/nodetool/config/environment.py:562` - `import torch`
- `src/nodetool/config/environment.py:568` - `import torch.mps`
- `src/nodetool/workflows/base_node.py:154` - `import torch`
- `src/nodetool/workflows/memory_utils.py:35` - `import torch`
- `src/nodetool/workflows/memory_utils.py:84` - `import torch`
- `src/nodetool/workflows/processing_context.py:32` - `from sklearn.base import BaseEstimator`

**Why**: These are optional dependencies that may not be installed in all environments. The code correctly uses try/except blocks to handle missing dependencies. The type checker flags these because it cannot find the modules in its search path.

**Solution**: These are known and expected behaviors for optional dependencies. The runtime code handles missing dependencies gracefully. No changes needed to the code itself.

**Workaround for type checking**: Install torch and sklearn in the type checking environment, or configure pyproject.toml to exclude these modules from type checking.

**Date**: 2026-01-20
