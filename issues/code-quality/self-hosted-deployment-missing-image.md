# SelfHostedDeployment Missing Required Image Parameter

**Problem**: The `deploy add` CLI command for self-hosted deployments was missing the required `image` parameter when creating a `SelfHostedDeployment` instance, causing a runtime error.

**Solution**: Added prompts for Docker image name and tag, and passed an `ImageConfig` object to the `SelfHostedDeployment` constructor.

**Files**: `src/nodetool/cli.py:2574-2579`, `src/nodetool/cli.py:2610`

**Date**: 2026-01-12
