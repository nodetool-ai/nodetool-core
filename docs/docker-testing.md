[← Back to Docs Index](index.md)

# Docker Job Execution Testing Guide

This guide covers manual testing, automated testing, and debugging for Docker-based job execution.

## Prerequisites

- Docker installed and running
- `nodetool` Docker image built
- `nodetool-base` package included in image

## Quick Verification

### Frontend E2E API Server Image

Use the dedicated `Dockerfile.e2e` to run a lightweight FastAPI server for
frontend end-to-end tests. The container installs `nodetool-base` plus the
current `nodetool-core` checkout, exposes port `8000`, and launches
`uvicorn nodetool.api.app:app` bound to `0.0.0.0` so it can share the host
network with the UI container.

```bash
# Build a public image that will be pushed to GHCR
docker build -f Dockerfile.e2e -t ghcr.io/<org>/nodetool-e2e:latest .

# Recommended: run on the same host/network namespace as the UI container
docker run --rm --net host ghcr.io/<org>/nodetool-e2e:latest

# Alternate: publish the API on localhost for UI running directly on the host
docker run --rm -p 8000:8000 ghcr.io/<org>/nodetool-e2e:latest
```

**Publishing to GHCR** (the image should remain public for CI to consume):

```bash
export GHCR_USER=<github-username>
export GHCR_TOKEN=<github-personal-access-token-with-write-packages>

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
docker push ghcr.io/<org>/nodetool-e2e:latest
```

In CI, `.github/workflows/docker-publish-e2e.yaml` automatically builds the
same image from `Dockerfile.e2e` and publishes it to the public
`ghcr.io/nodetool-ai/nodetool-e2e` repository on every push to `main` (or when
the workflow is manually dispatched). Local pushes should only be needed when
testing changes before they merge to the default branch.

Override the refs that get installed if you need a different branch/tag:

```bash
docker build \
  --build-arg NODETOOL_BASE_REF=release/2025.01 \
  -f Dockerfile.e2e \
  -t ghcr.io/<org>/nodetool-e2e:release-2025-01 .
```

Once the container is running, point the frontend to `http://localhost:8000`
and keep both UI and API on the same host to avoid browser security issues
during Cypress/Playwright runs.

### Check Image Status

```bash
# Verify nodetool command exists
docker run --rm --entrypoint which nodetool nodetool

# Verify --jsonl flag is available
docker run --rm --entrypoint nodetool nodetool run --help | grep jsonl

# List available commands
docker run --rm --entrypoint nodetool nodetool --help
```

### Issue: Node type not found

**Symptom**: Error about unknown node type

**Solutions**:

1. Check node type format (no `nodes.` prefix)
1. Ensure `nodetool-base` is installed in image
1. Verify node package is imported

### Issue: Container startup timeout

**Symptom**: Job hangs in "starting" status

**Solutions**:

1. Check Docker daemon is running
1. Verify image exists: `docker images | grep nodetool`
1. Check Docker logs: `docker logs <container_id>`

### Issue: GPU not available

**Symptom**: CUDA/GPU errors in container

**Solutions**:

1. Verify nvidia-docker: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
1. Check GPU devices: `nvidia-smi`
1. Verify container GPU access: `docker run --rm --gpus all nodetool nvidia-smi`

## CI Testing

The GitHub Actions workflow automatically tests Docker execution:

### Workflow Steps

1. Checkout code
1. Set up Docker Buildx with layer caching
1. Set up Python 3.11
1. Install Python dependencies
1. **Build Docker image with caching**
1. **Verify image has required features**
1. Run non-Docker tests
1. Run Docker-specific tests

### Local CI Simulation

Run the same tests as CI locally:

```bash
# Build image
docker build -t nodetool -f Dockerfile .

# Verify features
docker run --rm --entrypoint nodetool nodetool run --help | grep jsonl

# Run tests
pytest tests/workflows/test_docker_job_execution.py -v
```

## Rebuilding After Changes

When developing locally and making changes to CLI or core functionality:

```bash
# 1. Make your changes
vim src/nodetool/cli.py

# 2. Rebuild Docker image
docker build -t nodetool -f Dockerfile .

# 3. Verify the build
docker run --rm --entrypoint nodetool nodetool run --help | grep jsonl

# 4. Run tests
pytest tests/workflows/test_docker_job_execution.py -v
```

**Note**: In CI, this happens automatically on every push/PR.

## Related Documentation

- [Docker Execution Guide](docker-execution.md) - Main documentation
- [Docker Resource Management](docker-resource-management.md) - Multi-user deployment
- [CLI Reference](cli.md) - Command-line interface

## Test Files

- `tests/workflows/test_docker_job_execution.py` - Automated tests
- `src/nodetool/workflows/docker_job_execution.py` - Smoke test (run with `--help`)
- `test_workflow.json` - Example workflow
