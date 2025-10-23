[← Back to Docs Index](index.md)

# Docker Job Execution Testing Guide

This guide covers manual testing, automated testing, and debugging for Docker-based job execution.

## Prerequisites

- Docker installed and running
- `nodetool` Docker image built
- `nodetool-base` package included in image

## Quick Verification

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
