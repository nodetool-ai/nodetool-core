# Docker Job Execution

Docker job execution provides isolated, containerized workflow execution for nodetool. This is ideal for production deployments where security, resource management, and multi-tenancy are important.

## Overview

Docker job execution runs workflows in isolated Docker containers, providing:

- **Process Isolation**: Each workflow runs in its own container
- **Resource Control**: CPU, memory, and GPU limits per job
- **Security**: Isolated filesystem and network namespaces
- **Reproducibility**: Consistent execution environment
- **Multi-tenancy**: Safe execution of untrusted user workflows

## Architecture

```
User Request
    ↓
JobExecutionManager
    ↓
DockerJobExecution.create_and_start()
    ↓
Worker Thread:
  - Creates Docker container
  - Attaches hijacked socket
  - Sends RunJobRequest JSON to stdin
  - Starts container
  - Streams stdout/stderr via DockerHijackMultiplexDemuxer
  - Parses JSONL messages
  - Forwards to AsyncIO queue
    ↓
Main Thread (Async):
  - Monitors queue
  - Forwards messages to ProcessingContext
  - Updates job status
  - Updates database
```

## Configuration

### Environment Variables

Configure default resource limits for all Docker jobs:

```bash
# Memory limit (default: 2g)
export DOCKER_MEM_LIMIT="4g"

# CPU limit in cores (default: 2.0)
export DOCKER_CPU_LIMIT="2.0"

# GPU device IDs (comma-separated, empty = no GPUs)
export DOCKER_GPU_DEVICES="0,1"

# GPU memory limit per device
export DOCKER_GPU_MEMORY_LIMIT="8g"
```

## Docker Image Requirements

### Building the Image

The nodetool Docker image must include:

1. **nodetool-core** - Core workflow engine and CLI
2. **Node packages** - nodetool-base and other node libraries

Build from the repository root:

```bash
docker build -t nodetool -f Dockerfile .
```

### Verify the Image

Check that the image has the required CLI features:

```bash
# Verify nodetool command exists
docker run --rm --entrypoint which nodetool nodetool

# Verify --jsonl flag is available
docker run --rm --entrypoint nodetool nodetool run --help | grep jsonl

# Test a simple workflow
cat test_workflow.json | docker run --rm -i nodetool run --stdin --jsonl
```

## Environment Variable Passing

Docker containers automatically receive all nodetool settings and secrets:

### Automatic Injection

All registered settings are passed to containers:

- **API Keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, etc.
- **Service Credentials**: `AIME_USER`, `AIME_API_KEY`, `ELEVENLABS_API_KEY`, `FAL_API_KEY`, etc.
- **Configuration**: `FONT_PATH`, `COMFY_FOLDER`, `CHROMA_PATH`, `OLLAMA_API_URL`
- **Integrations**: `SERPAPI_API_KEY`, `BROWSER_URL`, `DATA_FOR_SEO_*`, etc.

## Continuous Integration

The GitHub Actions workflow ensures Docker images stay in sync with code changes:

### CI Workflow Features

- **Automatic Builds**: Builds Docker image on every push/PR
- **Feature Verification**: Verifies the `--jsonl` flag and other CLI features
- **Test Execution**: Runs all tests including Docker-specific tests
- **Docker Layer Caching**: Speeds up builds (~60% faster after first build)

## Message Flow

Containers communicate via JSONL (JSON Lines) over stdin/stdout:

### Message Types

1. **JobUpdate**: Overall job status changes
2. **NodeUpdate**: Individual node status changes
3. **NodeProgress**: Node execution progress

### Message Deserialization

The system automatically converts JSONL messages into Python objects:

```python
# Container outputs JSONL
{"type": "job_update", "status": "running", "job_id": "..."}
{"type": "node_update", "node_id": "input_text", "status": "completed"}

# Converted to Python objects
JobUpdate(status="running", job_id="...")
NodeUpdate(node_id="input_text", status="completed")
```

## Status Flow

1. **Job created** → status="starting"
2. **Container started** → status="running"
3. **Workflow completes** → status="completed"
4. **On error** → status="error"
5. **On cancel** → status="cancelled"

## Container Lifecycle

1. Create container with `auto_remove=True`
2. Attach hijacked socket (before starting!)
3. Start container
4. Feed RunJobRequest JSON to stdin
5. Stream stdout via DockerHijackMultiplexDemuxer
6. Wait for container to finish
7. Container auto-removes on completion

## Performance Characteristics

- **Container startup**: ~1-3 seconds
- **Workflow execution**: Depends on workflow complexity
- **Cleanup**: Instant (due to `auto_remove=True`)
- **Memory overhead**: Minimal (hijacked socket streaming)
- **CPU overhead**: Docker daemon handles isolation

## Security Considerations

### Current Implementation

✅ **Process isolation** via Docker  
✅ **Auto-remove containers** (no persistence)  
✅ **Environment variable isolation**  
✅ **Resource limits** (CPU, memory, GPU)

### Future Enhancements

- Network isolation options
- Volume mount security policies
- User namespace remapping
- Signed image verification

## Related Documentation

- [Docker Testing Guide](docker-testing.md) - Testing commands and examples
- [Docker Resource Management](docker-resource-management.md) - Multi-user deployment
- [Deployment Guide](deployment.md) - Production deployment
- [CLI Reference](cli.md) - Command-line interface

## References

- Implementation: `src/nodetool/workflows/docker_job_execution.py`
- Docker SDK: `DockerHijackMultiplexDemuxer` for multiplexed I/O
- CLI: `nodetool run --stdin --jsonl`
- Tests: `tests/workflows/test_docker_job_execution.py`
