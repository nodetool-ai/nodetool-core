[← Back to Docs Index](index.md)

# Job Execution System Refactoring

> **Note**  
> This document records the design changes introduced during the job execution refactor. For the up-to-date execution strategies and operational guidance, see `src/nodetool/workflows/threaded_job_execution.py`, `src/nodetool/workflows/subprocess_job_execution.py`, and `src/nodetool/workflows/docker_job_execution.py`, as well as the [Deployment Guide](deployment.md).

## Summary

The `JobExecution` abstraction has been refactored to properly support multiple execution strategies (threaded,
subprocess, Docker) without coupling the base class to the in-process `WorkflowRunner`.

## Changes Made

### 1. Decoupled Base Class from In-Process Runner

**File:** `nodetool-core/src/nodetool/workflows/job_execution_manager.py`

#### Before:

- `JobExecution.__init__` required a `WorkflowRunner` instance
- `status` and `is_running()` were tightly coupled to `runner.status` and `runner.is_running()`
- Subclasses could not implement subprocess/Docker without a dummy runner

#### After:

- `JobExecution` now has an internal `_status: str` field
- `runner` is optional and only used by `ThreadedJobExecution`
- `status` property returns `_status` (mutable by subclasses)
- `is_running()` is now abstract, implemented by each subclass

```python
class JobExecution(ABC):
    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        runner: WorkflowRunner | None = None,  # Optional!
    ):
        self.job_id = job_id
        self.context = context
        self.request = request
        self.job_model = job_model
        self.runner = runner  # Only set for threaded execution
        self.created_at = datetime.now()
        self._status: str = "starting"

    @property
    def status(self) -> str:
        """Get the current status of the job."""
        return self._status

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the job is still running."""
        pass
```

### 2. Updated ThreadedJobExecution

- Implements `is_running()` by delegating to `runner.is_running()` if runner exists
- Updates `_status` field in addition to `runner.status` during lifecycle transitions
- `cancel()` method now updates both `runner.status` and `_status`

```python
class ThreadedJobExecution(JobExecution):
    def is_running(self) -> bool:
        """Check if the job is still running."""
        if self.runner:
            return self.runner.is_running()
        return not self.is_completed()

    def cancel(self) -> bool:
        """Cancel the running job."""
        if not self.is_completed():
            self.future.cancel()
            if self.runner:
                self.runner.status = "cancelled"
            self._status = "cancelled"  # Update internal status
            return True
        return False
```

### 3. Updated Subprocess and Docker Stubs

Both `SubprocessJobExecution` and `DockerJobExecution` now:

- Call `super().__init__` with `runner=None`
- Have abstract method stubs for `is_running()` with implementation notes
- Include TODO comments explaining how to implement status tracking

```python
class SubprocessJobExecution(JobExecution):
    def __init__(
        self,
        job_id: str,
        context: ProcessingContext,
        request: RunJobRequest,
        job_model: Job,
        # process: subprocess.Popen,  # TODO
        # pid: int,  # TODO
    ):
        super().__init__(job_id, context, request, job_model, runner=None)
        # TODO: Store process handle and pid for status checks and recovery

    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        # TODO: Check process.poll() is None
        raise NotImplementedError(...)
```

### 4. Added Execution Strategy Selection

**File:** `nodetool-core/src/nodetool/workflows/run_job_request.py`

Added `ExecutionStrategy` enum and `execution_strategy` field:

```python
class ExecutionStrategy(str, Enum):
    """Execution strategy for workflow jobs."""
    THREADED = "threaded"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"

class RunJobRequest(BaseModel):
    # ...
    execution_strategy: ExecutionStrategy = ExecutionStrategy.THREADED
    # ...
```

### 5. Updated JobExecutionManager

The `start_job()` method now switches on `request.execution_strategy`:

```python
async def start_job(
    self, request: RunJobRequest, context: ProcessingContext
) -> JobExecution:
    """
    Start a new job execution using the requested execution strategy.
    """
    from nodetool.workflows.run_job_request import ExecutionStrategy

    # Switch on execution strategy
    if request.execution_strategy == ExecutionStrategy.THREADED:
        job = await ThreadedJobExecution.create_and_start(request, context)
    elif request.execution_strategy == ExecutionStrategy.SUBPROCESS:
        # TODO: Implement subprocess execution
        raise NotImplementedError(
            f"Execution strategy '{request.execution_strategy}' is not yet implemented"
        )
    elif request.execution_strategy == ExecutionStrategy.DOCKER:
        # TODO: Implement Docker execution
        raise NotImplementedError(
            f"Execution strategy '{request.execution_strategy}' is not yet implemented"
        )
    else:
        raise ValueError(f"Unknown execution strategy: {request.execution_strategy}")

    self._jobs[job.job_id] = job
    log.info(f"Started job {job.job_id} with strategy {request.execution_strategy}")
    return job
```

## Benefits

### For Subprocess Implementation

The abstraction now cleanly supports subprocess execution:

1. **Status Management:** Subprocess can update `_status` based on `process.poll()` without needing a runner
1. **Cancellation:** Can implement `cancel()` by sending SIGTERM and updating `_status`
1. **Recovery:** Can store `pid` in `Job.params` and reattach via `os.kill(pid, 0)` check
1. **Streaming:** Can use stdout/stderr pipes similar to `StreamRunnerBase` pattern

Example implementation outline:

```python
class SubprocessJobExecution(JobExecution):
    def __init__(self, ..., process: subprocess.Popen, pid: int):
        super().__init__(job_id, context, request, job_model, runner=None)
        self.process = process
        self.pid = pid

    def is_running(self) -> bool:
        return self.process.poll() is None

    def cancel(self) -> bool:
        if self.is_running():
            self.process.terminate()
            self._status = "cancelled"
            return True
        return False
```

### For Docker Implementation

The abstraction supports Docker containers similarly:

1. **Status Management:** Poll container status via `docker_client.containers.get(container_id).status`
1. **Cancellation:** Stop container via `container.stop()` and update `_status`
1. **Recovery:** Store `container_id` in `Job.params` for reconnection
1. **Streaming:** Use container logs API or attach similar to existing `StreamRunnerBase._docker_run`

Example implementation outline:

```python
class DockerJobExecution(JobExecution):
    def __init__(self, ..., container_id: str, docker_client):
        super().__init__(job_id, context, request, job_model, runner=None)
        self.container_id = container_id
        self.docker_client = docker_client

    def is_running(self) -> bool:
        container = self.docker_client.containers.get(self.container_id)
        return container.status == "running"

    def cancel(self) -> bool:
        if self.is_running():
            container = self.docker_client.containers.get(self.container_id)
            container.stop()
            self._status = "cancelled"
            return True
        return False
```

## Backward Compatibility

- Default execution strategy is `THREADED`, so existing code continues to work
- `ThreadedJobExecution` still uses `runner` internally, maintaining all existing behavior
- All existing tests pass without modification
- WebSocket and HTTP runners transparently use the new strategy field

## Next Steps

To implement subprocess or Docker execution:

1. **Choose strategy** (subprocess recommended first for simplicity)
1. **Create entrypoint script** that:
   - Accepts `RunJobRequest` as JSON via stdin or file
   - Creates `WorkflowRunner` and `ProcessingContext`
   - Runs workflow via `runner.run(request, context)`
   - Emits messages as JSON lines to stdout (JobUpdate, NodeProgress, etc.)
1. **Implement `create_and_start` factory** similar to `ThreadedJobExecution.create_and_start`:
   - Spawn subprocess with entrypoint
   - Create job in database
   - Start background thread to read stdout and bridge messages to `ProcessingContext`
   - Return `SubprocessJobExecution` instance
1. **Handle cancellation** via SIGTERM
1. **Handle recovery** by storing `pid` in `Job.params` and checking `os.kill(pid, 0)`

### Example Entrypoint Script

```python
#!/usr/bin/env python3
"""
Subprocess entrypoint for workflow execution.
Usage: workflow_subprocess_runner.py --request-file /path/to/request.json
"""
import sys
import json
import asyncio
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest

async def main():
    # Load request from file or stdin
    request = RunJobRequest.model_validate_json(sys.stdin.read())

    # Create runner and context
    runner = WorkflowRunner(job_id=request.workflow_id)
    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
        workflow_id=request.workflow_id,
    )

    # Run workflow and emit messages as JSON lines
    try:
        await runner.run(request, context)
        while context.has_messages():
            msg = await context.pop_message_async()
            print(json.dumps(msg.model_dump()), flush=True)
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"type": "error", "error": str(e)}), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

All existing tests in `tests/workflows/test_background_job_manager.py` pass successfully:

```bash
cd /Users/mg/workspace/nodetool-core
conda run -n nodetool pytest tests/workflows/test_background_job_manager.py -v
# Result: 9 passed in 3.52s
```

The refactoring maintains 100% backward compatibility with the existing test suite.

### New Strategy-Specific Tests (To Be Added)

When implementing subprocess or Docker strategies, add these tests:

- Test subprocess execution with simple workflow
- Test Docker execution with container isolation
- Test strategy selection from request
- Test error handling for unimplemented strategies
- Test recovery/reconnection for subprocess/Docker jobs
- Test cancellation behavior for each strategy

## References

- `nodetool-core/docs/background-jobs.md` - Original job execution documentation
- `nodetool-core/src/nodetool/code_runners/runtime_base.py` - Existing subprocess/Docker pattern for nodes
- `nodetool-core/src/nodetool/code_runners/server_subprocess_runner.py` - Example of subprocess streaming
