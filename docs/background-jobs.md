# Job Execution System

## Overview

The job execution system provides robust, resilient workflow execution that survives WebSocket disconnections and network interruptions. Jobs run in dedicated execution environments managed by the `JobExecutionManager`, allowing clients to disconnect and reconnect without losing job progress.

## Key Features

- **Persistent Execution**: Jobs continue running even when clients disconnect
- **Job Recovery**: Clients can reconnect to running jobs using job IDs
- **Multiple Execution Strategies**: Support for threaded, subprocess, and Docker-based execution
- **Automatic Cleanup**: Completed jobs are automatically cleaned up after a configurable timeout
- **Job Persistence**: All jobs are saved to the database for monitoring and recovery

## Architecture

### Components

1. **JobExecutionManager** (`workflows/job_execution_manager.py`)

   - Singleton manager for all job executions
   - Handles job lifecycle (start, cancel, cleanup)
   - Maintains in-memory registry of running jobs
   - Automatic cleanup of completed jobs

2. **JobExecution** Abstract Base Class

   - Defines the interface for job execution strategies
   - Common behavior for status tracking and state finalization
   - Subclasses implement specific execution methods:
     - `ThreadedJobExecution`: Runs in a dedicated thread with event loop
     - `SubprocessJobExecution`: Runs in a separate process (planned)
     - `DockerJobExecution`: Runs in Docker container (planned)

3. **WebSocketRunner** Updates

   - Support for job recovery via `RECONNECT_JOB` command
   - Jobs run independently of WebSocket connection
   - Streams messages from background jobs to connected clients

4. **Job API** (`api/job.py`)

   - List jobs (completed and running)
   - Get job details by ID
   - List currently running background jobs
   - Cancel jobs

5. **Job Model** Updates
   - Added `params` field to track job parameters
   - Stores job metadata for recovery

## Usage

### Backend

#### Starting a Job

Jobs are automatically started via the WebSocket `RUN_JOB` command and managed by `JobExecutionManager`:

```python
from nodetool.workflows.job_execution_manager import JobExecutionManager
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.processing_context import ProcessingContext

# Create job request
request = RunJobRequest(
    workflow_id="workflow_123",
    user_id="user_123",
    auth_token="token",
    params={"param1": "value1"}
)

# Create processing context
context = ProcessingContext(
    user_id=request.user_id,
    auth_token=request.auth_token,
    workflow_id=request.workflow_id,
)

# Start job execution
job_manager = JobExecutionManager.get_instance()
job = await job_manager.start_job(request, context)

print(f"Job started: {job.job_id}")
```

#### Reconnecting to a Job

Use the WebSocket `RECONNECT_JOB` command:

```python
# Send reconnect command via WebSocket
{
    "command": "reconnect_job",
    "data": {
        "job_id": "abc123def456"
    }
}
```

The WebSocket runner will:

1. Look up the background job by ID
2. Attach to the job's context and runner
3. Stream any remaining messages to the client

#### Listing Jobs

Via the API:

```python
# List all jobs for a user
GET /api/jobs/?user_id=user_123&limit=50

# Get specific job
GET /api/jobs/job_123

# List running background jobs
GET /api/jobs/running/all

# Cancel a job
POST /api/jobs/job_123/cancel
```

### Frontend

#### Running a Workflow

```typescript
import { useWebsocketRunner } from "./stores/WorkflowRunner";

const runner = useWebsocketRunner((state) => state);

// Run workflow
await runner.run(params, workflow, nodes, edges);

// Job ID will be automatically captured
console.log("Job ID:", runner.job_id);
```

#### Reconnecting to a Job

```typescript
import { useWebsocketRunner } from "./stores/WorkflowRunner";

const runner = useWebsocketRunner((state) => state);

// Reconnect to existing job
await runner.reconnect("abc123def456");

// Messages will continue streaming from where the job is
```

#### Monitoring Job Status

```typescript
const state = useWebsocketRunner((state) => state.state);
const jobId = useWebsocketRunner((state) => state.job_id);

// States: "idle" | "connecting" | "connected" | "running" | "error" | "cancelled"
console.log(`Job ${jobId} is ${state}`);
```

## WebSocket Commands

### RUN_JOB

Start a new workflow job:

```json
{
  "command": "run_job",
  "data": {
    "workflow_id": "workflow_123",
    "user_id": "user_123",
    "auth_token": "token",
    "job_type": "workflow",
    "params": {},
    "graph": { ... }
  }
}
```

Response:

```json
{
  "message": "Job started",
  "job_id": "abc123def456"
}
```

### RECONNECT_JOB

Reconnect to an existing job:

```json
{
  "command": "reconnect_job",
  "data": {
    "job_id": "abc123def456"
  }
}
```

Response:

```json
{
  "message": "Reconnecting to job abc123def456",
  "job_id": "abc123def456"
}
```

### CANCEL_JOB

Cancel a running job:

```json
{
  "command": "cancel_job",
  "data": {
    "job_id": "abc123def456"
  }
}
```

## API Endpoints

### GET `/api/jobs/`

List jobs for the current user.

**Query Parameters:**

- `workflow_id` (optional): Filter by workflow
- `limit` (default: 100): Max results
- `start_key` (optional): Pagination key

**Response:**

```json
[
  {
    "id": "job_123",
    "user_id": "user_123",
    "job_type": "workflow",
    "status": "completed",
    "workflow_id": "workflow_123",
    "started_at": "2025-09-30T10:00:00Z",
    "finished_at": "2025-09-30T10:05:00Z",
    "error": null,
    "cost": 0.15
  }
]
```

### GET `/api/jobs/{job_id}`

Get specific job details.

**Response:**

```json
{
  "id": "job_123",
  "user_id": "user_123",
  "job_type": "workflow",
  "status": "running",
  "workflow_id": "workflow_123",
  "started_at": "2025-09-30T10:00:00Z",
  "finished_at": null,
  "error": null,
  "cost": null
}
```

### GET `/api/jobs/running/all`

List all currently running background jobs for the current user.

**Response:**

```json
[
  {
    "job_id": "abc123",
    "status": "running",
    "workflow_id": "workflow_123",
    "created_at": "2025-09-30T10:00:00Z",
    "is_running": true,
    "is_completed": false
  }
]
```

### POST `/api/jobs/{job_id}/cancel`

Cancel a running job.

**Response:**

```json
{
  "message": "Job cancelled successfully",
  "job_id": "abc123"
}
```

## Job Lifecycle

1. **Starting**: Job is created and scheduled
2. **Running**: Job is actively executing
3. **Completed**: Job finished successfully
4. **Failed**: Job encountered an error
5. **Cancelled**: Job was manually cancelled

Jobs persist in the database with their status, allowing recovery and monitoring.

## Configuration

### Cleanup Settings

Configure automatic cleanup in `JobExecutionManager`:

```python
# Start cleanup task with custom settings
await job_manager.start_cleanup_task(
    interval_seconds=300  # Run cleanup every 5 minutes
)

# Cleanup jobs older than 1 hour
await job_manager.cleanup_completed_jobs(
    max_age_seconds=3600
)
```

### Server Integration

The job execution manager is automatically initialized on server startup and shutdown:

```python
# In server.py lifespan
async def lifespan(app: FastAPI):
    # Startup
    job_manager = JobExecutionManager.get_instance()
    await job_manager.start_cleanup_task()

    yield

    # Shutdown
    await job_manager.shutdown()
```

## Error Handling

### Job Failures

Failed jobs are marked with status "failed" and error message:

```python
try:
    await runner.run(request, context)
except Exception as e:
    # Error is captured and saved to job record
    await Job.update(
        job_id,
        status="failed",
        error=str(e),
        finished_at=datetime.now()
    )
```

### Connection Loss Recovery

If a client loses connection:

1. Job continues running in the background
2. Client can reconnect using the job ID
3. Any queued messages are streamed to the reconnected client
4. Job status is maintained throughout

### Timeout Handling

Configure timeouts in the workflow runner or processing context as needed.

## Best Practices

1. **Save Job IDs**: Always save job IDs on the client for reconnection
2. **Monitor Status**: Poll the jobs API or use WebSocket updates to monitor progress
3. **Handle Reconnection**: Implement automatic reconnection logic for network failures
4. **Clean Up**: Let the automatic cleanup handle old jobs, or manually trigger as needed
5. **Error Recovery**: Implement proper error handling and retry logic

## Troubleshooting

### Job Not Found on Reconnect

- Check that the job ID is correct
- Verify the job hasn't been cleaned up (check age)
- Ensure the user has permission to access the job

### Jobs Not Cleaning Up

- Check the cleanup task is running
- Verify cleanup interval and max age settings
- Check server logs for cleanup errors

### WebSocket Disconnections

- Implement automatic reconnection with exponential backoff
- Save job ID immediately when received
- Use the reconnect command to resume streaming

## Example: Full Workflow with Recovery

```typescript
// Start workflow and save job ID
const runner = useWebsocketRunner((state) => state);

try {
  await runner.run(params, workflow, nodes, edges);
  const jobId = runner.job_id;

  // Save to localStorage for recovery
  if (jobId) {
    localStorage.setItem("lastJobId", jobId);
  }
} catch (error) {
  console.error("Job failed to start:", error);
}

// On reconnection or page reload
const savedJobId = localStorage.getItem("lastJobId");
if (savedJobId) {
  try {
    await runner.reconnect(savedJobId);
    console.log("Successfully reconnected to job");
  } catch (error) {
    console.error("Failed to reconnect:", error);
    localStorage.removeItem("lastJobId");
  }
}
```
