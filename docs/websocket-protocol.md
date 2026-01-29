# WebSocket Protocol Reference

This document provides a detailed reference for the WebSocket protocol used by NodeTool, including all message types, schemas, state machines, and message flows.

## Overview

The NodeTool WebSocket protocol (`/ws`) uses MessagePack binary encoding by default for efficiency, with JSON text encoding available for debugging. The protocol supports bidirectional streaming of workflow execution updates and chat messages.

All messages are plain objects (no command wrapper for responses). Client-to-server messages use a `command` wrapper for operations like `run_job`, `chat_message`, etc.

## Message Format

### Binary (MessagePack - Default)

```javascript
const msg = msgpack.decode(new Uint8Array(event.data));
```

### Text (JSON - Debug Mode)

```json
{
  "type": "job_update",
  "status": "running",
  "job_id": "uuid-of-job"
}
```

### Switching Protocol Mode

```json
{
  "command": "set_mode",
  "data": { "mode": "text" }
}
```

---

## Processing Messages (Server → Client)

All processing messages share a common structure with a `type` discriminator field.

### Core Workflow Messages

#### JobUpdate

Represents a change in job status. This is the primary message for tracking workflow execution lifecycle.

```typescript
{
  type: "job_update",
  status: string,              // "pending" | "running" | "completed" | "failed" | "cancelled" | "error"
  job_id?: string,
  workflow_id?: string,
  message?: string,            // Human-readable status message
  result?: dict,               // Final workflow result on completion
  error?: string,              // Error message on failure
  traceback?: string,          // Python traceback for debugging
  run_state?: {                // Execution state info
    status: string,
    suspended_node_id?: string,
    suspension_reason?: string,
    error_message?: string,
    execution_strategy?: string,
    is_resumable: boolean
  }
}
```

**Status Values:**
- `pending` - Job queued but not yet started
- `running` - Job actively executing
- `completed` - Job finished successfully
- `failed` - Job encountered a non-recoverable error
- `cancelled` - Job was cancelled by user request
- `error` - Job encountered an error condition

#### NodeUpdate

Represents a change in node status within a workflow.

```typescript
{
  type: "node_update",
  node_id: string,
  node_name: string,
  node_type: string,
  status: string,              // "pending" | "running" | "completed" | "failed"
  error?: string,              // Error message if failed
  result?: dict,               // Node output result
  properties?: dict,           // Additional node properties
  workflow_id?: string
}
```

#### NodeProgress

Reports progress for long-running node operations.

```typescript
{
  type: "node_progress",
  node_id: string,
  progress: number,            // Current progress value
  total: number,               // Total/denominator value
  chunk?: string,              // Partial output data
  workflow_id?: string
}
```

#### EdgeUpdate

Reports edge processing status between nodes.

```typescript
{
  type: "edge_update",
  workflow_id: string,
  edge_id: string,
  status: string,
  counter?: number             // Data chunk counter for streaming
}
```

### Output and Preview Messages

#### OutputUpdate

Streams output values from workflow nodes to clients in real-time.

```typescript
{
  type: "output_update",
  node_id: string,
  node_name: string,
  output_name: string,
  value: any,                  // Output value (string, bytes, or structured)
  output_type: string,         // "string" | "image" | "audio" | "video" | etc.
  metadata?: dict,             // Additional metadata
  workflow_id?: string
}
```

#### PreviewUpdate

Sends preview data (e.g., images, intermediate results) during execution.

```typescript
{
  type: "preview_update",
  node_id: string,
  value: any
}
```

#### SaveUpdate

Reports saved output data with metadata.

```typescript
{
  type: "save_update",
  node_id: string,
  name: string,
  value: any,
  output_type: string,
  metadata?: dict
}
```

### Binary Data Messages

#### BinaryUpdate

Transmits binary data (images, audio, video) between nodes or to clients.

```typescript
{
  type: "binary_update",
  node_id: string,
  output_name: string,
  binary: bytes                // Raw binary data
}
```

**Encoding Format:** `node_id\0output_name\0binary_data`

### Chat and Streaming Messages

#### Chunk

Streaming text content from AI responses.

```typescript
{
  type: "chunk",
  node_id?: string,
  thread_id?: string,
  workflow_id?: string,
  content_type?: "text" | "audio" | "image" | "video" | "document",
  content?: string,
  content_metadata?: dict,
  done: boolean
}
```

#### Prediction

Complete prediction result from a model.

```typescript
{
  type: "prediction",
  // Full prediction schema - see API types
}
```

### Tool and Agent Messages

#### ToolCallUpdate

Reports when an AI provider executes a tool call.

```typescript
{
  type: "tool_call_update",
  node_id?: string,
  thread_id?: string,
  workflow_id?: string,
  tool_call_id?: string,
  name: string,
  args: dict,
  message?: string,
  step_id?: string,
  agent_execution_id?: string
}
```

#### ToolResultUpdate

Reports tool execution results.

```typescript
{
  type: "tool_result_update",
  node_id: string,
  thread_id?: string,
  workflow_id?: string,
  result: dict
}
```

#### TaskUpdate

Reports agent task status changes.

```typescript
{
  type: "task_update",
  node_id?: string,
  thread_id?: string,
  workflow_id?: string,
  task: Task,
  step?: Step,
  event: TaskUpdateEvent       // "task_created" | "step_started" | "step_completed" | etc.
}
```

#### PlanningUpdate

Reports planning stage information for multi-step agent processes.

```typescript
{
  type: "planning_update",
  node_id?: string,
  thread_id?: string,
  workflow_id?: string,
  phase: string,
  status: string,
  content?: string
}
```

#### StepResult

Reports step result from a task.

```typescript
{
  type: "step_result",
  step: Step,
  result: any,
  error?: string,
  is_task_result: boolean,
  thread_id?: string,
  workflow_id?: string
}
```

### Log and Notification Messages

#### LogUpdate

Reports log messages from node execution.

```typescript
{
  type: "log_update",
  node_id: string,
  node_name: string,
  content: string,
  severity: "info" | "warning" | "error"
}
```

#### Notification

Reports notifications from nodes.

```typescript
{
  type: "notification",
  node_id: string,
  content: string,
  severity: "info" | "warning" | "error"
}
```

### Error Messages

#### Error

General error message when no node context is available.

```typescript
{
  type: "error",
  message: string,
  thread_id?: string,
  workflow_id?: string
}
```

---

## Client Commands (Client → Server)

All client commands use a wrapped format:

```typescript
{
  command: string,
  data: dict
}
```

### Workflow Commands

#### run_job

Start a new workflow execution.

```json
{
  "command": "run_job",
  "data": {
    "workflow_id": "uuid-of-workflow",
    "params": { "input_name": "value" },
    "job_type": "workflow",
    "explicit_types": false,
    "graph": null
  }
}
```

#### reconnect_job

Reconnect to a running job to receive updates.

```json
{
  "command": "reconnect_job",
  "data": {
    "job_id": "uuid-of-job",
    "workflow_id": "uuid-of-workflow"
  }
}
```

#### cancel_job

Cancel a running job.

```json
{
  "command": "cancel_job",
  "data": {
    "job_id": "uuid-of-job"
  }
}
```

#### get_status

Get status of active jobs.

```json
{
  "command": "get_status",
  "data": { "job_id": "uuid-of-job" }
}
```

#### stream_input

Stream input data to a running job.

```json
{
  "command": "stream_input",
  "data": {
    "job_id": "uuid-of-job",
    "input": "input_name",
    "value": "chunk of data",
    "handle": "source_handle_name"
  }
}
```

#### end_input_stream

Signal end of streaming input.

```json
{
  "command": "end_input_stream",
  "data": {
    "job_id": "uuid-of-job",
    "input": "input_name",
    "handle": "source_handle_name"
  }
}
```

### Chat Commands

#### chat_message

Send a chat message. Requires `thread_id`.

```json
{
  "command": "chat_message",
  "data": {
    "role": "user",
    "content": "Hello, can you help me?",
    "thread_id": "uuid-of-thread",
    "model": "gpt-4",
    "provider": "openai",
    "tools": ["web_search", "code_execution"],
    "collections": ["collection-uuid"],
    "agent_mode": false,
    "help_mode": false,
    "workflow_id": "uuid"
  }
}
```

### Control Commands

#### stop

Stop current chat or workflow operation.

```json
{
  "command": "stop",
  "data": {
    "job_id": "uuid-of-job"
  }
}
```
or
```json
{
  "command": "stop",
  "data": {
    "thread_id": "uuid-of-thread"
  }
}
```

#### set_mode

Change protocol encoding mode.

```json
{
  "command": "set_mode",
  "data": {
    "mode": "text"
  }
}
```

#### clear_models

Clear ML models from memory (dev only).

```json
{
  "command": "clear_models",
  "data": {}
}
```

### Control Messages (No Wrapper)

These messages are sent without the `command` wrapper for protocol-level communication.

#### ping

Keep-alive ping.

```json
{
  "type": "ping"
}
```

Response:
```json
{
  "type": "pong",
  "ts": 1703936000.123
}
```

#### client_tools_manifest

Register client-side tools.

```json
{
  "type": "client_tools_manifest",
  "tools": [
    {
      "name": "select_file",
      "description": "Prompt user to select a file",
      "parameters": {
        "type": "object",
        "properties": {
          "accept": { "type": "string", "description": "File type filter" }
        }
      }
    }
  ]
}
```

#### tool_result

Send tool execution result.

```json
{
  "type": "tool_result",
  "tool_call_id": "uuid-of-tool-call",
  "result": { "file_path": "/path/to/file" },
  "ok": true
}
```

---

## Job State Machine

```
                        ┌─────────────┐
                        │   pending   │
                        └──────┬──────┘
                               │
                       start_job()
                               │
                               ▼
                        ┌─────────────┐
              ┌────────►│   running   │◄────────┐
              │         └──────┬──────┘         │
              │                │                │
        cancel_job()    node_progress    cancel_job()
              │                │                │
              │                ▼                │
              │         ┌─────────────┐         │
              │         │   failed    │         │
              │         └─────────────┘         │
              │                │                │
              │                ▼                │
              │         ┌─────────────┐         │
              │         │   error     │         │
              │         └─────────────┘         │
              │                │                │
              │                ▼                │
              │         ┌─────────────┐         │
              └─────────│ cancelled   │─────────┘
                        └─────────────┘
                               │
                    all_nodes_complete()
                               │
                               ▼
                        ┌─────────────┐
                        │  completed  │
                        └─────────────┘
```

### State Transitions

| From | To | Trigger |
|------|-----|---------|
| pending | running | Internal scheduler starts job |
| running | running | Node progress updates |
| running | completed | All nodes completed successfully |
| running | failed | Non-recoverable node error |
| running | error | Recoverable error condition |
| running | cancelled | User requested cancellation |
| failed | completed | N/A (terminal state) |
| error | completed | N/A (terminal state) |
| cancelled | completed | N/A (terminal state) |

---

## Message Flow Examples

### Successful Workflow Execution

```
Client → Server:
{
  "command": "run_job",
  "data": { "workflow_id": "wf-123", "params": { "input": "test" } }
}

Server → Client:
{ "type": "job_update", "status": "running", "job_id": "job-123", "workflow_id": "wf-123" }

Server → Client:
{ "type": "node_update", "node_id": "node-1", "node_name": "Input", "status": "running" }
{ "type": "node_update", "node_id": "node-1", "node_name": "Input", "status": "completed" }

Server → Client:
{ "type": "node_progress", "node_id": "node-2", "progress": 50, "total": 100 }

Server → Client:
{ "type": "node_update", "node_id": "node-2", "node_name": "Process", "status": "completed", "result": {...} }

Server → Client:
{ "type": "output_update", "node_id": "node-3", "output_name": "result", "value": "...", "output_type": "string" }

Server → Client:
{ "type": "job_update", "status": "completed", "job_id": "job-123", "result": {...} }
```

### Workflow with Error

```
Client → Server:
{
  "command": "run_job",
  "data": { "workflow_id": "wf-456" }
}

Server → Client:
{ "type": "job_update", "status": "running", "job_id": "job-456", "workflow_id": "wf-456" }

Server → Client:
{ "type": "node_update", "node_id": "node-1", "node_name": "HF Model", "status": "running" }

Server → Client:
{ "type": "node_update", "node_id": "node-1", "node_name": "HF Model", "status": "failed", "error": "HuggingFace token not configured" }

Server → Client:
{ "type": "job_update", "status": "failed", "job_id": "job-456", "error": "HuggingFace token not configured", "traceback": "..." }
```

### Job Cancellation

```
Client → Server:
{ "command": "cancel_job", "data": { "job_id": "job-789" } }

Server → Client:
{ "type": "job_update", "status": "cancelled", "job_id": "job-789", "message": "Job cancelled by user" }
```

### Chat Message with Streaming

```
Client → Server:
{
  "command": "chat_message",
  "data": { "role": "user", "content": "Explain quantum computing", "thread_id": "thread-123" }
}

Server → Client:
{ "type": "chunk", "content": "Quantum", "done": false }
{ "type": "chunk", "content": " computing", "done": false }
{ "type": "chunk", "content": " is...", "done": true }

Server → Client:
{ "type": "message", "role": "assistant", "content": "Quantum computing is...", "id": "msg-456", "thread_id": "thread-123" }
```

### Agent Task with Tool Calls

```
Server → Client:
{ "type": "task_update", "task": {...}, "event": "task_created" }
{ "type": "task_update", "step": {...}, "event": "step_started" }

Server → Client:
{ "type": "tool_call_update", "name": "search_web", "args": {"query": "..."}, "tool_call_id": "tc-123" }

Client → Server:
{ "type": "tool_result", "tool_call_id": "tc-123", "result": {...}, "ok": true }

Server → Client:
{ "type": "task_update", "step": {...}, "event": "step_completed" }
{ "type": "task_update", "task": {...}, "event": "task_completed" }
```

### Error Without Node Context

```
Server → Client:
{ "type": "error", "message": "Connection to provider failed: timeout" }
```

---

## Complete Message Type Reference

| Type | Direction | Description |
|------|-----------|-------------|
| `job_update` | Server→Client | Job status changes |
| `node_update` | Server→Client | Node status changes |
| `node_progress` | Server→Client | Node progress updates |
| `edge_update` | Server→Client | Edge processing status |
| `output_update` | Server→Client | Real-time output streaming |
| `preview_update` | Server→Client | Preview data |
| `save_update` | Server→Client | Saved output |
| `binary_update` | Server→Client | Binary data |
| `chunk` | Server→Client | Streaming text |
| `prediction` | Server→Client | Model prediction |
| `tool_call_update` | Server→Client | Tool execution |
| `tool_result_update` | Server→Client | Tool result |
| `task_update` | Server→Client | Agent task status |
| `planning_update` | Server→Client | Planning stage |
| `step_result` | Server→Client | Step result |
| `log_update` | Server→Client | Log message |
| `notification` | Server→Client | Notification |
| `error` | Server→Client | Error message |
| `run_job` | Client→Server | Start workflow |
| `reconnect_job` | Client→Server | Reconnect to job |
| `cancel_job` | Client→Server | Cancel job |
| `get_status` | Client→Server | Get job status |
| `stream_input` | Client→Server | Stream input |
| `end_input_stream` | Client→Server | End stream |
| `chat_message` | Client→Server | Send chat |
| `stop` | Client→Server | Stop operation |
| `set_mode` | Client→Server | Set protocol mode |
| `clear_models` | Client→Server | Clear models |
| `ping` | Both | Keep-alive |
| `pong` | Server→Client | Ping response |
| `client_tools_manifest` | Client→Server | Register tools |
| `tool_result` | Client→Server | Tool result |

---

## Error Handling

### Validation Errors

```json
{
  "error": "job_id is required"
}
```

### Processing Errors

```json
{
  "type": "error",
  "message": "Error description"
}
```

### Job Failure

```json
{
  "type": "job_update",
  "status": "failed",
  "error": "Error message",
  "job_id": "...",
  "workflow_id": "...",
  "traceback": "..."
}
```

### Node Failure

```json
{
  "type": "node_update",
  "node_id": "...",
  "node_name": "Process Node",
  "status": "failed",
  "error": "Node-specific error"
}
```

---

## File Locations

- **Server Types:** `/Users/mg/workspace/nodetool-core/src/nodetool/workflows/types.py`
- **Job Types:** `/Users/mg/workspace/nodetool-core/src/nodetool/types/job.py`
- **Protocol Handler:** `/Users/mg/workspace/nodetool-core/src/nodetool/api/websocket.py`
- **Frontend Protocol:** `/Users/mg/workspace/nodetool/web/src/core/chat/chatProtocol.ts`
- **API Types:** `/Users/mg/workspace/nodetool/web/src/stores/ApiTypes.ts`
- **High-Level Docs:** `/Users/mg/workspace/nodetool-core/docs/websocket-api.md`
