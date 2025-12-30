# Unified WebSocket API

This document describes the unified WebSocket API for NodeTool, which combines workflow execution and chat communications into a single endpoint.

## Overview

The unified WebSocket endpoint (`/ws`) provides a single connection point for:
- **Workflow execution**: Running, monitoring, and managing workflow jobs
- **Chat communications**: Real-time AI chat with message persistence
- **Bidirectional updates**: Receive workflow updates during chat or chat updates during workflows

### Benefits of the Unified Endpoint
- Single connection for all real-time features
- Reduced connection overhead
- Ability to mix workflow and chat operations
- Simpler client implementation

## Endpoints

| Endpoint | Description | Status |
|----------|-------------|--------|
| `/ws` | Unified WebSocket for workflows and chat | **Recommended** |
| `/ws/predict` | Legacy workflow-only endpoint | Deprecated |
| `/ws/chat` | Legacy chat-only endpoint | Deprecated |
| `/ws/updates` | System stats broadcasts | Active |
| `/ws/terminal` | Terminal access (dev only) | Active |

## Protocol

The WebSocket supports both binary (MessagePack) and text (JSON) protocols:

- **Binary (default)**: More efficient, uses MessagePack encoding
- **Text**: JSON encoding, easier to debug

The server auto-detects the format from incoming messages and responds in the same format.

### Switching Protocol Mode

```json
{
  "command": "set_mode",
  "data": {
    "mode": "text"  // or "binary"
  }
}
```

## Authentication

Include the authentication token in the WebSocket handshake:

```javascript
const ws = new WebSocket(`wss://api.example.com/ws?token=${authToken}`);
// or via Authorization header if supported by your client
```

In development mode (`REMOTE_AUTH=0`), authentication is optional and defaults to user ID "1".

## Message Structure

**All messages must be wrapped in a command structure.** This ensures proper routing and requires valid references (job_id or thread_id) for operations.

```json
{
  "command": "<command_type>",
  "data": {
    // command-specific data
  }
}
```

### Available Commands

| Command | Description | Required Fields |
|---------|-------------|-----------------|
| `run_job` | Start workflow execution | `workflow_id` |
| `reconnect_job` | Reconnect to running job | `job_id` |
| `cancel_job` | Cancel running job | `job_id` |
| `get_status` | Get job status | `job_id` (optional) |
| `stream_input` | Stream input to job | `job_id`, `input` |
| `end_input_stream` | End streaming input | `job_id`, `input` |
| `chat_message` | Send chat message | `thread_id` |
| `stop` | Stop current operation | `job_id` or `thread_id` |
| `set_mode` | Change protocol mode | `mode` |
| `clear_models` | Clear ML models | none |

### Control Messages (No Command Wrapper)

These special messages are handled without the command wrapper for backward compatibility:

| Type | Description |
|------|-------------|
| `ping` | Keep-alive ping (responds with `pong`) |
| `client_tools_manifest` | Register client-side tools |
| `tool_result` | Return tool execution result |

---

## Workflow Commands

### Run Job

Start a new workflow execution.

**Request:**
```json
{
  "command": "run_job",
  "data": {
    "workflow_id": "uuid-of-workflow",
    "params": {
      "input_name": "value"
    },
    "job_type": "workflow",
    "explicit_types": false,
    "graph": null  // Optional: provide graph for unsaved workflows
  }
}
```

**Response:**
```json
{
  "message": "Job started",
  "workflow_id": "uuid-of-workflow"
}
```

**Streaming Updates:**
After starting, you'll receive streaming updates:
```json
{"type": "job_update", "status": "running", "job_id": "...", "workflow_id": "..."}
{"type": "node_update", "node_id": "...", "status": "running", ...}
{"type": "node_progress", "node_id": "...", "progress": 50, "total": 100}
{"type": "node_update", "node_id": "...", "status": "completed", "result": {...}}
{"type": "job_update", "status": "completed", "job_id": "...", "workflow_id": "..."}
```

### Reconnect to Job

Reconnect to an existing running job.

**Request:**
```json
{
  "command": "reconnect_job",
  "data": {
    "job_id": "uuid-of-job",
    "workflow_id": "uuid-of-workflow"  // optional
  }
}
```

**Response:**
```json
{
  "message": "Reconnecting to job ...",
  "job_id": "...",
  "workflow_id": "..."
}
```

### Cancel Job

Cancel a running workflow job.

**Request:**
```json
{
  "command": "cancel_job",
  "data": {
    "job_id": "uuid-of-job"
  }
}
```

**Response:**
```json
{
  "message": "Job cancellation requested",
  "job_id": "...",
  "workflow_id": "..."
}
```

### Get Status

Get status of active jobs.

**Request (all jobs):**
```json
{
  "command": "get_status",
  "data": {}
}
```

**Response:**
```json
{
  "active_jobs": [
    {
      "job_id": "...",
      "workflow_id": "...",
      "status": "running"
    }
  ]
}
```

**Request (specific job):**
```json
{
  "command": "get_status",
  "data": {
    "job_id": "uuid-of-job"
  }
}
```

### Stream Input

Push streaming input to a running job.

**Request:**
```json
{
  "command": "stream_input",
  "data": {
    "job_id": "uuid-of-job",
    "input": "input_name",
    "value": "chunk of data",
    "handle": "source_handle_name"  // optional
  }
}
```

### End Input Stream

Signal end of streaming input.

**Request:**
```json
{
  "command": "end_input_stream",
  "data": {
    "job_id": "uuid-of-job",
    "input": "input_name",
    "handle": "source_handle_name"  // optional
  }
}
```

### Clear Models

Clear ML models from memory (development only).

**Request:**
```json
{
  "command": "clear_models",
  "data": {}
}
```

---

## Chat Messages

Chat messages **must** be sent using the `chat_message` command with a valid `thread_id`.

### Send Chat Message

**Request:**
```json
{
  "command": "chat_message",
  "data": {
    "role": "user",
    "content": "Hello, can you help me?",
    "thread_id": "uuid-of-thread",  // REQUIRED
    "model": "gpt-4",
    "provider": "openai",
    "tools": ["web_search", "code_execution"],  // optional
    "collections": ["collection-uuid"],  // optional, for RAG
    "agent_mode": false,  // optional, enables agent behavior
    "help_mode": false,   // optional, enables help mode
    "workflow_id": "uuid"  // optional, for workflow-specific chat
  }
}
```

**Response:**
```json
{
  "message": "Chat message processing started",
  "thread_id": "uuid-of-thread"
}
```

**Streaming Response:**
```json
{"type": "chunk", "content": "Hello", "done": false}
{"type": "chunk", "content": "!", "done": false}
{"type": "chunk", "content": " How can", "done": false}
{"type": "chunk", "content": " I help you?", "done": true}
{"type": "message", "role": "assistant", "content": "Hello! How can I help you?", ...}
```

### Chat with Workflow

Send a message that triggers workflow processing:

**Request:**
```json
{
  "command": "chat_message",
  "data": {
    "role": "user",
    "content": "Process this image",
    "thread_id": "uuid-of-thread",
    "workflow_id": "uuid-of-workflow",
    "workflow_target": "workflow"  // Routes to workflow processor
  }
}
```

---

## Control Commands

### Stop Current Operation

Stop the current chat or workflow processing. **Requires** a `job_id` or `thread_id`.

**Request (stop chat):**
```json
{
  "command": "stop",
  "data": {
    "thread_id": "uuid-of-thread"
  }
}
```

**Request (stop job):**
```json
{
  "command": "stop",
  "data": {
    "job_id": "uuid-of-job"
  }
}
```

**Response:**
```json
{
  "type": "generation_stopped",
  "message": "Generation stopped by user",
  "job_id": "...",
  "thread_id": "..."
}
```

### Ping (Keep-Alive)

Client can send pings to keep the connection alive:

**Request:**
```json
{
  "type": "ping"
}
```

**Response:**
```json
{
  "type": "pong",
  "ts": 1703936000.123
}
```

The server also sends periodic heartbeat pings (every 25 seconds).

### Client Tools Manifest

Register client-side tools that can be called by the AI:

**Request:**
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
          "accept": {"type": "string", "description": "File type filter"}
        }
      }
    }
  ]
}
```

### Tool Result

Send result of a client-side tool execution:

**Request:**
```json
{
  "type": "tool_result",
  "tool_call_id": "uuid-of-tool-call",
  "result": {"file_path": "/path/to/file"}
}
```

---

## Update Message Types

### Workflow Updates

| Type | Description | Fields |
|------|-------------|--------|
| `job_update` | Job status change | `status`, `job_id`, `workflow_id`, `error?`, `result?` |
| `node_update` | Node status change | `node_id`, `node_name`, `node_type`, `status`, `result?`, `error?` |
| `node_progress` | Node progress | `node_id`, `progress`, `total`, `chunk?` |
| `edge_update` | Edge status | `edge_id`, `status`, `counter?` |
| `output_update` | Output value | `node_id`, `node_name`, `output_name`, `value`, `output_type` |
| `preview_update` | Preview data | `node_id`, `value` |
| `error` | General error | `error` |

### Chat Updates

| Type | Description | Fields |
|------|-------------|--------|
| `chunk` | Streaming text | `content`, `done`, `content_type`, `node_id?` |
| `message` | Complete message | `role`, `content`, `id`, `thread_id`, `tool_calls?`, ... |
| `tool_call_update` | Tool being called | `name`, `args`, `tool_call_id?`, `node_id?` |
| `task_update` | Agent task status | `task`, `step?`, `event`, `node_id?` |
| `planning_update` | Planning status | `phase`, `status`, `content?`, `node_id?` |

---

## Error Handling

Errors are returned in a consistent format:

**Validation Error:**
```json
{
  "error": "job_id is required"
}
```

**Processing Error:**
```json
{
  "type": "error",
  "message": "Error description"
}
```

**Job Failure:**
```json
{
  "type": "job_update",
  "status": "failed",
  "error": "Error message",
  "job_id": "...",
  "workflow_id": "..."
}
```

---

## Migration Guide

### From `/ws/predict` to `/ws`

The unified endpoint supports all workflow commands identically. No changes needed to your command structure.

**Before:**
```javascript
const ws = new WebSocket('wss://api.example.com/ws/predict');
```

**After:**
```javascript
const ws = new WebSocket('wss://api.example.com/ws');
```

### From `/ws/chat` to `/ws`

Chat messages must now use the `chat_message` command with a `thread_id`.

**Before:**
```javascript
const chatWs = new WebSocket('wss://api.example.com/ws/chat');
chatWs.send(JSON.stringify({ role: 'user', content: 'Hello' }));
```

**After:**
```javascript
const ws = new WebSocket('wss://api.example.com/ws');
ws.send(JSON.stringify({
  command: 'chat_message',
  data: { role: 'user', content: 'Hello', thread_id: 'uuid-of-thread' }
}));
```

### Combined Usage

With the unified endpoint, you can use both features on the same connection:

```javascript
const ws = new WebSocket('wss://api.example.com/ws');

// Start a workflow
ws.send(JSON.stringify({
  command: 'run_job',
  data: { workflow_id: '...', params: {} }
}));

// Send a chat message (while workflow is running)
ws.send(JSON.stringify({
  command: 'chat_message',
  data: {
    role: 'user',
    content: 'What\'s the status of my workflow?',
    thread_id: 'uuid-of-thread'
  }
}));

// Stop a chat operation
ws.send(JSON.stringify({
  command: 'stop',
  data: { thread_id: 'uuid-of-thread' }
}));

// Handle all updates in one place
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  if (msg.type === 'job_update') {
    // Handle workflow update
  } else if (msg.type === 'chunk' || msg.type === 'message') {
    // Handle chat response
  }
};
```

---

## JavaScript Client Example

```javascript
class NodeToolWebSocket {
  constructor(url, authToken) {
    this.url = url;
    this.authToken = authToken;
    this.ws = null;
    this.handlers = new Map();
  }

  connect() {
    this.ws = new WebSocket(`${this.url}?token=${this.authToken}`);
    
    this.ws.onmessage = (event) => {
      let msg;
      if (event.data instanceof ArrayBuffer) {
        // Binary - decode with msgpack
        msg = msgpack.decode(new Uint8Array(event.data));
      } else {
        // Text - parse JSON
        msg = JSON.parse(event.data);
      }
      this.handleMessage(msg);
    };

    this.ws.onclose = () => {
      console.log('WebSocket closed');
    };

    return new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
    });
  }

  handleMessage(msg) {
    const type = msg.type || msg.command;
    const handler = this.handlers.get(type);
    if (handler) {
      handler(msg);
    }
  }

  on(type, handler) {
    this.handlers.set(type, handler);
  }

  // Workflow methods
  runJob(workflowId, params = {}) {
    this.send({
      command: 'run_job',
      data: { workflow_id: workflowId, params, job_type: 'workflow' }
    });
  }

  cancelJob(jobId) {
    this.send({
      command: 'cancel_job',
      data: { job_id: jobId }
    });
  }

  // Chat methods - requires thread_id
  sendChat(content, threadId, options = {}) {
    this.send({
      command: 'chat_message',
      data: {
        role: 'user',
        content,
        thread_id: threadId,
        ...options
      }
    });
  }

  // Stop requires job_id or thread_id
  stopChat(threadId) {
    this.send({
      command: 'stop',
      data: { thread_id: threadId }
    });
  }

  stopJob(jobId) {
    this.send({
      command: 'stop',
      data: { job_id: jobId }
    });
  }

  send(msg) {
    this.ws.send(JSON.stringify(msg));
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const client = new NodeToolWebSocket('wss://api.example.com/ws', 'auth-token');
await client.connect();

// Handle updates
client.on('job_update', (msg) => console.log('Job:', msg.status));
client.on('node_update', (msg) => console.log('Node:', msg.node_id, msg.status));
client.on('chunk', (msg) => process.stdout.write(msg.content));
client.on('message', (msg) => console.log('Assistant:', msg.content));

// Run a workflow
client.runJob('workflow-uuid', { input: 'value' });

// Send a chat message (thread_id required)
client.sendChat('Hello, world!', 'thread-uuid');

// Stop a chat operation
client.stopChat('thread-uuid');
```

---

## Python Client Example

```python
import asyncio
import json
import websockets

class NodeToolClient:
    def __init__(self, url: str, auth_token: str):
        self.url = url
        self.auth_token = auth_token
        self.ws = None
        
    async def connect(self):
        self.ws = await websockets.connect(
            f"{self.url}?token={self.auth_token}"
        )
        
    async def send(self, msg: dict):
        await self.ws.send(json.dumps(msg))
        
    async def receive(self) -> dict:
        data = await self.ws.recv()
        return json.loads(data)
    
    async def run_job(self, workflow_id: str, params: dict = None):
        await self.send({
            "command": "run_job",
            "data": {
                "workflow_id": workflow_id,
                "params": params or {},
                "job_type": "workflow"
            }
        })
        
    async def send_chat(self, content: str, thread_id: str, **kwargs):
        """Send a chat message. thread_id is required."""
        await self.send({
            "command": "chat_message",
            "data": {
                "role": "user",
                "content": content,
                "thread_id": thread_id,
                **kwargs
            }
        })
        
    async def stop_chat(self, thread_id: str):
        """Stop a chat operation. Requires thread_id."""
        await self.send({
            "command": "stop",
            "data": {"thread_id": thread_id}
        })
        
    async def stop_job(self, job_id: str):
        """Stop a job. Requires job_id."""
        await self.send({
            "command": "stop",
            "data": {"job_id": job_id}
        })
        
    async def close(self):
        if self.ws:
            await self.ws.close()

# Usage
async def main():
    client = NodeToolClient("wss://api.example.com/ws", "auth-token")
    await client.connect()
    
    # Run workflow
    await client.run_job("workflow-uuid", {"input": "value"})
    
    # Process updates
    while True:
        msg = await client.receive()
        if msg.get("type") == "job_update":
            print(f"Job status: {msg['status']}")
            if msg["status"] in ("completed", "failed", "cancelled"):
                break
        elif msg.get("type") == "chunk":
            print(msg["content"], end="")
    
    # Send a chat message (thread_id required)
    await client.send_chat("Hello!", "thread-uuid")
    
    # Stop a chat operation
    await client.stop_chat("thread-uuid")
            
    await client.close()

asyncio.run(main())
```

---

## Troubleshooting

### Connection Refused
- Verify the server is running
- Check authentication token
- Ensure correct endpoint URL

### Messages Not Routing Correctly
- Workflow commands must have `command` field
- Chat messages should have `role` or `type: "message"`
- Check for typos in field names

### Jobs Not Starting
- Verify workflow_id exists
- Check user has permission to run the workflow
- Review server logs for errors

### Chat Not Responding
- Verify provider API keys are configured
- Check model name is valid for the provider
- Ensure thread_id is valid or omit for auto-creation
