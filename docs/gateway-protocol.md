# OpenClaw Gateway Protocol

This document describes the WebSocket-based gateway protocol implementation for NodeTool, which allows NodeTool instances to act as nodes in a distributed gateway architecture.

## Overview

The Gateway protocol enables:
- **Distributed workflow execution**: NodeTool instances can connect to a gateway server and execute workflows remotely
- **MCP-like command interface**: Gateway can send commands to query and manage workflows, jobs, assets, etc.
- **Real-time updates**: Streaming workflow execution updates back to the gateway
- **Automatic reconnection**: Resilient connection handling with exponential backoff

## Architecture

```
┌─────────────────┐
│  Gateway Server │
│   (OpenClaw)    │
└────────┬────────┘
         │
         │ WebSocket
         │
    ┌────┴─────┐
    │          │
┌───▼───┐  ┌──▼────┐
│Node 1 │  │Node 2 │
│NodeT. │  │NodeT. │
└───────┘  └───────┘
```

## Message Types

All messages are JSON-encoded and follow this structure:

```json
{
  "type": "message_type",
  "timestamp": "2024-01-30T12:00:00Z",
  ...additional fields...
}
```

### Node → Gateway Messages

#### Node Registration

Sent on initial connection to register node capabilities.

```json
{
  "type": "node_registration",
  "node_id": "node-abc123",
  "capabilities": {
    "workflow_execution": true,
    "commands": ["list_workflows", "run_workflow", "list_jobs", ...]
  },
  "metadata": {
    "user_id": "1"
  },
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Heartbeat

Periodic message to maintain connection and signal node health.

```json
{
  "type": "heartbeat",
  "node_id": "node-abc123",
  "status": "active",
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Workflow Response

Response to a workflow execution request.

```json
{
  "type": "workflow_response",
  "request_id": "req-123",
  "status": "completed",
  "result": {
    "job_id": "job-456",
    "outputs": {...}
  },
  "error": null,
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Workflow Update

Streaming update during workflow execution.

```json
{
  "type": "workflow_update",
  "request_id": "req-123",
  "job_id": "job-456",
  "update_type": "node_update",
  "data": {
    "node_id": "node-1",
    "status": "running",
    ...
  },
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Command Response

Response to a command request.

```json
{
  "type": "command_response",
  "request_id": "req-789",
  "status": "success",
  "result": {
    "workflows": [...]
  },
  "error": null,
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Acknowledgment

Quick acknowledgment that a request was received.

```json
{
  "type": "ack",
  "request_id": "req-123",
  "message": "Workflow execution started",
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Error

Error message for issues not tied to a specific request.

```json
{
  "type": "error",
  "error": "Connection lost to database",
  "details": {...},
  "timestamp": "2024-01-30T12:00:00Z"
}
```

### Gateway → Node Messages

#### Workflow Request

Request to execute a workflow.

```json
{
  "type": "workflow_request",
  "request_id": "req-123",
  "workflow_id": "wf-abc",
  "params": {
    "input": "value"
  },
  "user_id": "1",
  "timestamp": "2024-01-30T12:00:00Z"
}
```

Or with inline graph:

```json
{
  "type": "workflow_request",
  "request_id": "req-123",
  "graph": {
    "nodes": [...],
    "edges": [...]
  },
  "params": {...},
  "user_id": "1",
  "timestamp": "2024-01-30T12:00:00Z"
}
```

#### Command Request

Request to execute a command (MCP-like).

```json
{
  "type": "command_request",
  "request_id": "req-789",
  "command": "list_workflows",
  "args": {
    "workflow_type": "user",
    "limit": 10
  },
  "timestamp": "2024-01-30T12:00:00Z"
}
```

## Supported Commands

The gateway client supports the following commands:

### Workflow Commands
- `list_workflows` - List workflows
- `get_workflow` - Get workflow details
- `run_workflow` - Execute a workflow
- `validate_workflow` - Validate workflow structure

### Job Commands
- `list_jobs` - List workflow execution jobs
- `get_job` - Get job details
- `get_job_logs` - Get job execution logs
- `start_background_job` - Start a background job

### Asset Commands
- `list_assets` - List assets
- `get_asset` - Get asset details

### Node Commands
- `list_nodes` - List available node types
- `search_nodes` - Search for node types
- `get_node_info` - Get node metadata

### Model Commands
- `list_models` - List available AI models

### Collection Commands
- `list_collections` - List vector database collections
- `get_collection` - Get collection details

## Connection Flow

1. **Connect**: Client establishes WebSocket connection to gateway
2. **Register**: Client sends `node_registration` message with capabilities
3. **Heartbeat**: Client starts sending periodic heartbeat messages
4. **Ready**: Gateway can now send workflow and command requests
5. **Execute**: Client processes requests and sends responses/updates
6. **Disconnect**: On shutdown, client gracefully closes connection

## Error Handling

The client implements several error handling strategies:

- **Automatic reconnection**: If connection is lost, client attempts to reconnect with exponential backoff
- **Request acknowledgment**: Client sends quick ACK for received requests
- **Error responses**: Failed executions return error information in response
- **Graceful shutdown**: Client properly cleans up resources on disconnect

## Configuration

The gateway client can be configured via:

- **Environment variables**: Gateway URL, auth token, etc.
- **CLI arguments**: When using `nodetool gateway connect` command
- **Python API**: When instantiating `GatewayClient` directly

## Usage Examples

### CLI Usage

```bash
# Connect to a gateway server
nodetool gateway connect --url ws://gateway.example.com:8080

# With authentication
nodetool gateway connect \
  --url ws://gateway.example.com:8080 \
  --auth-token your-token

# With custom node ID
nodetool gateway connect \
  --url ws://gateway.example.com:8080 \
  --node-id production-node-1 \
  --user-id prod-user
```

### Python API Usage

```python
import asyncio
from nodetool.gateway import GatewayClient

async def main():
    client = GatewayClient(
        gateway_url="ws://gateway.example.com:8080",
        node_id="my-node",
        auth_token="your-token",
        user_id="1",
    )
    
    # Run the client (blocks until disconnected)
    await client.run()

asyncio.run(main())
```

### Custom Message Handlers

You can extend the client with custom message handlers:

```python
from nodetool.gateway import GatewayClient

class CustomGatewayClient(GatewayClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom handlers
        self._handlers["custom_message"] = self._handle_custom_message
    
    async def _handle_custom_message(self, data: dict):
        # Handle custom message type
        print(f"Received custom message: {data}")

# Use custom client
client = CustomGatewayClient(gateway_url="ws://gateway.example.com:8080")
await client.run()
```

## Security Considerations

- **Authentication**: Always use authentication tokens in production
- **TLS/SSL**: Use `wss://` (secure WebSocket) for production deployments
- **Token rotation**: Implement token rotation for long-running nodes
- **Network isolation**: Run gateway nodes in isolated network segments
- **Access control**: Use user_id to control workflow execution permissions

## Monitoring and Debugging

The gateway client uses the standard NodeTool logging system:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Gateway client will log:
# - Connection events
# - Message send/receive
# - Error conditions
# - Workflow execution progress
```

## Future Enhancements

Potential future improvements:

- **Binary protocol**: Support for MessagePack encoding
- **Compression**: Message compression for large payloads
- **Load balancing**: Gateway-side load balancing across nodes
- **Node discovery**: Automatic node discovery and registration
- **Health checks**: Enhanced health checking and monitoring
- **Metrics**: Built-in metrics collection and reporting
