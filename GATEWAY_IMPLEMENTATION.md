# OpenClaw Gateway Protocol Implementation

This document summarizes the implementation of the WebSocket client for the OpenClaw Gateway protocol.

## Overview

The implementation allows NodeTool to act as a node in a distributed gateway architecture, receiving and executing workflows from a central gateway server. The design follows NodeTool's existing patterns (MCP tools, WebSocket protocol) and provides a robust, production-ready client.

## Architecture

```
┌─────────────────┐
│  Gateway Server │
│   (OpenClaw)    │
└────────┬────────┘
         │
         │ WebSocket (JSON messages)
         │
    ┌────┴─────┐
    │          │
┌───▼───┐  ┌──▼────┐
│Node 1 │  │Node 2 │
│NodeT. │  │NodeT. │
└───────┘  └───────┘
```

## Components

### 1. Protocol Definitions (`src/nodetool/gateway/protocol.py`)

Pydantic models for all message types:

**Node → Gateway:**
- `NodeRegistration` - Initial node registration with capabilities
- `NodeHeartbeat` - Periodic heartbeat to maintain connection
- `WorkflowResponse` - Response to workflow execution request
- `WorkflowUpdate` - Streaming updates during workflow execution
- `CommandResponse` - Response to command request
- `AckMessage` - Quick acknowledgment of received requests
- `ErrorMessage` - Error notifications

**Gateway → Node:**
- `WorkflowRequest` - Request to execute a workflow (by ID or inline graph)
- `CommandRequest` - Request to execute a command (MCP-like)

### 2. Gateway Client (`src/nodetool/gateway/client.py`)

Main `GatewayClient` class with features:

- **Connection Management**
  - Automatic reconnection with exponential backoff
  - Configurable reconnection delay (default 5s, max 60s)
  - Graceful disconnect handling

- **Registration & Heartbeat**
  - Automatic node registration on connect
  - Periodic heartbeat messages (default 30s)
  - Advertises capabilities (workflow execution, supported commands)

- **Workflow Execution**
  - Handles `workflow_request` messages
  - Executes workflows by ID or inline graph
  - Streams updates back to gateway
  - Error handling with detailed error responses

- **Command Dispatch**
  - Routes commands to appropriate tool functions
  - Supports all MCP-like commands:
    - Workflow: list_workflows, get_workflow, run_workflow, validate_workflow
    - Job: list_jobs, get_job, get_job_logs, start_background_job
    - Asset: list_assets, get_asset
    - Node: list_nodes, search_nodes, get_node_info
    - Model: list_models
    - Collection: list_collections, get_collection

### 3. CLI Command (`src/nodetool/cli.py`)

Added `nodetool gateway connect` command:

```bash
nodetool gateway connect \
  --url ws://gateway.example.com:8080 \
  --auth-token your-token \
  --node-id my-node \
  --user-id user-1 \
  --heartbeat-interval 30
```

### 4. Tests (`tests/gateway/`)

Comprehensive test suite:
- `test_protocol.py` - Tests for all message types (14 tests)
- `test_client.py` - Tests for client functionality (16 tests)

All tests use proper async/await patterns and mocking for isolation.

## Message Flow Examples

### Successful Workflow Execution

```
1. Node connects to Gateway
   Node → Gateway: NodeRegistration

2. Gateway acknowledges
   Gateway → Node: (implicit ack)

3. Node sends heartbeat
   Node → Gateway: NodeHeartbeat

4. Gateway requests workflow execution
   Gateway → Node: WorkflowRequest(workflow_id="wf-123")

5. Node acknowledges
   Node → Gateway: AckMessage("Workflow execution started")

6. Node executes workflow
   Node → Gateway: WorkflowUpdate(status="running", node_id="node-1")
   Node → Gateway: WorkflowUpdate(status="completed", node_id="node-1")

7. Node sends final response
   Node → Gateway: WorkflowResponse(status="completed", result={...})
```

### Command Execution

```
1. Gateway sends command
   Gateway → Node: CommandRequest(command="list_workflows", args={})

2. Node acknowledges
   Node → Gateway: AckMessage("Executing command: list_workflows")

3. Node executes command
   (internal call to WorkflowTools.list_workflows())

4. Node sends response
   Node → Gateway: CommandResponse(status="success", result={workflows: [...]})
```

## Key Design Decisions

### 1. Protocol Design

- **JSON-based**: Easy to debug, human-readable
- **Pydantic models**: Strong typing, validation, serialization
- **Message type discriminator**: `type` field for routing
- **Timestamps**: All messages include UTC timestamps

### 2. Async Architecture

- Full async/await using `asyncio`
- Background tasks for heartbeat and message processing
- Non-blocking workflow execution (fire-and-forget pattern)

### 3. Error Handling

- Automatic reconnection on connection loss
- Exponential backoff to avoid overwhelming the gateway
- Graceful degradation (logs errors, continues processing)
- Detailed error responses with traceback information

### 4. Integration

- Reuses existing `WorkflowTools`, `JobTools`, etc.
- No duplication of workflow execution logic
- Consistent with MCP server patterns

## Security Considerations

Implemented:
- Optional authentication token support
- User ID-based access control

Recommended for production:
- Use `wss://` (secure WebSocket)
- Implement token rotation
- Network isolation for gateway nodes
- TLS certificate validation

## Extensibility

The client is designed to be easily extended:

```python
class CustomGatewayClient(GatewayClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom handlers
        self._handlers["custom_message"] = self._handle_custom_message
    
    async def _handle_custom_message(self, data: dict):
        # Custom logic here
        pass
```

## Testing Strategy

- **Unit tests**: Protocol message creation/serialization
- **Integration tests**: Client functionality with mocked WebSocket
- **Manual testing**: Requires actual gateway server (not implemented)

## Future Enhancements

Potential improvements:
- MessagePack encoding for efficiency
- Compression for large payloads
- Enhanced monitoring/metrics
- Load balancing hints
- Node discovery mechanism
- Health check endpoints

## Documentation

- **Protocol documentation**: `docs/gateway-protocol.md`
- **Usage examples**: `examples/gateway-client-example.md`
- **API documentation**: Docstrings in all modules

## Dependencies

New dependencies added:
- `websockets` - WebSocket client library (already a dependency)
- `pydantic` - Data validation (already a dependency)

No new external dependencies required.

## Deployment

The client can be deployed in several ways:

1. **Standalone process**: `nodetool gateway connect ...`
2. **Docker container**: Package with NodeTool image
3. **Python script**: Import and use `GatewayClient` directly
4. **Systemd service**: Create service unit for auto-restart

## Monitoring

The client provides logging at multiple levels:

- **INFO**: Connection events, requests received
- **DEBUG**: Message send/receive, detailed workflow progress
- **ERROR**: Connection failures, execution errors

Use standard Python logging configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Status

**✅ Complete and ready for testing**

The implementation is fully functional and passes all tests. It awaits testing with an actual OpenClaw Gateway server to verify protocol compatibility.

## Next Steps

1. Test with OpenClaw Gateway server (or mock server)
2. Add integration tests with actual WebSocket communication
3. Performance testing under load
4. Production deployment documentation
5. Consider adding metrics/monitoring instrumentation
