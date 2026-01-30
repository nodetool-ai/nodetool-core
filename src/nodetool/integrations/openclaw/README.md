# OpenClaw Integration for nodetool-core

This module provides integration with the OpenClaw Gateway Protocol, allowing nodetool-core to function as an OpenClaw node.

## Overview

The OpenClaw integration enables nodetool-core to:
- Register as a node in the OpenClaw decentralized architecture
- Expose capabilities to other nodes via the Gateway
- Execute tasks requested by the Gateway
- Report health and status information
- Maintain heartbeat connections with the Gateway

## Architecture

```
┌─────────────────────────────────┐
│      OpenClaw Gateway           │
│   (Message Bus + API)           │
└────────────┬────────────────────┘
             │                     
             │ Registration        
             │ Task Execution      
             │ Heartbeat           
             │                     
┌────────────▼────────────────────┐
│   nodetool-core OpenClaw API    │
│                                 │
│  - /openclaw/register           │
│  - /openclaw/execute            │
│  - /openclaw/capabilities       │
│  - /openclaw/status             │
│  - /openclaw/health             │
└─────────────────────────────────┘
```

## Configuration

The OpenClaw integration is configured via environment variables:

### Core Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENCLAW_ENABLED` | Enable OpenClaw integration | `false` | Yes |
| `OPENCLAW_GATEWAY_URL` | URL of OpenClaw Gateway | `https://gateway.openclaw.ai` | No |
| `OPENCLAW_GATEWAY_TOKEN` | Authentication token for Gateway | - | Yes (for production) |

### Node Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCLAW_NODE_ID` | Unique node identifier | Auto-generated |
| `OPENCLAW_NODE_NAME` | Human-readable node name | `nodetool-core` |
| `OPENCLAW_NODE_ENDPOINT` | Base URL for this node | Constructed from `NODETOOL_API_URL` |

### Operational Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCLAW_AUTO_REGISTER` | Auto-register on startup | `true` |
| `OPENCLAW_HEARTBEAT_INTERVAL` | Heartbeat interval in seconds | `60` |
| `OPENCLAW_MAX_CONCURRENT_TASKS` | Max concurrent tasks | `10` |
| `OPENCLAW_TASK_TIMEOUT` | Task timeout in seconds | `300` |

## Setup

### 1. Enable OpenClaw Integration

Add to your `.env` file or set environment variables:

```bash
# Enable OpenClaw
OPENCLAW_ENABLED=true

# Gateway configuration
OPENCLAW_GATEWAY_URL=https://gateway.openclaw.ai
OPENCLAW_GATEWAY_TOKEN=your-gateway-token-here

# Optional: Node identification
OPENCLAW_NODE_ID=nodetool-prod-1
OPENCLAW_NODE_NAME=nodetool-production

# Optional: Operational settings
OPENCLAW_HEARTBEAT_INTERVAL=30
OPENCLAW_MAX_CONCURRENT_TASKS=20
```

### 2. Start the Server

```bash
nodetool serve --port 7777
```

The OpenClaw API endpoints will be available at `http://localhost:7777/openclaw/`.

### 3. Register with Gateway

#### Manual Registration

```bash
curl -X POST http://localhost:7777/openclaw/register \
  -H "Content-Type: application/json"
```

#### Automatic Registration

Set `OPENCLAW_AUTO_REGISTER=true` to register on startup.

## API Endpoints

### GET /openclaw/capabilities

Returns the list of capabilities this node provides.

**Response:**
```json
[
  {
    "name": "workflow_execution",
    "description": "Execute AI workflows defined in nodetool format",
    "input_schema": {...},
    "output_schema": {...}
  },
  {
    "name": "chat_completion",
    "description": "Generate text completions using various AI models",
    "input_schema": {...},
    "output_schema": {...}
  },
  {
    "name": "asset_processing",
    "description": "Process and transform media assets",
    "input_schema": {...},
    "output_schema": {...}
  }
]
```

### GET /openclaw/status

Returns the current status and health of this node.

**Response:**
```json
{
  "node_id": "nodetool-hostname-12345",
  "status": "online",
  "uptime_seconds": 3600.5,
  "active_tasks": 2,
  "total_tasks_completed": 100,
  "total_tasks_failed": 5,
  "system_info": {
    "platform": "Linux-5.15.0-x86_64-with-glibc2.35",
    "python_version": "3.12.3",
    "cpu_percent": 45.2,
    "memory_percent": 60.1,
    "disk_percent": 35.8
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### POST /openclaw/register

Register this node with the OpenClaw Gateway.

**Request:**
```json
{
  "node_id": "nodetool-prod-1",
  "node_name": "nodetool Production",
  "node_version": "0.6.3",
  "capabilities": [...],
  "endpoint": "https://nodetool.example.com/openclaw",
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "node_id": "nodetool-prod-1",
  "token": "node-auth-token-xyz",
  "message": "Registration successful"
}
```

### POST /openclaw/execute

Execute a task on this node.

**Request:**
```json
{
  "task_id": "task-123",
  "capability_name": "workflow_execution",
  "parameters": {
    "workflow_data": {...},
    "params": {...}
  },
  "callback_url": "https://gateway.example.com/callback",
  "metadata": {}
}
```

**Response:**
```json
{
  "task_id": "task-123",
  "status": "running",
  "message": "Task execution started"
}
```

### GET /openclaw/tasks/{task_id}

Get the status of a specific task.

**Response:**
```json
{
  "task_id": "task-123",
  "status": "completed",
  "message": "Task completed successfully",
  "result": {
    "output": "...",
    "data": {...}
  }
}
```

### GET /openclaw/health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "node_id": "nodetool-hostname-12345",
  "uptime_seconds": 3600.5,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Node Capabilities

The nodetool-core node provides the following capabilities:

### workflow_execution

Execute AI workflows defined in nodetool format.

**Input:**
- `workflow_id` (optional): ID of workflow to execute
- `workflow_data` (required): Workflow graph definition
- `params` (optional): Input parameters for the workflow

**Output:**
- `job_id`: ID of the created job
- `status`: Job status
- `result`: Execution results

### chat_completion

Generate text completions using various AI models.

**Input:**
- `messages` (required): Chat messages in OpenAI format
- `model` (optional): Model to use
- `temperature` (optional): Sampling temperature

**Output:**
- `response`: Generated text
- `model`: Model used
- `usage`: Token usage statistics

### asset_processing

Process and transform media assets (images, audio, video).

**Input:**
- `asset_url` (required): URL of asset to process
- `operation` (required): Operation to perform (resize, convert, transform)
- `parameters` (optional): Operation-specific parameters

**Output:**
- `result_url`: URL of processed asset
- `metadata`: Processing metadata

## Development

### Running Tests

```bash
# Run all OpenClaw tests
uv run pytest tests/integrations/openclaw/ -v

# Run specific test file
uv run pytest tests/integrations/openclaw/test_config.py -v
```

### Linting

```bash
# Check code style
uv run ruff check src/nodetool/integrations/openclaw/

# Auto-fix issues
uv run ruff check --fix src/nodetool/integrations/openclaw/
```

## Security Considerations

1. **Authentication**: Always use `OPENCLAW_GATEWAY_TOKEN` in production
2. **TLS/HTTPS**: Use HTTPS for Gateway communication
3. **Rate Limiting**: Respect Gateway rate limits
4. **Input Validation**: All task parameters are validated
5. **Resource Limits**: Configured via `OPENCLAW_MAX_CONCURRENT_TASKS`

## Troubleshooting

### Integration Not Enabled

**Error:** `OpenClaw integration is not enabled`

**Solution:** Set `OPENCLAW_ENABLED=true` in your environment.

### Registration Failed

**Error:** `Failed to register with Gateway`

**Solution:** Check:
- Gateway URL is correct
- Gateway token is valid
- Network connectivity to Gateway
- Gateway is operational

### Tasks Not Executing

**Problem:** Tasks remain in "pending" status

**Solution:** Check:
- Node is not at max capacity (`OPENCLAW_MAX_CONCURRENT_TASKS`)
- Task capability is supported
- Task parameters are valid

## References

- [OpenClaw Architecture](https://docs.openclaw.ai/concepts/architecture)
- [OpenClaw Gateway Protocol](https://docs.openclaw.ai/gateway/protocol)
- [nodetool-core Documentation](https://docs.nodetool.ai)
