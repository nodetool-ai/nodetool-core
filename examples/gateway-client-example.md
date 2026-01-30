# Gateway Client Example

This example demonstrates how to use the NodeTool gateway client to connect to an OpenClaw Gateway server.

## Overview

The gateway client allows NodeTool to act as a node in a distributed architecture, receiving and executing workflow requests from a central gateway server.

## Basic Usage

### Using the CLI

The simplest way to start the gateway client is using the CLI:

```bash
# Connect to a gateway server
nodetool gateway connect --url ws://gateway.example.com:8080

# With authentication
nodetool gateway connect \
  --url ws://gateway.example.com:8080 \
  --auth-token your-token

# With custom node configuration
nodetool gateway connect \
  --url ws://gateway.example.com:8080 \
  --node-id production-node-1 \
  --user-id prod-user \
  --heartbeat-interval 60
```

### Using the Python API

You can also use the Python API directly:

```python
import asyncio
from nodetool.gateway import GatewayClient

async def main():
    # Create the client
    client = GatewayClient(
        gateway_url="ws://gateway.example.com:8080",
        node_id="my-node",
        auth_token="your-token",
        user_id="1",
        heartbeat_interval=30.0,
    )
    
    # Run the client (blocks until disconnected)
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

The gateway client provides:

- **Automatic connection management**: Reconnects automatically with exponential backoff
- **Heartbeat mechanism**: Maintains connection health with periodic heartbeats
- **Workflow execution**: Executes workflows received from the gateway
- **Command handling**: Supports MCP-like commands for managing workflows, jobs, assets, etc.
- **Real-time updates**: Streams workflow execution updates back to the gateway

## Supported Commands

The gateway client can execute these commands:

### Workflow Commands
- `list_workflows` - List available workflows
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

## Custom Client

You can extend the gateway client with custom message handlers:

```python
from nodetool.gateway import GatewayClient
from typing import Any

class CustomGatewayClient(GatewayClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom handlers
        self._handlers["custom_message"] = self._handle_custom_message
    
    async def _handle_custom_message(self, data: dict[str, Any]):
        # Handle custom message type
        self.log.info(f"Received custom message: {data}")
        # Process the message...

# Use custom client
async def main():
    client = CustomGatewayClient(
        gateway_url="ws://gateway.example.com:8080"
    )
    await client.run()
```

## Configuration

### Environment Variables

You can configure the gateway client using environment variables:

```bash
export GATEWAY_URL="ws://gateway.example.com:8080"
export GATEWAY_AUTH_TOKEN="your-token"
export GATEWAY_NODE_ID="my-node"
export GATEWAY_USER_ID="1"
```

### Configuration File

You can also use a configuration file (future enhancement).

## Security

For production deployments:

- **Always use TLS**: Use `wss://` instead of `ws://`
- **Authentication**: Always provide an authentication token
- **Network isolation**: Run gateway nodes in isolated network segments
- **Token rotation**: Implement token rotation for long-running nodes

## Monitoring

The gateway client uses the standard NodeTool logging system:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure specific logger
logger = logging.getLogger("nodetool.gateway.client")
logger.setLevel(logging.DEBUG)
```

## Testing

To test the gateway client without a real gateway server, you can use the test utilities:

```python
import pytest
from unittest.mock import AsyncMock
from nodetool.gateway import GatewayClient

@pytest.mark.asyncio
async def test_gateway_client():
    client = GatewayClient(gateway_url="ws://localhost:8080")
    
    # Mock the websocket
    client.websocket = AsyncMock()
    client.connected = True
    
    # Test registration
    await client._send_registration()
    assert client.websocket.send.called
```

## Troubleshooting

### Connection Issues

If you're having trouble connecting:

1. **Check the URL**: Ensure the gateway URL is correct
2. **Verify network connectivity**: Make sure you can reach the gateway server
3. **Check authentication**: Ensure the auth token is valid
4. **Review logs**: Enable debug logging to see detailed connection information

### Performance Issues

If workflows are executing slowly:

1. **Check system resources**: Ensure adequate CPU, memory, and GPU resources
2. **Review node configuration**: Ensure the node is configured optimally
3. **Monitor network latency**: High network latency can impact performance
4. **Check database performance**: Slow database queries can impact workflow execution

### Error Handling

The client automatically handles most errors:

- **Connection lost**: Automatically reconnects with exponential backoff
- **Invalid messages**: Logs errors and continues processing
- **Workflow failures**: Sends error responses back to the gateway
- **Command errors**: Returns error status in command responses

## Next Steps

- Read the [Gateway Protocol Documentation](../docs/gateway-protocol.md)
- Explore the [WebSocket API Documentation](../docs/websocket-api.md)
- See the [MCP Server Documentation](../src/nodetool/api/mcp_server.py) for similar patterns
