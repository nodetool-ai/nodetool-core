[← Back to Docs Index](index.md)

# NodeTool Chat Server

The NodeTool CLI now includes a `chat-server` command that allows you to run standalone chat servers using either
WebSocket or Server-Sent Events (SSE) protocols.

## Quick Start

### WebSocket Server

```bash
# Start WebSocket server on default port 8080
nodetool chat-server

# Start on custom port
nodetool chat-server --port 3000 --protocol websocket
```

### SSE Server

```bash
# Start SSE server on port 8080
nodetool chat-server --protocol sse

# Start on custom port  
nodetool chat-server --port 3000 --protocol sse
```

### With Authentication

```bash
# Enable remote authentication (requires Supabase configuration)
nodetool chat-server --remote-auth
```

### Database-Free Mode

```bash
# Run WebSocket server without database (uses in-memory storage)
nodetool chat-server --no-database

# Run SSE server without database (expects history in requests)
nodetool chat-server --protocol sse --no-database
```

## Usage Examples

### WebSocket Protocol

Connect to: `ws://127.0.0.1:8080/chat`

Send messages in this format:

```json
{
  "thread_id": "thread_123",
  "role": "user",
  "content": "Hello, world!",
  "model": "gpt-4",
  "provider": "openai",
  "tools": ["optional_tool_list"]
}
```

Authentication via query parameter:

```
ws://127.0.0.1:8080/chat?token=YOUR_AUTH_TOKEN
```

### SSE Protocol

Send POST requests to: `http://127.0.0.1:8080/chat/sse`

Request format:

```json
{
  "thread_id": "thread_456",
  "role": "user",
  "content": "Hello via SSE!",
  "model": "gpt-4",
  "provider": "openai"
}
```

#### SSE with Database-Free Mode

When using `--no-database`, include the full conversation history in the request:

```json
{
  "thread_id": "thread_456",
  "role": "user",
  "content": "What's the capital of France?",
  "model": "gpt-4",
  "provider": "openai",
  "history": [
    {
      "role": "user",
      "content": "Hello!",
      "thread_id": "thread_456"
    },
    {
      "role": "assistant",
      "content": "Hi there! How can I help you today?",
      "thread_id": "thread_456"
    }
  ]
}
```

Headers:

```
Content-Type: application/json
Accept: text/event-stream
Authorization: Bearer YOUR_AUTH_TOKEN
```

Response format (SSE):

```
data: {"type": "content", "content": "Response text"}

event: error
data: {"type": "error", "message": "Error description"}

event: end  
data: {"type": "end"}
```

## Protocol Comparison

| Feature               | WebSocket                          | SSE                              |
| --------------------- | ---------------------------------- | -------------------------------- |
| **Direction**         | Bidirectional                      | Server-to-client only            |
| **Connection**        | Persistent                         | Request-response with streaming  |
| **Protocol**          | `ws://`                            | `http://`                        |
| **Message Format**    | JSON or MessagePack                | JSON over SSE                    |
| **Authentication**    | Query param or header              | Authorization header             |
| **Browser Support**   | Requires WebSocket API             | Uses standard EventSource API    |
| **Firewall Friendly** | Sometimes blocked                  | HTTP-based, rarely blocked       |
| **Database Mode**     | Persistent or in-memory per thread | Persistent or history in request |
| **Use Case**          | Real-time chat apps                | Web dashboards, notifications    |

## Health Check

Both protocols expose a health endpoint:

```bash
curl http://127.0.0.1:8080/health
```

Response:

```json
{
  "status": "healthy",
  "protocol": "websocket"
}
```

## Testing

Run the example client:

```bash
# Start the server in one terminal
nodetool chat-server --protocol websocket

# Test in another terminal
python examples/chat_server_examples.py
```

## Configuration

### Environment Variables

- `NODETOOL_ENVIRONMENT`: Set to `production` for production mode
- `SUPABASE_URL`: Supabase project URL (for remote auth)
- `SUPABASE_KEY`: Supabase API key (for remote auth)

### Authentication Modes

**Local Mode (default):**

- No authentication required
- Uses `user_id = "1"` for all requests
- Suitable for development and testing

**Remote Mode (`--remote-auth`):**

- Requires valid Supabase JWT tokens
- Validates tokens against Supabase auth
- Suitable for production deployments

### Storage Modes

**Database Mode (default):**

- Messages are persisted to database
- Chat history is maintained automatically
- Suitable for multi-session conversations

**Database-Free Mode (`--no-database`):**

- WebSocket: Messages stored in-memory per thread_id (lost on disconnect)
- SSE: Full chat history must be included in each request
- Suitable for stateless deployments or testing

## Integration Examples

### JavaScript WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8080/chat?token=AUTH_TOKEN');

ws.onopen = () => {
  ws.send(JSON.stringify({
    thread_id: 'thread_123',
    role: 'user',
    content: 'Hello from JavaScript!',
    model: 'gpt-4',
    provider: 'openai'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### JavaScript SSE Client

```javascript
const response = await fetch('http://localhost:8080/chat/sse', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer AUTH_TOKEN'
  },
  body: JSON.stringify({
    thread_id: 'thread_456',
    role: 'user',
    content: 'Hello from SSE!',
    model: 'gpt-4',
    provider: 'openai'
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  console.log('SSE chunk:', chunk);
}
```

### Python Client Examples

See [`examples/chat_server_examples.py`](../examples/chat_server_examples.py) for complete WebSocket and SSE client
implementations.

## Error Handling

### WebSocket Errors

- Connection errors are logged to console
- Authentication failures close connection with code 1008
- Runtime errors send error messages to client

### SSE Errors

- HTTP 401 for authentication failures
- HTTP 500 for server errors
- Error events in SSE stream for runtime errors

## Performance Considerations

- **WebSocket**: Better for high-frequency bidirectional communication
- **SSE**: Better for server-initiated updates with simple HTTP infrastructure
- Both support concurrent connections limited by system resources
- Memory usage scales with number of active connections and message history

## Deployment

The chat server can be deployed standalone or integrated into existing FastAPI applications by importing the runner
classes directly:

```python
from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner
from nodetool.chat.chat_sse_runner import ChatSSERunner
```
