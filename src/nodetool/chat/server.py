"""
NodeTool Chat Server

This module provides a FastAPI-based chat server that supports both WebSocket and Server-Sent Events (SSE) protocols
for real-time chat communication with AI providers.

## API Endpoints

### WebSocket Protocol

#### WebSocket Chat Endpoint: `/chat`

**Connection URL:** `ws://host:port/chat?token=YOUR_TOKEN`

**Authentication:**
- Query parameter: `?token=YOUR_TOKEN`
- Header: `Authorization: Bearer YOUR_TOKEN`

**Message Format:**
Messages are sent and received as JSON objects over the WebSocket connection.

**Client to Server Messages:**
```json
{
  "type": "message",
  "content": "Your message here",
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Server to Client Messages:**
```json
{
  "type": "response",
  "content": "AI response content",
  "status": "streaming|complete|error"
}
```

### OpenAI-Compatible Protocol

#### Chat Completions Endpoint: `POST /v1/chat/completions`

**URL:** `http://host:port/v1/chat/completions`

**Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer YOUR_TOKEN`

**Request Body:**
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": true
}
```

**Response (Streaming):**
Server-Sent Events stream with OpenAI-compatible format:

```
data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1694268190, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1694268190, "model": "gpt-4", "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1694268190, "model": "gpt-4", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

**Response (Non-streaming):**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1694268190,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 15,
    "total_tokens": 27
  }
}
```

### Models Endpoint: `GET /v1/models`

**URL:** `http://host:port/v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4",
      "object": "model",
      "created": 0,
      "owned_by": "openai"
    }
  ]
}
```

#### CORS Preflight: `OPTIONS /v1/chat/completions`

Handles CORS preflight requests for web browser compatibility.

### Health Check

#### Health Check Endpoint: `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "protocol": "websocket|sse"
}
```

## Authentication Modes

### Local Authentication (default)
- Uses a fixed user ID of "1"
- No external authentication required
- Suitable for development and testing

### Remote Authentication (Supabase)
- Requires valid auth tokens from Supabase
- Validates tokens against remote authentication service
- Suitable for production deployments

## Database Modes

### With Database (default)
- Stores conversation history in database
- Persistent chat sessions
- Full context retention

## Usage Examples

### Start Chat Server
```bash
nodetool chat-server
nodetool chat-server --host 0.0.0.0 --port 8080
```

### With Authentication
```bash
nodetool chat-server --remote-auth
```

### With Custom Default Model
```bash
nodetool chat-server --default-model gpt-4 --default-provider openai
nodetool chat-server --default-model claude-3-opus-20240229 --default-provider anthropic
```

### Client Example (JavaScript)
```javascript
// Streaming request using fetch
const response = await fetch('http://localhost:8080/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_TOKEN'
    },
    body: JSON.stringify({
        model: 'gpt-4',
        messages: [
            {role: 'user', content: 'Hello, AI!'}
        ],
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') break;
            
            try {
                const parsed = JSON.parse(data);
                const content = parsed.choices[0]?.delta?.content;
                if (content) {
                    console.log('AI Response:', content);
                }
            } catch (e) {
                // Skip malformed JSON
            }
        }
    }
}

// Non-streaming request
const nonStreamingResponse = await fetch('http://localhost:8080/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_TOKEN'
    },
    body: JSON.stringify({
        model: 'gpt-4',
        messages: [
            {role: 'user', content: 'Hello, AI!'}
        ],
        stream: false
    })
});

const result = await nonStreamingResponse.json();
console.log('AI Response:', result.choices[0].message.content);
```

### cURL Examples

#### Chat Completions (Streaming)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, AI!"}
    ],
    "stream": true
  }'
```

#### Chat Completions (Non-streaming)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, AI!"}
    ],
    "stream": false
  }'
```

#### List Models
```bash
curl http://localhost:8080/v1/models \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Health Check
```bash
curl http://localhost:8080/health
```

## Error Handling

### WebSocket Errors
- Connection errors are logged and the connection is closed gracefully
- Authentication failures result in connection termination
- Malformed messages return error responses

### SSE Errors
- HTTP 500 status for server errors
- HTTP 401 for authentication failures
- Error events in the SSE stream for processing errors

## Security Considerations

- Always use HTTPS in production
- Validate auth tokens properly
- Implement rate limiting if needed
- Consider CORS configuration for web clients
- Use secure token storage practices
"""

import uvicorn
import asyncio
import platform
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from nodetool.config.environment import Environment
from nodetool.chat.chat_sse_runner import ChatSSERunner
from rich.console import Console
from nodetool.api.model import get_language_models
import json

from nodetool.types.workflow import Workflow

console = Console()


if platform.system() == "Windows":
    # Ensure subprocess support in asyncio on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def create_chat_server(
    remote_auth: bool,
    provider: str,
    default_model: str = "gpt-oss:20b",
    tools: list[str] = [],
    workflows: list[Workflow] = [],
) -> FastAPI:
    """Create a FastAPI chat server instance.

    Args:
        remote_auth: Whether to use remote authentication
        provider: Provider to use
        default_model: Default model to use when not specified by client

    Returns:
        FastAPI application instance
    """
    # Set authentication mode
    Environment.set_remote_auth(remote_auth)

    app = FastAPI(title="NodeTool Chat Server", version="1.0.0")

    # Include OpenAI-compatible routes via router
    from nodetool.api.openai import create_openai_compatible_router

    app.include_router(
        create_openai_compatible_router(
            provider=provider,
            default_model=default_model,
            tools=tools,
        )
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    # Add graceful shutdown
    @app.on_event("startup")
    async def startup_event():
        console.print(f"Chat server started successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        console.print("Chat server shutting down...")

    return app


def run_chat_server(
    host: str,
    port: int,
    remote_auth: bool,
    provider: str,
    default_model: str = "gpt-oss:20b",
    tools: list[str] = [],
    workflows: list[Workflow] = [],
):
    """Run the chat server.

    Args:
        host: Host address to serve on
        port: Port to serve on
        protocol: Protocol to use ('websocket' or 'sse')
        remote_auth: Whether to use remote authentication
        provider: Provider to use
        default_model: Default model to use when not specified by client
        tools: List of tool names to use
        workflows: List of workflows to use
    """
    import dotenv

    dotenv.load_dotenv()

    app = create_chat_server(remote_auth, provider, default_model, tools, workflows)

    console.print(f"🚀 Starting OpenAI-compatible chat server on {host}:{port}")
    console.print(
        f"Chat completions endpoint: http://{host}:{port}/v1/chat/completions"
    )
    console.print(f"Models endpoint: http://{host}:{port}/v1/models")
    console.print(
        "Authentication mode:",
        "Remote (Supabase)" if remote_auth else "Local (user_id=1)",
    )
    console.print("Default model:", f"{default_model} (provider: {provider})")
    console.print("Tools:", tools)
    console.print("Workflows:", [w.name for w in workflows])
    console.print("\nSend POST requests with Authorization: Bearer YOUR_TOKEN header")

    # Run the server
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\n👋 Chat server stopped by user")
    except Exception as e:
        console.print(f"❌ Server error: {e}")
        import sys

        sys.exit(1)
