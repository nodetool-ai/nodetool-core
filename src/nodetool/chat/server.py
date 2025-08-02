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

### SSE Protocol

#### SSE Chat Endpoint: `POST /chat/sse`

**URL:** `http://host:port/chat/sse`

**Headers:**
- `Content-Type: application/json`
- `Authorization: Bearer YOUR_TOKEN` (optional)

**Request Body:**
```json
{
  "message": "Your message here",
  "auth_token": "YOUR_TOKEN",
  "history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```

**Response:**
Server-Sent Events stream with the following event types:

```
event: message
data: {"type": "response", "content": "Partial response...", "status": "streaming"}

event: message
data: {"type": "response", "content": "Complete response", "status": "complete"}

event: error
data: {"type": "error", "message": "Error description"}

event: close
data: {"type": "close"}
```

#### CORS Preflight: `OPTIONS /chat/sse`

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

### Without Database (--no-database)
- **WebSocket:** In-memory conversation storage
- **SSE:** History must be included in each request
- Suitable for stateless deployments

## Usage Examples

### Start WebSocket Server
```bash
nodetool chat-server
nodetool chat-server --host 0.0.0.0 --port 8080
```

### Start SSE Server
```bash
nodetool chat-server --protocol sse
nodetool chat-server --protocol sse --port 3000
```

### With Authentication
```bash
nodetool chat-server --remote-auth
```

### Without Database
```bash
nodetool chat-server --no-database
```

### With Custom Default Model
```bash
nodetool chat-server --default-model gpt-4 --default-provider openai
nodetool chat-server --default-model claude-3-opus-20240229 --default-provider anthropic
```

### WebSocket Client Example (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8080/chat?token=YOUR_TOKEN');

ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'message',
        content: 'Hello, AI!',
        history: []
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('AI Response:', data.content);
};
```

### SSE Client Example (JavaScript)
```javascript
const eventSource = new EventSource('http://localhost:8080/chat/sse');

eventSource.addEventListener('message', function(event) {
    const data = JSON.parse(event.data);
    console.log('AI Response:', data.content);
});

// Send message via fetch
fetch('http://localhost:8080/chat/sse', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_TOKEN'
    },
    body: JSON.stringify({
        message: 'Hello, AI!',
        history: []
    })
});
```

### cURL Examples

#### SSE Request
```bash
curl -X POST http://localhost:8080/chat/sse \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"message": "Hello, AI!", "history": []}'
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

import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import StreamingResponse
from nodetool.common.environment import Environment
from nodetool.chat.chat_websocket_runner import ChatWebSocketRunner
from nodetool.chat.chat_sse_runner import ChatSSERunner
from rich.console import Console

console = Console()


def create_chat_server(protocol: str, remote_auth: bool, no_database: bool, default_model: str = "gemma3n:latest", default_provider: str = "ollama") -> FastAPI:
    """Create a FastAPI chat server instance.
    
    Args:
        protocol: Protocol to use ('websocket' or 'sse')
        remote_auth: Whether to use remote authentication
        no_database: Whether to run without database
        default_model: Default model to use when not specified by client
        default_provider: Default provider to use when not specified by client
        
    Returns:
        FastAPI application instance
    """
    # Set authentication mode
    Environment.set_remote_auth(remote_auth)

    app = FastAPI(title="NodeTool Chat Server", version="1.0.0")

    if protocol == "websocket":
        @app.websocket("/chat")
        async def websocket_chat(websocket: WebSocket):
            """WebSocket endpoint for chat communication."""
            # Get auth token from query parameters or headers
            auth_token = websocket.query_params.get("token")
            if not auth_token and "authorization" in websocket.headers:
                auth_header = websocket.headers["authorization"]
                if auth_header.startswith("Bearer "):
                    auth_token = auth_header[7:]
            
            runner = ChatWebSocketRunner(auth_token, use_database=not no_database, default_model=default_model, default_provider=default_provider)
            try:
                await runner.run(websocket)
            except Exception as e:
                console.print(f"WebSocket error: {e}")
            finally:
                await runner.disconnect()

    else:  # SSE
        @app.post("/chat/sse")
        async def sse_chat(request: Request):
            """SSE endpoint for chat communication."""
            try:
                # Get request data
                data = await request.json()
                
                # Extract auth token from headers
                auth_header = request.headers.get("authorization", "")
                auth_token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None
                
                # Add auth token to request data if provided
                if auth_token:
                    data["auth_token"] = auth_token
                
                # Create SSE runner and stream response
                runner = ChatSSERunner(auth_token, use_database=not no_database, default_model=default_model, default_provider=default_provider)
                
                return StreamingResponse(
                    runner.process_single_request(data),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Authorization, Content-Type",
                        "Access-Control-Allow-Methods": "POST, OPTIONS"
                    }
                )
            except Exception as e:
                console.print(f"SSE error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.options("/chat/sse")
        async def sse_chat_options():
            """Handle CORS preflight requests."""
            return {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
            }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "protocol": protocol}

    # Add graceful shutdown
    @app.on_event("startup")
    async def startup_event():
        console.print(f"Chat server started successfully using {protocol.upper()} protocol")

    @app.on_event("shutdown")
    async def shutdown_event():
        console.print("Chat server shutting down...")

    return app


def run_chat_server(host: str, port: int, protocol: str, remote_auth: bool, no_database: bool, default_model: str = "gemma3n:latest", default_provider: str = "ollama"):
    """Run the chat server.
    
    Args:
        host: Host address to serve on
        port: Port to serve on
        protocol: Protocol to use ('websocket' or 'sse')
        remote_auth: Whether to use remote authentication
        no_database: Whether to run without database
        default_model: Default model to use when not specified by client
        default_provider: Default provider to use when not specified by client
    """
    app = create_chat_server(protocol, remote_auth, no_database, default_model, default_provider)
    
    if protocol == "websocket":
        console.print(f"🚀 Starting WebSocket chat server on {host}:{port}")
        console.print(f"WebSocket endpoint: ws://{host}:{port}/chat")
        console.print("Authentication mode:", "Remote (Supabase)" if remote_auth else "Local (user_id=1)")
        console.print("Database mode:", "Disabled (in-memory)" if no_database else "Enabled")
        console.print("Default model:", f"{default_model} (provider: {default_provider})")
        console.print("\nTo connect with auth token: ws://host:port/chat?token=YOUR_TOKEN")
    else:  # SSE
        console.print(f"🚀 Starting SSE chat server on {host}:{port}")
        console.print(f"SSE endpoint: http://{host}:{port}/chat/sse")
        console.print("Authentication mode:", "Remote (Supabase)" if remote_auth else "Local (user_id=1)")
        console.print("Database mode:", "Disabled (history in request)" if no_database else "Enabled")
        console.print("Default model:", f"{default_model} (provider: {default_provider})")
        console.print("\nSend POST requests with Authorization: Bearer YOUR_TOKEN header")
        if no_database:
            console.print("Include 'history' field in request payload with full conversation history")

    # Run the server
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\n👋 Chat server stopped by user")
    except Exception as e:
        console.print(f"❌ Server error: {e}")
        import sys
        sys.exit(1)