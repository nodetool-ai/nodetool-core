# Chat WebSocket API

NodeTool exposes a `/chat` WebSocket endpoint for real time conversations.
The server side is implemented by `ChatWebSocketRunner` which handles message
parsing, tool execution and streaming responses.

The connection supports both binary (MessagePack) and text (JSON) messages.
Authentication can be provided via `Authorization: Bearer <token>` headers or
an `api_key` query parameter.

## Example usage

```javascript
const socket = new WebSocket("ws://localhost:8000/chat?api_key=YOUR_KEY");

// Send a chat message
const message = {
  role: "user",
  content: "Hello world",
  model: "gpt-3.5-turbo" // or any supported model
};

socket.onmessage = async (event) => {
  const data = msgpack.decode(new Uint8Array(await event.data.arrayBuffer()));
  if (data.type === "chunk") {
    console.log(data.content);
  }
};

socket.onopen = () => {
  socket.send(msgpack.encode(message));
};
```

### Server responses

Responses from the server may include:

- `chunk` – streamed text from the model
- `tool_call` – a request to execute a tool
- `tool_result` – the result of a tool execution
- `job_update` – status updates when running a workflow
- `error` – error messages

The runner also supports workflow execution by sending a message with a
`workflow_id`. In that case the WebSocket will stream job updates in addition
to regular chat responses.
