# nodetool.common.runpod_websocket_runner

RunPod WebSocket Runner
=======================

The `RunPodWebSocketRunner` class allows executing workflows on a [RunPod](https://www.runpod.io/) endpoint over a WebSocket connection.
It mirrors the behaviour of `WebSocketRunner` but delegates job execution to a remote RunPod endpoint using the RunPod HTTP API.

Key features include:

- WebSocket interface for running, cancelling and monitoring jobs
- Streaming of RunPod job output in either MessagePack (binary) or JSON (text) format
- Automatic mapping of RunPod status values to NodeTool job status
- Support for cancelling jobs and switching modes at runtime

Example usage:

```python
from nodetool.common.runpod_websocket_runner import RunPodWebSocketRunner

runner = RunPodWebSocketRunner(endpoint_id="YOUR_ENDPOINT_ID")
await runner.run(websocket)
```

See the source code for full details on all available methods.

