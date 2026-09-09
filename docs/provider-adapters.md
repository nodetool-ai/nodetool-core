# External provider adapter protocol

Container images can expose a dependency-isolated provider to the Python
worker without importing that provider into the NodeTool interpreter.

For provider id `example`, configure:

```text
NODETOOL_PROVIDER_ADAPTER_COMMAND_EXAMPLE=/isolated/python /adapter.py
NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_EXAMPLE=text_to_image,image_to_image,text_to_video,image_to_video,text_to_audio,text_to_speech_encoded
NODETOOL_PROVIDER_ADAPTER_DISPLAY_NAME_EXAMPLE=Example
```

The worker owns the command. An authenticated client can select the provider
and operation, but cannot supply an executable. One adapter process is started
per request. It receives one JSON object followed by a newline on stdin and
must write JSON-lines events to stdout. Human-readable logs belong on stderr.

Events use one of these shapes:

```json
{"type":"progress","data":{"progress":25,"status":"Denoising"}}
{"type":"result","data":{"models":[]}}
{"type":"result","data":{"path":"/absolute/path/to/output.mp4"}}
{"type":"error","data":{"error":"generation failed"}}
```

The `models` operation receives `provider` and `model_type`; its result must
contain a `models` array using the regular provider model shape. Generation
receives `provider`, `operation`, and `params`. Image-conditioned operations
add an absolute `image_path`; `tts_encoded` uses the
`text_to_speech_encoded` capability and adds `reference_audio_path` when the
request includes reference audio. Staged paths are owned by the worker and
valid only for that request. A successful media operation returns an existing
output `path`.

The worker reads that file and transfers the media through the authenticated
bridge's chunked, integrity-checked blob protocol. Cancellation terminates the
adapter's process group. Adapter JSON lines are limited to 1 MiB and retained
stderr is bounded to 32 KiB; binary data must therefore use the file boundary.
