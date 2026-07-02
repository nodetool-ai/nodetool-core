# src/nodetool/worker/__init__.py
"""NodeTool worker package.

Bridge protocol version: bumped only when the JS↔Python stdio protocol
changes in a non-backward-compatible way. The Electron app declares the
minimum version it can speak; if the Python worker reports a lower number
the JS bridge refuses to use it and asks the user to reinstall the
Python environment.

History:
  1 - Initial stdio protocol (msgpack length-prefixed framing,
      discover/execute/result/error/chunk/progress + provider.* messages).
  2 - Added models.* messages (models.list_cached / models.download /
      models.delete) for worker-side HuggingFace cache management.
  3 - Added comfy.* messages (ComfyUI proxy: comfy.execute /
      comfy.queue / comfy.interrupt / comfy.cancel / comfy.upload /
      comfy.view / comfy.object_info / comfy.system_stats / comfy.free /
      comfy.status + comfy.models.* volume management), the `comfy`
      capability block in worker.status, and the `comfy.event` frame
      type carrying streamed ComfyUI execution events during
      comfy.execute.
"""

BRIDGE_PROTOCOL_VERSION = 3
