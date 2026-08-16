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
  4 - Run identity + run boundary. `execute` / `execute.stream` carry five
      new OPTIONAL keys (`node_id`, `job_id`, `workflow_id`, `user_id`,
      `requires_vram_gb`); `job.start` / `job.end` bracket a run so the
      worker has a caller for `ModelManager.release_nodes()`;
      `models.evict` drops loaded weights on request; `discover` node
      entries may carry `requires_vram_gb`. Every part is additive — the
      identity keys are extra dict entries a pre-v4 worker ignores, and
      the new message types are capability-gated on the JS side — so the
      floor (`MIN_BRIDGE_PROTOCOL_VERSION` = 1) does not move.
"""

BRIDGE_PROTOCOL_VERSION = 4
