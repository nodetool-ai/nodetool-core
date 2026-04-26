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
"""

BRIDGE_PROTOCOL_VERSION = 1
