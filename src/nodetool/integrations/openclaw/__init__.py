"""OpenClaw integration module for nodetool-core.

This module provides integration with the OpenClaw Gateway Protocol,
allowing nodetool-core to function as an OpenClaw node.
"""

from .node_api import router as openclaw_router

__all__ = ["openclaw_router"]
