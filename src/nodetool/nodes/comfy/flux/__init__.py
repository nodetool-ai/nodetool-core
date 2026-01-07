"""
Flux model nodes for ComfyUI template execution.

This subpackage provides nodes for various Flux model workflows:
- FluxDevSimple: Basic Flux Dev text-to-image
- FluxSchnellSimple: Fast Flux Schnell text-to-image
"""

from nodetool.nodes.comfy.flux.flux_dev_simple import FluxDevSimple
from nodetool.nodes.comfy.flux.flux_schnell_simple import FluxSchnellSimple

__all__ = [
    "FluxDevSimple",
    "FluxSchnellSimple",
]
