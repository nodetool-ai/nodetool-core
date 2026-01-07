"""
ComfyUI Template Nodes Package
==============================

This package provides Python node implementations for ComfyUI workflow templates.
Each node class represents a specific ComfyUI workflow and defines its input/output
mapping to the JSON template structure through class-level configuration.

The nodes in this package use the ComfyTemplateNode base class which handles:
- Loading JSON workflow templates
- Injecting field values into templates
- Uploading images to ComfyUI
- Executing workflows
- Extracting and converting results

Subpackages:
- flux/: Flux model nodes (FluxDevSimple, FluxSchnell, etc.)
- sdxl/: SDXL model nodes (SDXLBaseTextToImage, etc.)
- video/: Video generation nodes (LTXVTextToVideo, etc.)
"""

# Import subpackages to register nodes
from nodetool.nodes.comfy import flux, sdxl, video
from nodetool.nodes.comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)

__all__ = [
    "ComfyTemplateNode",
    "ModelNodeMapping",
    "NodeInputMapping",
    "NodeOutputMapping",
    "flux",
    "sdxl",
    "video",
]
