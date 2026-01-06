"""
ComfyUI provider package.

This package provides YAML-based template mappings for ComfyUI workflows,
allowing easy configuration of text-to-image, image-to-image, and image-to-video
generation through template files.
"""

from nodetool.providers.comfy.template_loader import TemplateLoader
from nodetool.providers.comfy.template_models import (
    InputMapping,
    NodeMapping,
    OutputMapping,
    TemplateInfo,
    TemplateMapping,
)

__all__ = [
    "InputMapping",
    "NodeMapping",
    "OutputMapping",
    "TemplateInfo",
    "TemplateLoader",
    "TemplateMapping",
]
