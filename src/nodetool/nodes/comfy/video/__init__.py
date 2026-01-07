"""
Video generation nodes for ComfyUI template execution.

This subpackage provides nodes for various video generation workflows:
- LTXVTextToVideo: LTXV text-to-video generation
"""

from nodetool.nodes.comfy.video.ltxv_text_to_video import LTXVTextToVideo

__all__ = [
    "LTXVTextToVideo",
]
