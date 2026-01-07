"""
LTXV Text to Video node for video generation.

This node uses the LTXV model for text-to-video generation.
"""

from typing import ClassVar, Dict

from pydantic import Field

from nodetool.metadata.types import (
    LTXVideoModel,
    VideoRef,
)
from nodetool.nodes.comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)


class LTXVTextToVideo(ComfyTemplateNode):
    """
    LTXV model for text-to-video generation.

    This node uses the LTXV model to generate videos from text prompts.

    **Template:** workflow_templates/templates/video/ltxv_text_to_video.json

    **Model Requirements:**
    - LTXV model file

    **Recommended Settings:**
    - Steps: 20-40
    - CFG: 7.0
    - Frames: 16-48
    """

    # ========================================================================
    # Template Configuration
    # ========================================================================

    template_path: ClassVar[str] = "video/ltxv_text_to_video.json"

    # Map model fields to loader nodes
    model_mapping: ClassVar[Dict[str, ModelNodeMapping]] = {
        "video_model": ModelNodeMapping(
            node_id="10",
            input_name="model_name",
            loader_type="LTXVideoLoader"
        ),
    }

    # Map input fields to template nodes
    input_mapping: ClassVar[Dict[str, NodeInputMapping]] = {
        "prompt": NodeInputMapping(
            node_id="6",
            input_name="text"
        ),
        "negative_prompt": NodeInputMapping(
            node_id="7",
            input_name="text"
        ),
        "width": NodeInputMapping(
            node_id="5",
            input_name="width",
            transform="int"
        ),
        "height": NodeInputMapping(
            node_id="5",
            input_name="height",
            transform="int"
        ),
        "frames": NodeInputMapping(
            node_id="5",
            input_name="frames",
            transform="int"
        ),
        "seed": NodeInputMapping(
            node_id="3",
            input_name="seed",
            transform="int"
        ),
        "steps": NodeInputMapping(
            node_id="3",
            input_name="steps",
            transform="int"
        ),
        "cfg": NodeInputMapping(
            node_id="3",
            input_name="cfg",
            transform="float"
        ),
    }

    # Map outputs
    output_mapping: ClassVar[Dict[str, NodeOutputMapping]] = {
        "video": NodeOutputMapping(
            node_id="9",
            output_type="video",
            output_name="video"
        )
    }

    # ========================================================================
    # Model Fields
    # ========================================================================

    video_model: LTXVideoModel = Field(
        default=LTXVideoModel(value="ltxv.safetensors"),
        description="LTXV video model"
    )

    # ========================================================================
    # Input Fields
    # ========================================================================

    prompt: str = Field(
        default="",
        description="Text prompt describing the video to generate"
    )

    negative_prompt: str = Field(
        default="",
        description="Negative prompt (what to avoid)"
    )

    width: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Video width in pixels"
    )

    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Video height in pixels"
    )

    frames: int = Field(
        default=24,
        ge=1,
        le=128,
        description="Number of frames to generate"
    )

    seed: int = Field(
        default=0,
        description="Random seed for reproducibility (0 = random)"
    )

    steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of denoising steps"
    )

    cfg: float = Field(
        default=7.0,
        ge=0.0,
        le=30.0,
        description="Classifier-free guidance scale"
    )

    # ========================================================================
    # Output Definition
    # ========================================================================

    async def process(self, context) -> VideoRef:
        """Execute the LTXV workflow and return generated video."""
        result = await super().process(context)
        return result.get("video", VideoRef())
