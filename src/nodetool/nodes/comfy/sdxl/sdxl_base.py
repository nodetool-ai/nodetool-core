"""
SDXL Base node for text-to-image generation.

This node uses the SDXL Base model with a standard interface for
high-quality image generation.
"""

from typing import ClassVar, Dict

from pydantic import Field

from nodetool.metadata.types import (
    ComfyCheckpointSDXL,
    ImageRef,
)
from nodetool.nodes.comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)


class SDXLBaseTextToImage(ComfyTemplateNode):
    """
    SDXL Base model for high-quality text-to-image generation.

    This node uses the SDXL 1.0 base model for text-to-image generation.
    Supports both positive and negative prompts.

    **Template:** workflow_templates/templates/sdxl/sdxl_base.json

    **Model Requirements:**
    - Checkpoint: sd_xl_base_1.0.safetensors (~6.9GB)

    **Recommended Settings:**
    - Steps: 20-40
    - CFG: 7.0-12.0
    - Sampler: euler or dpmpp_2m
    """

    # ========================================================================
    # Template Configuration
    # ========================================================================

    template_path: ClassVar[str] = "sdxl/sdxl_base.json"

    # Map model fields to loader nodes
    model_mapping: ClassVar[Dict[str, ModelNodeMapping]] = {
        "checkpoint": ModelNodeMapping(
            node_id="4",
            input_name="ckpt_name",
            loader_type="CheckpointLoaderSimple"
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
        "sampler": NodeInputMapping(
            node_id="3",
            input_name="sampler_name"
        ),
        "scheduler": NodeInputMapping(
            node_id="3",
            input_name="scheduler"
        ),
    }

    # Map outputs
    output_mapping: ClassVar[Dict[str, NodeOutputMapping]] = {
        "image": NodeOutputMapping(
            node_id="9",
            output_type="image",
            output_name="images"
        )
    }

    # ========================================================================
    # Model Fields
    # ========================================================================

    checkpoint: ComfyCheckpointSDXL = Field(
        default=ComfyCheckpointSDXL(value="sd_xl_base_1.0.safetensors"),
        description="SDXL checkpoint model"
    )

    # ========================================================================
    # Input Fields
    # ========================================================================

    prompt: str = Field(
        default="",
        description="Positive prompt describing the image to generate"
    )

    negative_prompt: str = Field(
        default="",
        description="Negative prompt (what to avoid in the image)"
    )

    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Image width in pixels (SDXL works best at 1024)"
    )

    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Image height in pixels (SDXL works best at 1024)"
    )

    seed: int = Field(
        default=0,
        description="Random seed for reproducibility (0 = random)"
    )

    steps: int = Field(
        default=30,
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

    sampler: str = Field(
        default="euler",
        description="Sampling algorithm (euler, dpmpp_2m, etc.)"
    )

    scheduler: str = Field(
        default="normal",
        description="Noise scheduler (normal, karras, exponential)"
    )

    # ========================================================================
    # Output Definition
    # ========================================================================

    async def process(self, context) -> ImageRef:
        """Execute the SDXL workflow and return generated image."""
        result = await super().process(context)
        return result.get("image", ImageRef())
