"""
Flux Schnell Simple node for fast text-to-image generation.

This node uses the Flux.1-schnell model with a simplified interface for
fast image generation.
"""

from typing import ClassVar, Dict

from pydantic import Field

from nodetool.metadata.types import (
    FluxCLIP,
    FluxUNET,
    FluxVAE,
    ImageRef,
)
from nodetool.nodes.comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)


class FluxSchnellSimple(ComfyTemplateNode):
    """
    Flux Schnell model for fast text-to-image generation.

    This node uses the Flux.1-schnell model with a simplified interface.
    Ideal for fast image generation with good quality in fewer steps.

    **Template:** workflow_templates/templates/flux/flux_schnell_simple.json

    **Model Requirements:**
    - UNET: flux1-schnell.safetensors (~23GB)
    - VAE: ae.safetensors (~335MB)
    - CLIP: t5xxl_fp8_e4m3fn.safetensors (~4.9GB)

    **Recommended Settings:**
    - Steps: 4-8 (Schnell is optimized for few steps)
    - CFG: 1.0 (Flux doesn't need high CFG)
    """

    # ========================================================================
    # Template Configuration
    # ========================================================================

    template_path: ClassVar[str] = "flux/flux_schnell_simple.json"

    # Map model fields to loader nodes in JSON
    model_mapping: ClassVar[Dict[str, ModelNodeMapping]] = {
        "unet": ModelNodeMapping(
            node_id="38",
            input_name="unet_name",
            loader_type="UNETLoader"
        ),
        "vae": ModelNodeMapping(
            node_id="39",
            input_name="vae_name",
            loader_type="VAELoader"
        ),
        "clip": ModelNodeMapping(
            node_id="40",
            input_name="clip_name1",
            loader_type="DualCLIPLoader"
        ),
    }

    # Map input fields to template nodes
    input_mapping: ClassVar[Dict[str, NodeInputMapping]] = {
        "prompt": NodeInputMapping(
            node_id="41",
            input_name="text"
        ),
        "width": NodeInputMapping(
            node_id="27",
            input_name="width",
            transform="int"
        ),
        "height": NodeInputMapping(
            node_id="27",
            input_name="height",
            transform="int"
        ),
        "seed": NodeInputMapping(
            node_id="31",
            input_name="seed",
            transform="int"
        ),
        "steps": NodeInputMapping(
            node_id="31",
            input_name="steps",
            transform="int"
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

    unet: FluxUNET = Field(
        default=FluxUNET(value="flux1-schnell.safetensors"),
        description="Flux UNET diffusion model (schnell)"
    )

    vae: FluxVAE = Field(
        default=FluxVAE(value="ae.safetensors"),
        description="Flux VAE autoencoder"
    )

    clip: FluxCLIP = Field(
        default=FluxCLIP(value="t5xxl_fp8_e4m3fn.safetensors"),
        description="Flux CLIP text encoder (T5-XXL FP8)"
    )

    # ========================================================================
    # Input Fields
    # ========================================================================

    prompt: str = Field(
        default="",
        description="Text prompt describing the image to generate"
    )

    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Image width in pixels (must be multiple of 8)"
    )

    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        description="Image height in pixels (must be multiple of 8)"
    )

    seed: int = Field(
        default=0,
        description="Random seed for reproducibility (0 = random)"
    )

    steps: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of denoising steps (Schnell optimized for 4-8 steps)"
    )

    # ========================================================================
    # Output Definition
    # ========================================================================

    async def process(self, context) -> ImageRef:
        """Execute the Flux Schnell workflow and return generated image."""
        result = await super().process(context)
        return result.get("image", ImageRef())
