"""
Flux Dev Simple node for text-to-image generation.

This node uses the Flux.1-dev model with a simplified interface for
high-quality photorealistic image generation.
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


class FluxDevSimple(ComfyTemplateNode):
    """
    Flux Dev model for high-quality text-to-image generation.

    This node uses the Flux.1-dev model with a simplified interface.
    Ideal for photorealistic image generation with strong prompt following.

    **Template:** workflow_templates/templates/flux/flux_dev_simple.json

    **Model Requirements:**
    - UNET: flux1-dev.safetensors (~23GB)
    - VAE: ae.safetensors (~335MB)
    - CLIP: t5xxl_fp16.safetensors (~9.5GB)

    **Recommended Settings:**
    - Steps: 20-30 for quality, 8-12 for speed
    - CFG: 1.0 (Flux doesn't need high CFG)
    - Sampler: euler or euler_ancestral
    """

    # ========================================================================
    # Template Configuration
    # ========================================================================

    template_path: ClassVar[str] = "flux/flux_dev_simple.json"

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
        "cfg": NodeInputMapping(
            node_id="31",
            input_name="cfg",
            transform="float"
        ),
        "sampler": NodeInputMapping(
            node_id="31",
            input_name="sampler_name"
        ),
        "scheduler": NodeInputMapping(
            node_id="31",
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

    unet: FluxUNET = Field(
        default=FluxUNET(value="flux1-dev.safetensors"),
        description="Flux UNET diffusion model"
    )

    vae: FluxVAE = Field(
        default=FluxVAE(value="ae.safetensors"),
        description="Flux VAE autoencoder"
    )

    clip: FluxCLIP = Field(
        default=FluxCLIP(value="t5xxl_fp16.safetensors"),
        description="Flux CLIP text encoder (T5-XXL)"
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
        default=20,
        ge=1,
        le=100,
        description="Number of denoising steps (more = higher quality but slower)"
    )

    cfg: float = Field(
        default=1.0,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (Flux works well at 1.0)"
    )

    sampler: str = Field(
        default="euler",
        description="Sampling algorithm (euler, euler_ancestral, heun, etc.)"
    )

    scheduler: str = Field(
        default="simple",
        description="Noise scheduler (simple, normal, karras, exponential)"
    )

    # ========================================================================
    # Output Definition
    # ========================================================================

    # The return type defines the output
    async def process(self, context) -> ImageRef:
        """Execute the Flux Dev workflow and return generated image."""
        result = await super().process(context)
        return result.get("image", ImageRef())
