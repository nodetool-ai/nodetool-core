"""
Image generation types for the ImageProvider system.

This module defines canonical request and response types for text-to-image and image-to-image generation.
"""

from pydantic import BaseModel, Field
from nodetool.metadata.types import (
    ImageModel,
)


ImageBytes = bytes


class TextToImageParams(BaseModel):
    """Parameters for text-to-image generation."""

    model: ImageModel = Field(
        description="Provider and model ID for the text-to-image model"
    )
    prompt: str = Field(description="Text prompt describing the desired image")
    negative_prompt: str | None = Field(
        default=None, description="Text prompt describing what to avoid in the image"
    )
    width: int = Field(default=512, description="Width of the generated image")
    height: int = Field(default=512, description="Height of the generated image")
    guidance_scale: float | None = Field(
        default=None,
        description="Classifier-free guidance scale (higher values = closer to prompt)",
    )
    num_inference_steps: int | None = Field(
        default=None, description="Number of denoising steps"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility (None for random)"
    )
    scheduler: str | None = Field(
        default=None, description="Scheduler to use for generation"
    )
    safety_check: bool | None = Field(
        default=None,
        description="Whether to enable safety checking (provider-specific)",
    )
    image_format: str | None = Field(
        default=None, description="Output image format (png, jpg, etc.)"
    )


class ImageToImageParams(BaseModel):
    """Parameters for image-to-image generation."""

    model: ImageModel = Field(
        description="Provider and model ID for the image-to-image model"
    )
    prompt: str = Field(description="Text prompt describing the desired transformation")
    negative_prompt: str | None = Field(
        default=None, description="Text prompt describing what to avoid"
    )
    guidance_scale: float | None = Field(
        default=None, description="Classifier-free guidance scale"
    )
    num_inference_steps: int | None = Field(
        default=None, description="Number of denoising steps"
    )
    strength: float | None = Field(
        default=None,
        description="Transformation strength (0.0 = original, 1.0 = completely new)",
    )
    target_width: int | None = Field(
        default=None, description="Target width of the output image"
    )
    target_height: int | None = Field(
        default=None, description="Target height of the output image"
    )
    seed: int | None = Field(
        default=None, description="Random seed for reproducibility (None for random)"
    )
    scheduler: str | None = Field(
        default=None, description="Scheduler to use for generation"
    )
