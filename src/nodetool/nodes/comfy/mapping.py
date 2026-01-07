"""
Mapping configuration classes for ComfyUI template nodes.

These classes define how node fields map to ComfyUI template nodes for
inputs, outputs, and model loaders.
"""

from typing import Literal

from pydantic import BaseModel, Field


class NodeInputMapping(BaseModel):
    """
    Maps a node field to a ComfyUI template node input.

    Specifies where in the JSON template to inject the value.
    """

    node_id: str = Field(
        description="Node ID in ComfyUI JSON (e.g., '6', '38')"
    )

    input_name: str = Field(
        description="Input field name in the ComfyUI node (e.g., 'text', 'seed')"
    )

    transform: Literal["direct", "image_upload", "int", "float", "bool"] | None = Field(
        default="direct",
        description="Optional value transformation before injection"
    )


class ModelNodeMapping(BaseModel):
    """
    Maps a model field to a ComfyUI model loader node.

    Specifies which node loads the model and what input receives the filename.
    """

    node_id: str = Field(
        description="Model loader node ID (e.g., '38' for UNETLoader)"
    )

    input_name: str = Field(
        description="Input field for model filename (e.g., 'unet_name', 'ckpt_name')"
    )

    loader_type: str = Field(
        description="ComfyUI loader node type (e.g., 'UNETLoader', 'CheckpointLoader')"
    )


class NodeOutputMapping(BaseModel):
    """
    Maps a ComfyUI template output to a node output field.

    Specifies where to extract results from the executed workflow.
    """

    node_id: str = Field(
        description="Output node ID in ComfyUI JSON (e.g., '9' for SaveImage)"
    )

    output_type: Literal["image", "video", "latent", "audio"] = Field(
        description="Type of output to extract"
    )

    output_name: str = Field(
        default="images",
        description="Output field name in the ComfyUI node"
    )
