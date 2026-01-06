"""
Pydantic models for ComfyUI YAML template mappings.

These models define the schema for YAML files that map ComfyUI JSON workflow
templates to a standardized input/output interface.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class InputMapping(BaseModel):
    """Mapping definition for a template input parameter.

    Defines how user-provided values map to specific nodes and fields
    in the ComfyUI workflow graph.
    """

    node_id: int = Field(description="ID of the node in the ComfyUI workflow")
    node_type: str = Field(description="Type/class of the node (e.g., 'CLIPTextEncode')")
    input_field: str = Field(description="Name of the input field on the node")
    input_type: Literal["IMAGE", "STRING", "INT", "FLOAT", "BOOLEAN"] = Field(
        description="Data type of the input"
    )
    required: bool = Field(default=False, description="Whether this input is required")
    default: Any = Field(default=None, description="Default value if not provided")
    description: str = Field(default="", description="Human-readable description")


class OutputMapping(BaseModel):
    """Mapping definition for a template output.

    Defines which node and output slot to retrieve results from
    after workflow execution.
    """

    node_id: int = Field(description="ID of the output node in the workflow")
    node_type: str = Field(description="Type/class of the output node (e.g., 'SaveImage')")
    output_field: str = Field(description="Name of the output field on the node")
    output_type: Literal["IMAGE", "VIDEO", "LATENT", "AUDIO"] = Field(
        description="Data type of the output"
    )
    description: str = Field(default="", description="Human-readable description")


class NodeMapping(BaseModel):
    """Additional metadata about nodes in the workflow.

    Used to provide extra context about specific nodes that may be needed
    during graph building or execution.
    """

    type: str = Field(description="Logical type name for the node")
    class_type: str = Field(description="ComfyUI class_type identifier")
    images_directory: str | None = Field(
        default=None, description="Directory for image inputs (for LoadImage nodes)"
    )
    filename_prefix: str | None = Field(
        default=None, description="Prefix for output filenames (for SaveImage nodes)"
    )
    widgets_values_order: list[str | int | float] | None = Field(
        default=None,
        description="Order of widget values for positional parameter mapping",
    )


class TemplateMapping(BaseModel):
    """Complete mapping definition for a ComfyUI workflow template.

    Combines template metadata, input/output definitions, and node metadata
    into a single configuration that defines how to execute a workflow.
    """

    template_id: str = Field(description="Unique identifier matching the JSON filename")
    template_name: str = Field(description="Human-readable display name")
    template_type: Literal["text_to_image", "image_to_image", "image_to_video", "text_to_video"] = Field(
        description="Type of generation task this template performs"
    )
    description: str = Field(default="", description="Detailed description of the template")

    inputs: dict[str, InputMapping] = Field(
        description="Map of input names to their node mappings"
    )
    outputs: dict[str, OutputMapping] = Field(
        description="Map of output names to their node mappings"
    )
    nodes: dict[str, NodeMapping] = Field(
        default_factory=dict,
        description="Additional node metadata keyed by node ID",
    )

    presets: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Named parameter presets (e.g., 'fast', 'quality')",
    )


class TemplateInfo(BaseModel):
    """Lightweight template information for model discovery.

    Used by providers to expose available templates as selectable "models"
    without loading the full template mapping.
    """

    template_id: str = Field(description="Unique template identifier")
    template_name: str = Field(description="Human-readable display name")
    template_type: Literal["text_to_image", "image_to_image", "image_to_video", "text_to_video"] = Field(
        description="Type of generation task"
    )
    description: str = Field(default="", description="Template description")
    inputs: list[str] = Field(description="Available input parameter names")
    outputs: list[str] = Field(description="Available output names")
