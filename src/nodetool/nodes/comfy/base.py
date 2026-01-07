"""
Abstract base class for ComfyUI template nodes.

This module provides the ComfyTemplateNode class which handles:
- Loading JSON workflow templates from workflow_templates submodule
- Building ComfyUI graphs by injecting field values into templates
- Uploading images to ComfyUI
- Executing workflows via the ComfyUI API
- Extracting and converting results
"""

import json
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Dict

from pydantic import Field

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)
from nodetool.workflows.base_node import BaseNode

log = get_logger(__name__)


class ComfyTemplateNode(BaseNode, ABC):
    """
    Abstract base class for ComfyUI template nodes.

    Subclasses must define:
    1. template_path: Path to JSON workflow file relative to workflow_templates/templates/
    2. input_mapping: Map field names to template nodes
    3. model_mapping: Map model fields to loader nodes
    4. output_mapping: Map outputs to result extraction

    The base class handles:
    - Loading JSON workflow template
    - Injecting field values into template
    - Uploading images to ComfyUI
    - Executing workflow
    - Extracting and converting results
    """

    # ========================================================================
    # Class Variables (must be set by subclasses)
    # ========================================================================

    template_path: ClassVar[str] = ""  # e.g., "flux/flux_dev_simple.json"

    input_mapping: ClassVar[Dict[str, NodeInputMapping]] = {}
    model_mapping: ClassVar[Dict[str, ModelNodeMapping]] = {}
    output_mapping: ClassVar[Dict[str, NodeOutputMapping]] = {}

    # Cached template per class
    _template_cache: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def is_visible(cls) -> bool:
        """Only show concrete subclasses in UI."""
        return cls is not ComfyTemplateNode and cls.template_path != ""

    # ========================================================================
    # Template Loading
    # ========================================================================

    @classmethod
    def get_template_base_dir(cls) -> Path:
        """Get base directory for workflow templates.

        Returns the path to workflow_templates/templates/ directory,
        which should contain the JSON template files.
        """
        # Navigate from src/nodetool/nodes/comfy/base.py to repo root
        current_file = Path(__file__)
        # Go up: base.py -> comfy -> nodes -> nodetool -> src -> repo_root
        repo_root = current_file.parent.parent.parent.parent.parent
        return repo_root / "workflow_templates" / "templates"

    @classmethod
    def load_template_json(cls) -> Dict[str, Any]:
        """
        Load and cache the ComfyUI workflow JSON template.

        Returns:
            Parsed JSON workflow structure

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template_path not set
        """
        # Use class name as cache key to support inheritance
        cache_key = cls.__name__

        if cache_key in cls._template_cache:
            return cls._template_cache[cache_key]

        if not cls.template_path:
            raise ValueError(
                f"{cls.__name__} must define template_path class variable"
            )

        template_file = cls.get_template_base_dir() / cls.template_path

        if not template_file.exists():
            raise FileNotFoundError(
                f"Template not found: {template_file}\n"
                f"Make sure workflow_templates submodule is initialized:\n"
                f"  git submodule update --init --recursive"
            )

        with open(template_file, encoding="utf-8") as f:
            template = json.load(f)

        cls._template_cache[cache_key] = template
        return template

    # ========================================================================
    # Field Extraction
    # ========================================================================

    def get_model_fields(self) -> Dict[str, Any]:
        """
        Extract model type fields from this node instance.

        Returns:
            Dict mapping field name to ComfyModelFile instance
        """
        from nodetool.metadata.types import ComfyModelFile

        models = {}
        for field_name in self.model_fields:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, ComfyModelFile):
                models[field_name] = field_value
        return models

    def get_input_fields(self) -> Dict[str, Any]:
        """
        Extract non-model input fields from this node instance.

        Returns:
            Dict mapping field name to value (excluding ComfyModelFile fields)
        """
        from nodetool.metadata.types import ComfyModelFile

        inputs = {}
        for field_name in self.model_fields:
            # Skip internal fields
            if field_name.startswith("_"):
                continue

            field_value = getattr(self, field_name, None)

            # Skip model types
            if isinstance(field_value, ComfyModelFile):
                continue

            # Skip None values
            if field_value is None:
                continue

            inputs[field_name] = field_value

        return inputs

    # ========================================================================
    # Graph Building
    # ========================================================================

    def build_comfy_graph(self) -> Dict[str, Any]:
        """
        Build ComfyUI API graph by injecting field values into template.

        Process:
        1. Load base template JSON
        2. Inject model filenames into loader nodes
        3. Inject input values into parameter nodes
        4. Return modified graph

        Returns:
            ComfyUI API-compatible workflow graph
        """
        # Start with template
        template = self.load_template_json()
        graph = json.loads(json.dumps(template))  # Deep copy

        # Inject model filenames
        models = self.get_model_fields()
        for field_name, model_value in models.items():
            if field_name not in self.model_mapping:
                continue

            mapping = self.model_mapping[field_name]
            node_id = mapping.node_id

            if node_id not in graph:
                # Create loader node if not in template
                graph[node_id] = {
                    "inputs": {},
                    "class_type": mapping.loader_type
                }

            # Inject filename
            graph[node_id]["inputs"][mapping.input_name] = model_value.value

        # Inject input values
        inputs = self.get_input_fields()
        for field_name, field_value in inputs.items():
            if field_name not in self.input_mapping:
                continue

            mapping = self.input_mapping[field_name]
            node_id = mapping.node_id

            if node_id not in graph:
                raise ValueError(
                    f"Node {node_id} not found in template for input {field_name}"
                )

            # Apply transformation if specified
            if mapping.transform == "int":
                field_value = int(field_value)
            elif mapping.transform == "float":
                field_value = float(field_value)
            elif mapping.transform == "bool":
                field_value = bool(field_value)
            # image_upload handled separately in execute_comfy

            # Inject value
            graph[node_id]["inputs"][mapping.input_name] = field_value

        return graph

    # ========================================================================
    # Image Upload Handling
    # ========================================================================

    async def prepare_image_inputs(
        self,
        context: Any
    ) -> Dict[str, str]:
        """
        Upload any ImageRef inputs to ComfyUI and return filename mappings.

        Returns:
            Dict mapping field name to uploaded filename
        """
        uploaded = {}

        for field_name, field_value in self.get_input_fields().items():
            if not isinstance(field_value, ImageRef):
                continue

            if field_name not in self.input_mapping:
                continue

            mapping = self.input_mapping[field_name]
            if mapping.transform != "image_upload":
                continue

            # Get image bytes from the ImageRef
            image_data = await context.asset_to_bytes(field_value)

            # Upload to ComfyUI
            filename = await self._upload_image_to_comfy(image_data)
            uploaded[field_name] = filename

        return uploaded

    async def _upload_image_to_comfy(self, image_data: bytes) -> str:
        """
        Upload image to ComfyUI input folder.

        Returns:
            Filename assigned by ComfyUI
        """
        import asyncio
        import base64
        import uuid

        from nodetool.providers.comfy_api import upload_images

        filename = f"nodetool_input_{uuid.uuid4().hex}.png"
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        upload_payload = [{
            "name": filename,
            "image": f"data:image/png;base64,{image_b64}"
        }]

        result = await asyncio.to_thread(upload_images, upload_payload)

        if result.get("status") == "error":
            details = ", ".join(result.get("details", []))
            raise RuntimeError(f"Failed to upload image to ComfyUI: {details}")

        return filename

    # ========================================================================
    # Execution
    # ========================================================================

    async def execute_comfy_graph(
        self,
        context: Any,
        graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute ComfyUI workflow graph.

        Delegates to ComfyTemplateProvider for actual execution,
        reusing WebSocket connection, result extraction, etc.

        Returns:
            Dict mapping output names to result data
        """
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers import get_provider
        from nodetool.providers.comfy_template_provider import ComfyTemplateProvider

        provider = await get_provider(ProviderEnum.ComfyTemplate, context.user_id)
        assert isinstance(provider, ComfyTemplateProvider)

        # Execute graph
        images = await provider._execute_graph(
            graph,
            template_id=self.__class__.__name__
        )

        return {"images": images}

    async def extract_outputs(
        self,
        context: Any,
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract outputs from execution result based on output_mapping.

        Returns:
            Dict mapping output field names to typed results
        """
        outputs = {}

        for field_name, mapping in self.output_mapping.items():
            if mapping.output_type == "image":
                # Get first image
                images = execution_result.get("images", [])
                if images:
                    # Create ImageRef from bytes
                    image_ref = await context.image_from_bytes(images[0])
                    outputs[field_name] = image_ref

            elif mapping.output_type == "video":
                # Get video bytes
                videos = execution_result.get("videos", [])
                if videos:
                    video_ref = await context.video_from_bytes(videos[0])
                    outputs[field_name] = video_ref

        return outputs

    # ========================================================================
    # Main Process Method
    # ========================================================================

    async def process(self, context: Any) -> Dict[str, Any]:
        """
        Main processing method - builds and executes ComfyUI workflow.

        Steps:
        1. Upload any image inputs to ComfyUI
        2. Build graph from template + field values
        3. Inject uploaded image filenames
        4. Execute graph
        5. Extract and convert outputs

        Returns:
            Dict with output field values
        """
        # Upload images if needed
        uploaded_images = await self.prepare_image_inputs(context)

        # Build graph
        graph = self.build_comfy_graph()

        # Inject uploaded image filenames
        for field_name, filename in uploaded_images.items():
            mapping = self.input_mapping[field_name]
            graph[mapping.node_id]["inputs"][mapping.input_name] = filename

        # Execute
        execution_result = await self.execute_comfy_graph(context, graph)

        # Extract outputs
        outputs = await self.extract_outputs(context, execution_result)

        return outputs
