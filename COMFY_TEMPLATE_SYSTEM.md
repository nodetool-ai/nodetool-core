# ComfyUI Template Node System - Design Document (Pure Python)

## Executive Summary

This design proposes a **pure Python approach** where each ComfyUI workflow template is represented by a strongly-typed BaseNode subclass. Templates are loaded from the existing JSON workflow files in the `workflow_templates` submodule.  Each node class defines its input/output mapping to the JSON template structure through class-level configuration.  ComfyUI model types are represented as dedicated BaseType subclasses, enabling proper type validation and UI rendering.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ComfyTemplateNode                        │
│                  (Abstract Base Class)                      │
├─────────────────────────────────────────────────────────────┤
│  - template_path: ClassVar[str]                             │
│  - input_mapping: ClassVar[Dict[str, NodeInputMapping]]    │
│  - output_mapping: ClassVar[Dict[str, NodeOutputMapping]]  │
│  - model_mapping: ClassVar[Dict[str, ModelNodeMapping]]    │
│  - load_template_json() → dict                              │
│  - build_comfy_graph() → dict                               │
│  - execute_comfy() → result                                 │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────────────┐  ┌────────────────┐  ┌───────────────┐
│ FluxDevNode   │  │  SDXLBaseNode  │  │  LTXVideoNode │
├───────────────┤  ├────────────────┤  ├───────────────┤
│ prompt:  str   │  │ prompt: str    │  │ prompt: str   │
│ unet: FluxUNET│  │ ckpt:  SDXLCKPT │  │ model: LTXVid │
│ vae: FluxVAE  │  │ vae:  SDXLVAE   │  │ vae: VideoVAE │
│ clip:  FluxCLIP│  │ clip: SDXLCLIP │  │ clip: T5Enc   │
│ width: int    │  │ width: int     │  │ frames: int   │
│ seed: int     │  │ seed: int      │  │ fps: int      │
└───────────────┘  └────────────────┘  └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
    flux_dev. json      sdxl_base.json      ltxv. json
  (workflow_templates/)(workflow_templates/)(workflow_templates/)
```

## Core Components

### 1. ComfyUI Model Type System

**Purpose:** Type-safe model file references with proper validation and UI rendering

```python
# src/nodetool/metadata/types.py

from typing import ClassVar, Literal
from pydantic import Field, field_validator
from nodetool.metadata.types import BaseType


class ComfyModelType(BaseType):
    """
    Base class for all ComfyUI model types. 
    
    Each subclass represents a specific model category and defines: 
    - The ComfyUI folder where models are stored
    - File extensions supported
    - Optional validation rules
    """
    
    type: ClassVar[str] = "comfy. model"
    model_folder: ClassVar[str] = "checkpoints"  # Override in subclasses
    extensions:  ClassVar[list[str]] = [".safetensors", ". ckpt", ".pt", ".pth"]
    
    value: str = Field(
        description="Model filename (e.g., 'flux1-dev. safetensors')"
    )
    
    @field_validator("value")
    @classmethod
    def validate_extension(cls, v: str) -> str:
        """Validate file has correct extension."""
        if not any(v.endswith(ext) for ext in cls.extensions):
            raise ValueError(
                f"Model file must have one of these extensions: {cls.extensions}"
            )
        return v
    
    def get_comfy_path(self, comfy_base_dir: str) -> str:
        """Get full path in ComfyUI directory."""
        return f"{comfy_base_dir}/models/{self.model_folder}/{self.value}"
    
    @classmethod
    def get_folder_path(cls, comfy_base_dir: str) -> str:
        """Get folder path for this model type."""
        return f"{comfy_base_dir}/models/{cls.model_folder}"


# ============================================================================
# Checkpoint Models (Full Models)
# ============================================================================

class ComfyCheckpoint(ComfyModelType):
    """Generic checkpoint model (SD1.5/SD2.x/SDXL)."""
    type: ClassVar[str] = "comfy.checkpoint"
    model_folder: ClassVar[str] = "checkpoints"


class ComfyCheckpointSDXL(ComfyModelType):
    """SDXL checkpoint model."""
    type: ClassVar[str] = "comfy.checkpoint.sdxl"
    model_folder:  ClassVar[str] = "checkpoints"


class ComfyCheckpointSD15(ComfyModelType):
    """Stable Diffusion 1.5 checkpoint."""
    type: ClassVar[str] = "comfy. checkpoint.sd15"
    model_folder: ClassVar[str] = "checkpoints"


# ============================================================================
# UNET Models (Diffusion Models)
# ============================================================================

class ComfyUNET(ComfyModelType):
    """Generic UNET diffusion model."""
    type: ClassVar[str] = "comfy.unet"
    model_folder: ClassVar[str] = "unet"


class FluxUNET(ComfyModelType):
    """Flux UNET model (flux1-dev, flux1-schnell)."""
    type: ClassVar[str] = "comfy.unet.flux"
    model_folder: ClassVar[str] = "unet"


class SDXLUNET(ComfyModelType):
    """SDXL UNET model."""
    type: ClassVar[str] = "comfy.unet.sdxl"
    model_folder: ClassVar[str] = "unet"


class SD3UNET(ComfyModelType):
    """Stable Diffusion 3 UNET."""
    type: ClassVar[str] = "comfy.unet.sd3"
    model_folder: ClassVar[str] = "unet"


# ============================================================================
# VAE Models (Autoencoders)
# ============================================================================

class ComfyVAE(ComfyModelType):
    """Generic VAE model."""
    type: ClassVar[str] = "comfy.vae"
    model_folder: ClassVar[str] = "vae"


class FluxVAE(ComfyModelType):
    """Flux VAE (ae. safetensors)."""
    type: ClassVar[str] = "comfy.vae.flux"
    model_folder: ClassVar[str] = "vae"


class SDXLVAE(ComfyModelType):
    """SDXL VAE."""
    type: ClassVar[str] = "comfy.vae.sdxl"
    model_folder: ClassVar[str] = "vae"


class SD15VAE(ComfyModelType):
    """SD 1.5 VAE."""
    type: ClassVar[str] = "comfy.vae. sd15"
    model_folder: ClassVar[str] = "vae"


# ============================================================================
# CLIP Models (Text Encoders)
# ============================================================================

class ComfyCLIP(ComfyModelType):
    """Generic CLIP text encoder."""
    type: ClassVar[str] = "comfy. clip"
    model_folder:  ClassVar[str] = "clip"


class FluxCLIP(ComfyModelType):
    """Flux CLIP (dual CLIP - T5 + CLIP-L)."""
    type: ClassVar[str] = "comfy. clip.flux"
    model_folder: ClassVar[str] = "clip"


class SDXLCLIP(ComfyModelType):
    """SDXL CLIP (dual CLIP)."""
    type: ClassVar[str] = "comfy.clip.sdxl"
    model_folder: ClassVar[str] = "clip"


class T5TextEncoder(ComfyModelType):
    """T5 text encoder."""
    type: ClassVar[str] = "comfy.clip.t5"
    model_folder: ClassVar[str] = "clip"
    extensions: ClassVar[list[str]] = [".safetensors"]


# ============================================================================
# ControlNet Models
# ============================================================================

class ComfyControlNet(ComfyModelType):
    """Generic ControlNet model."""
    type: ClassVar[str] = "comfy.controlnet"
    model_folder: ClassVar[str] = "controlnet"


class CannyControlNet(ComfyModelType):
    """Canny edge ControlNet."""
    type: ClassVar[str] = "comfy.controlnet.canny"
    model_folder: ClassVar[str] = "controlnet"


class DepthControlNet(ComfyModelType):
    """Depth ControlNet."""
    type: ClassVar[str] = "comfy.controlnet.depth"
    model_folder: ClassVar[str] = "controlnet"


class PoseControlNet(ComfyModelType):
    """OpenPose ControlNet."""
    type: ClassVar[str] = "comfy.controlnet.pose"
    model_folder: ClassVar[str] = "controlnet"


# ============================================================================
# LoRA Models
# ============================================================================

class ComfyLoRA(ComfyModelType):
    """Generic LoRA model."""
    type: ClassVar[str] = "comfy.lora"
    model_folder: ClassVar[str] = "loras"


class FluxLoRA(ComfyModelType):
    """Flux LoRA."""
    type: ClassVar[str] = "comfy.lora.flux"
    model_folder: ClassVar[str] = "loras"


class SDXLLoRA(ComfyModelType):
    """SDXL LoRA."""
    type: ClassVar[str] = "comfy.lora.sdxl"
    model_folder: ClassVar[str] = "loras"


# ============================================================================
# Upscale Models
# ============================================================================

class ComfyUpscaleModel(ComfyModelType):
    """Upscale model (ESRGAN, RealESRGAN, etc.)."""
    type: ClassVar[str] = "comfy. upscale"
    model_folder: ClassVar[str] = "upscale_models"


# ============================================================================
# Video Models
# ============================================================================

class ComfyVideoModel(ComfyModelType):
    """Generic video generation model."""
    type: ClassVar[str] = "comfy. video"
    model_folder:  ClassVar[str] = "video_models"


class LTXVideoModel(ComfyModelType):
    """LTX Video model."""
    type: ClassVar[str] = "comfy. video.ltxv"
    model_folder:  ClassVar[str] = "video_models"


class CogVideoModel(ComfyModelType):
    """CogVideo model."""
    type: ClassVar[str] = "comfy.video.cogvideo"
    model_folder: ClassVar[str] = "video_models"
```

### 2. Template Mapping Configuration

```python
# src/nodetool/nodes/comfy/mapping.py

from typing import Any, Literal
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
    
    transform:  Literal["direct", "image_upload", "int", "float", "bool"] | None = Field(
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
    
    output_type:  Literal["image", "video", "latent", "audio"] = Field(
        description="Type of output to extract"
    )
    
    output_name: str = Field(
        default="images",
        description="Output field name in the ComfyUI node"
    )
```

### 3. Abstract Base Class

```python
# src/nodetool/nodes/comfy/base.py

import json
from abc import ABC
from pathlib import Path
from typing import ClassVar, Dict, Any, Type

from pydantic import Field

from nodetool.metadata.types import ImageRef, VideoRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.nodes.comfy.mapping import (
    NodeInputMapping,
    ModelNodeMapping,
    NodeOutputMapping,
)


class ComfyTemplateNode(BaseNode, ABC):
    """
    Abstract base class for ComfyUI template nodes.
    
    Subclasses must define:
    1. template_path: Path to JSON workflow file
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
    
    template_path: ClassVar[str] = None  # e.g., "flux/flux_dev_simple.json"
    
    input_mapping: ClassVar[Dict[str, NodeInputMapping]] = {}
    model_mapping:  ClassVar[Dict[str, ModelNodeMapping]] = {}
    output_mapping: ClassVar[Dict[str, NodeOutputMapping]] = {}
    
    # Cached template
    _template_json: ClassVar[Dict[str, Any] | None] = None
    
    # ========================================================================
    # Template Loading
    # ========================================================================
    
    @classmethod
    def get_template_base_dir(cls) -> Path:
        """Get base directory for workflow templates."""
        # Assumes workflow_templates submodule is checked out
        repo_root = Path(__file__).parent.parent. parent. parent
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
        if cls._template_json is not None:
            return cls._template_json
        
        if cls.template_path is None:
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
        
        with open(template_file, 'r') as f:
            cls._template_json = json.load(f)
        
        return cls._template_json
    
    # ========================================================================
    # Field Extraction
    # ========================================================================
    
    def get_model_fields(self) -> Dict[str, Any]:
        """
        Extract model type fields from this node instance.
        
        Returns:
            Dict mapping field name to ComfyModelType instance
        """
        from nodetool.metadata.types import ComfyModelType
        
        models = {}
        for field_name, field_info in self.__fields__.items():
            field_value = getattr(self, field_name)
            if isinstance(field_value, ComfyModelType):
                models[field_name] = field_value
        return models
    
    def get_input_fields(self) -> Dict[str, Any]:
        """
        Extract non-model input fields from this node instance.
        
        Returns:
            Dict mapping field name to value (excluding ComfyModelType fields)
        """
        from nodetool.metadata.types import ComfyModelType
        
        inputs = {}
        for field_name, field_info in self.__fields__.items():
            # Skip internal fields
            if field_name. startswith('_'):
                continue
            
            field_value = getattr(self, field_name)
            
            # Skip model types
            if isinstance(field_value, ComfyModelType):
                continue
            
            # Skip None values unless required
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
            graph[node_id]["inputs"][mapping.input_name] = model_value. value
        
        # Inject input values
        inputs = self.get_input_fields()
        for field_name, field_value in inputs.items():
            if field_name not in self.input_mapping:
                continue
            
            mapping = self. input_mapping[field_name]
            node_id = mapping. node_id
            
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
        context: ProcessingContext
    ) -> Dict[str, str]:
        """
        Upload any ImageRef inputs to ComfyUI and return filename mappings.
        
        Returns:
            Dict mapping field name to uploaded filename
        """
        from nodetool.metadata.types import ImageRef
        
        uploaded = {}
        
        for field_name, field_value in self.get_input_fields().items():
            if not isinstance(field_value, ImageRef):
                continue
            
            if field_name not in self.input_mapping:
                continue
            
            mapping = self. input_mapping[field_name]
            if mapping.transform != "image_upload":
                continue
            
            # Get image bytes
            image_data = await field_value.get_data(context)
            
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
        import uuid
        import base64
        from nodetool.providers.comfy_api import upload_images
        
        filename = f"nodetool_input_{uuid.uuid4().hex}.png"
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        
        upload_payload = [{
            "name": filename,
            "image":  f"data:image/png;base64,{image_b64}"
        }]
        
        result = upload_images(upload_payload)
        
        if result.get("status") == "error":
            details = ", ".join(result.get("details", []))
            raise RuntimeError(f"Failed to upload image to ComfyUI: {details}")
        
        return filename
    
    # ========================================================================
    # Execution
    # ========================================================================
    
    async def execute_comfy_graph(
        self,
        context: ProcessingContext,
        graph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute ComfyUI workflow graph.
        
        Delegates to ComfyTemplateProvider for actual execution,
        reusing WebSocket connection, result extraction, etc.
        
        Returns:
            Dict mapping output names to result data
        """
        from nodetool.providers.comfy_template_provider import ComfyTemplateProvider
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers import get_provider
        
        provider = await get_provider(ProviderEnum.ComfyTemplate, context. user_id)
        assert isinstance(provider, ComfyTemplateProvider)
        
        # Execute graph
        images = await provider._execute_graph(
            graph,
            template_id=self.__class__.__name__
        )
        
        return {"images": images}
    
    def extract_outputs(
        self,
        context: ProcessingContext,
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
                    image_asset = context.create_asset_from_data(
                        images[0],
                        content_type="image/png"
                    )
                    outputs[field_name] = image_asset
            
            elif mapping.output_type == "video":
                # Get video bytes
                videos = execution_result.get("videos", [])
                if videos:
                    video_asset = context.create_asset_from_data(
                        videos[0],
                        content_type="video/mp4"
                    )
                    outputs[field_name] = video_asset
        
        return outputs
    
    # ========================================================================
    # Main Process Method
    # ========================================================================
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
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
        execution_result = await self. execute_comfy_graph(context, graph)
        
        # Extract outputs
        outputs = self. extract_outputs(context, execution_result)
        
        return outputs
```

### 4. Concrete Node Example

```python
# src/nodetool/nodes/comfy/flux/flux_dev_simple.py

from pydantic import Field

from nodetool.metadata.types import ImageRef, FluxUNET, FluxVAE, FluxCLIP
from nodetool.nodes.comfy. base import ComfyTemplateNode
from nodetool.nodes. comfy.mapping import (
    NodeInputMapping,
    ModelNodeMapping,
    NodeOutputMapping,
)


class FluxDevSimple(ComfyTemplateNode):
    """
    Flux Dev model for high-quality text-to-image generation.
    
    This node uses the Flux. 1-dev model with a simplified interface.
    Ideal for photorealistic image generation with strong prompt following.
    
    **Template:** workflow_templates/templates/flux/flux_dev_simple.json
    
    **Model Requirements:**
    - UNET: flux1-dev. safetensors (~23GB)
    - VAE: ae. safetensors (~335MB)
    - CLIP: t5xxl_fp16.safetensors (~9. 5GB)
    
    **Recommended Settings:**
    - Steps: 20-30 for quality, 8-12 for speed
    - CFG: 1. 0 (Flux doesn't need high CFG)
    - Sampler: euler or euler_ancestral
    """
    
    # ========================================================================
    # Template Configuration
    # ========================================================================
    
    template_path = "flux/flux_dev_simple. json"
    
    # Map model fields to loader nodes in JSON
    model_mapping = {
        "unet":  ModelNodeMapping(
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
    input_mapping = {
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
    output_mapping = {
        "image": NodeOutputMapping(
            node_id="9",
            output_type="image",
            output_name="images"
        )
    }
    
    # ========================================================================
    # Model Fields
    # ========================================================================
    
    unet:  FluxUNET = Field(
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
        description="Text prompt describing the image to generate"
    )
    
    width: int = Field(
        default=1024,
        ge=256,
        le=2048,
        multiple_of=8,
        description="Image width in pixels (must be multiple of 8)"
    )
    
    height: int = Field(
        default=1024,
        ge=256,
        le=2048,
        multiple_of=8,
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
    # Output
    # ========================================================================
    
    image: ImageRef = Field(
        description="Generated image"
    )
```

### 5. Another Example:  SDXL with ControlNet

```python
# src/nodetool/nodes/comfy/sdxl/sdxl_canny_controlnet.py

from pydantic import Field

from nodetool.metadata.types import (
    ImageRef,
    ComfyCheckpointSDXL,
    CannyControlNet,
)
from nodetool.nodes. comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    NodeInputMapping,
    ModelNodeMapping,
    NodeOutputMapping,
)


class SDXLCannyControlNet(ComfyTemplateNode):
    """
    SDXL with Canny edge ControlNet for controlled generation.
    
    Uses edge detection to guide image generation, maintaining
    structural composition from a reference image.
    """
    
    template_path = "sdxl/sdxl_canny_controlnet.json"
    
    model_mapping = {
        "checkpoint": ModelNodeMapping(
            node_id="4",
            input_name="ckpt_name",
            loader_type="CheckpointLoaderSimple"
        ),
        "controlnet": ModelNodeMapping(
            node_id="17",
            input_name="control_net_name",
            loader_type="ControlNetLoader"
        ),
    }
    
    input_mapping = {
        "prompt": NodeInputMapping(
            node_id="6",
            input_name="text"
        ),
        "negative_prompt": NodeInputMapping(
            node_id="7",
            input_name="text"
        ),
        "control_image": NodeInputMapping(
            node_id="18",
            input_name="image",
            transform="image_upload"
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
        "controlnet_strength": NodeInputMapping(
            node_id="17",
            input_name="strength",
            transform="float"
        ),
    }
    
    output_mapping = {
        "image": NodeOutputMapping(
            node_id="9",
            output_type="image"
        )
    }
    
    # Models
    checkpoint: ComfyCheckpointSDXL = Field(
        default=ComfyCheckpointSDXL(value="sd_xl_base_1.0.safetensors"),
        description="SDXL checkpoint model"
    )
    
    controlnet: CannyControlNet = Field(
        default=CannyControlNet(value="control-lora-canny-rank256.safetensors"),
        description="Canny edge ControlNet model"
    )
    
    # Inputs
    prompt: str = Field(
        description="Text prompt"
    )
    
    negative_prompt: str = Field(
        default="",
        description="Negative prompt"
    )
    
    control_image: ImageRef = Field(
        description="Reference image for edge detection"
    )
    
    seed: int = Field(default=0)
    steps: int = Field(default=20, ge=1, le=100)
    cfg: float = Field(default=7.0, ge=0.0, le=20.0)
    
    controlnet_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="ControlNet influence strength"
    )
    
    # Output
    image: ImageRef = Field(description="Generated image")
```

### 6. Video Generation Example

```python
# src/nodetool/nodes/comfy/video/ltxv_text_to_video.py

from pydantic import Field

from nodetool.metadata.types import VideoRef, LTXVideoModel, T5TextEncoder
from nodetool.nodes. comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    NodeInputMapping,
    ModelNodeMapping,
    NodeOutputMapping,
)


class LTXVideoTextToVideo(ComfyTemplateNode):
    """
    LTX Video model for text-to-video generation.
    
    Generates video clips from text prompts using the LTX Video model.
    """
    
    template_path = "video/ltxv_text_to_video.json"
    
    model_mapping = {
        "model": ModelNodeMapping(
            node_id="68",
            input_name="model_name",
            loader_type="LTXVLoader"
        ),
        "clip":  ModelNodeMapping(
            node_id="67",
            input_name="clip_name",
            loader_type="CLIPLoader"
        ),
    }
    
    input_mapping = {
        "prompt": NodeInputMapping(
            node_id="6",
            input_name="text"
        ),
        "negative_prompt": NodeInputMapping(
            node_id="7",
            input_name="text"
        ),
        "width": NodeInputMapping(
            node_id="70",
            input_name="width",
            transform="int"
        ),
        "height": NodeInputMapping(
            node_id="70",
            input_name="height",
            transform="int"
        ),
        "frames": NodeInputMapping(
            node_id="70",
            input_name="frames",
            transform="int"
        ),
        "seed": NodeInputMapping(
            node_id="72",
            input_name="seed",
            transform="int"
        ),
        "steps": NodeInputMapping(
            node_id="71",
            input_name="steps",
            transform="int"
        ),
    }
    
    output_mapping = {
        "video": NodeOutputMapping(
            node_id="79",
            output_type="video",
            output_name="video"
        )
    }
    
    # Models
    model:  LTXVideoModel = Field(
        default=LTXVideoModel(value="ltxv. safetensors"),
        description="LTX Video model"
    )
    
    clip: T5TextEncoder = Field(
        default=T5TextEncoder(value="t5xxl_fp16.safetensors"),
        description="T5 text encoder"
    )
    
    # Inputs
    prompt: str = Field(description="Text prompt for video")
    negative_prompt: str = Field(default="", description="Negative prompt")
    
    width: int = Field(
        default=768,
        ge=256,
        le=1024,
        multiple_of=8
    )
    
    height: int = Field(
        default=512,
        ge=256,
        le=1024,
        multiple_of=8
    )
    
    frames: int = Field(
        default=97,
        ge=25,
        le=257,
        description="Number of video frames"
    )
    
    seed: int = Field(default=0)
    steps: int = Field(default=30, ge=10, le=100)
    
    # Output
    video: VideoRef = Field(description="Generated video")
```

## Implementation Tasks

### Phase 1: Type System (Week 1)

#### TODO-1: Create ComfyModelType Base Class
**File:** `src/nodetool/metadata/types.py`
- [ ] Define ComfyModelType(BaseType) abstract class
- [ ] Add model_folder, extensions class variables
- [ ] Add value field for filename
- [ ] Add get_comfy_path() method
- [ ] Add get_folder_path() class method
- [ ] Add extension validation

**Acceptance Criteria:**
- ✓ ComfyModelType is BaseType subclass
- ✓ Can serialize/deserialize to/from JSON
- ✓ Extension validation works
- ✓ Path construction correct
- ✓ UI can render model type fields
- ✓ Unit tests cover all methods

#### TODO-2: Create All Model Type Classes
**File:** `src/nodetool/metadata/types. py`
- [ ] Create checkpoint types (3 classes)
- [ ] Create UNET types (4 classes)
- [ ] Create VAE types (4 classes)
- [ ] Create CLIP types (4 classes)
- [ ] Create ControlNet types (4 classes)
- [ ] Create LoRA types (3 classes)
- [ ] Create upscale types (1 class)
- [ ] Create video types (3 classes)

**Acceptance Criteria:**
- ✓ 26 model type classes defined
- ✓ Each has correct model_folder
- ✓ Each has appropriate extensions
- ✓ All follow naming convention
- ✓ Factory methods for common models
- ✓ Tests for each type

#### TODO-3: Register Model Types with Type System
**File:** `src/nodetool/metadata/type_metadata.py`
- [ ] Register all ComfyModelType classes
- [ ] Add type serialization handlers
- [ ] Add type metadata for UI rendering
- [ ] Add validation hooks

**Acceptance Criteria:**
- ✓ All model types in type registry
- ✓ Types serialize correctly
- ✓ Metadata includes folder info
- ✓ UI can query available types
- ✓ Tests verify registration

### Phase 2: Mapping System (Week 1-2)

#### TODO-4: Create Mapping Classes
**File:** `src/nodetool/nodes/comfy/mapping.py`
- [ ] Define NodeInputMapping
- [ ] Define ModelNodeMapping
- [ ] Define NodeOutputMapping
- [ ] Add validation
- [ ] Add helper methods

**Acceptance Criteria:**
- ✓ All mapping classes are Pydantic models
- ✓ Validation enforces required fields
- ✓ Clear documentation with examples
- ✓ Unit tests for each class

#### TODO-5: Create ComfyTemplateNode Base Class
**File:** `src/nodetool/nodes/comfy/base.py`
- [ ] Define abstract base class
- [ ] Implement load_template_json() with caching
- [ ] Implement get_model_fields()
- [ ] Implement get_input_fields()
- [ ] Implement build_comfy_graph()
- [ ] Implement prepare_image_inputs()
- [ ] Implement execute_comfy_graph()
- [ ] Implement extract_outputs()
- [ ] Implement process() orchestration

**Acceptance Criteria:**
- ✓ Base class cannot be instantiated
- ✓ Template loading works with caching
- ✓ Field extraction separates models from inputs
- ✓ Graph building injects all values correctly
- ✓ Image uploads work
- ✓ Execution delegates to provider
- ✓ Output extraction works for image/video
- ✓ Process method orchestrates correctly
- ✓ Error handling comprehensive
- ✓ 90%+ test coverage

### Phase 3: Core Templates (Week 2-3)

#### TODO-6: Flux Templates
**Files:** `src/nodetool/nodes/comfy/flux/*. py`
- [ ] FluxDevSimple
- [ ] FluxSchnellSimple
- [ ] FluxDevFull (all parameters)
- [ ] Flux2DevTextToImage
- [ ] FluxCannyControlNet
- [ ] FluxDepthControlNet
- [ ] FluxReduxStyleTransfer

**Acceptance Criteria:**
- ✓ 7 Flux nodes implemented
- ✓ All use correct template JSON files
- ✓ All mappings correct
- ✓ All have proper documentation
- ✓ All tested end-to-end
- ✓ All registered in node registry

#### TODO-7: SDXL Templates
**Files:** `src/nodetool/nodes/comfy/sdxl/*.py`
- [ ] SDXLBase
- [ ] SDXLRefiner
- [ ] SDXLTurbo
- [ ] SDXLCannyControlNet
- [ ] SDXLDepthControlNet

**Acceptance Criteria:**
- ✓ 5 SDXL nodes implemented
- ✓ Base + Refiner chaining works
- ✓ ControlNet variants work
- ✓ All mappings correct
- ✓ All tested

#### TODO-8: Video Templates
**Files:** `src/nodetool/nodes/comfy/video/*.py`
- [ ] LTXVideoTextToVideo
- [ ] WanImageToVideo

**Acceptance Criteria:**
- ✓ 2 video nodes implemented
- ✓ Video outputs work
- ✓ Frame parameters work
- ✓ All tested

#### TODO-9: Specialized Templates
**Files:** `src/nodetool/nodes/comfy/specialized/*.py`
- [ ] QwenImageEdit
- [ ] QwenImageTextToImage
- [ ] HiDreamI1Fast
- [ ] ImageChromaTextToImage
- [ ] ZImageTurboTextToImage

**Acceptance Criteria:**
- ✓ 5 specialized nodes implemented
- ✓ Multi-image inputs work (Qwen)
- ✓ All mappings correct
- ✓ All tested

### Phase 4: UI Integration (Week 3-4)

#### TODO-10: Model File Discovery API
**File:** `src/nodetool/api/comfy. py`
- [ ] Create GET /api/comfy/models/{model_type} endpoint
- [ ] Query ComfyUI filesystem for models
- [ ] Filter by model type
- [ ] Return list of filenames
- [ ] Handle errors gracefully
- [ ] Add caching

**Acceptance Criteria:**
- ✓ Endpoint returns model files
- ✓ Filters by model_folder
- ✓ Returns 404 for invalid types
- ✓ Returns 503 if ComfyUI down
- ✓ Results cached 60 seconds
- ✓ OpenAPI docs generated

#### TODO-11: Model Selector UI Component
**File:** `web/src/components/properties/ComfyModelSelector.tsx`
- [ ] Create React component for model selection
- [ ] Fetch models from API
- [ ] Display in dropdown
- [ ] Show model type badge
- [ ] Handle loading states
- [ ] Handle errors
- [ ] Update node data on selection

**Acceptance Criteria:**
- ✓ Component renders for ComfyModelType fields
- ✓ Shows model type (UNET, VAE, etc.)
- ✓ Populates dropdown from API
- ✓ Selection updates node
- ✓ Loading spinner shown
- ✓ Error messages displayed
- ✓ Works in node editor

#### TODO-12: Property Rendering Integration
**File:** `web/src/components/properties/PropertyRenderer.tsx`
- [ ] Detect ComfyModelType fields
- [ ] Render ComfyModelSelector
- [ ] Handle nested model type objects
- [ ] Display current selection
- [ ] Show validation errors

**Acceptance Criteria:**
- ✓ ComfyModelType fields use custom selector
- ✓ Regular fields use standard inputs
- ✓ Current values displayed
- ✓ Updates propagate correctly
- ✓ Validation shown inline

#### TODO-13: Node Metadata Enhancement
**File:** `src/nodetool/metadata/node_metadata.py`
- [ ] Add model_type info to property metadata
- [ ] Add model_folder to metadata
- [ ] Add default values to metadata
- [ ] Ensure UI can render correctly

**Acceptance Criteria:**
- ✓ Property metadata includes model info
- ✓ Frontend can determine render type
- ✓ Defaults displayed correctly

### Phase 5: Advanced Features (Week 4-5)

#### TODO-14: LoRA Support
**File:** `src/nodetool/nodes/comfy/base.py` (enhancement)
- [ ] Add optional lora field to nodes
- [ ] Add lora_strength field
- [ ] Update build_comfy_graph() to inject LoraLoader
- [ ] Add lora to relevant templates

**Acceptance Criteria:**
- ✓ Nodes can specify optional LoRA
- ✓ LoRA selector works
- ✓ Strength parameter works
- ✓ Graph builder injects LoraLoader node
- ✓ LoRA applies correctly
- ✓ Tested with Flux/SDXL

#### TODO-15: Batch Processing
**File:** `src/nodetool/nodes/comfy/base.py` (enhancement)
- [ ] Add batch_size field
- [ ] Update graph builder for batch
- [ ] Update output extraction for multiple images
- [ ] Return list[ImageRef]

**Acceptance Criteria:**
- ✓ batch_size parameter works
- ✓ Graph handles batch dimension
- ✓ Multiple images returned
- ✓ UI shows all outputs
- ✓ Tested with various nodes

#### TODO-16: Preset System
**File:** `src/nodetool/nodes/comfy/presets.py`
- [ ] Define preset system
- [ ] Add presets to node classes
- [ ] Create apply_preset() method
- [ ] Add preset selector to UI

**Acceptance Criteria:**
- ✓ Nodes can define presets as class variable
- ✓ Presets update multiple fields
- ✓ UI shows preset dropdown
- ✓ Selecting preset updates form
- ✓ Common presets defined (fast, quality, etc.)

#### TODO-17: Workflow Chaining
**File:** Examples and documentation
- [ ] Document multi-node workflows
- [ ] Create examples with base + refiner
- [ ] Create examples with upscaling
- [ ] Show ControlNet workflows

**Acceptance Criteria:**
- ✓ SDXL base + refiner example
- ✓ Generate + upscale example
- ✓ ControlNet workflow example
- ✓ All examples tested
- ✓ Documentation clear

### Phase 6: Testing & Documentation (Week 5-6)

#### TODO-18: Unit Tests - Type System
**File:** `tests/metadata/test_comfy_model_types.py`
- [ ] Test each model type class
- [ ] Test path construction
- [ ] Test validation
- [ ] Test serialization
- [ ] Test factory methods

**Acceptance Criteria:**
- ✓ Tests for all 26 model types
- ✓ 100% coverage of model type code
- ✓ Tests run in CI

#### TODO-19: Unit Tests - Base Class
**File:** `tests/nodes/comfy/test_base.py`
- [ ] Test template loading
- [ ] Test field extraction
- [ ] Test graph building
- [ ] Test image uploads
- [ ] Test output extraction
- [ ] Mock ComfyUI API

**Acceptance Criteria:**
- ✓ 90%+ coverage of base class
- ✓ All methods tested
- ✓ Error cases covered
- ✓ Mocking comprehensive

#### TODO-20: Integration Tests
**File:** `tests/nodes/comfy/test_integration.py`
- [ ] Test FluxDevSimple end-to-end
- [ ] Test SDXLBase end-to-end
- [ ] Test LTXVideo end-to-end
- [ ] Test ControlNet workflow
- [ ] Test LoRA application
- [ ] Mock ComfyUI server

**Acceptance Criteria:**
- ✓ Full workflow tests for each node type
- ✓ Verify graph structure
- ✓ Verify model injection
- ✓ Verify outputs
- ✓ Tests run in CI

#### TODO-21: Node Tests
**File:** `tests/nodes/comfy/test_all_nodes.py`
- [ ] Test instantiation of all nodes
- [ ] Test default values
- [ ] Test field validation
- [ ] Test metadata generation
- [ ] Test node registration

**Acceptance Criteria:**
- ✓ Test for every node class (30+)
- ✓ Parameterized tests for efficiency
- ✓ Verify all nodes in registry
- ✓ Verify metadata correct

#### TODO-22: Documentation - Architecture
**File:** `docs/comfy_template_system.md`
- [ ] Document architecture overview
- [ ] Document type system
- [ ] Document mapping system
- [ ] Document base class
- [ ] Include diagrams

**Acceptance Criteria:**
- ✓ Clear architecture explanation
- ✓ Diagrams for key concepts
- ✓ Code examples
- ✓ Links to API docs

#### TODO-23: Documentation - Adding Templates
**File:** `docs/adding_comfy_templates.md`
- [ ] Step-by-step guide
- [ ] Template JSON requirements
- [ ] Mapping configuration
- [ ] Testing checklist
- [ ] Complete example

**Acceptance Criteria:**
- ✓ Clear step-by-step process
- ✓ 2+ complete examples
- ✓ Common pitfalls documented
- ✓ Testing guidelines

#### TODO-24: Documentation - Model Types
**File:** `docs/comfy_model_types.md`
- [ ] List all model types
- [ ] Document folder structure
- [ ] Document extensions
- [ ] Document validation
- [ ] UI rendering notes

**Acceptance Criteria:**
- ✓ Complete model type reference
- ✓ Table of all types
- ✓ Folder mappings clear
- ✓ Examples for each category

#### TODO-25: API Documentation
**File:** Docstrings in all classes
- [ ] Add comprehensive docstrings to all public classes
- [ ] Add docstrings to all public methods
- [ ] Add parameter descriptions
- [ ] Add return value descriptions
- [ ] Add examples in docstrings

**Acceptance Criteria:**
- ✓ All public APIs documented
- ✓ Sphinx-compatible format
- ✓ Examples in critical classes
- ✓ API docs build without errors

## Acceptance Criteria Summary

### Functional Requirements

1. **Type System**
   - ✓ 26 ComfyModelType classes defined
   - ✓ All inherit from BaseType
   - ✓ Model folder mapping correct
   - ✓ Extension validation works
   - ✓ Path construction correct
   - ✓ Serialization/deserialization works
   - ✓ UI can render all types

2. **Mapping System**
   - ✓ Node input mapping works
   - ✓ Model node mapping works
   - ✓ Output mapping works
   - ✓ Transformations apply correctly
   - ✓ Image uploads handled

3. **Base Class**
   - ✓ Template loading from JSON works
   - ✓ Template caching works
   - ✓ Field extraction correct
   - ✓ Graph building injects all values
   - ✓ Model filenames injected
   - ✓ Input values injected
   - ✓ Image uploads work
   - ✓ Execution delegates to provider
   - ✓ Output extraction works
   - ✓ Error handling comprehensive

4. **Node Implementation**
   - ✓ 30+ template nodes implemented
   - ✓ All mappings correct
   - ✓ All use correct JSON templates
   - ✓ All have proper types
   - ✓ All have documentation
   - ✓ All registered in registry
   - ✓ All appear in /api/node/metadata
   - ✓ All searchable

5. **Graph Execution**
   - ✓ Generated graphs valid ComfyUI format
   - ✓ Graphs execute successfully
   - ✓ Model files loaded correctly
   - ✓ Parameters applied correctly
   - ✓ Images uploaded correctly
   - ✓ Results extracted correctly

6. **UI Integration**
   - ✓ Model selectors render correctly
   - ✓ Model files populated from API
   - ✓ Selection updates node data
   - ✓ Default values shown
   - ✓ Validation errors displayed
   - ✓ Works in node editor
   - ✓ Works in workflow runner

### Non-Functional Requirements

1. **Performance**
   - ✓ Template loading cached (< 1ms cached)
   - ✓ Graph building < 10ms
   - ✓ Model list API < 500ms
   - ✓ No memory leaks

2. **Reliability**
   - ✓ Handles missing templates
   - ✓ Handles invalid JSON
   - ✓ Handles missing models
   - ✓ Handles ComfyUI unavailable
   - ✓ Clear error messages
   - ✓ Graceful degradation

3. **Maintainability**
   - ✓ Code coverage > 85%
   - ✓ All public APIs documented
   - ✓ Architecture documented
   - ✓ Adding templates documented
   - ✓ Model types documented

4. **Extensibility**
   - ✓ Easy to add new model types
   - ✓ Easy to add new nodes
   - ✓ Base class handles common logic
   - ✓ Clear extension points

## Migration from Current Implementation

### Current State (PR #230)
- Provider-based with YAML templates
- 30 YAML files in `src/nodetool/providers/comfy/templates/`
- Templates as models in provider

### New State
- Node-based with JSON templates
- 30+ Python node classes in `src/nodetool/nodes/comfy/`
- Templates from `workflow_templates/` submodule
- Models as typed fields

### Migration Steps

1. **Keep provider for execution** - ComfyTemplateProvider remains for API calls
2. **Add model types** - New type system alongside existing types
3. **Add base class** - New node base class
4. **Implement nodes** - Create node classes for each template
5. **Update UI** - Add model selectors
6. **Deprecate YAML approach** - Phase out over 2-3 releases

### Backward Compatibility

- Provider API remains functional
- YAML templates kept for reference
- No breaking changes to existing workflows
- Migration guide provided

## Success Metrics

1. **30+ ComfyUI nodes** implemented
2. **26 model types** defined and working
3. **Model selectors** working in UI
4. **85%+ test coverage**
5. **Zero breaking changes** to provider
6. **Documentation complete**
7. **< 150 lines** per node class average
8. **Performance:** Graph building < 10ms

## Timeline

- **Week 1:** Type system + mapping system
- **Week 2:** Base class + Flux templates
- **Week 3:** SDXL/Video templates + UI start
- **Week 4:** UI completion + advanced features
- **Week 5:** Testing
- **Week 6:** Documentation + polish

**Total:** 6 weeks

## Conclusion

This pure Python approach provides:
- ✅ **Type safety** through Pydantic and dedicated type classes
- ✅ **Discovery** via standard node registry
- ✅ **Integration** with workflows and agents
- ✅ **Flexibility** through mapping system
- ✅ **Maintainability** with clear class structure
- ✅ **Extensibility** via base class and type system
- ✅ **Professional UI** with proper model selectors
- ✅ **Reusability** of existing JSON templates

The design eliminates YAML as an intermediate format, instead using the source JSON templates directly with Python-based mapping configuration. 
