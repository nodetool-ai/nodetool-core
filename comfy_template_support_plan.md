# Plan: YAML-Based Template Mapping for ComfyLocalProvider

## Problem Statement

The current `ComfyLocalProvider` implementation uses **hardcoded workflow graphs**. Instead of building complex runtime analyzers, we will use **coding agents** to create **YAML mapping files** that explicitly define how each template maps to inputs/outputs. Templates remain unchanged as JSON files in a submodule.

## Core Concept: YAML Template Mappings

1. **Templates**: Raw ComfyUI JSON workflows (unchanged, in `workflow_templates/templates/`)
2. **Mappings**: YAML files (one per template) that define:
   - Template type (text_to_image, image_to_image, etc.)
   - Input nodes and fields (what can user provide)
   - Output nodes and fields (what to return)
   - Default values for parameters
3. **Loader**: Simple YAML loader that reads mappings and builds execution graphs

### YAML Mapping Structure

```yaml
template_id: flux_dev_full_text_to_image
template_name: Flux Dev Text to Image
template_type: text_to_image
description: Standard Flux text to image workflow with full parameter control

inputs:
  prompt:
    node_id: 6
    node_type: CLIPTextEncode
    input_field: text
    input_type: STRING
    required: true
    description: "Positive prompt for image generation"

  negative_prompt:
    node_id: 7
    node_type: CLIPTextEncode
    input_field: text
    input_type: STRING
    required: false
    description: "Negative prompt for image generation"

  seed:
    node_id: 31
    node_type: KSampler
    input_field: seed
    input_type: INT
    required: false
    default: 0
    description: "Random seed for reproducibility"

  steps:
    node_id: 31
    node_type: KSampler
    input_field: steps
    input_type: INT
    required: false
    default: 20
    description: "Number of sampling steps"

  cfg:
    node_id: 31
    node_type: KSampler
    input_field: cfg
    input_type: FLOAT
    required: false
    default: 1.0
    description: "Classifier-free guidance scale"

  sampler:
    node_id: 31
    node_type: KSampler
    input_field: sampler_name
    input_type: STRING
    required: false
    default: "euler"
    description: "Sampling algorithm"

  scheduler:
    node_id: 31
    node_type: KSampler
    input_field: scheduler
    input_type: STRING
    required: false
    default: "simple"
    description: "Noise scheduler"

outputs:
  image:
    node_id: 9
    node_type: SaveImage
    output_field: images
    output_type: IMAGE
    description: "Generated image output"

nodes:
  6:
    type: CLIPTextEncode
    class_type: CLIPTextEncode
  7:
    type: CLIPTextEncode
    class_type: CLIPTextEncode
  31:
    type: KSampler
    class_type: KSampler
    widgets_values_order:
      - seed
      - "randomize"
      - steps
      - cfg
      - sampler_name
      - scheduler
      - 1
  8:
    type: VAEDecode
    class_type: VAEDecode
  9:
    type: SaveImage
    class_type: SaveImage
    filename_prefix: "ComfyUI"

presets:
  fast:
    steps: 10
    cfg: 1.0
  quality:
    steps: 30
    cfg: 3.0
```

### Image-to-Image Example

```yaml
template_id: image_qwen_image_edit_2509
template_name: Qwen Image Edit
template_type: image_to_image
description: Image editing with Qwen model and prompt

inputs:
  image:
    node_id: 78
    node_type: LoadImage
    input_field: image
    input_type: IMAGE
    required: true
    description: "Input image to edit"

  prompt:
    node_id: 435
    node_type: PrimitiveStringMultiline
    input_field: value
    input_type: STRING
    required: true
    description: "Edit prompt"

  mask:
    node_id: 79
    node_type: LoadImage
    input_field: image
    input_type: IMAGE
    required: false
    description: "Optional mask for inpainting"

  strength:
    node_id: 82
    node_type: KSampler
    input_field: denoise
    input_type: FLOAT
    required: false
    default: 0.75
    description: "Denoising strength"

outputs:
  image:
    node_id: 60
    node_type: SaveImage
    output_field: images
    output_type: IMAGE

nodes:
  78:
    type: LoadImage
    class_type: LoadImage
    images_directory: "input"
  79:
    type: LoadImage
    class_type: LoadImage
    images_directory: "input"
  435:
    type: PrimitiveStringMultiline
    class_type: PrimitiveStringMultiline
  # ... rest of nodes
```

---

## Workflow

### Step 1: Agent Analyzes Template

Coding agent reads JSON template and identifies:
- Node types and IDs
- Input/output slots
- Widget values and their meanings
- Link structure

### Step 2: Agent Creates YAML Mapping

Agent writes YAML file with explicit mappings:

```yaml
# template_id matches JSON filename (without .json)
template_id: flux_dev_full_text_to_image

inputs:
  # ... explicit input definitions
```

### Step 3: Loader Uses YAML for Execution

```python
class TemplateLoader:
    async def load(self, template_id: str) -> TemplateMapping:
        yaml_path = f"mappings/{template_id}.yaml"
        return parse_yaml(yaml_path)

    async def execute(
        self,
        mapping: TemplateMapping,
        inputs: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        # Build graph from mapping + inputs
        graph = self._build_graph(mapping, inputs, params)
        return await self._execute_graph(graph)
```

---

## Directory Structure

```
workflow_templates/
├── templates/                    # Raw JSON templates (submodule)
│   ├── flux_dev_full_text_to_image.json
│   ├── image_qwen_image_edit_2509.json
│   └── ...
│
src/nodetool/providers/comfy/
    ├── __init__.py
    ├── template_loader.py       # NEW: Load YAML mappings
    ├── template_models.py       # NEW: Pydantic models for YAML
    └── templates/               # YAML mappings (generated)
        ├── flux_dev_full_text_to_image.yaml
        ├── image_qwen_image_edit_2509.yaml
        └── ...
```

---

## YAML Schema Definition

```python
from pydantic import BaseModel, Field
from typing import Literal

class InputMapping(BaseModel):
    node_id: int
    node_type: str
    input_field: str
    input_type: Literal["IMAGE", "STRING", "INT", "FLOAT", "BOOLEAN"]
    required: bool = False
    default: Any = None
    description: str = ""

class OutputMapping(BaseModel):
    node_id: int
    node_type: str
    output_field: str
    output_type: Literal["IMAGE", "VIDEO", "LATENT"]
    description: str = ""

class NodeMapping(BaseModel):
    type: str
    class_type: str
    images_directory: str | None = None
    filename_prefix: str | None = None
    widgets_values_order: list[str] | None = None

class TemplateMapping(BaseModel):
    template_id: str
    template_name: str
    template_type: Literal["text_to_image", "image_to_image", "image_to_video"]
    description: str = ""

    inputs: dict[str, InputMapping]
    outputs: dict[str, OutputMapping]
    nodes: dict[str, NodeMapping]

    presets: dict[str, dict[str, Any]] | None = None
```

---

## Implementation Phases

### Phase 1: Schema and Loader (Week 1)

- [ ] Define Pydantic models in `template_models.py`
- [ ] Implement `TemplateLoader` class in `template_loader.py`
- [ ] Add YAML parsing with validation
- [ ] Write unit tests for loader

### Phase 2: Generate YAML Mappings (Week 1-2)

- [ ] Create coding agent prompt for template analysis
- [ ] Generate YAML mappings for all templates in `workflow_templates/templates/`
- [ ] Review and validate all mappings
- [ ] Add to `src/nodetool/providers/comfy/templates/`

### Phase 3: Execute Method (Week 2)

- [ ] Implement `_build_graph()` using YAML mapping
- [ ] Handle input injection (user values → node inputs)
- [ ] Handle parameter overrides (seeds, steps, etc.)
- [ ] Preserve internal links from original template

### Phase 4: Provider Integration (Week 2-3)

- [ ] Update `ComfyLocalProvider` to use `TemplateLoader`
- [ ] Implement `get_available_image_models()` for image templates
- [ ] Implement `get_available_video_models()` for video templates
- [ ] Implement `text_to_image` via templates
- [ ] Implement `image_to_image` via templates
- [ ] Add fallback to hardcoded logic

### Phase 5: Testing (Week 3)

- [ ] Test loader with all YAML files
- [ ] Test `get_available_image_models()` returns correct templates
- [ ] Test `get_available_video_models()` returns correct templates
- [ ] Test execution with real ComfyUI server
- [ ] Test edge cases (missing inputs, invalid types)
- [ ] Test CLI model listing

---

## Coding Agent Prompt for YAML Generation

```markdown
Analyze the ComfyUI JSON template at {template_path} and create a YAML mapping file.

Task:
1. Read the JSON template
2. Identify all input nodes (LoadImage, CLIPTextEncode, KSampler, etc.)
3. Identify output nodes (SaveImage, VAEDecode, etc.)
4. Map widget_values to parameter names
5. Write YAML to {output_path}

Output format:
```yaml
template_id: {filename_without_extension}
template_name: Human readable name
template_type: text_to_image | image_to_image | image_to_video
description: Brief description

inputs:
  prompt:
    node_id: 6
    node_type: CLIPTextEncode
    input_field: text
    input_type: STRING
    required: true
    description: "Positive prompt"

outputs:
  image:
    node_id: 9
    node_type: SaveImage
    output_field: images
    output_type: IMAGE

nodes:
  6:
    type: CLIPTextEncode
    class_type: CLIPTextEncode
  31:
    type: KSampler
    class_type: KSampler
    widgets_values_order:
      - seed
      - "randomize"
      - steps
      - cfg
      - sampler_name
      - scheduler
      - denoise
```

Guidelines:
- Match node_id from JSON to YAML
- For KSampler, map widgets_values by position to parameter names
- For Primitive nodes, use "value" as input_field
- Use exact class_type from JSON "class_type" field
- Set required: true for essential inputs (prompt, image)
- Set required: false for optional parameters (seed, negative_prompt)
```

---

## Benefits of YAML Approach

| Aspect | YAML Mappings | Runtime Analyzer |
|--------|---------------|------------------|
| Accuracy | Explicit, human-validated | Heuristic, may fail |
| Debugging | Readable YAML | Complex code paths |
| Maintenance | Update YAML per template | Fix analyzer bugs |
| Performance | Fast load | Runtime parsing overhead |
| Coverage | Per-template control | Generic patterns only |

---

## Constraints

1. **One YAML per template**: `flux_dev_full_text_to_image.json` → `flux_dev_full_text_to_image.yaml`
2. **No template modification**: Templates stay in submodule, untouched
3. **Validation required**: YAML must pass Pydantic validation before use
4. **Backward compatible**: Keep hardcoded fallback for unmapped templates

---

## Success Criteria

1. [ ] All templates have corresponding YAML mappings
2. [ ] YAML files pass validation
3. [ ] TemplateLoader correctly builds execution graphs from YAML
4. [ ] `get_available_image_models()` returns all image templates with correct metadata
5. [ ] `get_available_video_models()` returns all video templates with correct metadata
6. [ ] Users can select templates by ID for text_to_image
7. [ ] Users can select templates by ID for image_to_image
8. [ ] Fallback works for templates without YAML

---

## Template as Model Selection

Users should be able to select which template to use for generation. Templates serve as the "model" selection.

### TemplateInfo Model

```python
from pydantic import BaseModel
from typing import Literal

class TemplateInfo(BaseModel):
    template_id: str
    template_name: str
    template_type: Literal["text_to_image", "image_to_image", "image_to_video"]
    description: str
    inputs: list[str]  # Available input keys (e.g., ["prompt", "seed", "steps"])
    outputs: list[str]  # Available output keys (e.g., ["image"])
```

### TemplateLoader Methods for Model Discovery

```python
class TemplateLoader:
    def __init__(self, templates_dir: str = "templates/"):
        self.templates_dir = templates_dir

    async def list_templates(
        self,
        template_types: list[str] | None = None,
    ) -> list[TemplateInfo]:
        """
        List all available templates, optionally filtered by types.

        Args:
            template_types: Filter by template types (e.g., ["text_to_image", "image_to_image"])
                           If None, returns all templates.

        Returns:
            List of TemplateInfo for user selection.
        """
        templates = []
        for yaml_file in Path(self.templates_dir).glob("*.yaml"):
            mapping = self.load(yaml_file.stem)
            if template_types is None or mapping.template_type in template_types:
                templates.append(TemplateInfo(
                    template_id=mapping.template_id,
                    template_name=mapping.template_name,
                    template_type=mapping.template_type,
                    description=mapping.description,
                    inputs=list(mapping.inputs.keys()),
                    outputs=list(mapping.outputs.keys()),
                ))
        return templates

    async def get_template_info(self, template_id: str) -> TemplateInfo | None:
        """Get info for a single template."""
        mapping = self.load(template_id)
        if mapping:
            return TemplateInfo(
                template_id=mapping.template_id,
                template_name=mapping.template_name,
                template_type=mapping.template_type,
                description=mapping.description,
                inputs=list(mapping.inputs.keys()),
                outputs=list(mapping.outputs.keys()),
            )
        return None
```

### Provider Integration for Model Listing

```python
@register_provider(ProviderEnum.ComfyLocal)
class ComfyLocalProvider(BaseProvider):
    provider_name = "comfy_local"

    def __init__(self, secrets: dict[str, str] | None = None):
        super().__init__(secrets)
        self.template_loader = TemplateLoader(
            templates_dir=os.environ.get(
                "COMFY_TEMPLATE_DIR",
                "src/nodetool/providers/comfy/templates/"
            )
        )

    async def get_available_image_models(self) -> list[ModelInfo]:
        """List available image generation templates (text_to_image, image_to_image)."""
        templates = await self.template_loader.list_templates(
            template_types=["text_to_image", "image_to_image"]
        )
        return [
            ModelInfo(
                id=t.template_id,
                name=t.template_name,
                type="image",
                description=t.description,
                capabilities=[t.template_type],
            )
            for t in templates
        ]

    async def get_available_video_models(self) -> list[ModelInfo]:
        """List available video generation templates (image_to_video)."""
        templates = await self.template_loader.list_templates(
            template_types=["image_to_video", "text_to_video"]
        )
        return [
            ModelInfo(
                id=t.template_id,
                name=t.template_name,
                type="video",
                description=t.description,
                capabilities=[t.template_type],
            )
            for t in templates
        ]

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context=None,
        node_id: str | None = None,
    ) -> ImageBytes:
        template_id = params.model.id if params.model else None
        mapping = self.template_loader.load(template_id)
        if not mapping:
            raise ValueError(f"Template not found: {template_id}")

        adapter = TemplateAdapter(mapping)
        inputs = {"prompt": params.prompt}
        params_dict = self._params_to_dict(params)
        return await adapter.execute(self, inputs, params_dict, timeout_s)
```

### Usage Example

```python
# User lists available image models
provider = ComfyLocalProvider()
image_models = await provider.get_available_image_models()
for model in image_models:
    print(f"{model.id}: {model.name}")

# User lists available video models
video_models = await provider.get_available_video_models()
for model in video_models:
    print(f"{model.id}: {model.name}")

# User selects template by ID for text-to-image
params = TextToImageParams(
    model=Model(id="flux_dev_full_text_to_image"),
    prompt="A beautiful sunset",
)
result = await provider.text_to_image(params)

# User selects template by ID for image-to-image
params = ImageToImageParams(
    model=Model(id="image_qwen_image_edit_2509"),
    image=image_bytes,
    prompt="Make it blue",
)
result = await provider.image_to_image(params)
```

### CLI Integration

```bash
# List all image models
nodetool providers image-models comfy_local

# Output:
# flux_dev_full_text_to_image: Flux Dev Text to Image
# flux_schnell_text_to_image: Flux Schnell (Fast)
# image_qwen_image_edit_2509: Qwen Image Edit
# ...

# List all video models
nodetool providers video-models comfy_local

# Output:
# image_to_video_svd: Stable Video Diffusion
# ...
```

---

## File Changes

```
src/nodetool/providers/comfy/
├── __init__.py
├── template_models.py           # NEW: Pydantic models
├── template_loader.py           # NEW: YAML loader and executor
└── templates/                   # NEW: YAML mappings
    ├── flux_dev_full_text_to_image.yaml
    ├── image_qwen_image_edit_2509.yaml
    └── ... (all templates)
```
