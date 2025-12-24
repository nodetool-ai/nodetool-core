from huggingface_hub import ModelInfo
from huggingface_hub.inference._providers import PROVIDER_T
from pydantic import BaseModel


class UnifiedModel(BaseModel):
    id: str
    type: str | None
    name: str
    repo_id: str | None
    path: str | None = None
    # Optional artifact detection metadata (safetensors/gguf/bin/config).
    artifact_family: str | None = None
    artifact_component: str | None = None
    artifact_confidence: float | None = None
    artifact_evidence: list[str] | None = None
    cache_path: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    description: str | None = None
    readme: str | None = None
    downloaded: bool = False
    size_on_disk: int | None = None
    pipeline_tag: str | None = None
    tags: list[str] | None = None
    has_model_index: bool | None = None
    downloads: int | None = None
    likes: int | None = None
    trending_score: float | None = None


class ModelPack(BaseModel):
    """A curated bundle of models that work together.

    Model packs group related models (e.g., Flux checkpoint + CLIP + T5 + VAE)
    into a single downloadable unit with a clear title and description.
    """

    id: str  # Unique identifier, e.g., "flux_dev_fp8"
    title: str  # User-friendly title, e.g., "Flux Dev FP8"
    description: str  # Explains what this pack is for
    category: str = "image_generation"  # For grouping in UI
    tags: list[str] = []  # Searchable tags
    models: list[UnifiedModel] = []  # The actual models in the pack
    total_size: int | None = None  # Combined size in bytes (computed)

    def compute_total_size(self) -> int:
        """Compute total size from individual model sizes."""
        return sum(m.size_on_disk or 0 for m in self.models)


class RepoPath(BaseModel):
    repo_id: str
    path: str
    downloaded: bool = False


class CachedRepo(BaseModel):
    repo_id: str
    downloaded: bool = False


class CachedFileInfo(BaseModel):
    repo_id: str
    file_name: str
    size_on_disk: int
    model_info: ModelInfo | None = None


CachedFileInfo.model_rebuild()
