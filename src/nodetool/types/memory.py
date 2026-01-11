"""
Types for memory and model management API responses.
"""

from typing import Optional

from pydantic import BaseModel, Field


class LoadedModel(BaseModel):
    """Information about a loaded model/pipeline."""

    id: str = Field(description="Unique identifier (typically node_id)")
    type: str = Field(description="Model/pipeline type name (e.g., 'StableDiffusionXLPipeline')")
    memory_mb: float = Field(description="Estimated memory usage in MB")
    device: str = Field(description="Device location (cuda, cpu, mps)")
    offloaded: bool = Field(default=False, description="Whether CPU offload is enabled")
    model_id: Optional[str] = Field(default=None, description="HuggingFace model ID if applicable")


class LoadedModelsResponse(BaseModel):
    """Response for listing loaded models."""

    models: list[LoadedModel] = Field(default_factory=list)
    total_memory_mb: float = Field(default=0.0, description="Total memory used by all models")


class MemoryStats(BaseModel):
    """Current memory statistics."""

    ram_mb: float = Field(description="Process RAM usage in MB")
    ram_total_mb: Optional[float] = Field(default=None, description="Total system RAM in MB")
    gpu_allocated_mb: Optional[float] = Field(default=None, description="GPU memory allocated in MB")
    gpu_reserved_mb: Optional[float] = Field(default=None, description="GPU memory reserved in MB")
    gpu_total_mb: Optional[float] = Field(default=None, description="Total GPU memory in MB")
    memory_cache_count: int = Field(default=0, description="Items in memory URI cache")
    loaded_models_count: int = Field(default=0, description="Number of loaded models")
    loaded_models_memory_mb: float = Field(default=0.0, description="Total memory used by loaded models")


class MemoryCleanupResult(BaseModel):
    """Result of a memory cleanup operation."""

    success: bool = Field(default=True)
    message: str = Field(default="Cleanup completed")
    ram_freed_mb: float = Field(default=0.0, description="RAM freed in MB")
    models_unloaded: int = Field(default=0, description="Number of models unloaded")
    cache_items_cleared: int = Field(default=0, description="Cache items cleared")


class ModelUnloadResult(BaseModel):
    """Result of unloading a specific model."""

    success: bool = Field(default=True)
    model_id: str = Field(description="ID of the model that was unloaded")
    message: str = Field(default="Model unloaded successfully")
    memory_freed_mb: float = Field(default=0.0, description="Estimated memory freed in MB")
