"""
API endpoints for memory and model management.

Provides endpoints for:
- Listing loaded models and their memory usage
- Unloading specific models or all models
- Clearing GPU cache
- Full memory cleanup
- Getting memory statistics
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.types.memory import (
    LoadedModel,
    LoadedModelsResponse,
    MemoryCleanupResult,
    MemoryStats,
    ModelUnloadResult,
)
from nodetool.workflows.memory_utils import (
    clear_memory_uri_cache,
    get_gpu_memory_usage_mb,
    get_memory_uri_cache_stats,
    get_memory_usage_mb,
    run_gc,
)
from nodetool.workflows.model_registry import get_model_registry

log = get_logger(__name__)
router = APIRouter(prefix="/api/memory", tags=["memory"])


def _get_total_ram_mb() -> Optional[float]:
    """Get total system RAM in MB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return None


def _get_total_gpu_memory_mb() -> Optional[float]:
    """Get total GPU memory in MB."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 * 1024)
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _clear_gpu_cache() -> bool:
    """Clear GPU cache. Returns True if successful."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"Failed to clear GPU cache: {e}")
    return False


@router.get("", response_model=MemoryStats)
async def get_memory_stats(
    user: str = Depends(current_user),
) -> MemoryStats:
    """
    Get current memory statistics.

    Returns RAM usage, GPU memory usage, cache stats, and loaded model info.
    """
    registry = get_model_registry()
    models = registry.list_models()

    # Get GPU memory stats
    gpu_mem = get_gpu_memory_usage_mb()
    if gpu_mem is not None:
        gpu_allocated, gpu_reserved = gpu_mem
    else:
        gpu_allocated, gpu_reserved = None, None

    # Get cache stats
    cache_stats = get_memory_uri_cache_stats()

    return MemoryStats(
        ram_mb=get_memory_usage_mb(),
        ram_total_mb=_get_total_ram_mb(),
        gpu_allocated_mb=gpu_allocated,
        gpu_reserved_mb=gpu_reserved,
        gpu_total_mb=_get_total_gpu_memory_mb(),
        memory_cache_count=cache_stats.get("count", 0),
        loaded_models_count=len(models),
        loaded_models_memory_mb=registry.get_total_memory_mb(),
    )


@router.get("/models", response_model=LoadedModelsResponse)
async def list_loaded_models(
    user: str = Depends(current_user),
) -> LoadedModelsResponse:
    """
    List all loaded models with their memory usage.

    Returns a list of all models currently held in memory with
    their type, device, memory usage, and offload status.
    """
    registry = get_model_registry()
    models = registry.list_models()

    loaded_models = [
        LoadedModel(
            id=m.id,
            type=m.model_type,
            memory_mb=m.memory_mb,
            device=m.device,
            offloaded=m.offloaded,
            model_id=m.model_id,
        )
        for m in models
    ]

    return LoadedModelsResponse(
        models=loaded_models,
        total_memory_mb=registry.get_total_memory_mb(),
    )


@router.delete("/models/{model_id}", response_model=ModelUnloadResult)
async def unload_model(
    model_id: str,
    force: bool = False,
    user: str = Depends(current_user),
) -> ModelUnloadResult:
    """
    Unload a specific model by its ID.

    Args:
        model_id: The ID of the model to unload
        force: If True, unload even if the model is in use

    Returns:
        Result indicating success and memory freed

    Raises:
        404: Model not found
        409: Model is in use (unless force=True)
    """
    registry = get_model_registry()
    model_info = registry.get_model_info(model_id)

    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    if model_info.in_use and not force:
        raise HTTPException(
            status_code=409,
            detail=f"Model '{model_id}' is currently in use. Use force=true to unload anyway.",
        )

    memory_freed = model_info.memory_mb
    success = registry.unregister(model_id)

    if success:
        # Run GC to actually free the memory
        run_gc(f"unload_{model_id}", log_before_after=False)

    return ModelUnloadResult(
        success=success,
        model_id=model_id,
        message="Model unloaded successfully" if success else "Failed to unload model",
        memory_freed_mb=memory_freed if success else 0.0,
    )


@router.post("/models/clear", response_model=MemoryCleanupResult)
async def clear_all_models(
    force: bool = False,
    user: str = Depends(current_user),
) -> MemoryCleanupResult:
    """
    Unload all loaded models.

    Args:
        force: If True, unload even models that are in use

    Returns:
        Result indicating success and resources freed
    """
    registry = get_model_registry()
    models_cleared, memory_freed = registry.clear_all(force=force)

    # Run GC to actually free the memory
    gc_freed = run_gc("clear_all_models", log_before_after=False)

    return MemoryCleanupResult(
        success=True,
        message=f"Cleared {models_cleared} models",
        ram_freed_mb=memory_freed + gc_freed,
        models_unloaded=models_cleared,
        cache_items_cleared=0,
    )


@router.post("/gpu", response_model=MemoryCleanupResult)
async def clear_gpu_cache(
    user: str = Depends(current_user),
) -> MemoryCleanupResult:
    """
    Clear GPU (CUDA) cache only.

    This frees unused GPU memory without unloading models.
    """
    gpu_before = get_gpu_memory_usage_mb()
    success = _clear_gpu_cache()
    gpu_after = get_gpu_memory_usage_mb()

    # Calculate freed memory
    freed = 0.0
    if gpu_before and gpu_after:
        freed = max(0, gpu_before[1] - gpu_after[1])  # Compare reserved memory

    return MemoryCleanupResult(
        success=success,
        message="GPU cache cleared" if success else "No GPU available or cache clear failed",
        ram_freed_mb=freed,
        models_unloaded=0,
        cache_items_cleared=0,
    )


@router.post("/all", response_model=MemoryCleanupResult)
async def full_memory_cleanup(
    force: bool = False,
    user: str = Depends(current_user),
) -> MemoryCleanupResult:
    """
    Perform a full memory cleanup.

    This includes:
    - Unloading all models (respects in_use flag unless force=True)
    - Clearing the memory URI cache
    - Clearing GPU cache
    - Running garbage collection
    """
    # Clear models
    registry = get_model_registry()
    models_cleared, model_memory_freed = registry.clear_all(force=force)

    # Clear URI cache
    cache_items_cleared = clear_memory_uri_cache(log_stats=True)

    # Clear GPU cache
    _clear_gpu_cache()

    # Run GC
    gc_freed = run_gc("full_cleanup", log_before_after=True)

    total_freed = model_memory_freed + gc_freed

    return MemoryCleanupResult(
        success=True,
        message=f"Full cleanup: {models_cleared} models unloaded, {cache_items_cleared} cache items cleared",
        ram_freed_mb=total_freed,
        models_unloaded=models_cleared,
        cache_items_cleared=cache_items_cleared,
    )
