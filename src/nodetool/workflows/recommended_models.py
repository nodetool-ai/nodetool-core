from __future__ import annotations

import platform
from typing import Iterable

from nodetool.types.model import UnifiedModel


def get_recommended_models() -> dict[str, list[UnifiedModel]]:
    """Aggregate recommended models from all installed packages.

    Returns a map of repo_id -> models for that repo. Models are
    de-duplicated by `id` across all packages.
    """
    from nodetool.packages.registry import Registry

    registry = Registry()
    node_metadata_list = registry.get_all_installed_nodes()
    model_ids: set[str] = set()
    models: dict[str, list[UnifiedModel]] = {}

    for meta in node_metadata_list:
        for model in meta.recommended_models:
            if model is None or not isinstance(model, UnifiedModel):
                continue
            if model.id in model_ids:
                continue
            model_ids.add(model.id)
            if model.repo_id is None:
                continue
            models.setdefault(model.repo_id, []).append(model)

    return models


def _flatten_unique(models: dict[str, list[UnifiedModel]]) -> list[UnifiedModel]:
    """Flatten dict of models and de-dupe by id preserving order."""
    seen: set[str] = set()
    out: list[UnifiedModel] = []
    for group in models.values():
        for m in group:
            if m.id in seen:
                continue
            seen.add(m.id)
            out.append(m)
    return out


def _is_mlx_model(m: UnifiedModel) -> bool:
    t = (m.type or "").lower()
    if t == "mlx":
        return True
    # Fallback: tag hint
    tags = [tag.lower() for tag in (m.tags or [])]
    return "mlx" in tags or "mflux" in tags


def _is_lora_model(m: UnifiedModel) -> bool:
    """Return True if the model looks like a LoRA/adapter, not a full gen model.

    We exclude LoRA repos from generic image/text-to-image/image-to-image lists,
    since these are auxiliary weights used with base models, not standalone models.
    """
    t = (m.type or "").lower()
    if t in {"hf.lora_sd", "hf.lora_sdxl", "hf.lora_qwen_image"}:
        return True
    tags = [tag.lower() for tag in (m.tags or [])]
    if any("lora" in tag for tag in tags):
        return True
    ident = f"{m.id} {m.name} {m.repo_id}".lower()
    return "lora" in ident


def _platform_allows_model(m: UnifiedModel, system: str | None = None) -> bool:
    """Platform-aware filter.

    - On macOS: prefer MLX. Only include MLX; if none exist, caller can fallback.
    - On Windows/Linux: exclude MLX entries.
    """
    sysname = (system or platform.system()).lower()
    is_mlx = _is_mlx_model(m)
    if sysname == "darwin":
        # On macOS, include both MLX and non-MLX models
        return True
    # Windows/Linux/other: exclude MLX entries (not supported off Apple Silicon)
    return not is_mlx


def _is_image_model(m: UnifiedModel) -> bool:
    """Heuristically determine if a model is for image generation/manipulation."""
    t = (m.type or "").lower()
    pipe = (m.pipeline_tag or "").lower()
    # Pipeline tags that indicate image gen/edit
    image_pipelines = {
        "text-to-image",
        "image-to-image",
        "unconditional-image-generation",
    }
    if _is_lora_model(m):
        return False
    if pipe in image_pipelines:
        return True
    # Known type strings for image-capable repos
    image_types = {
        "hf.text_to_image",
        "hf.image_to_image",
        "hf.flux",
        "hf.stable_diffusion",
        "hf.stable_diffusion_xl",
        "hf.stable_diffusion_3",
        "hf.stable_diffusion_upscale",
        "hf.pixart_alpha",
    }
    # MLX image repos typically have text-to-image pipeline tags (handled above).
    return t in image_types


def _is_text_to_image_model(m: UnifiedModel) -> bool:
    """Identify text-to-image capable repos based on pipeline_tag or tags."""
    pipe = (m.pipeline_tag or "").lower()
    if _is_lora_model(m):
        return False
    if pipe in {"text-to-image", "unconditional-image-generation"}:
        return True
    tags = [t.lower() for t in (m.tags or [])]
    return any(t in {"text-to-image", "unconditional-image-generation"} for t in tags)


def _is_image_to_image_model(m: UnifiedModel) -> bool:
    pipe = (m.pipeline_tag or "").lower()
    if _is_lora_model(m):
        return False
    if pipe == "image-to-image":
        return True
    tags = [t.lower() for t in (m.tags or [])]
    return "image-to-image" in tags


def _is_language_model(m: UnifiedModel) -> bool:
    t = (m.type or "").lower()
    pipe = (m.pipeline_tag or "").lower()
    if pipe in {"text-generation", "text2text-generation", "conversational"}:
        return True
    # MLX includes other modalities; rely on pipeline_tag when available.
    # If pipeline_tag missing, treat generic MLX as language-capable by default.
    return t in {"llama_cpp", "hf.text_generation", "hf.text_to_text", "mlx"}


def _is_language_embedding_model(m: UnifiedModel) -> bool:
    """Heuristic to identify embedding-capable language models."""
    pipe = (m.pipeline_tag or "").lower()
    if pipe in {"feature-extraction", "sentence-similarity", "text-embedding"}:
        return True
    tags = [t.lower() for t in (m.tags or [])]
    patterns = {
        "embedding",
        "embeddings",
        "sentence-transformers",
        "sentence_transformers",
        "feature-extraction",
        "text-embedding",
    }
    if any(p in tags for p in patterns):
        return True
    # Name/id heuristics for common embed repos
    ident = f"{m.id} {m.name} {m.repo_id}".lower()
    hints = [
        "embed",
        "text-embedding",
        "all-minilm",
        "nomic-embed",
        "mxbai-embed",
        "bge-",
        "gte-",
        "e5-",
    ]
    return any(h in ident for h in hints)


def _is_language_text_generation_model(m: UnifiedModel) -> bool:
    """Text generation if it's a language model and not classified as embedding."""
    if not _is_language_model(m):
        return False
    return not _is_language_embedding_model(m)


def _is_asr_model(m: UnifiedModel) -> bool:
    t = (m.type or "").lower()
    pipe = (m.pipeline_tag or "").lower()
    return pipe == "automatic-speech-recognition" or t == "hf.automatic_speech_recognition"


def _is_tts_model(m: UnifiedModel) -> bool:
    t = (m.type or "").lower()
    pipe = (m.pipeline_tag or "").lower()
    return pipe == "text-to-speech" or t == "hf.text_to_speech"


def _filter_models(
    models: Iterable[UnifiedModel],
    *,
    predicate,
    system: str | None = None,
) -> list[UnifiedModel]:
    """Filter, platform-gate, and de-dupe models preserving order."""
    out: list[UnifiedModel] = []
    seen: set[str] = set()
    for m in models:
        if m.id in seen:
            continue
        if not _platform_allows_model(m, system=system):
            continue
        if not predicate(m):
            continue
        seen.add(m.id)
        out.append(m)
    # No additional fallback needed now that macOS includes non-MLX as well
    return out


def get_recommended_models_flat() -> list[UnifiedModel]:
    """Return a flattened, de-duplicated list of all recommended models."""
    return _flatten_unique(get_recommended_models())


def get_recommended_image_models(system: str | None = None) -> list[UnifiedModel]:
    """All image-capable recommended models across installed packages.

    Platform-aware: Mac returns MLX if available; Windows/Linux exclude MLX.
    """
    return _filter_models(
        get_recommended_models_flat(), predicate=_is_image_model, system=system
    )


def get_recommended_text_to_image_models(system: str | None = None) -> list[UnifiedModel]:
    return [
        m for m in get_recommended_image_models(system) if _is_text_to_image_model(m)
    ]


def get_recommended_image_to_image_models(system: str | None = None) -> list[UnifiedModel]:
    return [
        m for m in get_recommended_image_models(system) if _is_image_to_image_model(m)
    ]


def _is_text_to_video_model(m: UnifiedModel) -> bool:
    pipe = (m.pipeline_tag or "").lower()
    if pipe == "text-to-video":
        return True
    tags = [t.lower() for t in (m.tags or [])]
    return "text-to-video" in tags


def _is_image_to_video_model(m: UnifiedModel) -> bool:
    pipe = (m.pipeline_tag or "").lower()
    if pipe == "image-to-video":
        return True
    tags = [t.lower() for t in (m.tags or [])]
    return "image-to-video" in tags


def get_recommended_text_to_video_models(system: str | None = None) -> list[UnifiedModel]:
    # Reuse generic aggregation then filter by tag
    models = get_recommended_models_flat()
    return _filter_models(models, predicate=_is_text_to_video_model, system=system)


def get_recommended_image_to_video_models(system: str | None = None) -> list[UnifiedModel]:
    models = get_recommended_models_flat()
    return _filter_models(models, predicate=_is_image_to_video_model, system=system)


def get_recommended_language_models(system: str | None = None) -> list[UnifiedModel]:
    """All language-capable recommended models across installed packages.

    Platform-aware: Mac returns MLX if available; Windows/Linux exclude MLX.
    """
    return _filter_models(
        get_recommended_models_flat(), predicate=_is_language_model, system=system
    )


def get_recommended_language_text_generation_models(system: str | None = None) -> list[UnifiedModel]:
    return [
        m
        for m in get_recommended_language_models(system)
        if _is_language_text_generation_model(m)
    ]


def get_recommended_language_embedding_models(system: str | None = None) -> list[UnifiedModel]:
    return [
        m
        for m in get_recommended_language_models(system)
        if _is_language_embedding_model(m)
    ]


def get_recommended_asr_models(system: str | None = None) -> list[UnifiedModel]:
    """All ASR-capable recommended models across installed packages.

    Platform-aware: Mac returns MLX if available; Windows/Linux exclude MLX.
    """
    return _filter_models(
        get_recommended_models_flat(), predicate=_is_asr_model, system=system
    )


def get_recommended_tts_models(system: str | None = None) -> list[UnifiedModel]:
    """All TTS-capable recommended models across installed packages.

    Platform-aware: Mac returns MLX if available; Windows/Linux exclude MLX.
    """
    return _filter_models(
        get_recommended_models_flat(), predicate=_is_tts_model, system=system
    )


# Platform-agnostic variants for package metadata (include all models)
def _filter_models_no_platform(models: Iterable[UnifiedModel], *, predicate) -> list[UnifiedModel]:
    out: list[UnifiedModel] = []
    seen: set[str] = set()
    for m in models:
        if m.id in seen:
            continue
        if not predicate(m):
            continue
        seen.add(m.id)
        out.append(m)
    return out


def get_all_recommended_image_models() -> list[UnifiedModel]:
    """All image-capable models from all packages without platform filtering."""
    return _filter_models_no_platform(
        get_recommended_models_flat(), predicate=_is_image_model
    )


def get_all_recommended_language_models() -> list[UnifiedModel]:
    """All language-capable models from all packages without platform filtering."""
    return _filter_models_no_platform(
        get_recommended_models_flat(), predicate=_is_language_model
    )


def get_all_recommended_asr_models() -> list[UnifiedModel]:
    """All ASR-capable models from all packages without platform filtering."""
    return _filter_models_no_platform(
        get_recommended_models_flat(), predicate=_is_asr_model
    )


def get_all_recommended_tts_models() -> list[UnifiedModel]:
    """All TTS-capable models from all packages without platform filtering."""
    return _filter_models_no_platform(
        get_recommended_models_flat(), predicate=_is_tts_model
    )
