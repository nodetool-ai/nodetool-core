"""
Offline-first Hugging Face model discovery and typing utilities.

The workflow implemented here follows a predictable sequence:
1. Enumerate cached repositories and snapshots via `HfFastCache` without hitting the hub.
2. Read lightweight metadata (sizes, filenames, artifacts) from the local snapshot to
   build `UnifiedModel` records for repos and individual files.
3. Infer model type and task:
   - Prefer package-provided recommendations.
   - Use hub metadata when available (pipeline tags, tags, diffusers config `_class_name`).
   - Fall back to local `model_index.json` / `config.json` parsing and artifact inspection
     so we can classify models even when offline or gated.
4. Provide targeted search helpers (by hf.* type, repo/file patterns, artifact hints) for
   workflows and UI consumers.
5. Expose convenience lookups for common runtimes (text-to-image, llama.cpp, vLLM, MLX).

Wherever possible, the code avoids expensive I/O and network calls, preferring cached
information and shallow file inspections to keep UI interactions fast and reliable.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import aiofiles

if TYPE_CHECKING:
    from huggingface_hub import HfApi, ModelInfo

    from nodetool.integrations.huggingface.artifact_inspector import (
        ArtifactDetection,
        inspect_paths,
    )

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.async_downloader import async_hf_download
from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache
from nodetool.metadata.types import (
    CLASSNAME_TO_MODEL_TYPE,
    HuggingFaceModel,
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.security.secret_helper import get_secret
from nodetool.types.model import UnifiedModel
from nodetool.workflows.recommended_models import get_recommended_models

# Extensions that identify repo-root single-file diffusion checkpoints we surface directly.
SINGLE_FILE_DIFFUSION_EXTENSIONS = (
    ".safetensors",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
    ".svdq",
)

# Tags that hint at single-file diffusion checkpoints when hub metadata is present.
SINGLE_FILE_DIFFUSION_TAGS = {
    "diffusers",
    "diffusers:stablediffusionpipeline",
    "diffusers:stablediffusionxlpipeline",
    "diffusers:stablediffusion3pipeline",
    "diffusion-single-file",
    "stable-diffusion",
    "flux",
}

log = get_logger(__name__)

# Default globs used when scanning repos for general-purpose weight files.
HF_DEFAULT_FILE_PATTERNS = [
    "*.safetensors",
    "*.ckpt",
    "*.gguf",
    "*.bin",
    "*.svdq",
]

# Extra globs for torch weights common in control/adapters.
HF_PTH_FILE_PATTERNS = ["*.pth", "*.pt"]

# Known repo-id allowlists for supported model families.
# These repos are recognized for offline type matching without hub metadata.
KNOWN_REPO_PATTERNS = {
    "flux": [
        "Comfy-Org/flux1-dev",
        "Comfy-Org/flux1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ],
    "flux_kontext": [
        "black-forest-labs/FLUX.1-Kontext-dev",
        "nunchaku-tech/nunchaku-flux-kontext",
    ],
    "flux_canny": [
        "black-forest-labs/FLUX.1-Canny-dev",
        "nunchaku-tech/nunchaku-flux.1-canny-dev",
    ],
    "flux_depth": [
        "black-forest-labs/FLUX.1-Depth-dev",
        "nunchaku-tech/nunchaku-flux.1-depth-dev",
    ],
    "flux_vae": ["ffxvs/vae-flux"],
    "qwen_image": [
        "Comfy-Org/Qwen-Image_ComfyUI",
        "city96/Qwen-Image-gguf",
        "nunchaku-tech/nunchaku-qwen-image",
    ],
    "qwen_image_edit": ["Comfy-Org/Qwen-Image-Edit_ComfyUI"],
    "sd35": ["Comfy-Org/stable-diffusion-3.5-fp8"],
}

# Map hf.* types to repo-id allowlists so type matching can succeed offline.
KNOWN_TYPE_REPO_MATCHERS: dict[str, list[str]] = {
    "hf.flux": [*KNOWN_REPO_PATTERNS["flux"]],
    "hf.flux_fp8": [*KNOWN_REPO_PATTERNS["flux"]],
    "hf.flux_kontext": [*KNOWN_REPO_PATTERNS["flux_kontext"]],
    "hf.flux_canny": [*KNOWN_REPO_PATTERNS["flux_canny"]],
    "hf.flux_depth": [*KNOWN_REPO_PATTERNS["flux_depth"]],
    "hf.stable_diffusion_3": [*KNOWN_REPO_PATTERNS["sd35"]],
    "hf.qwen_image": [
        *KNOWN_REPO_PATTERNS["qwen_image"],
        *KNOWN_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.qwen_image_edit": [*KNOWN_REPO_PATTERNS["qwen_image_edit"]],
    "hf.qwen_vl": [
        *KNOWN_REPO_PATTERNS["qwen_image"],
        *KNOWN_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.unet": [
        *KNOWN_REPO_PATTERNS["flux"],
        *KNOWN_REPO_PATTERNS["flux_kontext"],
        *KNOWN_REPO_PATTERNS["flux_canny"],
        *KNOWN_REPO_PATTERNS["flux_depth"],
        *KNOWN_REPO_PATTERNS["qwen_image"],
        *KNOWN_REPO_PATTERNS["qwen_image_edit"],
        *KNOWN_REPO_PATTERNS["sd35"],
    ],
    "hf.vae": [
        *KNOWN_REPO_PATTERNS["flux_vae"],
        *KNOWN_REPO_PATTERNS["qwen_image"],
        *KNOWN_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.clip": [
        *KNOWN_REPO_PATTERNS["sd35"],
        *KNOWN_REPO_PATTERNS["qwen_image"],
        *KNOWN_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.t5": [*KNOWN_REPO_PATTERNS["sd35"]],
}

# Base → checkpoint variant mapping so heuristics propagate to single-file checkpoints.
_CHECKPOINT_BASES = {
    "hf.stable_diffusion": "hf.stable_diffusion_checkpoint",
    "hf.stable_diffusion_xl": "hf.stable_diffusion_xl_checkpoint",
    "hf.stable_diffusion_3": "hf.stable_diffusion_3_checkpoint",
    "hf.stable_diffusion_xl_refiner": "hf.stable_diffusion_xl_refiner_checkpoint",
    "hf.flux": "hf.flux_checkpoint",
    "hf.flux_kontext": "hf.flux_kontext_checkpoint",
    "hf.flux_canny": "hf.flux_canny_checkpoint",
    "hf.flux_depth": "hf.flux_depth_checkpoint",
    "hf.qwen_image": "hf.qwen_image_checkpoint",
    "hf.qwen_image_edit": "hf.qwen_image_edit_checkpoint",
}

# Keyword hints across repo id/tags/paths to associate models with hf.* types.
HF_TYPE_KEYWORD_MATCHERS: dict[str, list[str]] = {
    "hf.stable_diffusion": ["stable-diffusion", "sd15"],
    "hf.stable_diffusion_xl": ["sdxl", "stable-diffusion-xl"],
    "hf.stable_diffusion_xl_refiner": ["refiner", "sdxl"],
    "hf.stable_diffusion_3": ["sd3", "stable-diffusion-3"],
    "hf.flux": ["flux"],
    "hf.flux_fp8": ["flux", "fp8"],
    "hf.flux_kontext": ["flux", "kontext", "nunchaku"],
    "hf.flux_canny": ["flux", "canny", "nunchaku"],
    "hf.flux_depth": ["flux", "depth", "nunchaku"],
    "hf.qwen_image": ["qwen", "nunchaku"],
    "hf.qwen_image_edit": ["qwen"],
    "hf.qwen_vl": ["vl", "text_encoder", "text-encoder", "qwen"],
    "hf.controlnet": ["control"],
    "hf.controlnet_sdxl": ["control", "sdxl"],
    "hf.controlnet_flux": ["control", "flux"],
    "hf.ip_adapter": ["ip-adapter"],
    "hf.lora_sd": ["lora"],
    "hf.lora_sdxl": ["lora", "sdxl"],
    "hf.lora_qwen_image": ["lora", "qwen"],
    "hf.vae": ["vae"],
    "hf.unet": ["unet"],
    "hf.clip": ["clip"],
    "hf.t5": ["t5"],
    "hf.flux_redux": ["flux", "redux"],
    "hf.real_esrgan": ["esrgan", "real-esrgan"],
}
# Copy keyword matchers to checkpoint variants.
for _base, _ckpt in _CHECKPOINT_BASES.items():
    if _base in HF_TYPE_KEYWORD_MATCHERS and _ckpt not in HF_TYPE_KEYWORD_MATCHERS:
        HF_TYPE_KEYWORD_MATCHERS[_ckpt] = list(HF_TYPE_KEYWORD_MATCHERS[_base])


class RepoPackagingHint(str, Enum):
    """
    Hint describing how a HF repo should be presented to the user.

    - `REPO_BUNDLE` means treat the repo as a single installable unit (typical diffusers).
    - `PER_FILE` exposes individual weights (gguf, adapters, quant variants).
    - `UNKNOWN` indicates insufficient signal; callers can pick a sensible default.
    """

    REPO_BUNDLE = "repo_bundle"  # Treat repo as a single unit (diffusers-style)
    PER_FILE = "per_file"  # Present independent model files (gguf, loras, adapters)
    UNKNOWN = "unknown"  # Not enough signal to decide


async def get_hf_token(user_id: str | None = None) -> str | None:
    """
    Resolve an HF access token from env or per-user secrets.

    This keeps hub calls working for gated models without forcing callers to
    know where tokens are stored. The lookup is async because secrets may live
    in a database behind an async provider.
    """

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    if user_id:
        return await get_secret("HF_TOKEN", user_id)
    return None


# Fast HF cache view for local snapshot lookups.
HF_FAST_CACHE = HfFastCache()

# GGUF_MODELS_FILE = Path(__file__).parent / "gguf_models.json"
# MLX_MODELS_FILE = Path(__file__).parent / "mlx_models.json"

# Map transformer `model_type` values to hf.* types when configs are parsed offline.
_CONFIG_MODEL_TYPE_MAPPING = {
    "whisper": "hf.automatic_speech_recognition",
    "automatic-speech-recognition": "hf.automatic_speech_recognition",
    "audio-classification": "hf.audio_classification",
    "zero-shot-audio-classification": "hf.zero_shot_audio_classification",
    "image-classification": "hf.image_classification",
    "zero-shot-image-classification": "hf.zero_shot_image_classification",
    "image-segmentation": "hf.image_segmentation",
    "depth-estimation": "hf.depth_estimation",
    "object-detection": "hf.object_detection",
    "zero-shot-object-detection": "hf.zero_shot_object_detection",
    "visual-question-answering": "hf.visual_question_answering",
    "question-answering": "hf.question_answering",
    "table-question-answering": "hf.table_question_answering",
    "text-classification": "hf.text_classification",
    "zero-shot-classification": "hf.zero_shot_classification",
    "token-classification": "hf.token_classification",
    "feature-extraction": "hf.feature_extraction",
    "fill-mask": "hf.fill_mask",
    "text2text-generation": "hf.text2text_generation",
    "translation": "hf.translation",
    "image-text-to-text": "hf.image_text_to_text",
    "sentence-similarity": "hf.sentence_similarity",
    "reranker": "hf.reranker",
    "real-esrgan": "hf.real_esrgan",
    "flux-redux": "hf.flux_redux",
    "text-generation": "hf.text_generation",
    "text-to-audio": "hf.text_to_audio",
    "text-to-speech": "hf.text_to_speech",
    "llama": "hf.text_generation",
    "gemma3": "hf.text_generation",
    "gemma3n": "hf.text_generation",
    "qwen2": "hf.text_generation",
    "qwen3": "hf.text_generation",
    "qwen_2_5_vl": "hf.text_generation",
    "mistral3": "hf.text_generation",
    "gpt_oss": "hf.text_generation",
    "phi3": "hf.text_generation",
    "phi4": "hf.text_generation",
    "gemma2": "hf.text_generation",
    "qwen_vl": "hf.image_text_to_text",
    "qwen_3_vl": "hf.image_text_to_text",
    "qwen2_5_vl": "hf.image_text_to_text",
    "glm4v": "hf.image_text_to_text",
}

_CONFIG_MODEL_TYPE_ARCHITECTURE_MAPPING = {}


def size_on_disk(
    model_info: ModelInfo,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> int:
    """
    Calculate the total size of cached files for a repo using only hub metadata.

    The function intentionally works off of the `siblings` entries in the
    `ModelInfo` payload instead of hitting the filesystem so we can quickly
    size repos that may not be downloaded locally. Optional allow/ignore
    patterns mimic the client-side filters we use when listing files.
    """
    siblings = model_info.siblings or []
    total_size = 0

    for sib in siblings:
        if sib.size is None:
            continue

        if not sib.rfilename:
            continue

        if allow_patterns is not None and not any(fnmatch(sib.rfilename, pattern) for pattern in allow_patterns):
            continue

        if ignore_patterns is not None and any(fnmatch(sib.rfilename, pattern) for pattern in ignore_patterns):
            continue

        total_size += sib.size

    return total_size


def has_model_index(model_info: ModelInfo) -> bool:
    """Return True when hub metadata lists a `model_index.json` sibling."""
    siblings = getattr(model_info, "siblings", None)
    return any(sib.rfilename == "model_index.json" for sib in (siblings or []))


# Packaging heuristics ---------------------------------------------------------
# Filenames/extensions used to decide whether weights belong to a bundle or per-file list.
_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".ckpt",
    ".pt",
    ".pth",
    ".gguf",
    ".ggml",
    ".onnx",
    ".svdq",
)
_INDEX_FILENAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model.bin.index.json",
    "model.index.json",
}
# Size/keyword thresholds that help spot adapters vs base weights.
_SMALL_ADAPTER_MAX_BYTES = 30 * 1024 * 1024
_QUANT_MARKERS = (
    "gptq",
    "awq",
    "exl2",
    "exl",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q8",
    "svdq",
)
_ADAPTER_MARKERS = (
    "lora",
    "adapter",
    "embedding",
    "textual_inversion",
    "ti_",
    "control",
    "ip-adapter",
    "style",
)


def detect_repo_packaging(
    repo_id: str,
    model_info: ModelInfo | None,
    file_entries: Sequence[tuple[str, int]],
) -> RepoPackagingHint:
    """
    Guess whether a repo should be presented as a single bundle or per-file list.

    We prefer bundle presentation for diffusers-style repos (pipeline metadata,
    sharded checkpoints, consistent weight naming) and flip to per-file when
    we see multiple quantizations or adapter-style weights that likely represent
    independent choices for the user.
    """
    weight_entries = [(name, size) for name, size in file_entries if _is_weight_file(name)]
    weight_files = [name for name, _ in weight_entries]
    lower_weight_files = [name.lower() for name in weight_files]

    if _has_bundle_metadata(model_info):
        return RepoPackagingHint.REPO_BUNDLE
    if _has_sharded_weights(lower_weight_files):
        return RepoPackagingHint.REPO_BUNDLE
    if _has_quantized_variants(lower_weight_files):
        return RepoPackagingHint.PER_FILE
    if _has_adapter_candidates(weight_entries):
        return RepoPackagingHint.PER_FILE
    if len(weight_files) == 1:
        return RepoPackagingHint.REPO_BUNDLE
    if _all_same_family(weight_files):
        return RepoPackagingHint.REPO_BUNDLE
    if len(weight_files) >= 4 and not model_info:
        return RepoPackagingHint.PER_FILE
    return RepoPackagingHint.UNKNOWN


def _is_weight_file(file_name: str) -> bool:
    """Lightweight check for weight-like filenames used by packaging heuristics."""
    lower = file_name.lower()
    return lower.endswith(_WEIGHT_EXTENSIONS)


def _has_bundle_metadata(model_info: ModelInfo | None) -> bool:
    """Detect diffusers/transformers repos that advertise a full pipeline bundle."""
    if model_info is None:
        return False
    if model_info.pipeline_tag:
        return True
    if has_model_index(model_info):
        return True
    library_name = getattr(model_info, "library_name", None)
    return bool(library_name and str(library_name).lower() in ("diffusers", "transformers"))


def _has_sharded_weights(weight_files: Sequence[str]) -> bool:
    """Return True if filenames match common sharded checkpoint patterns."""
    for name in weight_files:
        lower = name.lower()
        if lower in _INDEX_FILENAMES:
            return True
        if "-00001-of-" in lower:
            return True
        if lower.endswith(".index.json"):
            return True
    return False


def _has_quantized_variants(weight_files: Sequence[str]) -> bool:
    """Identify multiple quantized flavors of a model (gguf/ggml/awq/gptq/etc.)."""
    quantized = 0
    for name in weight_files:
        lower = name.lower()
        if lower.endswith(".gguf") or "ggml" in lower:
            quantized += 1
            continue
        if any(marker in lower for marker in _QUANT_MARKERS):
            quantized += 1
    return quantized >= 2


def _has_adapter_candidates(weight_entries: Sequence[tuple[str, int]]) -> bool:
    """Flag repos containing small LoRA/adapter-like files instead of base weights."""
    adapter_like = []
    for name, size in weight_entries:
        lower = name.lower()
        if any(marker in lower for marker in _ADAPTER_MARKERS):
            adapter_like.append((name, size))
            continue
        if lower.endswith(".safetensors") and size and size < _SMALL_ADAPTER_MAX_BYTES and len(weight_entries) > 1:
            adapter_like.append((name, size))
    return len(adapter_like) >= 1


def _all_same_family(weight_files: Sequence[str]) -> bool:
    """
    Determine if a small set of weight files belong to the same family/variant.

    Useful to keep repos with a handful of shards/quantizations grouped together
    instead of forcing per-file selection.
    """
    if not weight_files or len(weight_files) > 3:
        return False
    normalized_stems: set[str] = set()
    for name in weight_files:
        stem = Path(name).stem.lower()
        for marker in _QUANT_MARKERS:
            stem = stem.replace(marker, "")
        normalized_stems.add(stem)
    return len(normalized_stems) == 1


def _is_single_file_diffusion_weight(file_name: str) -> bool:
    """
    Heuristically detect raw checkpoint files (e.g. Stable Diffusion .safetensors)
    that live at the repo root inside the HF cache.

    Excludes standard model weight files that are part of multi-file repos:
    - model.safetensors
    - pytorch_model.bin
    - model.bin
    - model.pt
    - model.pth
    """
    normalized = file_name.replace("\\", "/")
    if "/" in normalized:
        return False
    lower = normalized.lower()

    # Must have a supported extension
    if not lower.endswith(SINGLE_FILE_DIFFUSION_EXTENSIONS):
        return False

    # Exclude standard model weight filenames that are part of multi-file repos
    standard_weight_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "model.bin",
        "model.pt",
        "model.pth",
    }
    return lower not in standard_weight_names


_DIFFUSION_COMPONENTS = {"unet", "transformer_denoiser"}
_ARTIFACT_INSPECTION_LIMIT = 16
_DIFFUSION_REPO_CACHE: dict[str, bool] = {}


def _is_diffusion_artifact_candidate(file_name: str) -> bool:
    """Return True for files that might reveal diffusion components."""
    lower = file_name.lower()
    if lower.endswith(("model_index.json", "config.json")):
        return True
    return any(lower.endswith(ext) for ext in SINGLE_FILE_DIFFUSION_EXTENSIONS)


async def _repo_has_diffusion_artifacts(
    repo_id: str,
    snapshot_dir: str | Path | None,
    file_list: Sequence[str],
) -> bool:
    """Return True when artifact inspection identifies diffusion components."""
    cached = _DIFFUSION_REPO_CACHE.get(repo_id)
    if cached is not None:
        return cached

    if not snapshot_dir or not file_list:
        _DIFFUSION_REPO_CACHE[repo_id] = False
        return False

    base_path = Path(snapshot_dir)
    candidate_paths: list[str] = []
    for fname in file_list:
        if len(candidate_paths) >= _ARTIFACT_INSPECTION_LIMIT:
            break
        if not _is_diffusion_artifact_candidate(fname):
            continue
        candidate_paths.append(str(base_path / fname))

    if not candidate_paths:
        _DIFFUSION_REPO_CACHE[repo_id] = False
        return False

    try:
        from nodetool.integrations.huggingface.artifact_inspector import inspect_paths

        detection = await asyncio.to_thread(inspect_paths, candidate_paths)

    except Exception as exc:  # pragma: no cover - best effort
        log.debug("inspect_paths failed for %s: %s", repo_id, exc)
        _DIFFUSION_REPO_CACHE[repo_id] = False
        return False

    matches = bool(detection and detection.component in _DIFFUSION_COMPONENTS)
    _DIFFUSION_REPO_CACHE[repo_id] = matches
    return matches


async def unified_model(
    model: HuggingFaceModel,
    model_info: ModelInfo | None = None,
    size: int | None = None,
    user_id: str | None = None,
) -> UnifiedModel | None:
    """
    Build a `UnifiedModel` instance using only already-fetched metadata.

    The helper is used when the UI hands us recommended models or when we
    want to decorate hub results without triggering extra network calls.
    Size/pipeline/tag information is filled from `model_info` when provided;
    otherwise we return a minimal record so callers can still render choices.
    """

    model_id = f"{model.repo_id}:{model.path}" if model.path is not None else model.repo_id

    # Without hub lookups, size and metadata may be missing; rely on provided info only.
    if model_info is not None and size is None:
        if model.path:
            size = next(
                (sib.size for sib in (model_info.siblings or []) if sib.rfilename == model.path),
                None,
            )
        else:
            size = size_on_disk(
                model_info,
                allow_patterns=model.allow_patterns,
                ignore_patterns=model.ignore_patterns,
            )

    return UnifiedModel(
        id=model_id,
        repo_id=model.repo_id,
        path=model.path,
        type=model.type,
        name=model.repo_id,
        cache_path=None,
        allow_patterns=model.allow_patterns,
        ignore_patterns=model.ignore_patterns,
        description=None,
        readme=None,
        size_on_disk=size,
        pipeline_tag=model_info.pipeline_tag if model_info else None,
        tags=model_info.tags if model_info else None,
        has_model_index=has_model_index(model_info) if model_info else None,
        downloads=model_info.downloads if model_info else None,
        likes=model_info.likes if model_info else None,
        trending_score=model_info.trending_score if model_info else None,
    )


async def fetch_model_readme(model_id: str) -> str | None:
    """
    Retrieve README text for a repo, preferring the local HF cache and falling back to the hub.
    """
    from huggingface_hub import (
        _CACHED_NO_EXIST,
        try_to_load_from_cache,
    )

    cached_path = try_to_load_from_cache(repo_id=model_id, filename="README.md")

    if isinstance(cached_path, str):
        try:
            async with aiofiles.open(cached_path, encoding="utf-8") as handle:
                return await handle.read()
        except Exception as e:
            log.debug("Failed to read cached README for %s: %s", model_id, e)
    elif cached_path is _CACHED_NO_EXIST:
        return None

    try:
        token = await get_hf_token()
        if token:
            log.debug(
                "fetch_model_readme: Downloading README for %s with HF_TOKEN (token length: %d chars)",
                model_id,
                len(token),
            )
        else:
            log.debug(
                "fetch_model_readme: Downloading README for %s without HF_TOKEN - gated models may not be accessible",
                model_id,
            )

        readme_path = await async_hf_download(
            repo_id=model_id,
            filename="README.md",
            repo_type="model",
            token=token,
        )
        async with aiofiles.open(readme_path, encoding="utf-8") as handle:
            return await handle.read()
    except Exception as exc:  # pragma: no cover
        log.debug("Failed to download README for %s: %s", model_id, exc)
        return None


class _RecursiveNamespace:
    """Read-only view of a nested dict that mimics attribute access."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError:
            # Mirror standard object behavior for missing attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from None

        if isinstance(val, dict):
            return _RecursiveNamespace(val)
        if isinstance(val, list):
            return [_RecursiveNamespace(x) if isinstance(x, dict) else x for x in val]
        return val

    def __repr__(self) -> str:
        return f"_RecursiveNamespace({repr(self._data)})"


async def fetch_model_info(model_id: str) -> ModelInfo | None:
    """
    Fetch and cache `ModelInfo` for a repo, using the hub only when necessary.
    """
    from huggingface_hub import HfApi

    token = await get_hf_token()
    api = HfApi(token=token) if token else HfApi()

    try:
        model_info: ModelInfo = await asyncio.to_thread(
            api.model_info,
            model_id,
            files_metadata=True,
        )
    except Exception as exc:  # pragma: no cover
        log.debug("fetch_model_info: failed to fetch %s: %s", model_id, exc)
        return None

    return model_info


def model_type_from_model_info(
    recommended_models: dict[str, list[UnifiedModel]],
    repo_id: str,
    model_info: ModelInfo | None,
) -> str | None:
    """
    Resolve a model's canonical hf.* type using multiple sources of truth.

    Priority order:
    1) Package-provided recommended models (authoritative for known repos).
    2) Diffusers `_class_name` in the hub config.
    3) Hub pipeline tags and generic tag hints (mlx/gguf).
    The function intentionally returns None when lacking signal so downstream
    callers can try local config parsing or artifact inspection.
    """
    recommended = recommended_models.get(repo_id, [])
    if len(recommended) == 1:
        return recommended[0].type
    if model_info is None:
        return None
    if model_info.config and "diffusers" in model_info.config and "_class_name" in model_info.config["diffusers"]:
        return CLASSNAME_TO_MODEL_TYPE.get(
            model_info.config["diffusers"]["_class_name"],
            None,  # type: ignore[no-any-return]
        )
    if model_info.pipeline_tag:
        name = model_info.pipeline_tag.replace("-", "_")
        return f"hf.{name}"
    if model_info.tags:
        if "mlx" in model_info.tags:
            return "mlx"
        if "gguf" in model_info.tags:
            return "llama_cpp"
    return None


def _get_file_size(file_path: Path) -> int:
    """Get file size, handling symlinks."""
    try:
        if file_path.is_symlink():
            # Resolve symlink to get actual file size
            resolved = file_path.resolve(strict=False)
            if resolved.exists():
                return resolved.stat().st_size
        elif file_path.exists():
            return file_path.stat().st_size
    except OSError:
        pass
    return 0


def _safe_load_json(file_path: Path) -> dict:
    """Best-effort JSON loader that logs failures without interrupting discovery."""
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        log.debug("Failed to load JSON from %s: %s", file_path, exc)
        return {}


def _infer_model_type_from_local_configs(
    file_entries: Sequence[tuple[str, int]],
    snapshot_dir: Path | None,
) -> str | None:
    """
    Infer model type by reading cached config or model_index files when hub metadata
    is unavailable.

    The logic mirrors hub-side typing: prefer diffusers `_class_name`, fall back to
    `model_type`/`architectures` hints, and only touches local files that are already
    present in the snapshot. This keeps offline environments functional while avoiding
    unnecessary disk I/O.
    """
    if not snapshot_dir:
        return None

    config_candidates = [
        rel_path for rel_path, _ in file_entries if rel_path.lower().endswith(("model_index.json", "config.json"))
    ]
    if not config_candidates:
        return None

    for rel_path in sorted(config_candidates, key=lambda value: (value.count("/"), len(value))):
        config_path = snapshot_dir / rel_path
        data = _safe_load_json(config_path)
        if not data:
            continue

        class_name = data.get("_class_name")
        if isinstance(class_name, str):
            mapped = CLASSNAME_TO_MODEL_TYPE.get(class_name)
            if mapped:
                return mapped

        model_type = str(data.get("model_type", "")).lower() if isinstance(data, dict) else ""
        if model_type:
            mapped = _CONFIG_MODEL_TYPE_MAPPING.get(model_type)
            if mapped:
                return mapped

        architectures = data.get("architectures")
        if isinstance(architectures, list):
            for arch in architectures:
                mapped = _CONFIG_MODEL_TYPE_ARCHITECTURE_MAPPING.get(
                    str(arch),
                    None,
                )
                if mapped:
                    return mapped
    return None


async def _build_cached_repo_entry(
    repo_id: str,
    repo_dir: Path,
    recommended_models: dict[str, list[UnifiedModel]],
    snapshot_dir: Path | None = None,
    file_list: list[str] | None = None,
    model_info: ModelInfo | None = None,
) -> tuple[UnifiedModel, list[tuple[str, int]]]:
    """
    Build the repo-level `UnifiedModel` plus per-file metadata for a cached HF repo.
    """
    repo_root = await HF_FAST_CACHE.repo_root(repo_id, repo_type="model")
    file_entries: list[tuple[str, int]] = []
    size_on_disk = 0
    snapshot_path: Path | None = snapshot_dir
    if snapshot_path is None:
        resolved_snapshot = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        snapshot_path = Path(resolved_snapshot) if resolved_snapshot else None

    if snapshot_path:
        if file_list is None:
            file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")

        for file_name in file_list or []:
            file_path = snapshot_path / file_name
            file_size = _get_file_size(file_path)
            size_on_disk += file_size
            file_entries.append((file_name, file_size))

    if repo_id in recommended_models:
        model = recommended_models[repo_id][0]
        return model, file_entries

    artifact_detection: ArtifactDetection | None = None
    if file_entries and snapshot_path:
        artifact_paths = [str(snapshot_path / name) for name, _ in file_entries]
        try:
            from nodetool.integrations.huggingface.artifact_inspector import inspect_paths

            artifact_detection = await asyncio.to_thread(inspect_paths, artifact_paths)

        except Exception:
            artifact_detection = None

    model_type = _infer_model_type_from_local_configs(
        file_entries,
        snapshot_path,
    )

    repo_model = UnifiedModel(
        id=repo_id,
        type=model_type,
        name=repo_id,
        cache_path=str(repo_root) if repo_root else str(repo_dir),
        allow_patterns=None,
        ignore_patterns=None,
        description=None,
        readme=None,
        downloaded=repo_root is not None or repo_dir.exists(),
        repo_id=repo_id,
        path=None,
        size_on_disk=size_on_disk,
        artifact_family=artifact_detection.family if artifact_detection else None,
        artifact_component=artifact_detection.component if artifact_detection else None,
        artifact_confidence=(artifact_detection.confidence if artifact_detection else None),
        artifact_evidence=artifact_detection.evidence if artifact_detection else None,
    )

    return repo_model, file_entries


async def read_cached_hf_models() -> list[UnifiedModel]:
    """
    Enumerate all cached HF repos and return repo-level `UnifiedModel` entries.
    """

    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:
        log.debug("read_cached_hf_models: failed to discover repos: %s", exc)
        return []

    recommended_models = get_recommended_models()
    models: list[UnifiedModel] = []

    for repo_id, repo_dir in repo_list:
        repo_model, _ = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            recommended_models,
        )
        models.append(repo_model)

    return models


# Static search hints per hf.* type used to build repo/file queries (offline/hub).
HF_SEARCH_TYPE_CONFIG: dict[str, dict[str, list[str] | str]] = {
    "hf.stable_diffusion_3": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": KNOWN_REPO_PATTERNS["sd35"],
    },
    "hf.flux": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [*KNOWN_REPO_PATTERNS["flux"], "*flux*"],
    },
    "hf.flux_fp8": {
        "filename_pattern": [
            "*fp8*.safetensors",
            "*fp8*.ckpt",
            "*fp8*.bin",
            "*fp8*.pt",
            "*fp8*.pth",
        ],
        "repo_pattern": [*KNOWN_REPO_PATTERNS["flux"], "*flux*"],
    },
    "hf.flux_kontext": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["flux_kontext"],
            "*nunchaku*flux*",
            "*flux*kontext*",
        ],
    },
    "hf.flux_canny": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["flux_canny"],
            "*nunchaku*flux*canny*",
            "*flux*canny*",
        ],
    },
    "hf.flux_depth": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["flux_depth"],
            "*nunchaku*flux*depth*",
            "*flux*depth*",
        ],
    },
    "hf.qwen_image": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": KNOWN_REPO_PATTERNS["qwen_image"],
    },
    "hf.qwen_image_edit": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": KNOWN_REPO_PATTERNS["qwen_image_edit"],
    },
    "hf.qwen_vl": {
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["qwen_image"],
            *KNOWN_REPO_PATTERNS["qwen_image_edit"],
        ],
    },
    "hf.controlnet": {
        "repo_pattern": ["*control*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
    },
    "hf.controlnet_sdxl": {
        "repo_pattern": ["*control*"],
        "tag": ["*sdxl*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
    },
    "hf.controlnet_flux": {
        "repo_pattern": ["*control*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
    },
    "hf.ip_adapter": {
        "repo_pattern": ["*IP-Adapter*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
    },
    "hf.lora_sd": {"repo_pattern": ["*lora*"], "pipeline_tag": []},
    "hf.lora_sdxl": {
        "repo_pattern": ["*lora*sdxl*", "*sdxl*lora*"],
    },
    "hf.lora_qwen_image": {"repo_pattern": ["*lora*qwen*"], "pipeline_tag": []},
    "hf.unet": {
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["flux"],
            *KNOWN_REPO_PATTERNS["qwen_image"],
            *KNOWN_REPO_PATTERNS["qwen_image_edit"],
            *KNOWN_REPO_PATTERNS["sd35"],
            "*unet*",
            "*stable-diffusion*",
        ],
        "filename_pattern": [
            "*unet*.safetensors",
            "*unet*.bin",
            "*unet*.ckpt",
            "*flux*.safetensors",
            "*flux*.bin",
            "*flux*.ckpt",
        ],
    },
    "hf.vae": {
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["flux_vae"],
            *KNOWN_REPO_PATTERNS["qwen_image"],
            *KNOWN_REPO_PATTERNS["qwen_image_edit"],
            "*vae*",
            "*stable-diffusion*",
        ],
        "filename_pattern": [
            "*vae*.safetensors",
            "*vae*.bin",
            "*vae*.ckpt",
            "*vae*.pt",
        ],
    },
    "hf.clip": {
        "repo_pattern": [
            *KNOWN_REPO_PATTERNS["sd35"],
            *KNOWN_REPO_PATTERNS["qwen_image"],
            *KNOWN_REPO_PATTERNS["qwen_image_edit"],
            "*clip*",
            "*flux*",
        ],
        "filename_pattern": [
            "*clip*.safetensors",
            "*clip*.bin",
            "*clip*.gguf",
            "*clip*.ckpt",
        ],
    },
    "hf.t5": {
        "repo_pattern": [*KNOWN_REPO_PATTERNS["sd35"], "*t5*", "*flux*"],
        "filename_pattern": ["*t5*.safetensors", "*t5*.bin", "*t5*.gguf", "*t5*.ckpt"],
    },
    "hf.image_to_video": {"pipeline_tag": ["image-to-video"]},
    "hf.text_to_video": {"pipeline_tag": ["text-to-video"]},
    "hf.image_to_text": {"pipeline_tag": ["image-to-text"], "tag": ["*caption*"]},
    "hf.inpainting": {"pipeline_tag": ["image-inpainting"], "tag": ["*inpaint*"]},
    "hf.outpainting": {"tag": ["*outpaint*"]},
    "hf.flux_redux": {
        "repo_pattern": ["*flux*redux*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
    },
    "hf.real_esrgan": {
        "repo_pattern": ["*esrgan*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
    },
}

# Derive checkpoint variants (single-file) from base configs.
for _base, _ckpt in _CHECKPOINT_BASES.items():
    if _base in HF_SEARCH_TYPE_CONFIG and _ckpt not in HF_SEARCH_TYPE_CONFIG:
        _base_cfg = HF_SEARCH_TYPE_CONFIG[_base]
        HF_SEARCH_TYPE_CONFIG[_ckpt] = {k: (list(v) if isinstance(v, list) else v) for k, v in _base_cfg.items()}

HF_TYPE_STRUCTURAL_RULES: dict[str, dict[str, bool]] = {
    "hf.unet": {"file_only": True},
    "hf.vae": {"file_only": True},
    "hf.clip": {"file_only": True},
    "hf.t5": {"file_only": True},
    "hf.qwen_vl": {"file_only": True},
    "hf.stable_diffusion_checkpoint": {"file_only": True, "checkpoint": True},
    "hf.stable_diffusion_xl_checkpoint": {"file_only": True, "checkpoint": True},
    "hf.stable_diffusion_3_checkpoint": {"file_only": True, "checkpoint": True},
    "hf.stable_diffusion_xl_refiner_checkpoint": {
        "file_only": True,
        "checkpoint": True,
    },
    "hf.flux_checkpoint": {"file_only": True, "checkpoint": True},
    "hf.qwen_image_checkpoint": {"checkpoint": True, "nested_checkpoint": True},
    "hf.qwen_image_edit_checkpoint": {"checkpoint": True, "nested_checkpoint": True},
    "hf.flux": {"single_file_repo": True},
    "hf.flux_fp8": {"single_file_repo": True},
    "hf.flux_redux": {"single_file_repo": True},
    "hf.stable_diffusion": {"single_file_repo": True},
    "hf.stable_diffusion_xl": {"single_file_repo": True},
    "hf.stable_diffusion_3": {"single_file_repo": True},
    "hf.stable_diffusion_xl_refiner": {"single_file_repo": True},
}


GENERIC_HF_TYPES = {
    "hf.text_to_image",
    "hf.image_to_image",
    "hf.model",
    "hf.model_generic",
}


def _build_search_config_for_type(model_type: str) -> dict[str, list[str] | str] | None:
    """
    Get search configuration for a given hf.* type, or None if not found.

    Args:
        model_type: The model type (e.g., "hf.flux", "hf.stable_diffusion")

    Returns:
        Search configuration dict with patterns, or None if type is not configured
    """
    normalized = model_type.lower()
    config = HF_SEARCH_TYPE_CONFIG.get(normalized)
    if config is not None:
        return config
    if normalized.startswith("hf."):
        return {
            "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
            "repo_pattern": ["*"],
        }
    return None


def _derive_pipeline_tag(normalized_type: str, task: str | None = None) -> str | None:
    """
    Infer a sensible HF pipeline tag from an hf.* type or explicit task override.

    This keeps pipeline tags aligned with how the UI filters models without
    requiring every caller to supply both type and task inputs.
    """
    if task:
        return task.replace("_", "-")
    slug = normalized_type[3:] if normalized_type.startswith("hf.") else normalized_type
    if slug.endswith("_checkpoint"):
        slug = slug[: -len("_checkpoint")]
    if slug in {
        "stable_diffusion",
        "stable_diffusion_xl",
        "stable_diffusion_xl_refiner",
        "stable_diffusion_3",
        "flux",
        "flux_fp8",
        "flux_kontext",
        "flux_canny",
        "flux_depth",
        "flux_redux",
        "qwen_image",
        "ip_adapter",
    }:
        return "text-to-image"
    if slug in {"qwen_image_edit", "image_to_image", "inpainting", "outpainting"}:
        return "image-to-image"
    if slug == "text_to_video":
        return "text-to-video"
    if slug == "image_to_video":
        return "image-to-video"
    if "text_to_image" in slug:
        return "text-to-image"
    if "image_to_image" in slug:
        return "image-to-image"
    return slug.replace("_", "-")


def _matches_repo_for_type(normalized_type: str, repo_id: str, repo_id_from_id: str) -> bool:
    """Check if a repo id matches any hard-coded comfy-type mappings for a model type."""
    matchers = KNOWN_TYPE_REPO_MATCHERS.get(normalized_type)
    if not matchers:
        return False
    repo_lower = repo_id.lower()
    repo_from_id_lower = repo_id_from_id.lower()
    return any(repo_lower == candidate.lower() or repo_from_id_lower == candidate.lower() for candidate in matchers)


def _matches_artifact_detection(
    normalized_type: str,
    artifact_family: str | None = None,
    artifact_component: str | None = None,
) -> bool:
    """
    Use lightweight artifact inspection hints (family/component) to match a model type.

    This enables classification for cached repos that lack explicit tags or pipeline
    metadata but can be identified by file headers alone.
    """
    fam = artifact_family or ""
    comp = artifact_component or ""
    if normalized_type in {
        "hf.flux",
        "hf.flux_fp8",
        "hf.flux_kontext",
        "hf.flux_canny",
        "hf.flux_depth",
    }:
        return "flux" in fam
    if normalized_type == "hf.stable_diffusion":
        return fam.startswith("sd1") or fam.startswith("sd2") or "stable-diffusion" in fam
    if normalized_type == "hf.stable_diffusion_xl":
        return "sdxl" in fam
    if normalized_type == "hf.stable_diffusion_xl_refiner":
        return "refiner" in fam or ("sdxl" in fam and comp == "unet")
    if normalized_type == "hf.stable_diffusion_3":
        return "sd3" in fam or "stable-diffusion-3" in fam
    if normalized_type in {"hf.qwen_image", "hf.qwen_image_edit"}:
        return "qwen" in fam
    return False


def _matches_model_type(model: UnifiedModel, model_type: str) -> bool:
    """Semantic match for hf.* types (no structural checks here)."""
    normalized_type = model_type.lower()
    checkpoint_variant = None
    if normalized_type.endswith("_checkpoint"):
        checkpoint_variant = normalized_type
        normalized_type = normalized_type[: -len("_checkpoint")]

    model_type_lower = (model.type or "").lower()
    repo_id = (model.repo_id or "").lower()
    repo_id_from_id = (model.id or "").split(":", 1)[0].lower() if model.id else ""
    path_lower = (model.path or "").lower()

    def _is_qwen_text_encoder(path: str | None) -> bool:
        if not path:
            return False
        return "text_encoders" in path or "text_encoder" in path or "qwen_2.5_vl" in path

    def _is_qwen_vae(path: str | None) -> bool:
        if not path:
            return False
        return "vae" in path

    target_types = {normalized_type}
    if checkpoint_variant:
        target_types.add(checkpoint_variant)

    if model_type_lower:
        model_type_base = (
            model_type_lower[: -len("_checkpoint")] if model_type_lower.endswith("_checkpoint") else model_type_lower
        )
        if model_type_lower in target_types or model_type_base == normalized_type:
            return not (
                normalized_type in {"hf.qwen_image", "hf.qwen_image_edit"}
                and (_is_qwen_text_encoder(path_lower) or _is_qwen_vae(path_lower))
            )

        if model_type_lower not in GENERIC_HF_TYPES:
            qwen_family_types = {"hf.qwen_image", "hf.qwen_image_checkpoint"}
            allowed_family = normalized_type in {
                "hf.qwen_image_checkpoint",
                "hf.qwen_vl",
                "hf.vae",
            } and (model_type_lower in qwen_family_types)
            if not allowed_family:
                return False

    if normalized_type in {"hf.qwen_image", "hf.qwen_image_edit"}:
        if _is_qwen_text_encoder(path_lower) or _is_qwen_vae(path_lower):
            return False

    if normalized_type == "hf.qwen_vl":
        return _is_qwen_text_encoder(path_lower)

    if _matches_repo_for_type(normalized_type, repo_id, repo_id_from_id):
        return True

    artifact_family = (getattr(model, "artifact_family", None) or "").lower()
    artifact_component = (getattr(model, "artifact_component", None) or "").lower()
    if artifact_family or artifact_component:
        if _matches_artifact_detection(normalized_type, artifact_family, artifact_component):
            return True

    tags = [(tag or "").lower() for tag in (model.tags or [])]
    keywords = HF_TYPE_KEYWORD_MATCHERS.get(normalized_type, [])
    if keywords:
        if any(keyword in repo_id or any(keyword in tag for tag in tags) for keyword in keywords):
            return True
        if path_lower and any(keyword in path_lower for keyword in keywords):
            return True

    derived_pipeline = _derive_pipeline_tag(normalized_type)
    return bool(derived_pipeline and model.pipeline_tag == derived_pipeline)


async def get_models_by_hf_type(
    model_type: str,
) -> list[UnifiedModel]:
    """
    Return cached Hugging Face models matching a requested hf.* type.

    The search is entirely offline: build a search config, scan cached repos/files,
    then apply the same client-side heuristics (keyword matching, repo patterns,
    artifact hints) to label each result with the desired type.
    """

    config = _build_search_config_for_type(model_type) or {}

    def _filter_models(models: list[UnifiedModel]) -> list[UnifiedModel]:
        """Apply type-specific structural rules then semantic matching."""
        rules = HF_TYPE_STRUCTURAL_RULES.get(model_type, {})
        file_only = rules.get("file_only", False)
        checkpoint = rules.get("checkpoint", False) or model_type in set(_CHECKPOINT_BASES.values())
        nested_checkpoint = rules.get("nested_checkpoint", False)
        single_file_repo = rules.get("single_file_repo", False)

        seen: set[str] = set()
        filtered: list[UnifiedModel] = []

        for model in models:
            if model.id in seen:
                continue

            repo_lower = (model.repo_id or "").lower()
            path_value = getattr(model, "path", None)

            if file_only and path_value is None:
                continue

            if single_file_repo:
                if path_value is None and "gguf" in repo_lower:
                    continue
                if path_value:
                    path_lower = path_value.lower()
                    if not _is_single_file_diffusion_weight(path_value) and not path_lower.endswith(".gguf"):
                        continue

            if checkpoint:
                if not path_value:
                    continue
                if "/" in path_value and not nested_checkpoint:
                    continue

            if not _matches_model_type(model, model_type):
                continue

            filtered.append(model)
            seen.add(model.id)

        return filtered

    # Offline-first search to avoid network dependency when cache is present.
    offline_models = await search_cached_hf_models(
        repo_patterns=config.get("repo_pattern"),
        filename_patterns=config.get("filename_pattern"),
    )
    offline_filtered = _filter_models(offline_models)
    if offline_filtered:
        log.debug(
            "get_models_by_hf_type: returning %d models from offline cache (type=%s)",
            len(offline_filtered),
            model_type,
        )
        return offline_filtered

    return offline_filtered


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    """Case-sensitive glob check; empty pattern list means match everything."""
    if not patterns:
        return True
    return any(fnmatch(value, pattern) for pattern in patterns)


def _matches_any_pattern_ci(value: str, patterns: list[str]) -> bool:
    """Case-insensitive glob check used when filtering by repo id."""
    value_lower = value.lower()
    return any(fnmatch(value_lower, pattern.lower()) for pattern in patterns)


async def iter_cached_model_files(
    pre_resolved_repos: Sequence[tuple[str, Path]] | None = None,
) -> AsyncIterator[tuple[str, Path, Path, list[str]]]:
    """
    Yield (repo_id, repo_dir, snapshot_dir, file_list) for cached HF repos.

    Traversal is offline-only and best-effort: repos without an active snapshot
    or whose files cannot be listed are skipped.
    """
    repo_list = (
        list(pre_resolved_repos) if pre_resolved_repos is not None else await HF_FAST_CACHE.discover_repos("model")
    )

    for repo_id, repo_dir in repo_list:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        try:
            file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.debug("iter_cached_model_files: list_files failed for %s: %s", repo_id, exc)
            continue

        yield repo_id, Path(repo_dir), Path(snapshot_dir), file_list


async def search_cached_hf_models(
    repo_patterns: Sequence[str] | None = None,
    filename_patterns: Sequence[str] | None = None,
) -> list[UnifiedModel]:
    """
    Search the local HF cache for repos/files using offline data only.

    The function is hub-free: it discovers repos from disk, optionally pre-resolves
    specific repos to avoid globbing, and emits both repo-level and file-level
    `UnifiedModel` entries when filename patterns are provided.
    """

    recommended_models = get_recommended_models()
    results: list[UnifiedModel] = []
    repo_count = 0

    async for repo_id, repo_dir, snapshot_dir, file_list in iter_cached_model_files():
        repo_count += 1
        if repo_patterns and not _matches_any_pattern_ci(repo_id, list(repo_patterns)):
            continue

        repo_model, file_entries = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            recommended_models,
            snapshot_dir=snapshot_dir,
            file_list=file_list,
        )
        results.append(repo_model)

        if filename_patterns and file_entries:
            for relative_name, file_size in file_entries:
                if not _matches_any_pattern(relative_name, list(filename_patterns)):
                    continue

                file_model = UnifiedModel(
                    id=f"{repo_id}:{relative_name}",
                    type=repo_model.type,
                    name=f"{repo_id}/{relative_name}",
                    repo_id=repo_id,
                    path=relative_name,
                    cache_path=repo_model.cache_path,
                    allow_patterns=None,
                    ignore_patterns=None,
                    description=None,
                    readme=None,
                    size_on_disk=file_size,
                    downloaded=repo_model.downloaded,
                    pipeline_tag=repo_model.pipeline_tag,
                    tags=repo_model.tags,
                    has_model_index=repo_model.has_model_index,
                    downloads=repo_model.downloads,
                    likes=repo_model.likes,
                    trending_score=repo_model.trending_score,
                    artifact_family=repo_model.artifact_family,
                    artifact_component=repo_model.artifact_component,
                    artifact_confidence=repo_model.artifact_confidence,
                    artifact_evidence=repo_model.artifact_evidence,
                )
                results.append(file_model)

    log.debug(
        "search_cached_hf_models: returning %d results (repos scanned=%d)",
        len(results),
        repo_count,
    )
    return results


SUPPORTED_MODEL_TYPES = [
    "qwen2",
    "qwen3",
    "qwen_2_5_vl",
    "qwen_3_vl",
    "mistral3",
    "gpt_oss",
    "llama",
    "gemma3",
    "gemma3n",
    "phi3",
    "phi4",
    "gemma2",
]


async def get_hf_language_models_from_hf_cache() -> list[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos containing language models.
    """
    results: list[LanguageModel] = []
    repo_list = await HF_FAST_CACHE.discover_repos("model")
    for repo_id, _repo_dir in repo_list:
        repo_id.split("/")[-1]
        config = await HF_FAST_CACHE.resolve(repo_id, "config.json")
        if config:
            async with aiofiles.open(config) as f:
                config_data = json.loads(await f.read())
            model_type = config_data.get("model_type")
            if model_type in SUPPORTED_MODEL_TYPES:
                results.append(
                    LanguageModel(
                        id=repo_id,
                        name=model_type,
                        provider=Provider.HuggingFace,
                        supported_tasks=["text_generation"],
                    )
                )
    return results


async def get_llamacpp_language_models_from_hf_cache() -> list[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos containing GGUF files
    that look suitable for llama.cpp.

    Heuristics:
    - File ends with .gguf (case-insensitive)
    - Each GGUF file yields a LanguageModel with id "<repo_id>:<filename>"

    Returns:
        List[LanguageModel]: Llama.cpp-compatible models discovered in the HF cache
    """
    results: list[LanguageModel] = []

    async for repo_id, _repo_dir, _snapshot_dir, file_list in iter_cached_model_files():
        for fname in file_list:
            if not fname.lower().endswith(".gguf"):
                continue
            model_id = f"{repo_id}:{fname}"
            display = f"{repo_id.split('/')[-1]} • {fname}"
            results.append(
                LanguageModel(
                    id=model_id,
                    name=display,
                    path=fname,
                    provider=Provider.LlamaCpp,
                )
            )

    # Sort for stability: by repo then filename
    results.sort(key=lambda m: (m.id.split(":", 1)[0], m.id))
    return results


async def get_llama_cpp_models_from_cache() -> list[UnifiedModel]:
    """
    Enumerate GGUF models in the llama.cpp native cache directory.

    llama.cpp uses a flat file structure:
    - {org}_{repo}_{filename}.gguf
    - {org}_{repo}_{filename}.gguf.etag
    - manifest={org}={repo}={tag}.json

    Cache locations:
    - Linux: ~/.cache/llama.cpp/
    - macOS: ~/Library/Caches/llama.cpp/
    - Windows: %LOCALAPPDATA%/llama.cpp/

    Returns:
        List[UnifiedModel]: Models with type='llama_cpp_model' found in the cache.
    """
    from nodetool.providers.llama_server_manager import get_llama_cpp_cache_dir

    cache_dir = get_llama_cpp_cache_dir()
    if not os.path.isdir(cache_dir):
        return []

    models: list[UnifiedModel] = []

    # llama.cpp uses flat naming: {org}_{repo}_{filename}.gguf
    for entry in os.listdir(cache_dir):
        if not entry.lower().endswith(".gguf"):
            continue
        # Skip etag files and other metadata
        if entry.endswith(".etag"):
            continue

        file_path = os.path.join(cache_dir, entry)
        if not os.path.isfile(file_path):
            continue

        # Parse repo info from flat filename: org_repo_filename.gguf
        # Example: ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf
        parts = entry.rsplit("_", 2)  # Split from right: [org, repo, filename]
        if len(parts) >= 3:
            org = parts[0]
            repo = parts[1]
            filename = parts[2]
            repo_id = f"{org}/{repo}"
        else:
            # Fallback for unexpected format
            repo = ""
            repo_id = ""
            filename = entry

        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0

        models.append(
            UnifiedModel(
                id=f"{repo_id}:{filename}" if repo_id else filename,
                type="llama_cpp_model",
                name=f"{repo.replace('-', ' ').title()} • {filename}" if repo_id else filename,
                repo_id=repo_id,
                path=filename,
                cache_path=file_path,
                size_on_disk=size,
                downloaded=True,
            )
        )

    # Sort for stability
    models.sort(key=lambda m: (m.repo_id or "", m.path or ""))
    log.debug(f"Found {len(models)} models in llama.cpp cache at {cache_dir}")
    return models


async def get_vllm_language_models_from_hf_cache() -> list[LanguageModel]:
    """Return LanguageModel entries based on cached weight files (hub-free)."""
    seen_repos: set[str] = set()
    results: list[LanguageModel] = []

    SUPPORTED_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

    async for repo_id, _repo_dir, _snapshot_dir, file_list in iter_cached_model_files():
        if repo_id not in seen_repos and any(
            fname.lower().endswith(SUPPORTED_WEIGHT_EXTENSIONS) for fname in file_list
        ):
            seen_repos.add(repo_id)
            repo_display = repo_id.split("/")[-1]
            results.append(
                LanguageModel(
                    id=repo_id,
                    name=repo_display,
                    provider=Provider.VLLM,
                )
            )
    return results


async def get_mlx_language_models_from_hf_cache() -> list[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos that look suitable
    for MLX runtime (Apple Silicon).

    Each qualifying repo yields a LanguageModel with id "<repo_id>" (no file suffix),
    because MLX loaders typically resolve the correct shard/quantization internally.

    Returns:
        List[LanguageModel]: MLX-compatible models discovered in the HF cache
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("get_mlx_language_models_from_hf_cache: discover failed: %s", exc)
        return []

    result: dict[str, LanguageModel] = {}
    for repo_id, _repo_dir in repo_list:
        repo_lower = repo_id.lower()
        if "mlx" not in repo_lower:
            continue
        display = repo_id.split("/")[-1]
        result[repo_id] = LanguageModel(
            id=repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


def _is_component_only_repo(repo_id: str) -> bool:
    """
    Check if a repo is a component-only repo that can't be used as a standalone pipeline.

    These repos contain model components (transformers, text encoders) that need to be
    combined with base models and aren't usable as standalone image generation models.

    Examples:
    - nunchaku-tech/nunchaku-flux.1-schnell (Nunchaku FLUX transformer)
    - nunchaku-tech/nunchaku-t5 (T5 encoder for Nunchaku)
    """
    repo_lower = repo_id.lower()
    # Nunchaku repos are component-only (transformers, T5 encoders)
    return "nunchaku" in repo_lower


async def _get_diffusion_models_from_hf_cache(task: str) -> list[ImageModel]:
    """
    Shared helper to discover cached diffusion models for a specific task.

    Returns:
    - For component-only repos (like Nunchaku): only individual component files
    - For normal repos with single-file checkpoints: repo entry + file entries
    - For normal multi-file repos: just the repo entry
    """
    result: dict[str, ImageModel] = {}
    async for repo_id, _repo_dir, snapshot_dir, file_list in iter_cached_model_files():
        if not file_list:
            continue
        # Check if this is a component-only repo (e.g., Nunchaku transformers)
        is_component_only = _is_component_only_repo(repo_id)
        has_diffusion_artifacts = await _repo_has_diffusion_artifacts(repo_id, snapshot_dir, file_list)

        # Skip non-component repos that don't have diffusion artifacts
        if not is_component_only and not has_diffusion_artifacts:
            continue

        # Add individual single-file checkpoints
        added_single_file = False
        for fname in file_list:
            if not _is_single_file_diffusion_weight(fname):
                continue
            model_id = f"{repo_id}:{fname}"
            display = f"{repo_id.split('/')[-1]} • {fname}"
            result[model_id] = ImageModel(
                id=model_id,
                name=display,
                path=fname,
                provider=Provider.HuggingFace,
                supported_tasks=[task],
            )
            added_single_file = True

        # Add repo-level entry for non-component repos
        # - If they have single-file checkpoints, add as companion entry
        # - If they're multi-file repos with diffusion artifacts, add as the main entry
        if not is_component_only and (added_single_file or has_diffusion_artifacts):
            result.setdefault(
                repo_id,
                ImageModel(
                    id=repo_id,
                    name=repo_id.split("/")[-1],
                    provider=Provider.HuggingFace,
                    supported_tasks=[task],
                ),
            )

    return list(result.values())


async def get_text_to_image_models_from_hf_cache() -> list[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are text-to-image models,
    including single-file checkpoints stored at the repo root.
    """
    return await _get_diffusion_models_from_hf_cache("text_to_image")


async def get_image_to_image_models_from_hf_cache() -> list[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are image-to-image models,
    including single-file checkpoints stored at the repo root.
    """
    return await _get_diffusion_models_from_hf_cache("image_to_image")


async def get_mlx_image_models_from_hf_cache() -> list[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are mflux models
    (MLX-compatible image generation models).

    Returns:
        List[ImageModel]: MLX-compatible image models (mflux) discovered in the HF cache
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("get_mlx_image_models_from_hf_cache: discover failed: %s", exc)
        return []

    result: dict[str, ImageModel] = {}
    for repo_id, _repo_dir in repo_list:
        if "mflux" not in repo_id.lower():
            continue
        display = repo_id.split("/")[-1]
        result[repo_id] = ImageModel(
            id=repo_id,
            name=display,
            provider=Provider.MLX,
        )

    return list(result.values())


async def _fetch_models_by_author(user_id: str | None = None, **kwargs) -> list[ModelInfo]:
    """Fetch models list from HF API for a given author using HFAPI.

    Returns raw model dicts from the public API.
    """
    # Use HF_TOKEN from secrets if available for gated model downloads
    token = await get_hf_token(user_id)
    author = kwargs.get("author", "unknown")
    if token:
        log.debug(
            f"_fetch_models_by_author: Fetching models for author {author} with HF_TOKEN (token length: {len(token)} chars)"
        )
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    else:
        log.debug(
            f"_fetch_models_by_author: Fetching models for author {author} without HF_TOKEN - gated models may not be accessible"
        )
        from huggingface_hub import HfApi

        api = HfApi()
    # Run the blocking call in a thread executor
    models = await asyncio.get_running_loop().run_in_executor(None, lambda: api.list_models(**kwargs))
    return list(models)


async def get_gguf_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "downloads",
    tags: str = "gguf",
) -> list[UnifiedModel]:
    """
    Fetch all HF repos authored by the given authors that include GGUF files/tags.

    Heuristic: filter API results to those with a "gguf" tag, then for each
    author select the top 30 repos sorted by likes.

    Args:
        authors: List of HF author/org names (e.g., ["unsloth", "ggml-org"]).

    Returns:
        List[HuggingFaceModel]: One entry per matching repo.
    """
    # Fetch authors concurrently
    # Note: user_id would need to be passed from caller context
    results = await asyncio.gather(
        *(
            _fetch_models_by_author(
                user_id=None,
                author=a,
                limit=limit,
                sort=sort,
                tags=tags,
            )
            for a in authors
        )
    )
    repos = [item for sublist in results for item in sublist]
    model_infos = await asyncio.gather(*[fetch_model_info(repo.id) for repo in repos])

    # Collect all unified_model tasks
    tasks: list[tuple[HuggingFaceModel, ModelInfo, int | None]] = []
    seen_file: set[tuple[str, str]] = set()
    for info in model_infos:
        if info is None:
            continue
        sibs = info.siblings or []
        for sib in sibs:
            fname = getattr(sib, "rfilename", None)
            if not isinstance(fname, str) or not fname.lower().endswith(".gguf"):
                continue

            # Use (repo_id, filename) as unique key to allow same filename in different repos
            unique_key = (info.id, fname)
            if unique_key in seen_file:
                continue
            seen_file.add(unique_key)

            tasks.append(
                (
                    HuggingFaceModel(type="llama_cpp", repo_id=info.id, path=fname),
                    info,
                    sib.size,
                )
            )

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(*[unified_model(model, info, size) for model, info, size in tasks])

    # Sort for stability: repo then filename
    entries = [entry for entry in entries if entry is not None]
    return entries


async def get_mlx_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "trending_score",
    tags: str = "mlx",
) -> list[UnifiedModel]:
    """
    Fetch MLX-friendly repos authored by the given authors/orgs and return
    one LanguageModel per repo id.

    Heuristics:
    - Prefer orgs like "mlx-community" via the authors parameter
    - Filter API results to those with a tag containing "mlx"
    - Per author, take the top 30 repos sorted by likes

    Args:
        authors: List of HF author/org names (e.g., ["mlx-community"]).

    Returns:
        List[HuggingFaceModel]: One entry per qualifying repo.
    """
    # Fetch authors concurrently
    # Note: user_id would need to be passed from caller context
    results = await asyncio.gather(
        *(_fetch_models_by_author(user_id=None, author=a, limit=limit, sort=sort, tags=tags) for a in authors)
    )
    model_infos = [item for sublist in results for item in sublist]

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(
        *[unified_model(HuggingFaceModel(type="mlx", repo_id=info.id), info) for info in model_infos]
    )

    # Stable order
    return [entry for entry in entries if entry is not None]


async def delete_cached_hf_model(model_id: str) -> bool:
    """
    Deletes a model from the Hugging Face cache and the disk cache.

    Args:
        model_id (str): The ID of the model to delete.
    """
    # Use HfFastCache to resolve the repo root without walking the entire cache.
    repo_root = await HF_FAST_CACHE.repo_root(model_id, repo_type="model")
    if not repo_root:
        return False

    if not await asyncio.to_thread(os.path.exists, repo_root):
        return False

    shutil.rmtree(repo_root)

    await HF_FAST_CACHE.invalidate(model_id, repo_type="model")
    return True


# GGUF_AUTHORS = [
# "unsloth",
# "ggml-org",
# "LiquidAI",
# "gabriellarson",
# "openbmb",
# "zai-org",
# "vikhyatk",
# "01-ai",
# "BAAI",
# "Lin-Chen",
# "mtgv",
# "lm-sys",
# "NousResearch",
# ]
# MLX_AUTHORS = ["mlx-community"]


# async def save_gguf_models_to_file() -> None:
#     models = await get_gguf_language_models_from_authors(
#         GGUF_AUTHORS, limit=500, sort="downloads", tags="gguf"
#     )
#     with open(GGUF_MODELS_FILE, "w") as f:
#         json.dump(
#             [model.model_dump() for model in models if model is not None], f, indent=2
#         )


# async def save_mlx_models_to_file() -> None:
#     models = await get_mlx_language_models_from_authors(
#         MLX_AUTHORS, limit=1000, sort="downloads", tags="mlx"
#     )
#     with open(MLX_MODELS_FILE, "w") as f:
#         json.dump([model.model_dump() for model in models], f, indent=2)


# async def load_gguf_models_from_file() -> List[UnifiedModel]:
#     async with aiofiles.open(GGUF_MODELS_FILE, "r") as f:
#         content = await f.read()
#         return [UnifiedModel(**model) for model in json.loads(content)]


# async def load_mlx_models_from_file() -> List[UnifiedModel]:
#     async with aiofiles.open(MLX_MODELS_FILE, "r") as f:
#         content = await f.read()
#         return [UnifiedModel(**model) for model in json.loads(content)]


if __name__ == "__main__":

    async def main():
        """Debug helper: list cached image-to-image models containing IP-Adapter."""
        cached = await get_models_by_hf_type("hf.qwen2_5_vl")
        for model in cached:
            print(model.type, model.id)

    asyncio.run(main())
