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

import asyncio
import json
import os
import shutil
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, List, Sequence

from huggingface_hub import HfApi, ModelInfo
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.artifact_inspector import (
    ArtifactDetection,
    inspect_paths,
)
from nodetool.integrations.huggingface.hf_fast_cache import (
    DEFAULT_MODEL_INFO_CACHE_TTL,
    HfFastCache,
)
from nodetool.metadata.types import (
    CLASSNAME_TO_MODEL_TYPE,
    HuggingFaceModel,
    ImageModel,
    LanguageModel,
    Provider,
)
from nodetool.runtime.resources import maybe_scope
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

# Explicit repo-id allowlists for ComfyUI-flavored model families.
COMFY_REPO_PATTERNS = {
    "flux": [
        "Comfy-Org/flux1-dev",
        "Comfy-Org/flux1-schnell",
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ],
    "flux_vae": ["ffxvs/vae-flux"],
    "qwen_image": ["Comfy-Org/Qwen-Image_ComfyUI", "city96/Qwen-Image-gguf"],
    "qwen_image_edit": ["Comfy-Org/Qwen-Image-Edit_ComfyUI"],
    "sd35": ["Comfy-Org/stable-diffusion-3.5-fp8"],
}

# Map hf.* comfy types to repo-id allowlists so type matching can succeed offline.
COMFY_TYPE_REPO_MATCHERS: dict[str, list[str]] = {
    "hf.flux": [*COMFY_REPO_PATTERNS["flux"]],
    "hf.flux_fp8": [*COMFY_REPO_PATTERNS["flux"]],
    "hf.stable_diffusion_3": [*COMFY_REPO_PATTERNS["sd35"]],
    "hf.qwen_image": [
        *COMFY_REPO_PATTERNS["qwen_image"],
        *COMFY_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.qwen_image_edit": [*COMFY_REPO_PATTERNS["qwen_image_edit"]],
    "hf.qwen_vl": [
        *COMFY_REPO_PATTERNS["qwen_image"],
        *COMFY_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.unet": [
        *COMFY_REPO_PATTERNS["flux"],
        *COMFY_REPO_PATTERNS["qwen_image"],
        *COMFY_REPO_PATTERNS["qwen_image_edit"],
        *COMFY_REPO_PATTERNS["sd35"],
    ],
    "hf.vae": [
        *COMFY_REPO_PATTERNS["flux_vae"],
        *COMFY_REPO_PATTERNS["qwen_image"],
        *COMFY_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.clip": [
        *COMFY_REPO_PATTERNS["sd35"],
        *COMFY_REPO_PATTERNS["qwen_image"],
        *COMFY_REPO_PATTERNS["qwen_image_edit"],
    ],
    "hf.t5": [*COMFY_REPO_PATTERNS["sd35"]],
}

# Base → checkpoint variant mapping so heuristics propagate to single-file checkpoints.
_CHECKPOINT_BASES = {
    "hf.stable_diffusion": "hf.stable_diffusion_checkpoint",
    "hf.stable_diffusion_xl": "hf.stable_diffusion_xl_checkpoint",
    "hf.stable_diffusion_3": "hf.stable_diffusion_3_checkpoint",
    "hf.stable_diffusion_xl_refiner": "hf.stable_diffusion_xl_refiner_checkpoint",
    "hf.flux": "hf.flux_checkpoint",
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
    "hf.qwen_image": ["qwen"],
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
}
# Copy keyword matchers to checkpoint variants.
for _base, _ckpt in _CHECKPOINT_BASES.items():
    if _base in HF_TYPE_KEYWORD_MATCHERS and _ckpt not in HF_TYPE_KEYWORD_MATCHERS:
        HF_TYPE_KEYWORD_MATCHERS[_ckpt] = list(HF_TYPE_KEYWORD_MATCHERS[_base])

# hf.* types that should trigger filename-based searches when building search configs.
HF_FILE_PATTERN_TYPES = {
    "hf.text_to_image",
    "hf.image_to_image",
    "hf.stable_diffusion",
    "hf.stable_diffusion_xl",
    "hf.stable_diffusion_xl_refiner",
    "hf.stable_diffusion_3",
    "hf.qwen_image",
    "hf.qwen_image_edit",
    "hf.qwen_vl",
    "hf.controlnet",
    "hf.controlnet_sdxl",
    "hf.controlnet_flux",
    "hf.ip_adapter",
    "hf.lora_sd",
    "hf.lora_sdxl",
    "hf.lora_qwen_image",
    "hf.inpainting",
    "hf.outpainting",
    "hf.stable_diffusion_checkpoint",
    "hf.stable_diffusion_xl_checkpoint",
    "hf.stable_diffusion_3_checkpoint",
    "hf.stable_diffusion_xl_refiner_checkpoint",
    "hf.flux_checkpoint",
    "hf.vae",
    "hf.unet",
    "hf.clip",
    "hf.t5",
    "hf.image_to_video",
    "hf.text_to_video",
    "hf.text_to_speech",
    "hf.text_to_text",
    "hf.image_to_text",
    "hf.text_to_audio",
    "hf.text_generation",
    "hf.sentence_similarity",
}
# Cache key/TTL used to memoize full cached-model listings to speed UI refreshes.
CACHED_HF_MODELS_CACHE_KEY = "cached_hf_models"
CACHED_HF_MODELS_TTL = 3600  # 1 hour

CACHED_HF_MODELS_CACHE_KEY = "cached_hf_models"
CACHED_HF_MODELS_TTL = 3600  # 1 hour


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
}


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

        if allow_patterns is not None and not any(
            fnmatch(sib.rfilename, pattern) for pattern in allow_patterns
        ):
            continue

        if ignore_patterns is not None and any(
            fnmatch(sib.rfilename, pattern) for pattern in ignore_patterns
        ):
            continue

        total_size += sib.size

    return total_size


def has_model_index(model_info: ModelInfo) -> bool:
    """Return True when hub metadata lists a `model_index.json` sibling."""
    return any(
        sib.rfilename == "model_index.json" for sib in (model_info.siblings or [])
    )


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
    weight_entries = [
        (name, size)
        for name, size in file_entries
        if _is_weight_file(name)
    ]
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
    if library_name and str(library_name).lower() in ("diffusers", "transformers"):
        return True
    return False


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
        if (
            lower.endswith(".safetensors")
            and size
            and size < _SMALL_ADAPTER_MAX_BYTES
            and len(weight_entries) > 1
        ):
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


def _repo_supports_diffusion_checkpoint(model_info: ModelInfo | None) -> bool:
    """
    Return True if hub metadata suggests the repo contains a raw diffusion checkpoint.

    We look for known authors and tags so single-file checkpoints (safetensors/ckpt)
    can be surfaced even without diffusers metadata or README parsing.
    """
    if model_info is None:
        return False
    if model_info.author in ("lllyasviel", "bdsqlsz"):
        return True
    if not model_info.tags:
        return False
    tags = {tag.lower() for tag in model_info.tags}
    return any(tag in tags for tag in SINGLE_FILE_DIFFUSION_TAGS) or any(
        tag in tags for tag in model_info.tags
    )


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

    model_id = (
        f"{model.repo_id}:{model.path}" if model.path is not None else model.repo_id
    )

    # Without hub lookups, size and metadata may be missing; rely on provided info only.
    if model_info is not None and size is None:
        if model.path:
            size = next(
                (
                    sib.size
                    for sib in (model_info.siblings or [])
                    if sib.rfilename == model.path
                ),
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
        downloaded=False,
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

    We avoid network calls when the README is already cached. If we must download,
    we reuse the hub's own cache layer and optionally pass the user's HF token so
    gated models can still be read when permitted.
    """
    from huggingface_hub import (
        _CACHED_NO_EXIST,
        hf_hub_download,
        try_to_load_from_cache,
    )

    # First, try to load from the HF hub cache
    cached_path = try_to_load_from_cache(repo_id=model_id, filename="README.md")

    if isinstance(cached_path, str):
        # File exists in cache, read and return it
        try:
            with open(cached_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            log.debug(f"Failed to read cached README for {model_id}: {e}")
    elif cached_path is _CACHED_NO_EXIST:
        # Non-existence is cached, return None immediately
        return None

    # File not in cache, try to download it
    try:
        # Use HF_TOKEN from secrets if available for gated model downloads
        # Note: user_id would need to be passed from caller context
        token = await get_hf_token()
        if token:
            log.debug(
                f"fetch_model_readme: Downloading README for {model_id} with HF_TOKEN (token length: {len(token)} chars)"
            )
        else:
            log.debug(
                f"fetch_model_readme: Downloading README for {model_id} without HF_TOKEN - gated models may not be accessible"
            )
        readme_path = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: hf_hub_download(
                repo_id=model_id, filename="README.md", repo_type="model", token=token
            ),
        )
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        log.debug(f"Failed to download README for {model_id}: {e}")
        return None


async def fetch_model_info(model_id: str) -> ModelInfo | None:
    """
    Fetch and cache `ModelInfo` for a repo, using the hub only when necessary.

    Results are memoized in our fast cache to keep repeated lookups cheap. Any
    errors are treated as soft failures so the rest of discovery can proceed.
    """
    cache_key = f"model_info:{model_id}"
    cached_result = HF_FAST_CACHE.model_info_cache.get(cache_key)
    if cached_result is not None:
        log.debug("Cache hit for model info: %s", model_id)
        return cached_result

    token = await get_hf_token()
    api = HfApi(token=token) if token else HfApi()

    try:
        model_info: ModelInfo = await asyncio.get_event_loop().run_in_executor(
            None, lambda: api.model_info(model_id, files_metadata=True)
        )
    except Exception as exc:
        log.debug("fetch_model_info: failed to fetch %s: %s", model_id, exc)
        return None

    HF_FAST_CACHE.model_info_cache.set(
        cache_key,
        model_info,
        DEFAULT_MODEL_INFO_CACHE_TTL,
    )
    log.debug("Cached model info for: %s", model_id)
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
    if (
        model_info.config
        and "diffusers" in model_info.config
        and "_class_name" in model_info.config["diffusers"]
    ):
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


def _infer_model_type_from_architectures(architectures: Sequence[str]) -> str | None:
    """Map common architecture names to hf.* types when config.json exposes them."""
    for arch in architectures:
        lower = arch.lower()
        if "whisper" in lower:
            return "hf.automatic_speech_recognition"
    return None


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
        rel_path
        for rel_path, _ in file_entries
        if rel_path.lower().endswith(("model_index.json", "config.json"))
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
            mapped_arch = _infer_model_type_from_architectures([str(arch) for arch in architectures])
            if mapped_arch:
                return mapped_arch

    return None


async def _build_cached_repo_entry(
    repo_id: str,
    repo_dir: Path,
    model_info: ModelInfo | None,
    recommended_models: dict[str, list[UnifiedModel]],
) -> tuple[UnifiedModel, list[tuple[str, int]]]:
    """
    Build the repo-level `UnifiedModel` plus per-file metadata for a cached HF repo.

    The function gathers file sizes from the active snapshot, runs artifact inspection
    for family/component hints, infers model type (recommended → hub metadata → local
    configs), and derives a pipeline tag when none is provided. It returns both the
    assembled `UnifiedModel` and a list of file entries so callers can emit file-level
    models or perform additional heuristics.
    """
    repo_root = await HF_FAST_CACHE.repo_root(repo_id, repo_type="model")
    snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")

    file_entries: list[tuple[str, int]] = []
    size_on_disk = 0
    snapshot_path: Path | None = None

    if snapshot_dir:
        snapshot_path = Path(snapshot_dir)
        try:
            file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        except Exception as exc:  # pragma: no cover - defensive
            log.debug(f"Failed to list files for {repo_id}: {exc}")
            file_list = []

        for file_name in file_list:
            file_path = snapshot_path / file_name
            file_size = _get_file_size(file_path)
            size_on_disk += file_size
            file_entries.append((file_name, file_size))

    artifact_detection: ArtifactDetection | None = None
    if file_entries:
        artifact_paths = [
            str((snapshot_dir and Path(snapshot_dir) / name) or (repo_dir / name))
            for name, _ in file_entries
        ]
        try:
            artifact_detection = inspect_paths(artifact_paths)
        except Exception as exc:  # pragma: no cover - best effort only
            log.debug(f"Artifact detection failed for {repo_id}: {exc}")

    model_type = model_type_from_model_info(
        recommended_models,
        repo_id,
        model_info,
    )
    if model_type is None:
        model_type = _infer_model_type_from_local_configs(
            file_entries,
            snapshot_path if snapshot_dir else None,
        )

    pipeline_tag = model_info.pipeline_tag if model_info else None
    if pipeline_tag is None and model_type:
        pipeline_tag = _derive_pipeline_tag(model_type)

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
        pipeline_tag=pipeline_tag,
        tags=model_info.tags if model_info else None,
        has_model_index=has_model_index(model_info) if model_info else False,
        repo_id=repo_id,
        path=None,
        size_on_disk=size_on_disk,
        downloads=model_info.downloads if model_info else None,
        likes=model_info.likes if model_info else None,
        trending_score=model_info.trending_score if model_info else None,
        artifact_family=artifact_detection.family if artifact_detection else None,
        artifact_component=artifact_detection.component if artifact_detection else None,
        artifact_confidence=artifact_detection.confidence if artifact_detection else None,
        artifact_evidence=artifact_detection.evidence if artifact_detection else None,
    )

    return repo_model, file_entries


async def read_cached_hf_models() -> List[UnifiedModel]:
    """
    Enumerate all cached HF repos and return repo-level `UnifiedModel` entries.

    The scan is offline-only: discover repos via `HfFastCache`, build entries using
    `_build_cached_repo_entry`, and memoize the result for an hour to avoid repeated
    filesystem traversal during UI interactions.
    """

    cached_models = HF_FAST_CACHE.model_info_cache.get(CACHED_HF_MODELS_CACHE_KEY)
    if cached_models is not None:
        return cached_models

    try:
        # Discover repos by listing cache directory (lightweight)
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(f"Failed to discover cached HF repos: {exc}")
        return []

    recommended_models = get_recommended_models()
    models: list[UnifiedModel] = []
    model_infos = [None for _ in repo_list]

    for (repo_id, repo_dir), model_info in zip(repo_list, model_infos, strict=False):
        # Handle exceptions from individual fetch_model_info calls
        if isinstance(model_info, BaseException):
            log.debug(f"Failed to fetch model info for {repo_id}: {model_info}")
            # Still create a basic model entry without the extra metadata
            model_info = None

        repo_model, _ = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            model_info,
            recommended_models,
        )
        models.append(repo_model)

    HF_FAST_CACHE.model_info_cache.set(
        CACHED_HF_MODELS_CACHE_KEY,
        models,
        CACHED_HF_MODELS_TTL,
    )

    return models


# Static search hints per hf.* type used to build repo/file queries (offline/hub).
HF_SEARCH_TYPE_CONFIG: dict[str, dict[str, list[str] | str]] = {
    "hf.text_to_image": {"pipeline_tag": ["text-to-image"], "filename_pattern": HF_DEFAULT_FILE_PATTERNS},
    "hf.image_to_image": {"pipeline_tag": ["image-to-image"], "filename_pattern": HF_DEFAULT_FILE_PATTERNS},
    "hf.stable_diffusion": {
        "pipeline_tag": ["text-to-image"],
        "tag": ["*stable-diffusion*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
    },
    "hf.stable_diffusion_xl": {
        "tag": ["diffusers:StableDiffusionXLPipeline", "*stable-diffusion-xl*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
    },
    "hf.stable_diffusion_xl_refiner": {"tag": ["*refiner*"], "filename_pattern": HF_DEFAULT_FILE_PATTERNS},
    "hf.stable_diffusion_3": {
        "tag": ["*stable-diffusion-3*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": COMFY_REPO_PATTERNS["sd35"],
    },
    "hf.flux": {
        "tag": ["*flux*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [*COMFY_REPO_PATTERNS["flux"], "*flux*"],
    },
    "hf.flux_fp8": {
        "tag": ["*flux*"],
        "filename_pattern": [
            "*fp8*.safetensors",
            "*fp8*.ckpt",
            "*fp8*.bin",
            "*fp8*.pt",
            "*fp8*.pth",
        ],
        "repo_pattern": [*COMFY_REPO_PATTERNS["flux"], "*flux*"],
    },
    "hf.qwen_image": {
        "tag": ["*qwen*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": COMFY_REPO_PATTERNS["qwen_image"],
    },
    "hf.qwen_image_edit": {
        "pipeline_tag": ["image-to-image"],
        "tag": ["*qwen*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": COMFY_REPO_PATTERNS["qwen_image_edit"],
    },
    "hf.qwen_vl": {
        "tag": ["*qwen*"],
        "filename_pattern": HF_DEFAULT_FILE_PATTERNS,
        "repo_pattern": [
            *COMFY_REPO_PATTERNS["qwen_image"],
            *COMFY_REPO_PATTERNS["qwen_image_edit"],
        ],
    },
    "hf.controlnet": {
        "repo_pattern": ["*control*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
        "pipeline_tag": [],
    },
    "hf.controlnet_sdxl": {
        "repo_pattern": ["*control*"],
        "tag": ["*sdxl*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
        "pipeline_tag": [],
    },
    "hf.controlnet_flux": {
        "repo_pattern": ["*control*"],
        "tag": ["*flux*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
        "pipeline_tag": [],
    },
    "hf.ip_adapter": {
        "repo_pattern": ["*IP-Adapter*"],
        "filename_pattern": [*HF_DEFAULT_FILE_PATTERNS, *HF_PTH_FILE_PATTERNS],
        "pipeline_tag": [],
    },
    "hf.lora_sd": {"repo_pattern": ["*lora*"], "pipeline_tag": []},
    "hf.lora_sdxl": {"repo_pattern": ["*lora*sdxl*", "*sdxl*lora*"], "pipeline_tag": []},
    "hf.lora_qwen_image": {"repo_pattern": ["*lora*qwen*"], "pipeline_tag": []},
    "hf.unet": {
        "repo_pattern": [
            *COMFY_REPO_PATTERNS["flux"],
            *COMFY_REPO_PATTERNS["qwen_image"],
            *COMFY_REPO_PATTERNS["qwen_image_edit"],
            *COMFY_REPO_PATTERNS["sd35"],
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
        "pipeline_tag": [],
    },
    "hf.vae": {
        "repo_pattern": [
            *COMFY_REPO_PATTERNS["flux_vae"],
            *COMFY_REPO_PATTERNS["qwen_image"],
            *COMFY_REPO_PATTERNS["qwen_image_edit"],
            "*vae*",
            "*stable-diffusion*",
        ],
        "filename_pattern": ["*vae*.safetensors", "*vae*.bin", "*vae*.ckpt", "*vae*.pt"],
        "pipeline_tag": [],
    },
    "hf.clip": {
        "repo_pattern": [
            *COMFY_REPO_PATTERNS["sd35"],
            *COMFY_REPO_PATTERNS["qwen_image"],
            *COMFY_REPO_PATTERNS["qwen_image_edit"],
            "*clip*",
            "*flux*",
        ],
        "filename_pattern": ["*clip*.safetensors", "*clip*.bin", "*clip*.gguf", "*clip*.ckpt"],
        "pipeline_tag": [],
    },
    "hf.t5": {
        "repo_pattern": [*COMFY_REPO_PATTERNS["sd35"], "*t5*", "*flux*"],
        "filename_pattern": ["*t5*.safetensors", "*t5*.bin", "*t5*.gguf", "*t5*.ckpt"],
        "pipeline_tag": [],
    },
    "hf.image_to_video": {"pipeline_tag": ["image-to-video"]},
    "hf.text_to_video": {"pipeline_tag": ["text-to-video"]},
    "hf.image_to_text": {"pipeline_tag": ["image-to-text"], "tag": ["*caption*"]},
    "hf.inpainting": {"pipeline_tag": ["image-inpainting"], "tag": ["*inpaint*"]},
    "hf.outpainting": {"tag": ["*outpaint*"]},
}

# Derive checkpoint variants (single-file) from base configs.
for _base, _ckpt in _CHECKPOINT_BASES.items():
    if _base in HF_SEARCH_TYPE_CONFIG and _ckpt not in HF_SEARCH_TYPE_CONFIG:
        _base_cfg = HF_SEARCH_TYPE_CONFIG[_base]
        HF_SEARCH_TYPE_CONFIG[_ckpt] = {k: (list(v) if isinstance(v, list) else v) for k, v in _base_cfg.items()}

def get_supported_hf_types() -> list[tuple[str, bool]]:
    """
    Return supported hf.* model types and whether they have built-in search config.

    The boolean indicates if the type has a predefined search configuration
    (works without a task override). Types without a configuration can still be
    used with get_models_by_hf_type when a task is provided.
    """
    configured: set[str] = set(HF_SEARCH_TYPE_CONFIG.keys()) | set(COMFY_TYPE_REPO_MATCHERS.keys())
    task_only = set(HF_FILE_PATTERN_TYPES) - configured

    supported: list[tuple[str, bool]] = []
    for model_type in sorted(configured):
        supported.append((model_type, True))
    for model_type in sorted(task_only):
        supported.append((model_type, False))
    return supported

GENERIC_HF_TYPES = {
    "hf.text_to_image",
    "hf.image_to_image",
    "hf.model",
    "hf.model_generic",
}


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


def _build_search_config_for_type(
    model_type: str, task: str | None = None
) -> dict[str, list[str] | str] | None:
    """
    Construct a repository search configuration for a given hf.* type.

    Combines static search hints (repo patterns, filename patterns, pipeline tags)
    with derived pipeline tags so we can reuse the same logic for both hub and
    offline cache searches. Returns None when the type is unsupported.
    """
    normalized_type = model_type.lower()
    base_config = HF_SEARCH_TYPE_CONFIG.get(normalized_type)
    if not base_config and not task:
        return None
    config: dict[str, list[str] | str] = {}

    for key in ("repo_pattern", "filename_pattern", "pipeline_tag", "tag", "author", "library_name"):
        value = base_config.get(key) if base_config else None
        if value:
            config[key] = list(value) if isinstance(value, list) else value

    if "pipeline_tag" not in config:
        derived = _derive_pipeline_tag(normalized_type, task)
        if derived:
            config["pipeline_tag"] = [derived]

    if "filename_pattern" not in config and normalized_type in HF_FILE_PATTERN_TYPES:
        config["filename_pattern"] = list(HF_DEFAULT_FILE_PATTERNS)

    return config if config else None


def _matches_repo_for_type(normalized_type: str, repo_id: str, repo_id_from_id: str) -> bool:
    """Check if a repo id matches any hard-coded comfy-type mappings for a model type."""
    matchers = COMFY_TYPE_REPO_MATCHERS.get(normalized_type)
    if not matchers:
        return False
    repo_lower = repo_id.lower()
    repo_from_id_lower = repo_id_from_id.lower()
    return any(
        repo_lower == candidate.lower() or repo_from_id_lower == candidate.lower() for candidate in matchers
    )


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
    if normalized_type in {"hf.flux", "hf.flux_fp8"}:
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
    """
    Decide if a `UnifiedModel` fits the requested hf.* type using layered heuristics.

    Checks include:
    - Exact/variant type matches (including checkpoint variants)
    - Repo id allowlists for comfy-specific repos
    - Artifact detection hints (family/component)
    - Keyword matches across repo id, tags, and file paths
    - Derived pipeline tag alignment
    """
    normalized_type = model_type.lower()
    checkpoint_variant = None
    if normalized_type.endswith("_checkpoint"):
        checkpoint_variant = normalized_type
        normalized_type = normalized_type[: -len("_checkpoint")]
    model_type_lower = (model.type or "").lower()
    repo_id = (model.repo_id or "").lower()
    repo_id_from_id = (model.id or "").split(":")[0].lower() if model.id else ""
    path_lower = (model.path or "").lower()

    def _is_qwen_text_encoder(path: str | None) -> bool:
        """Detect qwen text encoder paths so we can exclude them from base model matches."""
        if not path:
            return False
        return "text_encoders" in path or "text_encoder" in path or "qwen_2.5_vl" in path

    def _is_qwen_vae(path: str | None) -> bool:
        """Detect qwen VAE paths to avoid misclassifying auxiliary components."""
        if not path:
            return False
        return "vae" in path

    qwen_family_types = {"hf.qwen_image", "hf.qwen_image_checkpoint"}
    target_types = {normalized_type}
    if checkpoint_variant:
        target_types.add(checkpoint_variant)
    model_type_base = (
        model_type_lower[: -len("_checkpoint")] if model_type_lower.endswith("_checkpoint") else model_type_lower
    )
    if model_type_lower:
        if model_type_lower in target_types or model_type_base == normalized_type:
            if normalized_type in {"hf.qwen_image", "hf.qwen_image_edit"} and (
                _is_qwen_text_encoder(path_lower) or _is_qwen_vae(path_lower)
            ):
                return False
            return True
        if model_type_lower not in GENERIC_HF_TYPES:
            allowed_family = normalized_type in {"hf.qwen_image_checkpoint", "hf.qwen_vl", "hf.vae"} and (
                model_type_lower in qwen_family_types
            )
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
    if keywords and any(keyword in repo_id or any(keyword in tag for tag in tags) for keyword in keywords):
        return True
    if keywords and path_lower and any(keyword in path_lower for keyword in keywords):
        return True

    derived_pipeline = _derive_pipeline_tag(normalized_type)
    if derived_pipeline and model.pipeline_tag == derived_pipeline:
        return True

    return False


async def get_models_by_hf_type(model_type: str, task: str | None = None) -> list[UnifiedModel]:
    """
    Return cached Hugging Face models matching a requested hf.* type.

    The search is entirely offline: build a search config, scan cached repos/files,
    then apply the same client-side heuristics (keyword matching, repo patterns,
    artifact hints) to label each result with the desired type.
    """

    normalized_type = (model_type or "").lower()
    config = _build_search_config_for_type(normalized_type, task)
    if config is None:
        return []
    log.debug(
        "get_models_by_hf_type: type=%s task=%s repo_pattern=%s filename_pattern=%s pipeline_tag=%s tags=%s authors=%s library_name=%s",
        normalized_type,
        task,
        config.get("repo_pattern"),
        config.get("filename_pattern"),
        config.get("pipeline_tag"),
        config.get("tag"),
        config.get("author"),
        config.get("library_name"),
    )

    repo_patterns = config.get("repo_pattern") or []
    literal_repo_ids = [
        repo for repo in repo_patterns if repo and not any(ch in repo for ch in ["*", "?", "["])
    ]

    pre_resolved_repos: list[tuple[str, Path]] = []
    if literal_repo_ids:
        for repo in literal_repo_ids:
            try:
                root = await HF_FAST_CACHE.repo_root(repo, repo_type="model")
            except Exception as exc:  # pragma: no cover - defensive guard
                log.debug(f"repo_root failed for {repo}: {exc}")
                root = None
            if root:
                pre_resolved_repos.append((repo, Path(root)))
    has_wildcards = any(any(ch in repo for ch in ["*", "?", "["]) for repo in repo_patterns)
    pre_resolved_for_search = None if has_wildcards else (pre_resolved_repos or None)

    def _filter_models(models: list[UnifiedModel]) -> list[UnifiedModel]:
        """Apply type-specific filters to remove mismatched repo/file entries."""
        filtered: list[UnifiedModel] = []
        file_only_types = {
            "hf.unet",
            "hf.vae",
            "hf.clip",
            "hf.t5",
            "hf.qwen_vl",
            "hf.stable_diffusion_checkpoint",
            "hf.stable_diffusion_xl_checkpoint",
            "hf.stable_diffusion_3_checkpoint",
            "hf.stable_diffusion_xl_refiner_checkpoint",
            "hf.flux_checkpoint",
        }
        checkpoint_types = set(_CHECKPOINT_BASES.values())
        nested_checkpoint_types = {"hf.qwen_image_checkpoint", "hf.qwen_image_edit_checkpoint"}
        single_file_repo_types = {
            "hf.flux",
            "hf.flux_fp8",
            "hf.stable_diffusion",
            "hf.stable_diffusion_xl",
            "hf.stable_diffusion_3",
            "hf.stable_diffusion_xl_refiner",
        }
        seen: set[str] = set()
        for model in models:
            if model.id in seen:
                continue
            repo_lower = (model.repo_id or "").lower()
            path_value = getattr(model, "path", None)
            if normalized_type in file_only_types and getattr(model, "path", None) is None:
                continue
            if normalized_type in single_file_repo_types:
                # Skip repo-only entries for gguf-style repos; prefer per-file entries there.
                if path_value is None and "gguf" in repo_lower:
                    continue
                if path_value:
                    # Only keep single-file checkpoints/gguf weights to avoid auxiliary components.
                    path_lower = path_value.lower()
                    if not _is_single_file_diffusion_weight(path_value) and not path_lower.endswith(".gguf"):
                        continue
            if normalized_type in checkpoint_types:
                if not model.path:
                    continue
                if "/" in model.path and normalized_type not in nested_checkpoint_types:
                    continue
            if _matches_model_type(model, normalized_type):
                try:
                    model.type = normalized_type  # type: ignore[assignment]
                except Exception:
                    model = model.copy(update={"type": normalized_type})
                filtered.append(model)
                seen.add(model.id)
        return filtered

    # Offline-first search to avoid network dependency when cache is present.
    offline_models = await search_cached_hf_models(
        repo_patterns=config.get("repo_pattern"),
        filename_patterns=config.get("filename_pattern"),
        pipeline_tags=config.get("pipeline_tag"),
        tags=config.get("tag"),
        authors=config.get("author"),
        library_name=config.get("library_name"),
        pre_resolved_repos=pre_resolved_for_search,
    )
    offline_filtered = _filter_models(offline_models)
    if offline_filtered:
        log.debug(
            "get_models_by_hf_type: returning %d models from offline cache (type=%s)",
            len(offline_filtered),
            normalized_type,
        )
        return offline_filtered

    return offline_filtered


def _fallback_unified_model(repo_id: str, model_type: str) -> UnifiedModel:
    """Build a minimal `UnifiedModel` placeholder when no metadata is available."""
    pipeline_tag = _derive_pipeline_tag(model_type)
    return UnifiedModel(
        id=repo_id,
        repo_id=repo_id,
        path=None,
        type=model_type,
        name=repo_id,
        cache_path=None,
        allow_patterns=None,
        ignore_patterns=None,
        description=None,
        readme=None,
        size_on_disk=None,
        downloaded=False,
        pipeline_tag=pipeline_tag,
        tags=None,
        has_model_index=None,
        downloads=None,
        likes=None,
        trending_score=None,
    )


def _build_offline_models_for_repos(
    repo_ids: Sequence[str],
    model_type: str,
    filename_patterns: Sequence[str] | None,
    pipeline_tags: Sequence[str] | None,
) -> list[UnifiedModel]:
    """
    Build `UnifiedModel` entries directly from cached snapshots (offline fallback).

    Used when hub metadata is unavailable; emits a repo-level model plus per-file
    models that match the provided filename patterns.
    """
    results: list[UnifiedModel] = []
    pipeline_tag = pipeline_tags[0] if pipeline_tags else _derive_pipeline_tag(model_type)
    for repo_id in repo_ids:
        snapshot_dir = _offline_snapshot_dir(repo_id)
        if snapshot_dir is None:
            continue
        repo_model = UnifiedModel(
            id=repo_id,
            repo_id=repo_id,
            path=None,
            type=model_type,
            name=repo_id,
            cache_path=str(snapshot_dir.parent),
            allow_patterns=None,
            ignore_patterns=None,
            description=None,
            readme=None,
            size_on_disk=None,
            downloaded=True,
            pipeline_tag=pipeline_tag,
            tags=None,
            has_model_index=None,
            downloads=None,
            likes=None,
            trending_score=None,
        )
        results.append(repo_model)
        if filename_patterns:
            rel_files = _offline_snapshot_files(snapshot_dir)
            for relpath in rel_files:
                if not any(fnmatch(relpath, pat) for pat in filename_patterns):
                    continue
                file_id = f"{repo_id}:{relpath}"
                results.append(
                    UnifiedModel(
                        id=file_id,
                        repo_id=repo_id,
                        path=relpath,
                        type=model_type,
                        name=f"{repo_id}/{relpath}",
                        cache_path=str(snapshot_dir.parent),
                        allow_patterns=None,
                        ignore_patterns=None,
                        description=None,
                        readme=None,
                        size_on_disk=None,
                        downloaded=True,
                        pipeline_tag=pipeline_tag,
                        tags=None,
                        has_model_index=None,
                        downloads=None,
                        likes=None,
                        trending_score=None,
                    )
                )
    return results


def _offline_snapshot_dir(repo_id: str) -> Path | None:
    """Return the newest snapshot directory for a repo if it exists locally."""
    repo_bits = [bit for bit in repo_id.split("/") if bit]
    if not repo_bits:
        return None
    repo_dir = HF_FAST_CACHE.cache_dir / ("models--" + "--".join(repo_bits))
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    newest = None
    newest_mtime = None
    for entry in snapshots_dir.iterdir():
        if not entry.is_dir():
            continue
        mt = entry.stat().st_mtime
        if newest_mtime is None or mt > newest_mtime:
            newest = entry
            newest_mtime = mt
    return newest


def _offline_snapshot_files(snapshot_dir: Path) -> list[str]:
    """List snapshot-relative file paths for a cached repo snapshot."""
    relpaths: list[str] = []
    for root, _, files in os.walk(snapshot_dir):
        for fname in files:
            full = Path(root) / fname
            relpaths.append(str(full.relative_to(snapshot_dir)))
    return relpaths


def _normalize_patterns(values: Sequence[str] | None, *, lower: bool = False) -> list[str]:
    """Trim/normalize pattern inputs and optionally lowercase them for matching."""
    normalized: list[str] = []
    for value in values or []:
        if value is None:
            continue
        trimmed = value.strip()
        if not trimmed:
            continue
        normalized.append(trimmed.lower() if lower else trimmed)
    return normalized


def _matches_any_pattern(value: str, patterns: list[str]) -> bool:
    """Case-sensitive glob check; empty pattern list means match everything."""
    if not patterns:
        return True
    return any(fnmatch(value, pattern) for pattern in patterns)


def _matches_any_pattern_ci(value: str, patterns: list[str]) -> bool:
    """Case-insensitive glob check used when filtering by repo id."""
    value_lower = value.lower()
    return any(fnmatch(value_lower, pattern.lower()) for pattern in patterns)


def _repo_tags_match_patterns(repo_tags: list[str], patterns: list[str]) -> bool:
    """Ensure all requested tag patterns are present on the repo (logical AND)."""
    if not patterns:
        return True
    if not repo_tags:
        return False
    for pattern in patterns:
        if not any(fnmatch(tag, pattern) for tag in repo_tags):
            return False
    return True


async def search_cached_hf_models(
    repo_patterns: Sequence[str] | None = None,
    filename_patterns: Sequence[str] | None = None,
    pipeline_tags: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    authors: Sequence[str] | None = None,
    library_name: str | None = None,
    *,
    skip_model_info: bool = False,
    pre_resolved_repos: Sequence[tuple[str, Path]] | None = None,
) -> List[UnifiedModel]:
    """
    Search the local HF cache for repos/files matching metadata and pattern filters.

    The function is hub-free: it discovers repos from disk, optionally pre-resolves
    specific repos to avoid globbing, applies pipeline/tag/author/library filters
    when metadata is available, and emits both repo-level and file-level `UnifiedModel`
    entries when filename patterns are provided.
    """
    # Always skip hub metadata to avoid network calls.
    skip_model_info = True
    if pre_resolved_repos is not None:
        repo_list = list(pre_resolved_repos)
    else:
        try:
            repo_list = await HF_FAST_CACHE.discover_repos("model")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.warning(f"Failed to discover cached HF repos: {exc}")
            return []

    if not repo_list:
        return []

    repo_pattern_list = _normalize_patterns(repo_patterns)
    filename_pattern_list = _normalize_patterns(filename_patterns)
    pipeline_tag_patterns = _normalize_patterns(pipeline_tags, lower=True)
    tag_patterns = _normalize_patterns(tags, lower=True)
    author_patterns = _normalize_patterns(authors, lower=True)
    library_pattern = (
        library_name.strip().lower() if library_name and library_name.strip() else None
    )

    recommended_models = get_recommended_models()
    results: list[UnifiedModel] = []
    requires_metadata = False
    log.debug(
        "search_cached_hf_models: repos=%s files=%s pipeline_tags=%s tags=%s authors=%s library=%s skip_info=%s pre_resolved=%d",
        repo_pattern_list,
        filename_pattern_list,
        pipeline_tag_patterns,
        tag_patterns,
        author_patterns,
        library_pattern,
        skip_model_info,
        len(repo_list),
    )

    model_infos = [None for _ in repo_list]

    for (repo_id, repo_dir), model_info in zip(repo_list, model_infos, strict=False):
        info: ModelInfo | None
        if isinstance(model_info, BaseException):
            log.debug(f"Failed to fetch model info for {repo_id}: {model_info}")
            info = None
        else:
            info = model_info

        if repo_pattern_list and not _matches_any_pattern_ci(repo_id, repo_pattern_list):
            continue

        if requires_metadata and info is None:
            continue

        if info:
            pipeline_value = (info.pipeline_tag or "").lower()
            if pipeline_tag_patterns and not _matches_any_pattern(
                pipeline_value, pipeline_tag_patterns
            ):
                continue

            repo_tags = [tag.lower() for tag in (info.tags or [])]
            if not _repo_tags_match_patterns(repo_tags, tag_patterns):
                continue

            author_value = (info.author or "").lower()
            if author_patterns and not _matches_any_pattern(
                author_value, author_patterns
            ):
                continue

            library_value = (getattr(info, "library_name", "") or "").lower()
            if library_pattern and not fnmatch(library_value, library_pattern):
                continue

        repo_model, file_entries = await _build_cached_repo_entry(
            repo_id,
            repo_dir,
            info,
            recommended_models,
        )
        results.append(repo_model)

        if filename_pattern_list and file_entries:
            for relative_name, file_size in file_entries:
                if not _matches_any_pattern(relative_name, filename_pattern_list):
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
        len(repo_list),
    )
    return results


async def _filter_repos_by_metadata(
    pipeline_tag: str | None = None,
    library_name: str | None = None,
    tags: list[str] | None = None,
    predicate: Callable[[ModelInfo], bool] | None = None,
) -> list[tuple[str, Path, ModelInfo]]:
    """Metadata filtering is disabled (hub-free mode); return empty list."""
    log.debug("_filter_repos_by_metadata: metadata filtering disabled")
    return []


async def get_llamacpp_language_models_from_hf_cache() -> List[LanguageModel]:
    """
    Return LanguageModel entries for cached Hugging Face repos containing GGUF files
    that look suitable for llama.cpp.

    Heuristics:
    - File ends with .gguf (case-insensitive)
    - Each GGUF file yields a LanguageModel with id "<repo_id>:<filename>"

    Returns:
        List[LanguageModel]: Llama.cpp-compatible models discovered in the HF cache
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(f"get_llamacpp_language_models_from_hf_cache: discover failed: {exc}")
        return []

    results: list[LanguageModel] = []

    for repo_id, _repo_dir in repo_list:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
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


async def get_vllm_language_models_from_hf_cache() -> List[LanguageModel]:
    """Return LanguageModel entries based on cached weight files (hub-free)."""
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(f"get_vllm_language_models_from_hf_cache: discover failed: {exc}")
        return []
    seen_repos: set[str] = set()
    results: list[LanguageModel] = []

    SUPPORTED_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")

    for repo_id, _repo_dir in repo_list:
        if repo_id in seen_repos:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        if any(fname.lower().endswith(SUPPORTED_WEIGHT_EXTENSIONS) for fname in file_list):
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


async def get_mlx_language_models_from_hf_cache() -> List[LanguageModel]:
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
        log.debug(f"get_mlx_language_models_from_hf_cache: discover failed: {exc}")
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


async def get_text_to_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are text-to-image models,
    including single-file checkpoints stored at the repo root (e.g. Stable Diffusion safetensors).
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(f"get_text_to_image_models_from_hf_cache: discover failed: {exc}")
        return []

    result: dict[str, ImageModel] = {}
    for repo_id, _repo_dir in repo_list:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        added_single_file = False
        for fname in file_list:
            if not _is_single_file_diffusion_weight(fname):
                continue
            model_id = f"{repo_id}:{fname}"
            display = f"{repo_id.split('/')[-1]} • {fname}"
            result[model_id] = ImageModel(
                id=model_id,
                name=display,
                provider=Provider.HuggingFace,
                supported_tasks=["text_to_image"],
            )
            added_single_file = True
        if added_single_file:
            result.setdefault(
                repo_id,
                ImageModel(
                    id=repo_id,
                    name=repo_id.split("/")[-1],
                    provider=Provider.HuggingFace,
                    supported_tasks=["text_to_image"],
                ),
            )

    return list(result.values())


async def get_image_to_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are image-to-image models,
    including single-file checkpoints stored at the repo root.
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(f"get_image_to_image_models_from_hf_cache: discover failed: {exc}")
        return []

    result: dict[str, ImageModel] = {}
    for repo_id, _repo_dir in repo_list:
        snapshot_dir = await HF_FAST_CACHE.active_snapshot_dir(repo_id, repo_type="model")
        if not snapshot_dir:
            continue
        file_list = await HF_FAST_CACHE.list_files(repo_id, repo_type="model")
        added_single_file = False
        for fname in file_list:
            if not _is_single_file_diffusion_weight(fname):
                continue
            model_id = f"{repo_id}:{fname}"
            display = f"{repo_id.split('/')[-1]} • {fname}"
            result[model_id] = ImageModel(
                id=model_id,
                name=display,
                provider=Provider.HuggingFace,
                supported_tasks=["image_to_image"],
            )
            added_single_file = True
        if added_single_file:
            result.setdefault(
                repo_id,
                ImageModel(
                    id=repo_id,
                    name=repo_id.split("/")[-1],
                    provider=Provider.HuggingFace,
                    supported_tasks=["image_to_image"],
                ),
            )

    return list(result.values())


async def get_mlx_image_models_from_hf_cache() -> List[ImageModel]:
    """
    Return ImageModel entries for cached Hugging Face repos that are mflux models
    (MLX-compatible image generation models).

    Returns:
        List[ImageModel]: MLX-compatible image models (mflux) discovered in the HF cache
    """
    try:
        repo_list = await HF_FAST_CACHE.discover_repos("model")
    except Exception as exc:  # pragma: no cover - defensive
        log.debug(f"get_mlx_image_models_from_hf_cache: discover failed: {exc}")
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


async def _fetch_models_by_author(
    user_id: str | None = None, **kwargs
) -> list[ModelInfo]:
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
        api = HfApi(token=token)
    else:
        log.debug(
            f"_fetch_models_by_author: Fetching models for author {author} without HF_TOKEN - gated models may not be accessible"
        )
        api = HfApi()
    # Run the blocking call in a thread executor
    models = await asyncio.get_event_loop().run_in_executor(
        None, lambda: api.list_models(**kwargs)
    )
    return list(models)


async def get_gguf_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "downloads",
    tags: str = "gguf",
) -> List[UnifiedModel]:
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
    seen_file: set[str] = set()
    for info in model_infos:
        if info is None:
            continue
        sibs = info.siblings or []
        for sib in sibs:
            fname = getattr(sib, "rfilename", None)
            if not isinstance(fname, str) or not fname.lower().endswith(".gguf"):
                continue
            if fname in seen_file:
                continue
            seen_file.add(fname)
            tasks.append(
                (
                    HuggingFaceModel(type="llama_cpp", repo_id=info.id, path=fname),
                    info,
                    sib.size,
                )
            )

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(
        *[unified_model(model, info, size) for model, info, size in tasks]
    )

    # Sort for stability: repo then filename
    entries = [entry for entry in entries if entry is not None]
    return entries


async def get_mlx_language_models_from_authors(
    authors: list[str],
    limit: int = 200,
    sort: str = "trending_score",
    tags: str = "mlx",
) -> List[UnifiedModel]:
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
        *(
            _fetch_models_by_author(
                user_id=None, author=a, limit=limit, sort=sort, tags=tags
            )
            for a in authors
        )
    )
    model_infos = [item for sublist in results for item in sublist]

    # Execute all unified_model calls in parallel
    entries = await asyncio.gather(
        *[
            unified_model(HuggingFaceModel(type="mlx", repo_id=info.id), info)
            for info in model_infos
        ]
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

    # Purge all HuggingFace caches after successful deletion
    log.info("Purging HuggingFace model caches after model deletion")
    HF_FAST_CACHE.model_info_cache.delete_pattern("cached_hf_*")
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
        cached = await get_image_to_image_models_from_hf_cache()
        for model in cached:
            if "IP-Adapter" in model.id:
                print(model.path)

    asyncio.run(main())
