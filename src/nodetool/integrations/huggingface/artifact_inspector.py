"""Lightweight artifact inspectors for cached model files.

All inspectors are header/metadata only and avoid loading full tensors. Detected
metadata is intended to annotate UnifiedModel entries so the UI can present
family/component hints (e.g., Flux, SDXL, LLaMA GGUF) without expensive I/O.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from nodetool.integrations.huggingface.safetensors_inspector import (
        DetectionResult as STFDetectionResult,
    )


@dataclass
class ArtifactDetection:
    family: str | None
    component: str | None
    confidence: float | None
    evidence: list[str]


def inspect_paths(paths: Sequence[str | Path]) -> ArtifactDetection | None:
    """Inspect a collection of cached artifact files and return the strongest signal."""
    if not paths:
        return None

    # Priority: safetensors → gguf → torch bin/pt/ckpt → config/model_index metadata
    safetensors = [p for p in paths if str(p).lower().endswith(".safetensors")]
    ggufs = [p for p in paths if str(p).lower().endswith(".gguf")]
    torch_bins = [p for p in paths if str(p).lower().endswith((".bin", ".pt", ".ckpt", ".pth"))]
    config_files = [p for p in paths if str(p).lower().endswith("config.json")]
    model_index_files = [p for p in paths if str(p).lower().endswith("model_index.json")]

    if safetensors:
        from nodetool.integrations.huggingface.safetensors_inspector import (
            detect_model as detect_safetensors_model,
        )

        res = _wrap_stf(detect_safetensors_model(safetensors, framework="np", max_shape_reads=6))

        if res:
            return res

    if ggufs:
        res = detect_gguf(ggufs)
        if res:
            return res

    if torch_bins:
        res = detect_torch_bin(torch_bins)
        if res:
            return res

    if config_files or model_index_files:
        res = detect_from_json(config_files, model_index_files)
        if res:
            return res

    return None


def _wrap_stf(result: STFDetectionResult | None) -> ArtifactDetection | None:
    if result is None:
        return None
    return ArtifactDetection(
        family=result.family,
        component=result.component,
        confidence=result.confidence,
        evidence=result.evidence or [],
    )


# ---- GGUF detection ---------------------------------------------------------


def detect_gguf(paths: Sequence[str | Path]) -> ArtifactDetection | None:
    """Parse gguf headers to extract arch / quant info."""
    if not paths:
        return None

    try:
        info = _read_gguf_header(Path(paths[0]))
    except Exception:
        return None

    arch = info.get("general.architecture", "").lower()
    quant = info.get("general.name", "").lower()
    family = None
    evidence = []
    if arch:
        evidence.append(f"arch={arch}")
        if "llama" in arch or "mistral" in arch or "gemma" in arch or "qwen" in arch:
            family = "llama-family"
        elif "phi" in arch:
            family = "phi"
        elif "gptneox" in arch:
            family = "gpt-neox"
    if not family and "qwen" in quant:
        family = "qwen-family"
    if quant:
        evidence.append(f"quant={quant}")
    return ArtifactDetection(
        family=family or "gguf-unknown",
        component="llm",
        confidence=0.55 if not arch else 0.75,
        evidence=evidence,
    )


# GGUF metadata value-type enum (v2/v3). Fixed-width scalar sizes in bytes.
# String(8) and Array(9) are variable-length and handled separately.
_GGUF_TYPE_SIZES = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


def _read_gguf_string(f) -> str:
    """Read a GGUF string: uint64 length prefix followed by UTF-8 bytes."""
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8", errors="replace")


def _read_gguf_value(f, value_type: int) -> str | None:
    """Consume a single GGUF metadata value, keeping the stream aligned.

    Returns the decoded string for STRING values, otherwise ``None`` after
    skipping the value's bytes (we only need string values to reach the
    ``general.*`` keys ``detect_gguf`` cares about).
    """
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(f)
    if value_type == _GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _read_gguf_value(f, elem_type)
        return None
    size = _GGUF_TYPE_SIZES.get(value_type)
    if size is None:
        raise ValueError(f"Unknown GGUF value type: {value_type}")
    f.read(size)
    return None


def _read_gguf_header(path: Path) -> dict[str, str]:
    """Minimal GGUF (v2/v3) header reader.

    Follows the real GGUF layout so the stream stays aligned across all value
    types and we can reach the ``general.*`` keys used for detection. Only
    string-valued metadata entries are recorded.
    """
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("Not a GGUF file")
        struct.unpack("<I", f.read(4))  # version (uint32)
        struct.unpack("<Q", f.read(8))  # tensor_count (uint64)
        (kv_count,) = struct.unpack("<Q", f.read(8))  # kv_count (uint64)

        info: dict[str, str] = {}
        for _ in range(kv_count):
            key = _read_gguf_string(f)
            (value_type,) = struct.unpack("<I", f.read(4))
            value = _read_gguf_value(f, value_type)
            if value is not None:
                info[key] = value
        return info


# ---- Torch bin detection ----------------------------------------------------


def detect_torch_bin(paths: Sequence[str | Path]) -> ArtifactDetection | None:
    """Try to infer family/component from torch state_dict keys without loading tensors."""
    if not paths:
        return None

    try:
        keys = _sample_torch_keys(Path(paths[0]), limit=200)
    except Exception:
        return None

    if not keys:
        return None

    evidence: list[str] = []
    # LLaMA/Qwen style
    if any(k.startswith("model.layers.") for k in keys):
        evidence.append("found model.layers.*")
        return ArtifactDetection(
            family="llama-family",
            component="llm",
            confidence=0.7,
            evidence=evidence,
        )
    if any("text_model.encoder.layers" in k for k in keys):
        evidence.append("found text_model.encoder.layers.*")
        return ArtifactDetection(
            family="clip-text-encoder",
            component="text_encoder",
            confidence=0.6,
            evidence=evidence,
        )
    if any(k.startswith("down_blocks.") for k in keys):
        evidence.append("found down_blocks.* (UNet)")
        return ArtifactDetection(
            family="sd-or-sdxl-unknown",
            component="unet",
            confidence=0.5,
            evidence=evidence,
        )

    return ArtifactDetection(
        family=None,
        component=None,
        confidence=None,
        evidence=[],
    )


def _sample_torch_keys(path: Path, limit: int = 200) -> list[str]:
    """Load only state_dict metadata and return a sample of keys."""
    try:
        import torch  # type: ignore
    except ImportError as err:
        raise RuntimeError("torch not available for bin inspection") from err

    # torch.load with weights_only avoids materializing tensors on PyTorch>=2.3
    sd = torch.load(path, map_location="meta", weights_only=True)
    keys = list(sd.keys())
    return keys[:limit]


# ---- JSON config/model_index detection --------------------------------------


def detect_from_json(
    config_files: Sequence[str | Path],
    model_index_files: Sequence[str | Path],
) -> ArtifactDetection | None:
    """Infer family/component from config.json or model_index.json."""
    configs = [_safe_load_json(Path(p)) for p in config_files]
    model_indexes = [_safe_load_json(Path(p)) for p in model_index_files]

    # Check model_index.json for diffusers components
    for mi in model_indexes:
        if not mi:
            continue
        pipelines = mi.get("pipelines") or mi.get("_class_name")
        if isinstance(pipelines, list):
            pipelines = [str(p).lower() for p in pipelines]
            if any("unet2dconditionmodel" in p for p in pipelines):
                return ArtifactDetection(
                    family="sd-or-sdxl-unknown",
                    component="unet",
                    confidence=0.6,
                    evidence=["model_index.json lists UNet2DConditionModel"],
                )
        if "transformers" in mi:
            tfms = mi["transformers"]
            if isinstance(tfms, list):
                names = [str(t).lower() for t in tfms]
                if any("cliptextmodel" in n for n in names):
                    return ArtifactDetection(
                        family="clip-text-encoder",
                        component="text_encoder",
                        confidence=0.55,
                        evidence=["model_index.json lists CLIPTextModel"],
                    )

    # Inspect configs (huggingface transformer configs)
    for cfg in configs:
        if not cfg:
            continue
        model_type = str(cfg.get("model_type", "")).lower()
        archs = [str(a).lower() for a in cfg.get("architectures", [])]
        if model_type or archs:
            fam = _family_from_model_type(model_type, archs)
            if fam:
                return fam
        # Vision encoders
        if "vision_config" in cfg and "text_config" in cfg:
            return ArtifactDetection(
                family="multimodal-vision-text",
                component="vision_text",
                confidence=0.5,
                evidence=["config contains both vision_config and text_config"],
            )

    return None


def _family_from_model_type(model_type: str, archs: Sequence[str]) -> ArtifactDetection | None:
    mt = model_type

    def has(target: str) -> bool:
        return target in mt or any(target in arch for arch in archs)

    if has("bert"):
        return ArtifactDetection("bert", "llm", 0.9, ["model_type/arch includes bert"])
    if has("roberta"):
        return ArtifactDetection("roberta", "llm", 0.9, ["model_type/arch includes roberta"])
    if has("deberta"):
        return ArtifactDetection("deberta", "llm", 0.8, ["model_type/arch includes deberta"])
    if has("bart"):
        return ArtifactDetection("opt", "llm", 0.7, ["model_type/arch includes bart/decoder"])
    if has("gpt2"):
        return ArtifactDetection("gpt2", "llm", 0.7, ["model_type/arch includes gpt2"])
    if has("clip"):
        return ArtifactDetection("clip-text-encoder", "text_encoder", 0.6, ["model_type/arch includes clip"])
    if has("whisper"):
        return ArtifactDetection("whisper", "llm", 0.7, ["model_type/arch includes whisper"])
    if has("vit"):
        return ArtifactDetection("vision", "vision_encoder", 0.5, ["model_type/arch includes vit"])
    if has("yolos") or has("detr"):
        return ArtifactDetection("vision", "detection", 0.6, ["model_type/arch includes yolos/detr"])
    if has("segformer"):
        return ArtifactDetection("vision", "segmentation", 0.6, ["model_type/arch includes segformer"])
    if has("blip"):
        return ArtifactDetection("blip", "vision_text", 0.6, ["model_type/arch includes blip"])
    if has("llama"):
        return ArtifactDetection("llama-family", "llm", 0.7, ["model_type/arch includes llama"])
    if has("mistral"):
        return ArtifactDetection("llama-family", "llm", 0.7, ["model_type/arch includes mistral"])
    if has("qwen"):
        return ArtifactDetection("qwen-family", "llm", 0.7, ["model_type/arch includes qwen"])
    return None


def _safe_load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
