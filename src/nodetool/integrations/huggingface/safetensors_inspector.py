# safetensors_inspector.py
# Copyright ...
# SPDX-License-Identifier: Apache-2.0
"""Detect model family/component from safetensors headers."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

try:
    from safetensors import safe_open  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("safetensors is required. Install with `pip install safetensors`.") from exc

PathLike = str | os.PathLike


@dataclass
class DetectionResult:
    """Structured result of model detection."""

    family: str
    component: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


def detect_model(
    src: PathLike | Sequence[PathLike],
    *,
    framework: str = "pt",
    max_shape_reads: int = 8,
) -> DetectionResult:
    """Detect model family from one or more .safetensors files."""
    files = _normalize_inputs(src)
    if not files:
        raise FileNotFoundError("No .safetensors files found.")

    index = _build_index(files, framework)
    component = _infer_component(index)

    if component == "lora_adapter":
        return DetectionResult(
            family="lora-adapter",
            component=component,
            confidence=0.98,
            evidence=["Found LoRA keys such as 'lora_up.weight' or 'lora_A.weight'"],
            details=_common_details(index, sample=12),
        )

    if component == "llm":
        result = _classify_llm(index)
        result.details.update(_common_details(index, sample=10))
        return result

    if component == "asr":
        result = _classify_asr(index)
        result.details.update(_common_details(index, sample=10))
        return result

    if component == "tts":
        result = _classify_tts(index)
        result.details.update(_common_details(index, sample=10))
        return result

    if component in ("unet", "transformer_denoiser", "text_encoder", "vae"):
        result = _classify_diffusion(index, framework, max_shape_reads)
        result.details.update(_common_details(index, sample=12))
        return result

    return DetectionResult(
        family="unknown",
        component=component,
        confidence=0.0,
        evidence=["No known component signatures found"],
        details=_common_details(index, sample=12),
    )


@dataclass
class _Index:
    files: list[Path]
    keys_per_file: dict[Path, list[str]]
    key_to_file: dict[str, Path]


def _normalize_inputs(src: PathLike | Sequence[PathLike]) -> list[Path]:
    paths = [Path(src)] if isinstance(src, str | os.PathLike) else [Path(p) for p in src]

    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("*.safetensors")))
        elif path.suffix == ".safetensors" and path.exists():
            out.append(path)
        else:
            continue
    uniq = sorted(set(out))
    return uniq


def _build_index(files: list[Path], framework: str) -> _Index:
    keys_per_file: dict[Path, list[str]] = {}
    key_to_file: dict[str, Path] = {}
    for fp in files:
        with safe_open(fp.as_posix(), framework=framework) as handle:
            keys = list(handle.keys())
        keys_per_file[fp] = keys
        for key in keys:
            key_to_file[key] = fp
    return _Index(files=files, keys_per_file=keys_per_file, key_to_file=key_to_file)


def _has_any(keys: Iterable[str], substr: str) -> bool:
    return any(substr in key for key in keys)


def _has_regex(keys: Iterable[str], pattern: str) -> bool:
    regex = re.compile(pattern)
    return any(regex.search(key) for key in keys)


def _find_first(keys: Iterable[str], pattern: str) -> str | None:
    regex = re.compile(pattern)
    for key in keys:
        if regex.search(key):
            return key
    return None


def _get_shape(index: _Index, key: str, framework: str) -> tuple[int, ...] | None:
    file_path = index.key_to_file.get(key)
    if not file_path:
        return None
    with safe_open(file_path.as_posix(), framework=framework) as handle:
        try:
            tensor = handle.get_tensor(key)
        except Exception:
            return None
    try:
        return tuple(int(val) for val in tensor.shape)  # type: ignore
    except Exception:
        return None


def _infer_component(index: _Index) -> str:
    all_keys = list(index.key_to_file.keys())

    if _has_regex(all_keys, r"(lora_(A|B|down|up)\.weight)$"):
        if not any(
            substr in key
            for key in all_keys
            for substr in ("down_blocks.", "up_blocks.", "model.layers.", "transformer.h.")
        ):
            return "lora_adapter"

    # Diffusers-style UNet (down_blocks, up_blocks, mid_block)
    if any(key.startswith(("down_blocks.", "up_blocks.", "mid_block.")) for key in all_keys):
        return "unet"

    # CompVis/SD-style single-file checkpoint (model.diffusion_model.*)
    # These are commonly found in Civitai checkpoints
    if any(key.startswith("model.diffusion_model.") for key in all_keys):
        return "unet"

    if any(key.startswith("transformer_blocks.") for key in all_keys) and not any(
        key.startswith(("down_blocks.", "up_blocks.", "mid_block.")) for key in all_keys
    ):
        return "transformer_denoiser"

    if _has_any(all_keys, "quant_conv.weight") or _has_regex(all_keys, r"^(encoder|decoder)\."):
        if _has_any(all_keys, "decoder.conv_out.weight"):
            return "vae"

    if _has_regex(
        all_keys,
        r"(?:^|\.)(text_model|transformer\.text_model)\.encoder\.layers\.\d+\.self_attn\.q_proj\.weight$",
    ):
        return "text_encoder"

    # Whisper ASR models
    if _has_regex(all_keys, r"^model\.encoder\.layers\.\d+\.self_attn\.q_proj\.weight$") and _has_regex(
        all_keys, r"^model\.decoder\.layers\.\d+\.self_attn\.q_proj\.weight$"
    ):
        return "asr"

    # TTS/Audio generation models
    if _has_any(all_keys, "text_encoder.") and _has_any(all_keys, "decoder."):
        if _has_regex(all_keys, r"(duration_predictor|pitch_predictor|energy_predictor)"):
            return "tts"

    if any(
        key.startswith(("model.layers.", "transformer.h.", "gpt_neox.layers.", "model.decoder.layers."))
        for key in all_keys
    ) or _has_regex(all_keys, r"(bert|roberta)\.encoder\.layer\.\d+"):
        return "llm"

    return "unknown"


def _classify_diffusion(index: _Index, framework: str, max_shape_reads: int) -> DetectionResult:
    keys = list(index.key_to_file.keys())
    evidence: list[str] = []
    confidence = 0.0
    family = "unknown"
    component = _infer_component(index)

    if component == "transformer_denoiser":
        dit_hints = [
            r"(?:^|\.)x_embedder\.",
            r"(?:^|\.)t_embedder\.",
            r"(?:^|\.)pe_embedder\.",
            r"(?:^|\.)pos_embed",
            r"(?:^|\.)patch_embed\.proj\.weight",
            r"(?:^|\.)adaln_",
            r"(?:^|\.)caption|context_(?:proj|embed)",
        ]
        if any(_has_regex(keys, hint) for hint in dit_hints):
            family = "flux"
            confidence = 0.98
            evidence.append("Found transformer_blocks.* without UNet blocks")
            evidence.append("Found DiT style embedder keys such as x_embedder or pe_embedder")
            return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

        family = "flux-like"
        confidence = 0.75
        evidence.append("Transformer denoiser detected by top level transformer_blocks.*")
        return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

    if component == "unet":
        # Check for CompVis/SD-style single-file checkpoint first
        # These use model.diffusion_model.* naming instead of diffusers-style down_blocks.*
        if any(key.startswith("model.diffusion_model.") for key in keys):
            # This is a single-file SD checkpoint (commonly from Civitai)
            # Check for SDXL indicators
            if _has_any(keys, "conditioner.embedders."):
                family = "sdxl-base"
                confidence = 0.90
                evidence.append("CompVis-style checkpoint with conditioner.embedders.* (SDXL hallmark)")
            elif _has_any(keys, "cond_stage_model.transformer."):
                family = "sd1"
                confidence = 0.90
                evidence.append("CompVis-style checkpoint with cond_stage_model.transformer.* (SD1.x hallmark)")
            elif _has_any(keys, "cond_stage_model.model."):
                family = "sd2"
                confidence = 0.88
                evidence.append("CompVis-style checkpoint with cond_stage_model.model.* (SD2.x hallmark)")
            else:
                family = "sd1"
                confidence = 0.80
                evidence.append("CompVis-style checkpoint with model.diffusion_model.* (likely SD1.x)")
            return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

        probe = _find_first(keys, r"^down_blocks\.0\.resnets\.0\.conv1\.weight$")
        read_shapes = 0
        if probe and read_shapes < max_shape_reads:
            shape = _get_shape(index, probe, framework)
            read_shapes += 1
            if shape and len(shape) == 4 and shape[1] >= 1024:
                family = "sdxl-refiner"
                confidence = 0.97
                evidence.append(f"{probe} second dim {shape[1]} suggests refiner input 1280")
                return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

        if _has_regex(
            keys,
            r"^down_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn1\.to_q\.weight$",
        ):
            family = "sdxl-base"
            confidence = 0.93
            evidence.append("UNet attentions include transformer_blocks.* which is characteristic of SDXL")
        else:
            evidence.append("UNet present without top level transformer denoiser")

        if _has_regex(
            keys,
            r"(?:^|\.)(text_model|transformer\.text_model)\.encoder\.layers\.0\.self_attn\.q_proj\.weight$",
        ):
            if _has_regex(
                keys,
                r"(^|\.)(text_model)\.encoder\.layers\.0\.self_attn\.q_proj\.weight$",
            ):
                if family == "unknown":
                    family = "sd2"
                    confidence = 0.92
                else:
                    confidence = max(confidence, 0.92)
                evidence.append("Found OpenCLIP naming: text_model.encoder.layers.0.self_attn.q_proj.weight")
            if _has_regex(
                keys,
                r"(^|\.)(transformer\.text_model)\.encoder\.layers\.0\.self_attn\.q_proj\.weight$",
            ):
                if family == "unknown":
                    family = "sd1"
                    confidence = 0.92
                else:
                    if family != "sdxl-base":
                        family = "sd1"
                    confidence = max(confidence, 0.92)
                evidence.append(
                    "Found OpenAI CLIP naming: transformer.text_model.encoder.layers.0.self_attn.q_proj.weight"
                )

        if family in ("unknown", "sd1", "sd2"):
            cross_k = _find_first(keys, r"\.attn2\.to_k\.weight$")
            if cross_k and read_shapes < max_shape_reads:
                shape = _get_shape(index, cross_k, framework)
                read_shapes += 1
                if shape and len(shape) == 2:
                    cross_dim = shape[1]
                    if cross_dim in (768, 1024):
                        pred = "sd1" if cross_dim == 768 else "sd2"
                        if family == "unknown":
                            family = pred
                        confidence = max(confidence, 0.88)
                        evidence.append(f"{cross_k} cross_dim={cross_dim} → {pred}")

        if family == "sdxl-base":
            return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

        if family in ("sd1", "sd2"):
            return DetectionResult(family=family, component=component, confidence=confidence, evidence=evidence)

        return DetectionResult(
            family="sd-or-sdxl-unknown",
            component=component,
            confidence=0.40,
            evidence=[*evidence, "UNet present but patterns were insufficient to decide"],
        )

    if component == "vae":
        return DetectionResult(
            family="sd-vae",
            component=component,
            confidence=0.90,
            evidence=["Found quant_conv and decoder.conv_out weights, typical SD VAE"],
        )

    if component == "text_encoder":
        if _has_regex(
            keys,
            r"(^|\.)(text_model)\.encoder\.layers\.0\.self_attn\.q_proj\.weight$",
        ):
            return DetectionResult(
                family="openclip-text-encoder",
                component=component,
                confidence=0.95,
                evidence=["OpenCLIP text encoder naming convention detected"],
            )
        if _has_regex(
            keys,
            r"(^|\.)(transformer\.text_model)\.encoder\.layers\.0\.self_attn\.q_proj\.weight$",
        ):
            return DetectionResult(
                family="clip-text-encoder",
                component=component,
                confidence=0.95,
                evidence=["OpenAI CLIP text encoder naming convention detected"],
            )
        return DetectionResult(
            family="text-encoder-unknown",
            component=component,
            confidence=0.40,
            evidence=["Text encoder present without recognizable CLIP naming"],
        )

    return DetectionResult(
        family="unknown",
        component=component,
        confidence=0.0,
        evidence=["No diffusion family rules fired"],
    )


def _classify_llm(index: _Index) -> DetectionResult:
    keys = list(index.key_to_file.keys())

    if _has_regex(keys, r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$"):
        return DetectionResult(
            family="gpt-neox",
            component="llm",
            confidence=0.98,
            evidence=["Found gpt_neox.layers.N.attention.query_key_value.weight"],
        )

    if _has_regex(keys, r"^transformer\.h\.\d+\.self_attention\.query_key_value\.weight$"):
        if _has_any(keys, "transformer.word_embeddings_layernorm.weight"):
            return DetectionResult(
                family="bloom",
                component="llm",
                confidence=0.95,
                evidence=[
                    "Found transformer.h.N.self_attention.query_key_value.weight",
                    "Found transformer.word_embeddings_layernorm.weight (BLOOM hallmark)",
                ],
            )
        return DetectionResult(
            family="falcon",
            component="llm",
            confidence=0.90,
            evidence=["Found transformer.h.N.self_attention.query_key_value.weight without BLOOM layernorm"],
        )

    if _has_regex(keys, r"^model\.layers\.\d+\.self_attn\.q_proj\.weight$"):
        if _has_regex(keys, r"^model\.layers\.\d+\.attention\.(wqkv|w_qkv)\.weight$"):
            return DetectionResult(
                family="qwen-family",
                component="llm",
                confidence=0.90,
                evidence=["Found model.layers.N.attention.wqkv or w_qkv fused projection"],
            )
        return DetectionResult(
            family="llama-family",
            component="llm",
            confidence=0.88,
            evidence=["Found model.layers.N.self_attn.q_proj.weight"],
        )

    if _has_regex(keys, r"^model\.decoder\.layers\.\d+\.self_attn\.q_proj\.weight$"):
        return DetectionResult(
            family="opt",
            component="llm",
            confidence=0.94,
            evidence=["Found model.decoder.layers.N.self_attn.q_proj.weight"],
        )

    if _has_regex(keys, r"^transformer\.h\.\d+\.attn\.q_proj\.weight$"):
        return DetectionResult(
            family="gpt-j",
            component="llm",
            confidence=0.85,
            evidence=["Found transformer.h.N.attn.q_proj.weight"],
        )

    if _has_regex(keys, r"^transformer\.blocks\.\d+\.attn\.Wqkv\.weight$"):
        return DetectionResult(
            family="mpt",
            component="llm",
            confidence=0.95,
            evidence=["Found transformer.blocks.N.attn.Wqkv.weight"],
        )

    if _has_regex(keys, r"^encoder\.block\.\d+\.layer\.0\.SelfAttention\.q\.weight$"):
        return DetectionResult(
            family="t5",
            component="llm",
            confidence=0.95,
            evidence=["Found encoder.block.N.layer.0.SelfAttention.q.weight"],
        )

    if _has_regex(keys, r"^bert\.encoder\.layer\.\d+\.attention\.self\.query\.weight$"):
        return DetectionResult(
            family="bert",
            component="llm",
            confidence=0.96,
            evidence=["Found bert.encoder.layer.N.attention.self.query.weight"],
        )
    if _has_regex(keys, r"^roberta\.encoder\.layer\.\d+\.attention\.self\.query\.weight$"):
        return DetectionResult(
            family="roberta",
            component="llm",
            confidence=0.96,
            evidence=["Found roberta.encoder.layer.N.attention.self.query.weight"],
        )

    return DetectionResult(
        family="llm-unknown",
        component="llm",
        confidence=0.40,
        evidence=["LLM component detected but no family specific signature matched"],
    )


def _classify_asr(index: _Index) -> DetectionResult:
    """Classify automatic speech recognition models."""
    keys = list(index.key_to_file.keys())

    # Whisper models have both encoder and decoder layers
    if _has_regex(keys, r"^model\.encoder\.layers\.\d+\.self_attn\.q_proj\.weight$") and _has_regex(
        keys, r"^model\.decoder\.layers\.\d+\.self_attn\.q_proj\.weight$"
    ):
        return DetectionResult(
            family="whisper",
            component="asr",
            confidence=0.95,
            evidence=["Found Whisper-style encoder and decoder layers"],
        )

    return DetectionResult(
        family="asr-unknown",
        component="asr",
        confidence=0.50,
        evidence=["ASR component detected but no specific family matched"],
    )


def _classify_tts(index: _Index) -> DetectionResult:
    """Classify text-to-speech models."""
    keys = list(index.key_to_file.keys())

    # TTS models typically have text encoders and duration/pitch predictors
    if _has_regex(keys, r"(duration_predictor|pitch_predictor|energy_predictor)"):
        return DetectionResult(
            family="tts-generic",
            component="tts",
            confidence=0.85,
            evidence=["Found duration/pitch/energy predictors typical of TTS models"],
        )

    return DetectionResult(
        family="tts-unknown",
        component="tts",
        confidence=0.50,
        evidence=["TTS component detected but no specific family matched"],
    )


def _common_details(index: _Index, sample: int = 10) -> dict[str, object]:
    keys = sorted(index.key_to_file.keys())
    return {
        "num_files": len(index.files),
        "num_tensors": len(keys),
        "sample_keys": keys[:sample],
    }


def _to_json(result: DetectionResult) -> str:
    return json.dumps(
        {
            "family": result.family,
            "component": result.component,
            "confidence": round(float(result.confidence), 4),
            "evidence": result.evidence,
            "details": result.details,
        },
        indent=2,
        ensure_ascii=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect model family from .safetensors using tensor names and shapes.")
    parser.add_argument("path", nargs="+", help="File or directory path(s)")
    parser.add_argument(
        "--framework",
        default="pt",
        choices=["pt", "np"],
        help="Safetensors backend for reading shapes",
    )
    parser.add_argument(
        "--max-shape-reads",
        type=int,
        default=8,
        help="Maximum number of tensors to load for shape inspection",
    )
    args = parser.parse_args(argv)

    result = detect_model(args.path, framework=args.framework, max_shape_reads=args.max_shape_reads)
    print(_to_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
