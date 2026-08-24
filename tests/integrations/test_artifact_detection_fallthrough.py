"""Regression tests for artifact/packaging detection fall-through bugs.

Covers:
* ``inspect_paths`` no longer short-circuits on unidentified safetensors or
  torch-bin results (config.json inspection still runs).
* OpenAI-CLIP vs OpenCLIP text encoders are actually distinguished.
* Shard index manifests reach the sharding check in ``detect_repo_packaging``.
* ``model_type_from_model_info`` does not short-circuit on a typeless entry.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("safetensors")

from safetensors.numpy import save_file

from nodetool.integrations.huggingface import artifact_inspector
from nodetool.integrations.huggingface.artifact_inspector import (
    detect_torch_bin,
    inspect_paths,
)
from nodetool.integrations.huggingface.huggingface_models import (
    RepoPackagingHint,
    detect_repo_packaging,
    model_type_from_model_info,
)
from nodetool.integrations.huggingface.safetensors_inspector import detect_model


def _zeros(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _write(path: Path, tensors: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


# ---- inspect_paths fall-through ---------------------------------------------


def test_unidentified_safetensors_falls_through_to_config_json(tmp_path: Path) -> None:
    _write(tmp_path / "model.safetensors", {"embeddings.word_embeddings.weight": _zeros(4, 4)})
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
    )

    detection = inspect_paths([tmp_path / "model.safetensors", tmp_path / "config.json"])

    assert detection is not None
    assert detection.family == "bert"
    assert detection.component == "llm"


def test_identified_safetensors_still_wins(tmp_path: Path) -> None:
    _write(
        tmp_path / "model.safetensors",
        {"model.layers.0.self_attn.q_proj.weight": _zeros(4, 4)},
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    detection = inspect_paths([tmp_path / "model.safetensors", tmp_path / "config.json"])

    assert detection is not None
    assert detection.family == "llama-family"


def test_unidentified_result_kept_as_fallback(tmp_path: Path) -> None:
    _write(tmp_path / "model.safetensors", {"embeddings.word_embeddings.weight": _zeros(4, 4)})

    detection = inspect_paths([tmp_path / "model.safetensors"])

    assert detection is not None
    assert detection.family == "unknown"


def test_unidentified_torch_bin_falls_through_to_config_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"stub")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "clip"}))
    monkeypatch.setattr(
        artifact_inspector, "_sample_torch_keys", lambda path, limit=200: ["some.random.key"]
    )

    detection = inspect_paths([tmp_path / "pytorch_model.bin", tmp_path / "config.json"])

    assert detection is not None
    assert detection.family == "clip-text-encoder"


def test_detect_torch_bin_returns_none_without_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"stub")
    monkeypatch.setattr(
        artifact_inspector, "_sample_torch_keys", lambda path, limit=200: ["some.random.key"]
    )

    assert detect_torch_bin([tmp_path / "pytorch_model.bin"]) is None


def test_detect_torch_bin_still_detects_llama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"stub")
    monkeypatch.setattr(
        artifact_inspector,
        "_sample_torch_keys",
        lambda path, limit=200: ["model.layers.0.self_attn.q_proj.weight"],
    )

    detection = detect_torch_bin([tmp_path / "pytorch_model.bin"])

    assert detection is not None
    assert detection.family == "llama-family"


# ---- CLIP vs OpenCLIP text encoders -----------------------------------------


def test_openai_clip_text_encoder_not_reported_as_openclip(tmp_path: Path) -> None:
    path = tmp_path / "text_encoder.safetensors"
    _write(
        path,
        {
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": _zeros(4, 4),
            "transformer.text_model.encoder.layers.0.self_attn.k_proj.weight": _zeros(4, 4),
        },
    )

    result = detect_model(path, framework="np")

    assert result.component == "text_encoder"
    assert result.family == "clip-text-encoder"
    assert "OpenAI CLIP" in result.evidence[0]


def test_openclip_text_encoder_still_detected(tmp_path: Path) -> None:
    path = tmp_path / "text_encoder_2.safetensors"
    _write(path, {"text_model.encoder.layers.0.self_attn.q_proj.weight": _zeros(4, 4)})

    result = detect_model(path, framework="np")

    assert result.component == "text_encoder"
    assert result.family == "openclip-text-encoder"


def test_unet_with_openai_clip_keys_has_only_openai_evidence(tmp_path: Path) -> None:
    path = tmp_path / "sd1.safetensors"
    _write(
        path,
        {
            "down_blocks.0.resnets.0.conv1.weight": _zeros(4, 4, 3, 3),
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": _zeros(4, 4),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": _zeros(4, 768),
        },
    )

    result = detect_model(path, framework="np")

    assert result.family == "sd1"
    assert not any("OpenCLIP naming" in item for item in result.evidence)
    assert any("OpenAI CLIP naming" in item for item in result.evidence)


def test_unet_with_openclip_keys_keeps_openclip_evidence(tmp_path: Path) -> None:
    path = tmp_path / "sd2.safetensors"
    _write(
        path,
        {
            "down_blocks.0.resnets.0.conv1.weight": _zeros(4, 4, 3, 3),
            "text_model.encoder.layers.0.self_attn.q_proj.weight": _zeros(4, 4),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": _zeros(4, 1024),
        },
    )

    result = detect_model(path, framework="np")

    assert result.family == "sd2"
    assert any("OpenCLIP naming" in item for item in result.evidence)


# ---- sharded repo packaging --------------------------------------------------


def test_index_file_marks_bundle_without_numbered_shards() -> None:
    file_entries = [
        ("pytorch_model.bin.index.json", 2048),
        ("pytorch_model-shard1.bin", 5 * 1024 * 1024),
        ("pytorch_model-shard2.bin", 5 * 1024 * 1024),
    ]

    assert detect_repo_packaging("repo", None, file_entries) == RepoPackagingHint.REPO_BUNDLE


def test_index_file_beats_quantized_variant_heuristic() -> None:
    file_entries = [
        ("model.safetensors.index.json", 2048),
        ("model-q4.safetensors", 1024),
        ("model-q5.safetensors", 1024),
    ]

    assert detect_repo_packaging("repo", None, file_entries) == RepoPackagingHint.REPO_BUNDLE


def test_quantized_variants_without_index_still_per_file() -> None:
    file_entries = [("qwen-q4_0.gguf", 1024), ("qwen-q5_1.gguf", 1024)]

    assert detect_repo_packaging("repo", None, file_entries) == RepoPackagingHint.PER_FILE


# ---- model type enrichment ---------------------------------------------------


def _info(**kwargs) -> SimpleNamespace:
    defaults = {"config": None, "pipeline_tag": None, "tags": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_typeless_single_recommendation_does_not_short_circuit() -> None:
    model = SimpleNamespace(repo_id="org/repo", type=None)
    info = _info(config={"diffusers": {"_class_name": "StableDiffusionXLPipeline"}})

    assert model_type_from_model_info({"org/repo": [model]}, "org/repo", info) is not None


def test_typeless_single_recommendation_falls_back_to_pipeline_tag() -> None:
    model = SimpleNamespace(repo_id="org/repo", type=None)
    info = _info(pipeline_tag="text-generation")

    result = model_type_from_model_info({"org/repo": [model]}, "org/repo", info)

    assert result == "hf.text_generation"


def test_typed_single_recommendation_still_authoritative() -> None:
    model = SimpleNamespace(repo_id="org/repo", type="hf.text_generation")
    info = _info(pipeline_tag="text-to-image")

    result = model_type_from_model_info({"org/repo": [model]}, "org/repo", info)

    assert result == "hf.text_generation"
