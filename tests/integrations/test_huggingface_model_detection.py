import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest

pytest.importorskip("huggingface_hub")

from nodetool.integrations.huggingface import huggingface_models
from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache
from nodetool.integrations.huggingface.huggingface_models import (
    HF_SEARCH_TYPE_CONFIG,
    get_models_by_hf_type,
)


@dataclass
class RepoSpec:
    repo_id: str
    hf_type: str
    model_type: str
    weight_name: str = "model.safetensors"


class FakeHuggingFaceCache:
    """
    Minimal on-disk HF cache builder that mirrors the hub layout:
    models--owner--name/refs/main -> snapshots/<commit> with files.
    """

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root

    def add_repo(self, spec: RepoSpec, *, commit: str = "fake-sha"):
        repo_dir = self.cache_root / f"models--{spec.repo_id.replace('/', '--')}"
        snapshot_dir = repo_dir / "snapshots" / commit
        refs_dir = repo_dir / "refs"

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(commit, encoding="utf-8")

        # Minimal config.json exposing model_type for offline inference.
        config_path = snapshot_dir / "config.json"
        config_path.write_text(
            json.dumps({"model_type": spec.model_type}),
            encoding="utf-8",
        )

        weight_path = snapshot_dir / spec.weight_name
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        weight_path.write_bytes(b"weights")

    def build(self, specs: Iterable[RepoSpec]) -> Path:
        for spec in specs:
            self.add_repo(spec)
        return self.cache_root


def _load_node_types_from_package(base_dir: Path) -> set[str]:
    """Extract hf.* node types from the nodetool-huggingface package."""
    types: set[str] = set()
    for path in base_dir.rglob("*.py"):
        text = path.read_text()
        for match in re.finditer(r'type="(hf\.[^"]+)"', text):
            types.add(match.group(1))
    return types


def test_all_node_types_have_search_config():
    """
    Ensure every hf.* type used by nodetool-huggingface nodes is supported by
    our search configuration so UI queries don't return empty results.
    """
    # nodetool-huggingface lives alongside nodetool-core in this workspace.
    project_root = Path(__file__).resolve().parents[3] / "nodetool-huggingface"
    if not project_root.exists():
        pytest.skip("nodetool-huggingface package not available in workspace")

    node_types = _load_node_types_from_package(project_root / "src" / "nodetool")
    missing = {
        node_type
        for node_type in node_types
        if node_type.lower() not in HF_SEARCH_TYPE_CONFIG
    }
    assert not missing, f"Missing search config for: {sorted(missing)}"


@pytest.mark.asyncio
async def test_get_models_by_hf_type_with_fake_cache(monkeypatch, tmp_path):
    """
    Create a fake HF cache on disk and validate type detection across a wide
    range of hf.* types using offline-only lookups.
    """
    specs = [
        RepoSpec("user/text-cls", "hf.text_classification", "text-classification"),
        RepoSpec("user/zero-shot-cls", "hf.zero_shot_classification", "zero-shot-classification"),
        RepoSpec("user/token-cls", "hf.token_classification", "token-classification"),
        RepoSpec("user/object-det", "hf.object_detection", "object-detection"),
        RepoSpec("user/zero-shot-od", "hf.zero_shot_object_detection", "zero-shot-object-detection"),
        RepoSpec("user/image-cls", "hf.image_classification", "image-classification"),
        RepoSpec("user/zero-shot-img", "hf.zero_shot_image_classification", "zero-shot-image-classification"),
        RepoSpec("user/audio-cls", "hf.audio_classification", "audio-classification"),
        RepoSpec("user/zero-shot-audio", "hf.zero_shot_audio_classification", "zero-shot-audio-classification"),
        RepoSpec("user/image-seg", "hf.image_segmentation", "image-segmentation"),
        RepoSpec("user/depth", "hf.depth_estimation", "depth-estimation"),
        RepoSpec("user/feature", "hf.feature_extraction", "feature-extraction"),
        RepoSpec("user/fill-mask", "hf.fill_mask", "fill-mask"),
        RepoSpec("user/translation", "hf.translation", "translation"),
        RepoSpec("user/vqa", "hf.visual_question_answering", "visual-question-answering"),
        RepoSpec("user/qa", "hf.question_answering", "question-answering"),
        RepoSpec("user/table-qa", "hf.table_question_answering", "table-question-answering"),
        RepoSpec("user/text2text", "hf.text2text_generation", "text2text-generation"),
        RepoSpec("user/image-text", "hf.image_text_to_text", "image-text-to-text"),
        RepoSpec("user/reranker", "hf.reranker", "reranker"),
        RepoSpec("user/real-esrgan", "hf.real_esrgan", "real-esrgan"),
        RepoSpec("user/flux-redux", "hf.flux_redux", "flux-redux", weight_name="flux-redux.safetensors"),
        # New: ASR, TTS, and text generation models
        RepoSpec("user/whisper-asr", "hf.automatic_speech_recognition", "whisper"),
        RepoSpec("user/llama-gen", "hf.text_generation", "text-generation"),
        RepoSpec("user/tts-model", "hf.text_to_speech", "text-to-speech"),
        RepoSpec("user/audio-gen", "hf.text_to_audio", "text-to-audio"),
    ]

    cache_root = tmp_path / "hf_cache"
    FakeHuggingFaceCache(cache_root).build(specs)

    # Replace global cache with one pointing to the fake on-disk layout.
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    fake_cache = HfFastCache(cache_dir=cache_root)
    monkeypatch.setattr(huggingface_models, "HF_FAST_CACHE", fake_cache)
    fake_cache.model_info_cache.delete_pattern("cached_hf_*")

    for spec in specs:
        models = await get_models_by_hf_type(spec.hf_type)
        assert any(model.repo_id == spec.repo_id for model in models), f"Missing repo {spec.repo_id} for type {spec.hf_type}"


@pytest.mark.asyncio
async def test_model_detection_with_safetensors_headers(tmp_path):
    """
    Test model detection using safetensors file headers.
    This validates that we can detect model types from tensor key patterns.
    """
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    # Test Whisper ASR model detection via safetensors headers
    whisper_repo = tmp_path / "models--user--whisper" / "snapshots" / "fake-sha"
    whisper_repo.mkdir(parents=True)

    # Create minimal Whisper-style safetensors file
    whisper_tensors = {
        "model.encoder.layers.0.self_attn.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
        "model.decoder.layers.0.self_attn.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
    }
    save_file(whisper_tensors, whisper_repo / "model.safetensors")

    # Create config.json for additional metadata
    config_path = whisper_repo / "config.json"
    config_path.write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")

    # Create refs/main pointer
    refs_dir = tmp_path / "models--user--whisper" / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text("fake-sha", encoding="utf-8")

    # Test LLaMA text generation model via architectures field
    llama_repo = tmp_path / "models--user--llama" / "snapshots" / "fake-sha2"
    llama_repo.mkdir(parents=True)

    llama_config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"]
    }
    (llama_repo / "config.json").write_text(json.dumps(llama_config), encoding="utf-8")
    (llama_repo / "model.safetensors").write_bytes(b"weights")

    refs_dir2 = tmp_path / "models--user--llama" / "refs"
    refs_dir2.mkdir(parents=True, exist_ok=True)
    (refs_dir2 / "main").write_text("fake-sha2", encoding="utf-8")

    # Now test detection
    from nodetool.integrations.huggingface.safetensors_inspector import detect_model

    # Test Whisper detection
    whisper_result = detect_model(whisper_repo / "model.safetensors", framework="np")
    assert whisper_result.component == "asr", f"Expected asr component, got {whisper_result.component}"
    assert whisper_result.family == "whisper", f"Expected whisper family, got {whisper_result.family}"
    assert whisper_result.confidence > 0.9, f"Expected high confidence, got {whisper_result.confidence}"
