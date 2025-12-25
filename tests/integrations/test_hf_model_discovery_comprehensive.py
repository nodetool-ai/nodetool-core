"""
Additional tests for HuggingFace model discovery to ensure comprehensive coverage.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from nodetool.integrations.huggingface import huggingface_models
from nodetool.integrations.huggingface.hf_fast_cache import HfFastCache
from nodetool.integrations.huggingface.huggingface_models import (
    _CONFIG_MODEL_TYPE_MAPPING,
    _build_search_config_for_type,
    get_models_by_hf_type,
)


@dataclass
class ModelTypeSpec:
    """Specification for testing a model type."""
    hf_type: str
    model_type: str
    description: str


@pytest.mark.asyncio
async def test_all_config_model_types_have_search_configs():
    """
    Verify that all model types in _CONFIG_MODEL_TYPE_MAPPING
    have valid search configurations.
    """
    missing_configs = []

    for model_type, hf_type in _CONFIG_MODEL_TYPE_MAPPING.items():
        config = _build_search_config_for_type(hf_type)
        if config is None:
            missing_configs.append((model_type, hf_type))

    assert not missing_configs, f"Missing search configs for: {missing_configs}"


@pytest.mark.asyncio
async def test_model_type_mapping_completeness():
    """
    Test that important model types are mapped in _CONFIG_MODEL_TYPE_MAPPING.
    """
    important_types = [
        # ASR models
        "whisper",
        "automatic-speech-recognition",
        # TTS models
        "text-to-speech",
        "text-to-audio",
        # LLM models
        "text-generation",
        "llama",
        "qwen2",
        "gemma2",
        "phi3",
        "phi4",
        # Vision models
        "image-classification",
        "object-detection",
        "image-segmentation",
        # NLP models
        "text-classification",
        "token-classification",
        "question-answering",
    ]

    missing = [t for t in important_types if t not in _CONFIG_MODEL_TYPE_MAPPING]
    assert not missing, f"Missing mappings for important types: {missing}"


@pytest.mark.asyncio
async def test_search_config_consistency():
    """
    Test that search configurations are consistent and don't have conflicts.
    """
    from nodetool.integrations.huggingface.huggingface_models import (
        HF_SEARCH_TYPE_CONFIG,
        HF_TYPE_KEYWORD_MATCHERS,
    )

    # Check that checkpoint variants inherit from base types correctly
    checkpoint_types = [k for k in HF_SEARCH_TYPE_CONFIG.keys() if k.endswith("_checkpoint")]

    for ckpt_type in checkpoint_types:
        base_type = ckpt_type.replace("_checkpoint", "")

        # If base type exists, checkpoint should have similar config
        if base_type in HF_SEARCH_TYPE_CONFIG:
            base_config = HF_SEARCH_TYPE_CONFIG[base_type]
            ckpt_config = HF_SEARCH_TYPE_CONFIG[ckpt_type]

            # Both should have repo_pattern if one does
            if "repo_pattern" in base_config:
                assert "repo_pattern" in ckpt_config, \
                    f"{ckpt_type} missing repo_pattern from {base_type}"


@pytest.mark.asyncio
async def test_model_detection_with_various_architectures(tmp_path):
    """
    Test that model detection works for various architecture patterns.
    """
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    from nodetool.integrations.huggingface.safetensors_inspector import detect_model

    test_cases = [
        # (model_type, tensor_keys, expected_family, expected_component)
        (
            "whisper_asr",
            {
                "model.encoder.layers.0.self_attn.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
                "model.decoder.layers.0.self_attn.q_proj.weight": np.zeros((768, 768), dtype=np.float32),
            },
            "whisper",
            "asr",
        ),
        (
            "llama_llm",
            {
                "model.layers.0.self_attn.q_proj.weight": np.zeros((4096, 4096), dtype=np.float32),
                "model.layers.0.self_attn.k_proj.weight": np.zeros((4096, 4096), dtype=np.float32),
            },
            "llama-family",
            "llm",
        ),
        (
            "flux_transformer",
            {
                "transformer_blocks.0.attn.to_q.weight": np.zeros((3072, 3072), dtype=np.float32),
                "x_embedder.proj.weight": np.zeros((3072, 64), dtype=np.float32),
            },
            "flux",
            "transformer_denoiser",
        ),
    ]

    for name, tensors, expected_family, expected_component in test_cases:
        model_dir = tmp_path / name
        model_dir.mkdir()
        model_path = model_dir / "model.safetensors"

        save_file(tensors, model_path)

        result = detect_model(model_path, framework="np")

        assert result.component == expected_component, \
            f"{name}: expected component {expected_component}, got {result.component}"
        assert result.family == expected_family, \
            f"{name}: expected family {expected_family}, got {result.family}"
        assert result.confidence > 0.8, \
            f"{name}: low confidence {result.confidence}"


@pytest.mark.asyncio
async def test_config_json_model_type_inference(tmp_path):
    """
    Test that model types can be inferred from config.json files.
    """
    from nodetool.integrations.huggingface.huggingface_models import (
        _infer_model_type_from_local_configs,
    )

    test_cases = [
        # (config_data, expected_type)
        ({"model_type": "whisper"}, "hf.automatic_speech_recognition"),
        ({"model_type": "llama"}, "hf.text_generation"),
        ({"model_type": "text-classification"}, "hf.text_classification"),
        ({"model_type": "qwen2"}, "hf.text_generation"),
        ({"model_type": "phi3"}, "hf.text_generation"),
        ({"model_type": "gemma2"}, "hf.text_generation"),
        ({"model_type": "text-to-speech"}, "hf.text_to_speech"),
        ({"model_type": "text-to-audio"}, "hf.text_to_audio"),
    ]

    for config_data, expected_type in test_cases:
        # Create a temporary config file
        config_dir = tmp_path / f"test_{config_data['model_type']}"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps(config_data))

        # Test inference
        file_entries = [("config.json", 100)]
        inferred_type = _infer_model_type_from_local_configs(file_entries, config_dir)

        assert inferred_type == expected_type, \
            f"Expected {expected_type} for {config_data['model_type']}, got {inferred_type}"


@pytest.mark.asyncio
async def test_fallback_search_config():
    """
    Test that the fallback search config works for types without explicit configs.
    """
    # These types should use fallback but still work
    fallback_types = [
        "hf.audio_classification",
        "hf.depth_estimation",
        "hf.feature_extraction",
        "hf.reranker",
    ]

    for hf_type in fallback_types:
        config = _build_search_config_for_type(hf_type)
        assert config is not None, f"No config (not even fallback) for {hf_type}"
        assert "filename_pattern" in config, f"Fallback config missing filename_pattern for {hf_type}"
        assert "repo_pattern" in config, f"Fallback config missing repo_pattern for {hf_type}"


@pytest.mark.asyncio
async def test_checkpoint_variant_detection(monkeypatch, tmp_path):
    """
    Test that checkpoint variants (single-file models) are detected correctly.
    """
    # Create a fake cache with checkpoint files
    cache_root = tmp_path / "hf_cache"

    # Create a repo with a single-file checkpoint
    repo_dir = cache_root / "models--user--sdxl-model"
    snapshot_dir = repo_dir / "snapshots" / "abc123"
    refs_dir = repo_dir / "refs"

    snapshot_dir.mkdir(parents=True)
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123")

    # Create a checkpoint file (not a standard model.safetensors)
    checkpoint_file = snapshot_dir / "sdxl_checkpoint.safetensors"
    checkpoint_file.write_bytes(b"fake checkpoint data")

    # Create a config to help with type detection
    config_file = snapshot_dir / "config.json"
    config_file.write_text(json.dumps({"model_type": "stable-diffusion-xl"}))

    # Set up the cache
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    fake_cache = HfFastCache(cache_dir=cache_root)
    monkeypatch.setattr(huggingface_models, "HF_FAST_CACHE", fake_cache)
    fake_cache.model_info_cache.delete_pattern("cached_hf_*")

    # Test that we can find the checkpoint
    from nodetool.integrations.huggingface.huggingface_models import (
        search_cached_hf_models,
    )

    models = await search_cached_hf_models(
        repo_patterns=["user/*"],
        filename_patterns=["*.safetensors"],
    )

    # Should find both the repo and the file
    assert len(models) >= 1, f"Expected at least 1 model, got {len(models)}"

    # Check that the checkpoint file is found
    file_models = [m for m in models if m.path is not None]
    assert any(m.path == "sdxl_checkpoint.safetensors" for m in file_models), \
        "Checkpoint file not found in results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
