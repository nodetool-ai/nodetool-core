"""Unit tests for Hugging Face repo packaging detector."""

from huggingface_hub import ModelInfo

from nodetool.integrations.huggingface.huggingface_models import (
    RepoPackagingHint,
    detect_repo_packaging,
)


def _model_info(**kwargs) -> ModelInfo:
    """Create a minimal ModelInfo stub for testing."""
    defaults = {"id": "repo", "siblings": []}
    defaults.update(kwargs)
    return ModelInfo(**defaults)


def test_pipeline_tag_prefers_bundle() -> None:
    model_info = _model_info(pipeline_tag="text-to-image")
    file_entries = [("model-00001-of-00002.safetensors", 1024)]

    hint = detect_repo_packaging("repo", model_info, file_entries)

    assert hint == RepoPackagingHint.REPO_BUNDLE


def test_sharded_weights_flag_bundle() -> None:
    file_entries = [
        ("model.safetensors.index.json", 2048),
        ("model-00001-of-00002.safetensors", 1024),
        ("model-00002-of-00002.safetensors", 1024),
    ]

    hint = detect_repo_packaging("repo", None, file_entries)

    assert hint == RepoPackagingHint.REPO_BUNDLE


def test_quantized_variants_use_per_file() -> None:
    file_entries = [
        ("qwen-q4_0.gguf", 1024),
        ("qwen-q5_1.gguf", 1024),
    ]

    hint = detect_repo_packaging("repo", None, file_entries)

    assert hint == RepoPackagingHint.PER_FILE


def test_adapter_like_files_use_per_file() -> None:
    file_entries = [
        ("loras/style-lora.safetensors", 10 * 1024 * 1024),
        ("loras/character-lora.safetensors", 12 * 1024 * 1024),
    ]

    hint = detect_repo_packaging("repo", None, file_entries)

    assert hint == RepoPackagingHint.PER_FILE


def test_single_weight_defaults_to_bundle() -> None:
    file_entries = [("some/model.safetensors", 1024)]

    hint = detect_repo_packaging("repo", None, file_entries)

    assert hint == RepoPackagingHint.REPO_BUNDLE
