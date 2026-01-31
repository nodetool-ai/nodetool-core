"""
Tests for the recommended_models module.
"""

from unittest.mock import patch

import pytest

from nodetool.types.model import UnifiedModel
from nodetool.workflows.recommended_models import (
    _check_server_health,
    _filter_models,
    _filter_models_no_platform,
    _flatten_unique,
    _is_asr_model,
    _is_image_model,
    _is_image_to_image_model,
    _is_image_to_video_model,
    _is_language_embedding_model,
    _is_language_model,
    _is_language_text_generation_model,
    _is_llama_server_available,
    _is_lora_model,
    _is_mlx_model,
    _is_ollama_available,
    _is_text_to_image_model,
    _is_text_to_video_model,
    _is_tts_model,
    _platform_allows_model,
    _server_allows_model,
    _server_status_cache,
    get_server_availability,
)


def make_model(
    id: str = "test-model",
    name: str = "Test Model",
    type: str | None = None,
    pipeline_tag: str | None = None,
    tags: list[str] | None = None,
    repo_id: str | None = "test/repo",
) -> UnifiedModel:
    """Helper to create a UnifiedModel for testing."""
    return UnifiedModel(
        id=id,
        name=name,
        type=type,
        pipeline_tag=pipeline_tag,
        tags=tags or [],
        repo_id=repo_id,
    )


class TestServerHealthCheck:
    """Tests for server health check functions."""

    def test_check_server_health_returns_false_on_connection_error(self):
        """Test that health check returns False when server is unreachable."""
        result = _check_server_health("http://localhost:99999")
        assert result is False


class TestOllamaAvailability:
    """Tests for Ollama server availability."""

    def test_is_ollama_available_returns_false_when_no_url(self):
        """Test that returns False when OLLAMA_API_URL is not set."""
        # Clear cache
        _server_status_cache.clear()

        # Save original env
        import os
        old_val = os.environ.get("OLLAMA_API_URL")
        try:
            if "OLLAMA_API_URL" in os.environ:
                del os.environ["OLLAMA_API_URL"]
            with patch("nodetool.config.environment.Environment.get", return_value=None):
                result = _is_ollama_available()
                assert result is False
        finally:
            if old_val is not None:
                os.environ["OLLAMA_API_URL"] = old_val


class TestLlamaServerAvailability:
    """Tests for llama-server availability."""

    def test_is_llama_server_available_returns_false_when_no_url(self):
        """Test that returns False when LLAMA_CPP_URL is not set."""
        # Clear cache
        _server_status_cache.clear()

        # Save original env
        import os
        old_val = os.environ.get("LLAMA_CPP_URL")
        try:
            if "LLAMA_CPP_URL" in os.environ:
                del os.environ["LLAMA_CPP_URL"]
            with patch("nodetool.config.environment.Environment.get", return_value=None):
                result = _is_llama_server_available()
                assert result is False
        finally:
            if old_val is not None:
                os.environ["LLAMA_CPP_URL"] = old_val


class TestGetServerAvailability:
    """Tests for get_server_availability function."""

    def test_get_server_availability_returns_dict(self):
        """Test that get_server_availability returns a dict with expected keys."""
        result = get_server_availability()

        assert isinstance(result, dict)
        assert "ollama" in result
        assert "llama_server" in result


class TestServerAllowsModel:
    """Tests for _server_allows_model function."""

    def test_server_allows_model_for_non_server_types(self):
        """Test that non-server model types are always allowed."""
        model = make_model(type="hf.text_generation")
        servers = {"ollama": False, "llama_server": False}

        result = _server_allows_model(model, servers)
        assert result is True

    def test_server_allows_model_llama_cpp_requires_server(self):
        """Test that llama_cpp models require llama_server."""
        model = make_model(type="llama_cpp")

        result = _server_allows_model(model, {"ollama": False, "llama_server": False})
        assert result is False

        result = _server_allows_model(model, {"ollama": False, "llama_server": True})
        assert result is True

    def test_server_allows_model_llama_model_requires_ollama(self):
        """Test that llama_model types require ollama server."""
        model = make_model(type="llama_model")

        result = _server_allows_model(model, {"ollama": False, "llama_server": False})
        assert result is False

        result = _server_allows_model(model, {"ollama": True, "llama_server": False})
        assert result is True


class TestFlattenUnique:
    """Tests for _flatten_unique function."""

    def test_flatten_unique_deduplicates_by_id(self):
        """Test that flatten deduplicates models by id."""
        models = {
            "repo1": [make_model(id="model-1"), make_model(id="model-2")],
            "repo2": [make_model(id="model-1"), make_model(id="model-3")],
        }

        result = _flatten_unique(models)

        assert len(result) == 3
        ids = [m.id for m in result]
        assert ids == ["model-1", "model-2", "model-3"]

    def test_flatten_unique_preserves_order(self):
        """Test that flatten preserves insertion order."""
        models = {
            "repo1": [make_model(id="a"), make_model(id="b")],
            "repo2": [make_model(id="c")],
        }

        result = _flatten_unique(models)
        ids = [m.id for m in result]
        assert ids == ["a", "b", "c"]


class TestIsMLXModel:
    """Tests for _is_mlx_model function."""

    def test_is_mlx_model_by_type(self):
        """Test detection by type."""
        model = make_model(type="mlx")
        assert _is_mlx_model(model) is True

        model = make_model(type="hf.text_generation")
        assert _is_mlx_model(model) is False

    def test_is_mlx_model_by_tags(self):
        """Test detection by tags."""
        model = make_model(type=None, tags=["mlx"])
        assert _is_mlx_model(model) is True

        model = make_model(type=None, tags=["mflux"])
        assert _is_mlx_model(model) is True

        model = make_model(type=None, tags=["other"])
        assert _is_mlx_model(model) is False


class TestIsLoraModel:
    """Tests for _is_lora_model function."""

    def test_is_lora_model_by_type(self):
        """Test detection by type."""
        model = make_model(type="hf.lora_sd")
        assert _is_lora_model(model) is True

        model = make_model(type="hf.lora_sdxl")
        assert _is_lora_model(model) is True

        model = make_model(type="hf.stable_diffusion")
        assert _is_lora_model(model) is False

    def test_is_lora_model_by_tags(self):
        """Test detection by tags."""
        model = make_model(type=None, tags=["lora", "sdxl"])
        assert _is_lora_model(model) is True

    def test_is_lora_model_by_name(self):
        """Test detection by name/id."""
        model = make_model(id="some-lora-model")
        assert _is_lora_model(model) is True

        model = make_model(id="stable-diffusion")
        assert _is_lora_model(model) is False


class TestPlatformAllowsModel:
    """Tests for _platform_allows_model function."""

    def test_platform_allows_model_darwin_includes_all(self):
        """Test that macOS includes both MLX and non-MLX models."""
        mlx_model = make_model(type="mlx")
        non_mlx_model = make_model(type="hf.text_generation")

        assert _platform_allows_model(mlx_model, system="darwin") is True
        assert _platform_allows_model(non_mlx_model, system="darwin") is True

    def test_platform_allows_model_linux_excludes_mlx(self):
        """Test that Linux excludes MLX models."""
        mlx_model = make_model(type="mlx")
        non_mlx_model = make_model(type="hf.text_generation")

        assert _platform_allows_model(mlx_model, system="linux") is False
        assert _platform_allows_model(non_mlx_model, system="linux") is True

    def test_platform_allows_model_windows_excludes_mlx(self):
        """Test that Windows excludes MLX models."""
        mlx_model = make_model(type="mlx")
        non_mlx_model = make_model(type="hf.text_generation")

        assert _platform_allows_model(mlx_model, system="windows") is False
        assert _platform_allows_model(non_mlx_model, system="windows") is True


class TestIsImageModel:
    """Tests for _is_image_model function."""

    def test_is_image_model_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="text-to-image")
        assert _is_image_model(model) is True

        model = make_model(pipeline_tag="image-to-image")
        assert _is_image_model(model) is True

        model = make_model(pipeline_tag="text-generation")
        assert _is_image_model(model) is False

    def test_is_image_model_by_type(self):
        """Test detection by type."""
        model = make_model(type="hf.text_to_image")
        assert _is_image_model(model) is True

        model = make_model(type="hf.stable_diffusion")
        assert _is_image_model(model) is True

        model = make_model(type="hf.text_generation")
        assert _is_image_model(model) is False

    def test_is_image_model_excludes_lora(self):
        """Test that LoRA models are excluded."""
        model = make_model(type="hf.lora_sd", pipeline_tag="text-to-image")
        assert _is_image_model(model) is False


class TestIsTextToImageModel:
    """Tests for _is_text_to_image_model function."""

    def test_is_text_to_image_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="text-to-image")
        assert _is_text_to_image_model(model) is True

        model = make_model(pipeline_tag="image-to-image")
        assert _is_text_to_image_model(model) is False

    def test_is_text_to_image_by_tags(self):
        """Test detection by tags."""
        model = make_model(tags=["text-to-image"])
        assert _is_text_to_image_model(model) is True


class TestIsImageToImageModel:
    """Tests for _is_image_to_image_model function."""

    def test_is_image_to_image_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="image-to-image")
        assert _is_image_to_image_model(model) is True

        model = make_model(pipeline_tag="text-to-image")
        assert _is_image_to_image_model(model) is False


class TestIsLanguageModel:
    """Tests for _is_language_model function."""

    def test_is_language_model_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="text-generation")
        assert _is_language_model(model) is True

        model = make_model(pipeline_tag="conversational")
        assert _is_language_model(model) is True

        model = make_model(pipeline_tag="text-to-image")
        assert _is_language_model(model) is False

    def test_is_language_model_by_type(self):
        """Test detection by type."""
        model = make_model(type="hf.text_generation")
        assert _is_language_model(model) is True

        model = make_model(type="llama_cpp")
        assert _is_language_model(model) is True

        model = make_model(type="mlx")
        assert _is_language_model(model) is True


class TestIsLanguageEmbeddingModel:
    """Tests for _is_language_embedding_model function."""

    def test_is_embedding_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="feature-extraction")
        assert _is_language_embedding_model(model) is True

        model = make_model(pipeline_tag="sentence-similarity")
        assert _is_language_embedding_model(model) is True

    def test_is_embedding_by_tags(self):
        """Test detection by tags."""
        model = make_model(tags=["embedding"])
        assert _is_language_embedding_model(model) is True

        model = make_model(tags=["sentence-transformers"])
        assert _is_language_embedding_model(model) is True

    def test_is_embedding_by_name(self):
        """Test detection by name/id."""
        model = make_model(id="nomic-embed-text-v1.5")
        assert _is_language_embedding_model(model) is True

        model = make_model(id="bge-large-en")
        assert _is_language_embedding_model(model) is True


class TestIsLanguageTextGenerationModel:
    """Tests for _is_language_text_generation_model function."""

    def test_is_text_generation_excludes_embedding(self):
        """Test that embedding models are excluded."""
        # Text generation model
        model = make_model(type="hf.text_generation", pipeline_tag="text-generation")
        assert _is_language_text_generation_model(model) is True

        # Embedding model
        model = make_model(type="hf.text_generation", pipeline_tag="feature-extraction")
        assert _is_language_text_generation_model(model) is False


class TestIsASRModel:
    """Tests for _is_asr_model function."""

    def test_is_asr_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="automatic-speech-recognition")
        assert _is_asr_model(model) is True

        model = make_model(pipeline_tag="text-generation")
        assert _is_asr_model(model) is False

    def test_is_asr_by_type(self):
        """Test detection by type."""
        model = make_model(type="hf.automatic_speech_recognition")
        assert _is_asr_model(model) is True


class TestIsTTSModel:
    """Tests for _is_tts_model function."""

    def test_is_tts_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="text-to-speech")
        assert _is_tts_model(model) is True

        model = make_model(pipeline_tag="text-generation")
        assert _is_tts_model(model) is False

    def test_is_tts_by_type(self):
        """Test detection by type."""
        model = make_model(type="hf.text_to_speech")
        assert _is_tts_model(model) is True


class TestIsTextToVideoModel:
    """Tests for _is_text_to_video_model function."""

    def test_is_text_to_video_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="text-to-video")
        assert _is_text_to_video_model(model) is True

        model = make_model(pipeline_tag="text-to-image")
        assert _is_text_to_video_model(model) is False

    def test_is_text_to_video_by_tags(self):
        """Test detection by tags."""
        model = make_model(tags=["text-to-video"])
        assert _is_text_to_video_model(model) is True


class TestIsImageToVideoModel:
    """Tests for _is_image_to_video_model function."""

    def test_is_image_to_video_by_pipeline_tag(self):
        """Test detection by pipeline_tag."""
        model = make_model(pipeline_tag="image-to-video")
        assert _is_image_to_video_model(model) is True

        model = make_model(pipeline_tag="image-to-image")
        assert _is_image_to_video_model(model) is False


class TestFilterModels:
    """Tests for _filter_models function."""

    def test_filter_models_applies_predicate(self):
        """Test that filter applies the predicate."""
        models = [
            make_model(id="1", pipeline_tag="text-generation"),
            make_model(id="2", pipeline_tag="text-to-image"),
            make_model(id="3", pipeline_tag="text-generation"),
        ]

        result = _filter_models(
            models,
            predicate=_is_language_model,
            system="linux",
            check_servers=False,
        )

        assert len(result) == 2
        assert all(_is_language_model(m) for m in result)

    def test_filter_models_deduplicates_by_id(self):
        """Test that filter deduplicates by id."""
        models = [
            make_model(id="same-id", pipeline_tag="text-generation"),
            make_model(id="same-id", pipeline_tag="text-generation"),
        ]

        result = _filter_models(
            models,
            predicate=_is_language_model,
            system="linux",
            check_servers=False,
        )

        assert len(result) == 1

    def test_filter_models_applies_platform_filter(self):
        """Test that filter applies platform filtering."""
        models = [
            make_model(id="1", type="mlx"),
            make_model(id="2", type="hf.text_generation"),
        ]

        # Linux should exclude MLX
        result = _filter_models(
            models,
            predicate=lambda m: True,
            system="linux",
            check_servers=False,
        )

        assert len(result) == 1
        assert result[0].id == "2"


class TestFilterModelsNoPlatform:
    """Tests for _filter_models_no_platform function."""

    def test_filter_models_no_platform_includes_all_platforms(self):
        """Test that filter includes models for all platforms."""
        models = [
            make_model(id="1", type="mlx"),
            make_model(id="2", type="hf.text_generation"),
        ]

        result = _filter_models_no_platform(
            models,
            predicate=lambda m: True,
        )

        assert len(result) == 2
