"""
Tests for llama_server_manager model resolution functions.

Covers _resolve_llama_cpp_cached_file and _parse_model_args integration
with the llama.cpp native cache directory.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from nodetool.providers.llama_server_manager import (
    _parse_model_args,
    _resolve_hf_cached_file,
    _resolve_llama_cpp_cached_file,
)


class TestResolveLlamaCppCachedFile:
    """Tests for _resolve_llama_cpp_cached_file."""

    def test_returns_none_when_cache_dir_missing(self, tmp_path):
        """Should return None when llama.cpp cache directory doesn't exist."""
        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(tmp_path / "nonexistent"),
        ):
            result = _resolve_llama_cpp_cached_file("org/repo", "model.gguf")
            assert result is None

    def test_returns_none_when_file_missing(self, tmp_path):
        """Should return None when the model file is not in the cache."""
        cache_dir = tmp_path / "llama.cpp"
        cache_dir.mkdir()
        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(cache_dir),
        ):
            result = _resolve_llama_cpp_cached_file("org/repo", "model.gguf")
            assert result is None

    def test_resolves_existing_file(self, tmp_path):
        """Should return the path when the model file exists in the cache."""
        cache_dir = tmp_path / "llama.cpp"
        cache_dir.mkdir()
        # Create a file matching llama.cpp flat naming convention
        model_file = cache_dir / "org_repo_model.gguf"
        model_file.write_bytes(b"fake gguf data")

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(cache_dir),
        ):
            result = _resolve_llama_cpp_cached_file("org/repo", "model.gguf")
            assert result is not None
            assert result == str(model_file)

    def test_flat_naming_convention(self, tmp_path):
        """Should use the flat naming convention: {org}_{repo}_{filename}."""
        cache_dir = tmp_path / "llama.cpp"
        cache_dir.mkdir()
        model_file = cache_dir / "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
        model_file.write_bytes(b"fake gguf data")

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(cache_dir),
        ):
            result = _resolve_llama_cpp_cached_file(
                "ggml-org/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"
            )
            assert result is not None
            assert result == str(model_file)


class TestParseModelArgsWithLlamaCppCache:
    """Tests for _parse_model_args with llama.cpp cache support."""

    def test_resolves_from_llama_cpp_cache(self, tmp_path):
        """Should resolve model from llama.cpp cache when available."""
        cache_dir = tmp_path / "llama.cpp"
        cache_dir.mkdir()
        model_file = cache_dir / "org_repo_model-Q4.gguf"
        model_file.write_bytes(b"fake gguf data")

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(cache_dir),
        ):
            args, alias = _parse_model_args("org/repo:model-Q4.gguf")
            assert args == ["-m", str(model_file)]
            assert alias == "org/repo:model-Q4.gguf"

    def test_falls_back_to_hf_cache(self, tmp_path):
        """Should fall back to HF cache when not in llama.cpp cache."""
        # Set up empty llama.cpp cache
        llama_cache = tmp_path / "llama.cpp"
        llama_cache.mkdir()

        # Set up HF cache with the model
        hf_cache = tmp_path / "hf_hub"
        snapshot_dir = hf_cache / "models--org--repo" / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        model_file = snapshot_dir / "model.gguf"
        model_file.write_bytes(b"fake gguf data")

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(llama_cache),
        ), patch(
            "nodetool.providers.llama_server_manager._hf_cache_dir",
            return_value=str(hf_cache),
        ):
            args, _alias = _parse_model_args("org/repo:model.gguf")
            assert args == ["-m", str(model_file)]

    def test_raises_when_not_in_any_cache(self, tmp_path):
        """Should raise FileNotFoundError when not in either cache."""
        llama_cache = tmp_path / "llama.cpp"
        llama_cache.mkdir()
        hf_cache = tmp_path / "hf_hub"
        hf_cache.mkdir()

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(llama_cache),
        ), patch(
            "nodetool.providers.llama_server_manager._hf_cache_dir",
            return_value=str(hf_cache),
        ):
            with pytest.raises(FileNotFoundError, match=r"llama\.cpp or Hugging Face"):
                _parse_model_args("org/repo:model.gguf")

    def test_prefers_llama_cpp_cache_over_hf(self, tmp_path):
        """Should prefer llama.cpp cache when file exists in both caches."""
        # Set up llama.cpp cache
        llama_cache = tmp_path / "llama.cpp"
        llama_cache.mkdir()
        llama_model = llama_cache / "org_repo_model.gguf"
        llama_model.write_bytes(b"llama cache version")

        # Set up HF cache
        hf_cache = tmp_path / "hf_hub"
        snapshot_dir = hf_cache / "models--org--repo" / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        hf_model = snapshot_dir / "model.gguf"
        hf_model.write_bytes(b"hf cache version")

        with patch(
            "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
            return_value=str(llama_cache),
        ), patch(
            "nodetool.providers.llama_server_manager._hf_cache_dir",
            return_value=str(hf_cache),
        ):
            args, _alias = _parse_model_args("org/repo:model.gguf")
            assert args == ["-m", str(llama_model)]
