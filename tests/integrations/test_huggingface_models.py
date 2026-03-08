import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nodetool.integrations.huggingface import huggingface_models
from nodetool.metadata.types import ImageModel, Provider
from nodetool.types.model import UnifiedModel


class TestHuggingFaceModels(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("/tmp/test_hf_models")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("nodetool.integrations.huggingface.huggingface_models.HF_FAST_CACHE")
    @patch("nodetool.integrations.huggingface.artifact_inspector.inspect_paths")
    @patch("nodetool.integrations.huggingface.huggingface_models._get_file_size")
    def test_build_cached_repo_entry(self, mock_get_size, mock_inspect, mock_cache):
        # Setup
        repo_id = "test/repo"
        repo_dir = self.test_dir / "repo"
        snapshot_dir = self.test_dir / "snapshot"

        mock_cache.repo_root = AsyncMock(return_value=repo_dir)
        mock_cache.active_snapshot_dir = AsyncMock(return_value=snapshot_dir)
        mock_cache.list_files = AsyncMock(return_value=["model.safetensors", "config.json"])

        mock_get_size.return_value = 100

        mock_inspect.return_value = MagicMock(family="sdxl", component="unet", confidence=0.9, evidence=[])

        # Execute
        async def run_test():
            return await huggingface_models._build_cached_repo_entry(
                repo_id=repo_id,
                repo_dir=repo_dir,
                model_info=None,
                recommended_models={},
                snapshot_dir=snapshot_dir,
                file_list=["model.safetensors", "config.json"],
            )

        model, files = asyncio.run(run_test())

        # Verify
        self.assertIsInstance(model, UnifiedModel)
        self.assertEqual(model.id, repo_id)
        self.assertEqual(model.artifact_family, "sdxl")
        self.assertEqual(model.artifact_component, "unet")
        self.assertEqual(len(files), 2)

        # Verify inspect_paths called with correct paths
        [str(snapshot_dir / "model.safetensors"), str(snapshot_dir / "config.json")]
        mock_inspect.assert_called()
        # Note: inspect_paths is called via asyncio.to_thread, so we check the mock directly
        # args = mock_inspect.call_args[0][0]
        # self.assertEqual(sorted(args), sorted(expected_paths))

    @patch("nodetool.integrations.huggingface.huggingface_models.iter_cached_model_files")
    @patch("nodetool.integrations.huggingface.huggingface_models._repo_has_diffusion_artifacts")
    def test_get_diffusion_models(self, mock_has_artifacts, mock_iter_files):
        # Setup
        repo_id = "test/diffusion"
        # model.safetensors is excluded by default as a standard weight name
        # so we use my_model.safetensors to test single file detection
        file_list = ["my_model.safetensors", "other.bin"]

        async def mock_iter():
            yield repo_id, Path("/tmp"), Path("/tmp/snap"), file_list

        mock_iter_files.return_value = mock_iter()
        mock_has_artifacts.return_value = True

        # Execute
        models = asyncio.run(huggingface_models.get_text_to_image_models_from_hf_cache())

        # Verify
        self.assertEqual(len(models), 3)  # 2 files + 1 repo bundle

        file_models = [m for m in models if ":" in m.id]
        repo_models = [m for m in models if ":" not in m.id]

        with open("/tmp/debug_output.txt", "w") as f:
            f.write(f"file_models: {[m.id for m in file_models]}\n")
            f.write(f"repo_models: {[m.id for m in repo_models]}\n")

        self.assertEqual(len(file_models), 2)
        self.assertEqual(len(repo_models), 1)

        target_file = next(m for m in file_models if m.path == "my_model.safetensors")
        self.assertEqual(target_file.id, f"{repo_id}:my_model.safetensors")
        self.assertEqual(repo_models[0].id, repo_id)
        self.assertIn("text_to_image", target_file.supported_tasks)

    @patch("nodetool.integrations.huggingface.huggingface_models._get_file_size")
    def test_calculate_repo_stats(self, mock_get_size):
        # Setup
        snapshot_dir = Path("/tmp/snapshot")
        file_list = ["file1.txt", "file2.txt"]
        mock_get_size.side_effect = [100, 200]

        # Execute
        size, entries = huggingface_models._calculate_repo_stats(snapshot_dir, file_list)

        # Verify
        self.assertEqual(size, 300)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0], ("file1.txt", 100))
        self.assertEqual(entries[1], ("file2.txt", 200))

        # Test empty
        size, entries = huggingface_models._calculate_repo_stats(snapshot_dir, [])
        self.assertEqual(size, 0)
        self.assertEqual(entries, [])

        # Test None
        size, entries = huggingface_models._calculate_repo_stats(snapshot_dir, None)
        self.assertEqual(size, 0)
        self.assertEqual(entries, [])


class TestParseGgufFlatFilename(unittest.TestCase):
    """Tests for _parse_gguf_flat_filename and _build_manifest_lookup."""

    def test_simple_filename_without_manifest(self):
        """Simple filenames where repo and filename have no underscores."""
        repo_id, repo, filename = huggingface_models._parse_gguf_flat_filename("ggml-org_gemma-GGUF_gemma-Q4.gguf", {})
        self.assertEqual(repo_id, "ggml-org/gemma-GGUF")
        self.assertEqual(repo, "gemma-GGUF")
        self.assertEqual(filename, "gemma-Q4.gguf")

    def test_filename_with_underscores_no_manifest(self):
        """Filenames containing underscores (e.g. Q4_K_M) rely on heuristic."""
        repo_id, repo, filename = huggingface_models._parse_gguf_flat_filename(
            "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf", {}
        )
        self.assertEqual(repo_id, "ggml-org/gemma-3-1b-it-GGUF")
        self.assertEqual(repo, "gemma-3-1b-it-GGUF")
        self.assertEqual(filename, "gemma-3-1b-it-Q4_K_M.gguf")

    def test_manifest_takes_priority(self):
        """Manifest lookup takes priority over heuristic parsing."""
        lookup = {
            "myorg_myrepo_model_v2.gguf": ("myorg/myrepo", "model_v2.gguf"),
        }
        repo_id, repo, filename = huggingface_models._parse_gguf_flat_filename("myorg_myrepo_model_v2.gguf", lookup)
        self.assertEqual(repo_id, "myorg/myrepo")
        self.assertEqual(repo, "myrepo")
        self.assertEqual(filename, "model_v2.gguf")

    def test_bare_filename_fallback(self):
        """Filenames without underscores fall back gracefully."""
        repo_id, repo, filename = huggingface_models._parse_gguf_flat_filename("standalone-model.gguf", {})
        self.assertEqual(repo_id, "")
        self.assertEqual(repo, "")
        self.assertEqual(filename, "standalone-model.gguf")

    def test_build_manifest_lookup(self):
        """_build_manifest_lookup reads manifest JSON files."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "name": "gemma-3-1b-it-GGUF",
                "version": "latest",
                "ggufFile": {"rfilename": "gemma-3-1b-it-Q4_K_M.gguf", "size": 12345},
                "metadata": {"author": "ggml-org", "repo_id": "ggml-org/gemma-3-1b-it-GGUF"},
            }
            manifest_path = Path(tmpdir) / "manifest=ggml-org=gemma-3-1b-it-GGUF=latest.json"
            manifest_path.write_text(json.dumps(manifest))

            lookup = huggingface_models._build_manifest_lookup(tmpdir)

        expected_key = "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf"
        self.assertIn(expected_key, lookup)
        self.assertEqual(lookup[expected_key], ("ggml-org/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"))

    def test_get_llama_cpp_models_from_cache_with_manifest(self):
        """Integration test: models discovered via manifest have correct IDs."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create manifest
            manifest = {
                "name": "gemma-3-1b-it-GGUF",
                "version": "latest",
                "ggufFile": {"rfilename": "gemma-3-1b-it-Q4_K_M.gguf", "size": 100},
                "metadata": {"author": "ggml-org", "repo_id": "ggml-org/gemma-3-1b-it-GGUF"},
            }
            (Path(tmpdir) / "manifest=ggml-org=gemma-3-1b-it-GGUF=latest.json").write_text(json.dumps(manifest))
            # Create GGUF file
            (Path(tmpdir) / "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

            with patch(
                "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
                return_value=tmpdir,
            ):
                models = asyncio.run(huggingface_models.get_llama_cpp_models_from_cache())

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "ggml-org/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
        self.assertEqual(models[0].repo_id, "ggml-org/gemma-3-1b-it-GGUF")
        self.assertEqual(models[0].path, "gemma-3-1b-it-Q4_K_M.gguf")

    def test_get_llamacpp_language_models_from_llama_cache_standalone_uses_abs_path(self):
        """Standalone GGUF files should use absolute path IDs so they are loadable."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            gguf = Path(tmpdir) / "standalone-model.gguf"
            gguf.write_bytes(b"\x00" * 8)

            with patch(
                "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
                return_value=tmpdir,
            ):
                models = asyncio.run(huggingface_models.get_llamacpp_language_models_from_llama_cache())

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, str(gguf))
        self.assertEqual(models[0].path, str(gguf))

    def test_get_llamacpp_language_models_from_llama_cache_manifest_keeps_repo_file_id(self):
        """Manifest-backed GGUF entries should keep repo:file model IDs."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {
                "name": "gemma-3-1b-it-GGUF",
                "version": "latest",
                "ggufFile": {"rfilename": "gemma-3-1b-it-Q4_K_M.gguf", "size": 100},
                "metadata": {"author": "ggml-org", "repo_id": "ggml-org/gemma-3-1b-it-GGUF"},
            }
            (Path(tmpdir) / "manifest=ggml-org=gemma-3-1b-it-GGUF=latest.json").write_text(json.dumps(manifest))
            (Path(tmpdir) / "ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf").write_bytes(b"\x00" * 100)

            with patch(
                "nodetool.providers.llama_server_manager.get_llama_cpp_cache_dir",
                return_value=tmpdir,
            ):
                models = asyncio.run(huggingface_models.get_llamacpp_language_models_from_llama_cache())

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "ggml-org/gemma-3-1b-it-GGUF:gemma-3-1b-it-Q4_K_M.gguf")
        self.assertEqual(models[0].path, "gemma-3-1b-it-Q4_K_M.gguf")
