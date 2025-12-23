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
    @patch("nodetool.integrations.huggingface.huggingface_models.inspect_paths")
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
