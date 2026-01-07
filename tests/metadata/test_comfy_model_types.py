"""
Tests for ComfyUI template model types.
"""

import pytest

from nodetool.metadata.types import (
    ComfyModelFile,
    ComfyCheckpoint,
    ComfyCheckpointSDXL,
    ComfyCheckpointSD15,
    ComfyUNET,
    FluxUNET,
    SDXLUNET,
    SD3UNET,
    ComfyVAE,
    FluxVAE,
    SDXLVAE,
    SD15VAE,
    ComfyCLIP,
    FluxCLIP,
    SDXLCLIP,
    T5TextEncoder,
    ComfyControlNet,
    CannyControlNet,
    DepthControlNet,
    PoseControlNet,
    ComfyLoRA,
    FluxLoRA,
    SDXLLoRA,
    ComfyUpscaleModel,
    ComfyVideoModel,
    LTXVideoModel,
    CogVideoModel,
    comfy_template_model_types,
)


class TestComfyModelFileBase:
    """Tests for the ComfyModelFile base class."""

    def test_base_class_default_values(self):
        """Test ComfyModelFile has correct defaults."""
        model = ComfyModelFile()
        assert model.value == ""
        assert model.get_model_folder() == "checkpoints"
        assert ".safetensors" in model.get_extensions()

    def test_is_set_empty(self):
        """Test is_set returns False for empty value."""
        model = ComfyModelFile()
        assert model.is_set() is False
        assert model.is_empty() is True

    def test_is_set_with_value(self):
        """Test is_set returns True for non-empty value."""
        model = ComfyModelFile(value="model.safetensors")
        assert model.is_set() is True
        assert model.is_empty() is False

    def test_get_comfy_path(self):
        """Test get_comfy_path constructs correct path."""
        model = ComfyModelFile(value="model.safetensors")
        path = model.get_comfy_path("/comfy")
        assert path == "/comfy/models/checkpoints/model.safetensors"

    def test_get_folder_path(self):
        """Test get_folder_path class method."""
        path = ComfyModelFile.get_folder_path("/comfy")
        assert path == "/comfy/models/checkpoints"


class TestCheckpointModels:
    """Tests for checkpoint model types."""

    def test_comfy_checkpoint(self):
        """Test ComfyCheckpoint type."""
        model = ComfyCheckpoint(value="sd_xl_base_1.0.safetensors")
        assert model.type == "comfy.checkpoint"
        assert model.get_model_folder() == "checkpoints"
        assert model.value == "sd_xl_base_1.0.safetensors"

    def test_comfy_checkpoint_sdxl(self):
        """Test ComfyCheckpointSDXL type."""
        model = ComfyCheckpointSDXL(value="sdxl.safetensors")
        assert model.type == "comfy.checkpoint.sdxl"
        assert model.get_model_folder() == "checkpoints"

    def test_comfy_checkpoint_sd15(self):
        """Test ComfyCheckpointSD15 type."""
        model = ComfyCheckpointSD15(value="sd15.safetensors")
        assert model.type == "comfy.checkpoint.sd15"
        assert model.get_model_folder() == "checkpoints"


class TestUNETModels:
    """Tests for UNET model types."""

    def test_comfy_unet(self):
        """Test ComfyUNET type."""
        model = ComfyUNET(value="unet.safetensors")
        assert model.type == "comfy.unet"
        assert model.get_model_folder() == "unet"

    def test_flux_unet(self):
        """Test FluxUNET type."""
        model = FluxUNET(value="flux1-dev.safetensors")
        assert model.type == "comfy.unet.flux"
        assert model.get_model_folder() == "unet"

    def test_sdxl_unet(self):
        """Test SDXLUNET type."""
        model = SDXLUNET(value="sdxl_unet.safetensors")
        assert model.type == "comfy.unet.sdxl"
        assert model.get_model_folder() == "unet"

    def test_sd3_unet(self):
        """Test SD3UNET type."""
        model = SD3UNET(value="sd3_unet.safetensors")
        assert model.type == "comfy.unet.sd3"
        assert model.get_model_folder() == "unet"


class TestVAEModels:
    """Tests for VAE model types."""

    def test_comfy_vae(self):
        """Test ComfyVAE type."""
        model = ComfyVAE(value="vae.safetensors")
        assert model.type == "comfy.vae"
        assert model.get_model_folder() == "vae"

    def test_flux_vae(self):
        """Test FluxVAE type."""
        model = FluxVAE(value="ae.safetensors")
        assert model.type == "comfy.vae.flux"
        assert model.get_model_folder() == "vae"

    def test_sdxl_vae(self):
        """Test SDXLVAE type."""
        model = SDXLVAE(value="sdxl_vae.safetensors")
        assert model.type == "comfy.vae.sdxl"
        assert model.get_model_folder() == "vae"

    def test_sd15_vae(self):
        """Test SD15VAE type."""
        model = SD15VAE(value="sd15_vae.safetensors")
        assert model.type == "comfy.vae.sd15"
        assert model.get_model_folder() == "vae"


class TestCLIPModels:
    """Tests for CLIP model types."""

    def test_comfy_clip(self):
        """Test ComfyCLIP type."""
        model = ComfyCLIP(value="clip.safetensors")
        assert model.type == "comfy.clip"
        assert model.get_model_folder() == "clip"

    def test_flux_clip(self):
        """Test FluxCLIP type."""
        model = FluxCLIP(value="t5xxl_fp16.safetensors")
        assert model.type == "comfy.clip.flux"
        assert model.get_model_folder() == "clip"

    def test_sdxl_clip(self):
        """Test SDXLCLIP type."""
        model = SDXLCLIP(value="sdxl_clip.safetensors")
        assert model.type == "comfy.clip.sdxl"
        assert model.get_model_folder() == "clip"

    def test_t5_text_encoder(self):
        """Test T5TextEncoder type."""
        model = T5TextEncoder(value="t5xxl.safetensors")
        assert model.type == "comfy.clip.t5"
        assert model.get_model_folder() == "clip"
        # T5 has restricted extensions
        assert model.get_extensions() == [".safetensors"]


class TestControlNetModels:
    """Tests for ControlNet model types."""

    def test_comfy_controlnet(self):
        """Test ComfyControlNet type."""
        model = ComfyControlNet(value="controlnet.safetensors")
        assert model.type == "comfy.controlnet"
        assert model.get_model_folder() == "controlnet"

    def test_canny_controlnet(self):
        """Test CannyControlNet type."""
        model = CannyControlNet(value="canny.safetensors")
        assert model.type == "comfy.controlnet.canny"
        assert model.get_model_folder() == "controlnet"

    def test_depth_controlnet(self):
        """Test DepthControlNet type."""
        model = DepthControlNet(value="depth.safetensors")
        assert model.type == "comfy.controlnet.depth"
        assert model.get_model_folder() == "controlnet"

    def test_pose_controlnet(self):
        """Test PoseControlNet type."""
        model = PoseControlNet(value="pose.safetensors")
        assert model.type == "comfy.controlnet.pose"
        assert model.get_model_folder() == "controlnet"


class TestLoRAModels:
    """Tests for LoRA model types."""

    def test_comfy_lora(self):
        """Test ComfyLoRA type."""
        model = ComfyLoRA(value="lora.safetensors")
        assert model.type == "comfy.lora"
        assert model.get_model_folder() == "loras"

    def test_flux_lora(self):
        """Test FluxLoRA type."""
        model = FluxLoRA(value="flux_lora.safetensors")
        assert model.type == "comfy.lora.flux"
        assert model.get_model_folder() == "loras"

    def test_sdxl_lora(self):
        """Test SDXLLoRA type."""
        model = SDXLLoRA(value="sdxl_lora.safetensors")
        assert model.type == "comfy.lora.sdxl"
        assert model.get_model_folder() == "loras"


class TestUpscaleModels:
    """Tests for upscale model types."""

    def test_comfy_upscale_model(self):
        """Test ComfyUpscaleModel type."""
        model = ComfyUpscaleModel(value="4x_ESRGAN.safetensors")
        assert model.type == "comfy.upscale"
        assert model.get_model_folder() == "upscale_models"


class TestVideoModels:
    """Tests for video model types."""

    def test_comfy_video_model(self):
        """Test ComfyVideoModel type."""
        model = ComfyVideoModel(value="video_model.safetensors")
        assert model.type == "comfy.video"
        assert model.get_model_folder() == "video_models"

    def test_ltx_video_model(self):
        """Test LTXVideoModel type."""
        model = LTXVideoModel(value="ltxv.safetensors")
        assert model.type == "comfy.video.ltxv"
        assert model.get_model_folder() == "video_models"

    def test_cog_video_model(self):
        """Test CogVideoModel type."""
        model = CogVideoModel(value="cogvideo.safetensors")
        assert model.type == "comfy.video.cogvideo"
        assert model.get_model_folder() == "video_models"


class TestModelTypeRegistry:
    """Tests for model type registry."""

    def test_all_types_registered(self):
        """Test that all model types are registered."""
        # Note: comfy.model_file is the base class type and is registered
        expected_types = {
            "comfy.checkpoint",
            "comfy.checkpoint.sdxl",
            "comfy.checkpoint.sd15",
            "comfy.unet",
            "comfy.unet.flux",
            "comfy.unet.sdxl",
            "comfy.unet.sd3",
            "comfy.vae",
            "comfy.vae.flux",
            "comfy.vae.sdxl",
            "comfy.vae.sd15",
            "comfy.clip",
            "comfy.clip.flux",
            "comfy.clip.sdxl",
            "comfy.clip.t5",
            "comfy.controlnet",
            "comfy.controlnet.canny",
            "comfy.controlnet.depth",
            "comfy.controlnet.pose",
            "comfy.lora",
            "comfy.lora.flux",
            "comfy.lora.sdxl",
            "comfy.upscale",
            "comfy.video",
            "comfy.video.ltxv",
            "comfy.video.cogvideo",
        }
        for type_name in expected_types:
            assert type_name in comfy_template_model_types, f"Type {type_name} not registered"


class TestModelSerialization:
    """Tests for model type serialization."""

    def test_model_to_dict(self):
        """Test model serializes to dict correctly."""
        model = FluxUNET(value="flux1-dev.safetensors")
        data = model.model_dump()
        assert data["type"] == "comfy.unet.flux"
        assert data["value"] == "flux1-dev.safetensors"
        # model_folder is a ClassVar, check via class method
        assert model.get_model_folder() == "unet"

    def test_model_from_dict(self):
        """Test model deserializes from dict correctly."""
        data = {
            "type": "comfy.unet.flux",
            "value": "flux1-dev.safetensors",
        }
        model = FluxUNET.model_validate(data)
        assert model.value == "flux1-dev.safetensors"
        assert model.type == "comfy.unet.flux"
        assert model.get_model_folder() == "unet"
