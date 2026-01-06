"""
Tests for ComfyUI template provider.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.providers.comfy.template_loader import TemplateLoader
from nodetool.providers.comfy.template_models import (
    InputMapping,
    OutputMapping,
    TemplateMapping,
)
from nodetool.providers.comfy_template_provider import ComfyTemplateProvider


class TestComfyTemplateProvider:
    """Tests for ComfyTemplateProvider class."""

    @pytest.fixture
    def provider(self):
        """Create a ComfyTemplateProvider instance for testing."""
        return ComfyTemplateProvider()

    @pytest.fixture
    def sample_template_mapping(self):
        """Create a sample TemplateMapping for testing."""
        return TemplateMapping(
            template_id="test_text_to_image",
            template_name="Test Text to Image",
            template_type="text_to_image",
            description="A test template",
            inputs={
                "prompt": InputMapping(
                    node_id=6,
                    node_type="CLIPTextEncode",
                    input_field="text",
                    input_type="STRING",
                    required=True,
                ),
                "width": InputMapping(
                    node_id=27,
                    node_type="EmptySD3LatentImage",
                    input_field="width",
                    input_type="INT",
                    required=False,
                    default=1024,
                ),
                "height": InputMapping(
                    node_id=28,
                    node_type="EmptySD3LatentImage",
                    input_field="height",
                    input_type="INT",
                    required=False,
                    default=1024,
                ),
                "seed": InputMapping(
                    node_id=31,
                    node_type="KSampler",
                    input_field="seed",
                    input_type="INT",
                    required=False,
                    default=0,
                ),
            },
            outputs={
                "image": OutputMapping(
                    node_id=9,
                    node_type="SaveImage",
                    output_field="images",
                    output_type="IMAGE",
                )
            },
        )

    def test_provider_initialization(self, provider):
        """Test that provider initializes correctly."""
        assert provider.provider_name == "comfy_template"
        assert isinstance(provider.template_loader, TemplateLoader)

    def test_extract_params_basic(self, provider):
        """Test parameter extraction with basic params."""
        mock_params = MagicMock()
        mock_params.prompt = "test prompt"
        mock_params.negative_prompt = None
        mock_params.width = 512
        mock_params.height = 512
        mock_params.seed = 12345
        mock_params.num_inference_steps = 20
        mock_params.guidance_scale = 7.5
        mock_params.strength = None

        result = provider._extract_params(mock_params)

        assert result["prompt"] == "test prompt"
        assert result["width"] == 512
        assert result["height"] == 512
        assert result["seed"] == 12345
        assert result["steps"] == 20
        assert result["guidance"] == 7.5
        assert "negative_prompt" not in result
        assert "strength" not in result

    def test_extract_params_empty(self, provider):
        """Test parameter extraction with empty params."""
        mock_params = MagicMock()
        mock_params.prompt = None
        mock_params.negative_prompt = None
        mock_params.width = None
        mock_params.height = None
        mock_params.seed = None
        mock_params.num_inference_steps = None
        mock_params.guidance_scale = None
        mock_params.strength = None
        mock_params.scheduler = None
        mock_params.cfg = None
        mock_params.sampler = None

        result = provider._extract_params(mock_params)

        assert result == {}

    def test_build_graph_basic(self, provider, sample_template_mapping):
        """Test graph building with user parameters."""
        user_params = {
            "prompt": "a beautiful sunset",
            "width": 512,
            "height": 768,
            "seed": 42,
        }

        graph = provider._build_graph(sample_template_mapping, user_params)

        assert "6" in graph
        assert graph["6"]["class_type"] == "CLIPTextEncode"
        assert graph["6"]["inputs"]["text"] == "a beautiful sunset"

        assert "27" in graph
        assert graph["27"]["class_type"] == "EmptySD3LatentImage"
        assert graph["27"]["inputs"]["width"] == 512

        assert "28" in graph
        assert graph["28"]["class_type"] == "EmptySD3LatentImage"
        assert graph["28"]["inputs"]["height"] == 768

        assert "31" in graph
        assert graph["31"]["class_type"] == "KSampler"
        assert graph["31"]["inputs"]["seed"] == 42

        assert "9" in graph
        assert graph["9"]["class_type"] == "SaveImage"

    def test_build_graph_with_defaults(self, provider, sample_template_mapping):
        """Test graph building uses defaults for missing optional params."""
        user_params = {
            "prompt": "a test prompt",
        }

        graph = provider._build_graph(sample_template_mapping, user_params)

        assert graph["27"]["inputs"]["width"] == 1024
        assert graph["28"]["inputs"]["height"] == 1024
        assert graph["31"]["inputs"]["seed"] == 0

    def test_build_graph_required_missing(self, provider, sample_template_mapping):
        """Test graph building raises error for missing required params."""
        user_params = {
            "width": 512,
        }

        with pytest.raises(ValueError, match="Required input 'prompt' not provided"):
            provider._build_graph(sample_template_mapping, user_params)

    @pytest.mark.asyncio
    async def test_get_available_image_models_empty(self, provider):
        """Test get_available_image_models returns empty list when no templates."""
        with patch.object(
            provider.template_loader, "get_image_templates", return_value=[]
        ):
            result = await provider.get_available_image_models()
            assert result == []

    @pytest.mark.asyncio
    async def test_get_available_image_models_with_templates(self, provider):
        """Test get_available_image_models returns models for templates."""
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.comfy.template_models import TemplateInfo

        mock_templates = [
            TemplateInfo(
                template_id="flux_schnell",
                template_name="Flux Schnell",
                template_type="text_to_image",
                description="Fast Flux model",
                inputs=["prompt", "width", "height"],
                outputs=["image"],
            ),
            TemplateInfo(
                template_id="flux_i2i",
                template_name="Flux Image to Image",
                template_type="image_to_image",
                description="Flux image transformation",
                inputs=["prompt", "image"],
                outputs=["image"],
            ),
        ]

        with patch.object(
            provider.template_loader, "get_image_templates", return_value=mock_templates
        ):
            result = await provider.get_available_image_models()

            assert len(result) == 2
            assert result[0].id == "flux_schnell"
            assert result[0].name == "Flux Schnell"
            assert result[0].provider == ProviderEnum.ComfyTemplate
            assert result[0].supported_tasks == ["text_to_image"]

            assert result[1].id == "flux_i2i"
            assert result[1].name == "Flux Image to Image"
            assert result[1].supported_tasks == ["image_to_image"]

    @pytest.mark.asyncio
    async def test_get_available_video_models(self, provider):
        """Test get_available_video_models returns video templates."""
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.comfy.template_models import TemplateInfo

        mock_templates = [
            TemplateInfo(
                template_id="wan_i2v",
                template_name="WAN Image to Video",
                template_type="image_to_video",
                description="WAN video generation",
                inputs=["image", "seed"],
                outputs=["video"],
            ),
        ]

        with patch.object(
            provider.template_loader, "get_video_templates", return_value=mock_templates
        ):
            result = await provider.get_available_video_models()

            assert len(result) == 1
            assert result[0].id == "wan_i2v"
            assert result[0].provider == ProviderEnum.ComfyTemplate
            assert result[0].supported_tasks == ["image_to_video"]

    @pytest.mark.asyncio
    async def test_text_to_image_template_not_found(self, provider):
        """Test text_to_image raises error when template not found."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import TextToImageParams

        mock_params = TextToImageParams(
            model=ImageModel(id="nonexistent", name="None", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        with patch.object(
            provider.template_loader, "load", return_value=None
        ), pytest.raises(ValueError, match="Template not found"):
            await provider.text_to_image(mock_params)

    @pytest.mark.asyncio
    async def test_text_to_image_wrong_type(self, provider):
        """Test text_to_image raises error for non-text_to_image template."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import TextToImageParams

        mock_params = TextToImageParams(
            model=ImageModel(id="test", name="Test", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        i2i_mapping = TemplateMapping(
            template_id="test",
            template_name="Test",
            template_type="image_to_image",
            inputs={},
            outputs={},
        )

        with patch.object(provider.template_loader, "load", return_value=i2i_mapping), \
             pytest.raises(ValueError, match="not a text_to_image template"):
            await provider.text_to_image(mock_params)

    @pytest.mark.asyncio
    async def test_image_to_image_template_not_found(self, provider):
        """Test image_to_image raises error when template not found."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import ImageToImageParams

        mock_params = ImageToImageParams(
            model=ImageModel(id="nonexistent", name="None", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        with patch.object(
            provider.template_loader, "load", return_value=None
        ), pytest.raises(ValueError, match="Template not found"):
            await provider.image_to_image(b"test_image", mock_params)

    @pytest.mark.asyncio
    async def test_image_to_image_wrong_type(self, provider):
        """Test image_to_image raises error for non-image_to_image template."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import ImageToImageParams

        mock_params = ImageToImageParams(
            model=ImageModel(id="test", name="Test", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        t2i_mapping = TemplateMapping(
            template_id="test",
            template_name="Test",
            template_type="text_to_image",
            inputs={},
            outputs={},
        )

        with patch.object(provider.template_loader, "load", return_value=t2i_mapping), \
             pytest.raises(ValueError, match="not an image_to_image template"):
            await provider.image_to_image(b"test_image", mock_params)

    @pytest.mark.asyncio
    async def test_text_to_image_requires_model_id(self, provider):
        """Test text_to_image raises error when model.id is not provided."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import TextToImageParams

        mock_params = TextToImageParams(
            model=ImageModel(id="", name="Test", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        with pytest.raises(ValueError, match=r"model.id.*template_id.*is required"):
            await provider.text_to_image(mock_params)

    @pytest.mark.asyncio
    async def test_image_to_image_requires_model_id(self, provider):
        """Test image_to_image raises error when model.id is not provided."""
        from nodetool.metadata.types import ImageModel
        from nodetool.metadata.types import Provider as ProviderEnum
        from nodetool.providers.types import ImageToImageParams

        mock_params = ImageToImageParams(
            model=ImageModel(id="", name="Test", provider=ProviderEnum.ComfyTemplate),
            prompt="test prompt",
        )

        with pytest.raises(ValueError, match=r"model.id.*template_id.*is required"):
            await provider.image_to_image(b"test_image", mock_params)

    @pytest.mark.asyncio
    async def test_upload_image(self, provider):
        """Test image upload returns filename."""
        test_image = b"fake_image_data"

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = {"status": "success"}

            result = await provider._upload_image(test_image)

            assert result.startswith("nodetool_comfy_input_")
            assert result.endswith(".png")
            mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_image_failure(self, provider):
        """Test image upload raises error on failure."""
        test_image = b"fake_image_data"

        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = {"status": "error", "details": ["Upload failed"]}

            with pytest.raises(ValueError, match="Failed to upload input image"):
                await provider._upload_image(test_image)

    @pytest.mark.asyncio
    async def test_execute_graph(self, provider):
        """Test graph execution returns images."""
        test_graph = {
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "test"}},
            "9": {"class_type": "SaveImage", "inputs": {}},
        }
        test_images = [b"fake_image_data"]

        with patch("websockets.connect") as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value.__aenter__.return_value = mock_ws
            mock_ws.recv = AsyncMock(side_effect=[
                '{"type": "executing", "data": {"node": null, "prompt_id": "test123"}}'
            ])

            with patch("asyncio.to_thread") as mock_thread:
                mock_thread.return_value = {"prompt_id": "test123"}

                with patch.object(provider, "_collect_ws", return_value=test_images):
                    result = await provider._execute_graph(test_graph, template_id="test_template")

                    assert result == test_images

    @pytest.mark.asyncio
    async def test_ensure_server_success(self, provider):
        """Test _ensure_server succeeds when server is available."""
        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = True

            await provider._ensure_server()

            mock_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_server_failure(self, provider):
        """Test _ensure_server raises error when server is unavailable."""
        with patch("asyncio.to_thread") as mock_thread:
            mock_thread.return_value = False

            with pytest.raises(RuntimeError, match="not reachable"):
                await provider._ensure_server()


class TestComfyTemplateProviderEdgeCases:
    """Tests for edge cases in ComfyTemplateProvider."""

    @pytest.fixture
    def provider(self):
        """Create a ComfyTemplateProvider instance for testing."""
        return ComfyTemplateProvider()

    def test_build_graph_with_all_input_types(self, provider):
        """Test graph building handles different input types correctly."""
        mapping = TemplateMapping(
            template_id="test",
            template_name="Test",
            template_type="text_to_image",
            inputs={
                "prompt": InputMapping(
                    node_id=6, node_type="CLIPTextEncode", input_field="text", input_type="STRING"
                ),
                "width": InputMapping(
                    node_id=27, node_type="EmptyLatentImage", input_field="width", input_type="INT"
                ),
                "guidance": InputMapping(
                    node_id=31, node_type="KSampler", input_field="guidance", input_type="FLOAT"
                ),
                "enable": InputMapping(
                    node_id=40, node_type="Bool", input_field="value", input_type="BOOLEAN"
                ),
            },
            outputs={
                "image": OutputMapping(
                    node_id=9, node_type="SaveImage", output_field="images", output_type="IMAGE"
                )
            },
        )

        user_params = {
            "prompt": "test",
            "width": 512,
            "guidance": 7.5,
            "enable": True,
        }

        graph = provider._build_graph(mapping, user_params)

        assert graph["6"]["inputs"]["text"] == "test"
        assert graph["27"]["inputs"]["width"] == 512
        assert graph["31"]["inputs"]["guidance"] == 7.5
        assert graph["40"]["inputs"]["value"] is True

    def test_build_graph_with_node_metadata(self, provider):
        """Test graph building includes node metadata."""
        from nodetool.providers.comfy.template_models import NodeMapping

        mapping = TemplateMapping(
            template_id="test",
            template_name="Test",
            template_type="text_to_image",
            inputs={
                "prompt": InputMapping(
                    node_id=6, node_type="CLIPTextEncode", input_field="text", input_type="STRING"
                ),
            },
            outputs={
                "image": OutputMapping(
                    node_id=9, node_type="SaveImage", output_field="images", output_type="IMAGE"
                )
            },
            nodes={
                "1": NodeMapping(
                    type="LoadImage",
                    class_type="LoadImage",
                    images_directory="input",
                    filename_prefix="output",
                )
            },
        )

        user_params = {"prompt": "test"}
        graph = provider._build_graph(mapping, user_params)

        assert "1" in graph
        assert graph["1"]["class_type"] == "LoadImage"
        assert graph["1"]["inputs"]["images_directory"] == "input"
        assert graph["1"]["inputs"]["filename_prefix"] == "output"

    @pytest.mark.asyncio
    async def test_extract_params_with_all_fields(self, provider):
        """Test parameter extraction includes all possible fields."""
        mock_params = MagicMock()
        mock_params.prompt = "test"
        mock_params.negative_prompt = "negative"
        mock_params.width = 512
        mock_params.height = 768
        mock_params.seed = 12345
        mock_params.num_inference_steps = 30
        mock_params.guidance_scale = 8.0
        mock_params.strength = 0.7
        mock_params.scheduler = "karras"
        mock_params.cfg = 1.0
        mock_params.sampler = "euler"

        result = provider._extract_params(mock_params)

        assert result["prompt"] == "test"
        assert result["negative_prompt"] == "negative"
        assert result["width"] == 512
        assert result["height"] == 768
        assert result["seed"] == 12345
        assert result["steps"] == 30
        assert result["guidance"] == 8.0
        assert result["strength"] == 0.7
        assert result["scheduler"] == "karras"
        assert result["cfg"] == 1.0
        assert result["sampler"] == "euler"
