"""
Tests for ComfyUI template loader and models.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from nodetool.providers.comfy.template_loader import TemplateLoader
from nodetool.providers.comfy.template_models import (
    InputMapping,
    NodeMapping,
    OutputMapping,
    TemplateInfo,
    TemplateMapping,
)


class TestTemplateModels:
    """Tests for Pydantic template models."""

    def test_input_mapping_creation(self):
        """Test creating an InputMapping with all fields."""
        mapping = InputMapping(
            node_id=6,
            node_type="CLIPTextEncode",
            input_field="text",
            input_type="STRING",
            required=True,
            default="",
            description="Positive prompt",
        )
        assert mapping.node_id == 6
        assert mapping.node_type == "CLIPTextEncode"
        assert mapping.input_field == "text"
        assert mapping.input_type == "STRING"
        assert mapping.required is True
        assert mapping.default == ""
        assert mapping.description == "Positive prompt"

    def test_input_mapping_defaults(self):
        """Test InputMapping default values."""
        mapping = InputMapping(
            node_id=1,
            node_type="KSampler",
            input_field="seed",
            input_type="INT",
        )
        assert mapping.required is False
        assert mapping.default is None
        assert mapping.description == ""

    def test_output_mapping_creation(self):
        """Test creating an OutputMapping."""
        mapping = OutputMapping(
            node_id=9,
            node_type="SaveImage",
            output_field="images",
            output_type="IMAGE",
            description="Generated image",
        )
        assert mapping.node_id == 9
        assert mapping.node_type == "SaveImage"
        assert mapping.output_field == "images"
        assert mapping.output_type == "IMAGE"

    def test_node_mapping_creation(self):
        """Test creating a NodeMapping with optional fields."""
        mapping = NodeMapping(
            type="LoadImage",
            class_type="LoadImage",
            images_directory="input",
        )
        assert mapping.type == "LoadImage"
        assert mapping.class_type == "LoadImage"
        assert mapping.images_directory == "input"
        assert mapping.filename_prefix is None
        assert mapping.widgets_values_order is None

    def test_node_mapping_with_widgets(self):
        """Test NodeMapping with widgets_values_order."""
        mapping = NodeMapping(
            type="KSampler",
            class_type="KSampler",
            widgets_values_order=["seed", "randomize", "steps", "cfg"],
        )
        assert mapping.widgets_values_order == ["seed", "randomize", "steps", "cfg"]

    def test_template_mapping_creation(self):
        """Test creating a complete TemplateMapping."""
        mapping = TemplateMapping(
            template_id="test_template",
            template_name="Test Template",
            template_type="text_to_image",
            description="A test template",
            inputs={
                "prompt": InputMapping(
                    node_id=6,
                    node_type="CLIPTextEncode",
                    input_field="text",
                    input_type="STRING",
                    required=True,
                )
            },
            outputs={
                "image": OutputMapping(
                    node_id=9,
                    node_type="SaveImage",
                    output_field="images",
                    output_type="IMAGE",
                )
            },
            nodes={
                "6": NodeMapping(type="CLIPTextEncode", class_type="CLIPTextEncode")
            },
        )
        assert mapping.template_id == "test_template"
        assert mapping.template_name == "Test Template"
        assert mapping.template_type == "text_to_image"
        assert "prompt" in mapping.inputs
        assert "image" in mapping.outputs
        assert "6" in mapping.nodes
        assert mapping.presets is None

    def test_template_info_creation(self):
        """Test creating a TemplateInfo."""
        info = TemplateInfo(
            template_id="test",
            template_name="Test",
            template_type="text_to_image",
            description="Test description",
            inputs=["prompt", "seed"],
            outputs=["image"],
        )
        assert info.template_id == "test"
        assert info.inputs == ["prompt", "seed"]
        assert info.outputs == ["image"]

    def test_input_types_validation(self):
        """Test that input_type accepts valid Literal values."""
        for input_type in ["IMAGE", "STRING", "INT", "FLOAT", "BOOLEAN"]:
            mapping = InputMapping(
                node_id=1,
                node_type="Test",
                input_field="test",
                input_type=input_type,  # type: ignore
            )
            assert mapping.input_type == input_type

    def test_output_types_validation(self):
        """Test that output_type accepts valid Literal values."""
        for output_type in ["IMAGE", "VIDEO", "LATENT", "AUDIO"]:
            mapping = OutputMapping(
                node_id=1,
                node_type="Test",
                output_field="test",
                output_type=output_type,  # type: ignore
            )
            assert mapping.output_type == output_type

    def test_template_types_validation(self):
        """Test that template_type accepts valid Literal values."""
        for template_type in [
            "text_to_image",
            "image_to_image",
            "image_to_video",
            "text_to_video",
        ]:
            mapping = TemplateMapping(
                template_id="test",
                template_name="Test",
                template_type=template_type,  # type: ignore
                inputs={},
                outputs={},
            )
            assert mapping.template_type == template_type


class TestTemplateLoader:
    """Tests for TemplateLoader class."""

    @pytest.fixture
    def temp_templates_dir(self):
        """Create a temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            templates_path = Path(tmpdir)

            # Create a valid text-to-image template
            t2i_template = {
                "template_id": "test_t2i",
                "template_name": "Test Text to Image",
                "template_type": "text_to_image",
                "description": "Test template for text to image",
                "inputs": {
                    "prompt": {
                        "node_id": 6,
                        "node_type": "CLIPTextEncode",
                        "input_field": "text",
                        "input_type": "STRING",
                        "required": True,
                    },
                    "seed": {
                        "node_id": 31,
                        "node_type": "KSampler",
                        "input_field": "seed",
                        "input_type": "INT",
                        "required": False,
                        "default": 0,
                    },
                },
                "outputs": {
                    "image": {
                        "node_id": 9,
                        "node_type": "SaveImage",
                        "output_field": "images",
                        "output_type": "IMAGE",
                    }
                },
                "nodes": {},
            }
            with open(templates_path / "test_t2i.yaml", "w") as f:
                yaml.dump(t2i_template, f)

            # Create an image-to-image template
            i2i_template = {
                "template_id": "test_i2i",
                "template_name": "Test Image to Image",
                "template_type": "image_to_image",
                "description": "Test template for image to image",
                "inputs": {
                    "image": {
                        "node_id": 1,
                        "node_type": "LoadImage",
                        "input_field": "image",
                        "input_type": "IMAGE",
                        "required": True,
                    },
                    "prompt": {
                        "node_id": 6,
                        "node_type": "CLIPTextEncode",
                        "input_field": "text",
                        "input_type": "STRING",
                        "required": True,
                    },
                },
                "outputs": {
                    "image": {
                        "node_id": 9,
                        "node_type": "SaveImage",
                        "output_field": "images",
                        "output_type": "IMAGE",
                    }
                },
                "nodes": {},
            }
            with open(templates_path / "test_i2i.yaml", "w") as f:
                yaml.dump(i2i_template, f)

            # Create an image-to-video template
            i2v_template = {
                "template_id": "test_i2v",
                "template_name": "Test Image to Video",
                "template_type": "image_to_video",
                "description": "Test template for image to video",
                "inputs": {
                    "image": {
                        "node_id": 1,
                        "node_type": "LoadImage",
                        "input_field": "image",
                        "input_type": "IMAGE",
                        "required": True,
                    },
                },
                "outputs": {
                    "video": {
                        "node_id": 30,
                        "node_type": "SaveVideo",
                        "output_field": "video",
                        "output_type": "VIDEO",
                    }
                },
                "nodes": {},
            }
            with open(templates_path / "test_i2v.yaml", "w") as f:
                yaml.dump(i2v_template, f)

            # Create an invalid YAML file
            with open(templates_path / "invalid.yaml", "w") as f:
                f.write("invalid: yaml: content: [")

            yield templates_path

    def test_loader_initialization_with_path(self, temp_templates_dir):
        """Test TemplateLoader initialization with explicit path."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        assert loader.templates_dir == temp_templates_dir

    def test_loader_initialization_default(self):
        """Test TemplateLoader uses default templates directory."""
        loader = TemplateLoader()
        assert loader.templates_dir.name == "templates"

    def test_load_valid_template(self, temp_templates_dir):
        """Test loading a valid template mapping."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        mapping = loader.load("test_t2i")

        assert mapping is not None
        assert mapping.template_id == "test_t2i"
        assert mapping.template_name == "Test Text to Image"
        assert mapping.template_type == "text_to_image"
        assert "prompt" in mapping.inputs
        assert "image" in mapping.outputs

    def test_load_nonexistent_template(self, temp_templates_dir):
        """Test loading a template that doesn't exist returns None."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        mapping = loader.load("nonexistent")
        assert mapping is None

    def test_load_invalid_yaml(self, temp_templates_dir):
        """Test loading invalid YAML returns None."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        mapping = loader.load("invalid")
        assert mapping is None

    def test_list_all_templates(self, temp_templates_dir):
        """Test listing all templates without filtering."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        templates = loader.list_templates()

        # Should have 3 valid templates (invalid.yaml is skipped)
        assert len(templates) == 3
        template_ids = {t.template_id for t in templates}
        assert "test_t2i" in template_ids
        assert "test_i2i" in template_ids
        assert "test_i2v" in template_ids

    def test_list_templates_by_type(self, temp_templates_dir):
        """Test listing templates filtered by type."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)

        # Filter for text_to_image only
        t2i_templates = loader.list_templates(template_types=["text_to_image"])
        assert len(t2i_templates) == 1
        assert t2i_templates[0].template_id == "test_t2i"

        # Filter for image templates
        image_templates = loader.list_templates(
            template_types=["text_to_image", "image_to_image"]
        )
        assert len(image_templates) == 2

        # Filter for video templates
        video_templates = loader.list_templates(
            template_types=["image_to_video", "text_to_video"]
        )
        assert len(video_templates) == 1
        assert video_templates[0].template_id == "test_i2v"

    def test_list_templates_empty_directory(self):
        """Test listing templates from empty directory returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = TemplateLoader(templates_dir=tmpdir)
            templates = loader.list_templates()
            assert templates == []

    def test_list_templates_nonexistent_directory(self):
        """Test listing templates from nonexistent directory returns empty list."""
        loader = TemplateLoader(templates_dir="/nonexistent/path")
        templates = loader.list_templates()
        assert templates == []

    def test_get_template_info(self, temp_templates_dir):
        """Test getting template info."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        info = loader.get_template_info("test_t2i")

        assert info is not None
        assert info.template_id == "test_t2i"
        assert info.template_name == "Test Text to Image"
        assert info.template_type == "text_to_image"
        assert "prompt" in info.inputs
        assert "seed" in info.inputs
        assert "image" in info.outputs

    def test_get_template_info_nonexistent(self, temp_templates_dir):
        """Test getting info for nonexistent template returns None."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        info = loader.get_template_info("nonexistent")
        assert info is None

    def test_get_image_templates(self, temp_templates_dir):
        """Test get_image_templates helper method."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        templates = loader.get_image_templates()

        assert len(templates) == 2
        template_types = {t.template_type for t in templates}
        assert "text_to_image" in template_types
        assert "image_to_image" in template_types

    def test_get_video_templates(self, temp_templates_dir):
        """Test get_video_templates helper method."""
        loader = TemplateLoader(templates_dir=temp_templates_dir)
        templates = loader.get_video_templates()

        assert len(templates) == 1
        assert templates[0].template_type == "image_to_video"


class TestTemplateLoaderWithPackageTemplates:
    """Tests using the actual templates shipped with the package."""

    def test_load_sample_text_to_image_template(self):
        """Test loading the sample text-to-image template."""
        loader = TemplateLoader()
        mapping = loader.load("sample_flux_text_to_image")

        if mapping is None:
            pytest.skip("Sample template not found in package")

        assert mapping.template_id == "sample_flux_text_to_image"
        assert mapping.template_type == "text_to_image"
        assert "prompt" in mapping.inputs
        assert mapping.inputs["prompt"].required is True
        assert "image" in mapping.outputs
        assert mapping.presets is not None
        assert "fast" in mapping.presets

    def test_load_sample_image_to_image_template(self):
        """Test loading the sample image-to-image template."""
        loader = TemplateLoader()
        mapping = loader.load("sample_image_to_image")

        if mapping is None:
            pytest.skip("Sample template not found in package")

        assert mapping.template_id == "sample_image_to_image"
        assert mapping.template_type == "image_to_image"
        assert "image" in mapping.inputs
        assert "prompt" in mapping.inputs

    def test_load_sample_image_to_video_template(self):
        """Test loading the sample image-to-video template."""
        loader = TemplateLoader()
        mapping = loader.load("sample_image_to_video")

        if mapping is None:
            pytest.skip("Sample template not found in package")

        assert mapping.template_id == "sample_image_to_video"
        assert mapping.template_type == "image_to_video"
        assert "video" in mapping.outputs
