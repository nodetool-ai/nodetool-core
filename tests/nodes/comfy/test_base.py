"""
Tests for ComfyTemplateNode base class.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Dict
from unittest.mock import MagicMock, patch

import pytest

from nodetool.metadata.types import FluxUNET, FluxVAE, FluxCLIP, ImageRef
from nodetool.nodes.comfy.base import ComfyTemplateNode
from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)
from pydantic import Field


class TestTemplateNode(ComfyTemplateNode):
    """Test implementation of ComfyTemplateNode."""

    template_path: ClassVar[str] = "test/test_template.json"

    model_mapping: ClassVar[Dict[str, ModelNodeMapping]] = {
        "unet": ModelNodeMapping(
            node_id="38",
            input_name="unet_name",
            loader_type="UNETLoader"
        ),
    }

    input_mapping: ClassVar[Dict[str, NodeInputMapping]] = {
        "prompt": NodeInputMapping(
            node_id="6",
            input_name="text"
        ),
        "seed": NodeInputMapping(
            node_id="31",
            input_name="seed",
            transform="int"
        ),
        "cfg": NodeInputMapping(
            node_id="31",
            input_name="cfg",
            transform="float"
        ),
    }

    output_mapping: ClassVar[Dict[str, NodeOutputMapping]] = {
        "image": NodeOutputMapping(
            node_id="9",
            output_type="image"
        )
    }

    unet: FluxUNET = Field(default=FluxUNET(value="flux1-dev.safetensors"))
    prompt: str = Field(default="test prompt")
    seed: int = Field(default=42)
    cfg: float = Field(default=1.0)


class TestComfyTemplateNodeBase:
    """Tests for ComfyTemplateNode base class."""

    def test_is_visible_base_class(self):
        """Test that base class is not visible."""
        assert ComfyTemplateNode.is_visible() is False

    def test_is_visible_subclass_no_template(self):
        """Test that subclass without template_path is not visible."""
        class NoTemplateNode(ComfyTemplateNode):
            template_path: ClassVar[str] = ""

        assert NoTemplateNode.is_visible() is False

    def test_is_visible_subclass_with_template(self):
        """Test that subclass with template_path is visible."""
        assert TestTemplateNode.is_visible() is True


class TestFieldExtraction:
    """Tests for field extraction methods."""

    def test_get_model_fields(self):
        """Test extracting model fields."""
        node = TestTemplateNode(id="test")
        models = node.get_model_fields()

        assert "unet" in models
        assert models["unet"].value == "flux1-dev.safetensors"

    def test_get_input_fields(self):
        """Test extracting input fields."""
        node = TestTemplateNode(id="test")
        inputs = node.get_input_fields()

        assert "prompt" in inputs
        assert inputs["prompt"] == "test prompt"
        assert "seed" in inputs
        assert inputs["seed"] == 42
        assert "cfg" in inputs
        assert inputs["cfg"] == 1.0
        # Model fields should not be included
        assert "unet" not in inputs

    def test_get_input_fields_excludes_none(self):
        """Test that None values are excluded."""
        class NodeWithOptional(ComfyTemplateNode):
            template_path: ClassVar[str] = "test.json"
            optional_field: str | None = None
            required_field: str = "value"

        node = NodeWithOptional(id="test")
        inputs = node.get_input_fields()

        assert "required_field" in inputs
        assert "optional_field" not in inputs


class TestGraphBuilding:
    """Tests for graph building functionality."""

    @pytest.fixture
    def mock_template(self):
        """Create a mock template JSON."""
        return {
            "6": {
                "inputs": {"text": ""},
                "class_type": "CLIPTextEncode"
            },
            "31": {
                "inputs": {"seed": 0, "cfg": 7.0},
                "class_type": "KSampler"
            },
            "38": {
                "inputs": {"unet_name": ""},
                "class_type": "UNETLoader"
            },
            "9": {
                "inputs": {},
                "class_type": "SaveImage"
            }
        }

    def test_build_comfy_graph_injects_models(self, mock_template):
        """Test that model filenames are injected."""
        node = TestTemplateNode(id="test")

        with patch.object(TestTemplateNode, "load_template_json", return_value=mock_template):
            graph = node.build_comfy_graph()

        assert graph["38"]["inputs"]["unet_name"] == "flux1-dev.safetensors"

    def test_build_comfy_graph_injects_inputs(self, mock_template):
        """Test that input values are injected."""
        node = TestTemplateNode(id="test", prompt="a beautiful sunset", seed=123)

        with patch.object(TestTemplateNode, "load_template_json", return_value=mock_template):
            graph = node.build_comfy_graph()

        assert graph["6"]["inputs"]["text"] == "a beautiful sunset"
        assert graph["31"]["inputs"]["seed"] == 123

    def test_build_comfy_graph_applies_transforms(self, mock_template):
        """Test that value transformations are applied."""
        node = TestTemplateNode(id="test", seed="42", cfg="1.5")

        with patch.object(TestTemplateNode, "load_template_json", return_value=mock_template):
            graph = node.build_comfy_graph()

        # seed should be converted to int
        assert graph["31"]["inputs"]["seed"] == 42
        assert isinstance(graph["31"]["inputs"]["seed"], int)
        # cfg should be converted to float
        assert graph["31"]["inputs"]["cfg"] == 1.5
        assert isinstance(graph["31"]["inputs"]["cfg"], float)

    def test_build_comfy_graph_creates_missing_loader_nodes(self, mock_template):
        """Test that missing loader nodes are created."""
        # Remove the loader node from template
        del mock_template["38"]

        node = TestTemplateNode(id="test")

        with patch.object(TestTemplateNode, "load_template_json", return_value=mock_template):
            graph = node.build_comfy_graph()

        # Should create the loader node
        assert "38" in graph
        assert graph["38"]["class_type"] == "UNETLoader"
        assert graph["38"]["inputs"]["unet_name"] == "flux1-dev.safetensors"

    def test_build_comfy_graph_error_missing_input_node(self, mock_template):
        """Test error when input node is missing."""
        # Remove the input node from template
        del mock_template["6"]

        node = TestTemplateNode(id="test")

        with patch.object(TestTemplateNode, "load_template_json", return_value=mock_template):
            with pytest.raises(ValueError, match="Node 6 not found"):
                node.build_comfy_graph()


class TestTemplateLoading:
    """Tests for template loading functionality."""

    def test_load_template_json_caching(self):
        """Test that templates are cached."""
        # Clear the cache first
        TestTemplateNode._template_cache.clear()

        mock_template = {"test": "template"}

        with patch("builtins.open", MagicMock()), \
             patch("json.load", return_value=mock_template), \
             patch.object(Path, "exists", return_value=True):

            # First load
            result1 = TestTemplateNode.load_template_json()
            # Second load should use cache
            result2 = TestTemplateNode.load_template_json()

            assert result1 == result2
            assert "TestTemplateNode" in TestTemplateNode._template_cache

    def test_load_template_json_no_template_path(self):
        """Test error when template_path is not set."""
        class NoPathNode(ComfyTemplateNode):
            template_path: ClassVar[str] = ""

        with pytest.raises(ValueError, match="must define template_path"):
            NoPathNode.load_template_json()

    def test_load_template_json_file_not_found(self):
        """Test error when template file doesn't exist."""
        # Clear cache
        TestTemplateNode._template_cache.clear()

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Template not found"):
                TestTemplateNode.load_template_json()


class TestMappingConfiguration:
    """Tests for mapping configuration."""

    def test_model_mapping_configuration(self):
        """Test model mapping is properly configured."""
        assert "unet" in TestTemplateNode.model_mapping
        mapping = TestTemplateNode.model_mapping["unet"]
        assert mapping.node_id == "38"
        assert mapping.input_name == "unet_name"
        assert mapping.loader_type == "UNETLoader"

    def test_input_mapping_configuration(self):
        """Test input mapping is properly configured."""
        assert "prompt" in TestTemplateNode.input_mapping
        prompt_mapping = TestTemplateNode.input_mapping["prompt"]
        assert prompt_mapping.node_id == "6"
        assert prompt_mapping.input_name == "text"

        assert "seed" in TestTemplateNode.input_mapping
        seed_mapping = TestTemplateNode.input_mapping["seed"]
        assert seed_mapping.transform == "int"

    def test_output_mapping_configuration(self):
        """Test output mapping is properly configured."""
        assert "image" in TestTemplateNode.output_mapping
        mapping = TestTemplateNode.output_mapping["image"]
        assert mapping.node_id == "9"
        assert mapping.output_type == "image"
