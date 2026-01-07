"""
Tests for ComfyUI template node mapping classes.
"""

import pytest

from nodetool.nodes.comfy.mapping import (
    ModelNodeMapping,
    NodeInputMapping,
    NodeOutputMapping,
)


class TestNodeInputMapping:
    """Tests for NodeInputMapping class."""

    def test_creation_minimal(self):
        """Test creating NodeInputMapping with required fields only."""
        mapping = NodeInputMapping(
            node_id="6",
            input_name="text"
        )
        assert mapping.node_id == "6"
        assert mapping.input_name == "text"
        assert mapping.transform == "direct"

    def test_creation_with_transform(self):
        """Test creating NodeInputMapping with transform."""
        mapping = NodeInputMapping(
            node_id="31",
            input_name="seed",
            transform="int"
        )
        assert mapping.node_id == "31"
        assert mapping.input_name == "seed"
        assert mapping.transform == "int"

    def test_all_transform_types(self):
        """Test all valid transform types."""
        for transform in ["direct", "image_upload", "int", "float", "bool"]:
            mapping = NodeInputMapping(
                node_id="1",
                input_name="test",
                transform=transform
            )
            assert mapping.transform == transform


class TestModelNodeMapping:
    """Tests for ModelNodeMapping class."""

    def test_creation(self):
        """Test creating ModelNodeMapping."""
        mapping = ModelNodeMapping(
            node_id="38",
            input_name="unet_name",
            loader_type="UNETLoader"
        )
        assert mapping.node_id == "38"
        assert mapping.input_name == "unet_name"
        assert mapping.loader_type == "UNETLoader"

    def test_different_loaders(self):
        """Test various loader types."""
        loaders = [
            ("UNETLoader", "unet_name"),
            ("VAELoader", "vae_name"),
            ("CheckpointLoaderSimple", "ckpt_name"),
            ("DualCLIPLoader", "clip_name1"),
            ("ControlNetLoader", "control_net_name"),
        ]
        for loader_type, input_name in loaders:
            mapping = ModelNodeMapping(
                node_id="1",
                input_name=input_name,
                loader_type=loader_type
            )
            assert mapping.loader_type == loader_type
            assert mapping.input_name == input_name


class TestNodeOutputMapping:
    """Tests for NodeOutputMapping class."""

    def test_creation_minimal(self):
        """Test creating NodeOutputMapping with required fields."""
        mapping = NodeOutputMapping(
            node_id="9",
            output_type="image"
        )
        assert mapping.node_id == "9"
        assert mapping.output_type == "image"
        assert mapping.output_name == "images"  # default

    def test_creation_with_output_name(self):
        """Test creating NodeOutputMapping with custom output_name."""
        mapping = NodeOutputMapping(
            node_id="30",
            output_type="video",
            output_name="video"
        )
        assert mapping.output_type == "video"
        assert mapping.output_name == "video"

    def test_all_output_types(self):
        """Test all valid output types."""
        for output_type in ["image", "video", "latent", "audio"]:
            mapping = NodeOutputMapping(
                node_id="1",
                output_type=output_type
            )
            assert mapping.output_type == output_type


class TestMappingSerialization:
    """Tests for mapping serialization."""

    def test_input_mapping_to_dict(self):
        """Test NodeInputMapping serializes correctly."""
        mapping = NodeInputMapping(
            node_id="6",
            input_name="text",
            transform="direct"
        )
        data = mapping.model_dump()
        assert data["node_id"] == "6"
        assert data["input_name"] == "text"
        assert data["transform"] == "direct"

    def test_model_mapping_to_dict(self):
        """Test ModelNodeMapping serializes correctly."""
        mapping = ModelNodeMapping(
            node_id="38",
            input_name="unet_name",
            loader_type="UNETLoader"
        )
        data = mapping.model_dump()
        assert data["node_id"] == "38"
        assert data["input_name"] == "unet_name"
        assert data["loader_type"] == "UNETLoader"

    def test_output_mapping_to_dict(self):
        """Test NodeOutputMapping serializes correctly."""
        mapping = NodeOutputMapping(
            node_id="9",
            output_type="image",
            output_name="images"
        )
        data = mapping.model_dump()
        assert data["node_id"] == "9"
        assert data["output_type"] == "image"
        assert data["output_name"] == "images"
