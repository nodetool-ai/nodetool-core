"""Regression tests for assigning explicitly empty values and default_factory defaults."""

from typing import Optional

import pytest
from pydantic import Field

from nodetool.metadata.types import ImageRef
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class EmptyValueNode(BaseNode):
    tags: list[str] = Field(default_factory=lambda: ["a"])
    meta: dict[str, str] = Field(default_factory=lambda: {"k": "v"})
    optional_image: Optional[ImageRef] = ImageRef(uri="http://example.com/a.png")
    image: ImageRef = ImageRef(uri="http://example.com/b.png")
    text: str = "hello"
    factory_tags: list[str] = Field(default_factory=list)

    async def process(self, context: ProcessingContext) -> str:
        return self.text


def test_assign_empty_list_clears_default():
    node = EmptyValueNode()
    assert node.assign_property("tags", []) is None
    assert node.tags == []


def test_assign_empty_dict_clears_default():
    node = EmptyValueNode()
    assert node.assign_property("meta", {}) is None
    assert node.meta == {}


def test_assign_none_clears_optional_property():
    node = EmptyValueNode()
    assert node.assign_property("optional_image", None) is None
    assert node.optional_image is None


def test_assign_none_to_non_optional_keeps_default():
    """None is not assignable to a non-optional property: keep the current value."""
    node = EmptyValueNode()
    default_image = node.image
    assert node.assign_property("image", None) is None
    assert node.image == default_image
    assert node.assign_property("text", None) is None
    assert node.text == "hello"


def test_set_node_properties_with_empty_values():
    node = EmptyValueNode()
    node.set_node_properties({"tags": [], "meta": {}})
    assert node.tags == []
    assert node.meta == {}


def test_assign_property_ignores_msgpack_ext_type_for_optional():
    """Undecodable wire values keep the default, even for optional properties."""
    msgpack = pytest.importorskip("msgpack")

    node = EmptyValueNode()
    default = node.optional_image
    assert node.assign_property("optional_image", msgpack.ExtType(0, b"\x00")) is None
    assert node.optional_image == default


def test_default_factory_property_is_not_required():
    prop = next(p for p in EmptyValueNode.properties() if p.name == "factory_tags")
    assert prop.required is False
    assert prop.default == []


def test_default_factory_property_not_in_control_action_required():
    actions = EmptyValueNode.get_control_actions()
    required = actions["run"].get("required", [])
    assert "factory_tags" not in required
