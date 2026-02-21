"""
Tests for the Property class (workflows/property.py).

Tests node property metadata, serialization, and JSON schema generation.
"""

import pytest
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.workflows.property import Property


class TestProperty:
    """Test Property class basic functionality."""

    def test_property_creation_minimal(self):
        """Test creating a Property with minimal parameters."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="test_prop", type=prop_type)

        assert prop.name == "test_prop"
        assert prop.type == prop_type
        assert prop.default is None
        assert prop.title is None
        assert prop.description is None
        assert prop.min is None
        assert prop.max is None
        assert prop.required is False

    def test_property_creation_full(self):
        """Test creating a Property with all parameters."""
        prop_type = TypeMetadata(type="int")
        prop = Property(
            name="count",
            type=prop_type,
            default=5,
            title="Count",
            description="Number of items",
            min=0,
            max=10,
            json_schema_extra={"exclusiveMinimum": True},
            required=True,
        )

        assert prop.name == "count"
        assert prop.default == 5
        assert prop.title == "Count"
        assert prop.description == "Number of items"
        assert prop.min == 0
        assert prop.max == 10
        assert prop.json_schema_extra == {"exclusiveMinimum": True}
        assert prop.required is True


class TestPropertyRepr:
    """Test Property __repr__ method."""

    def test_repr_minimal_property(self):
        """Test repr of minimal property."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="test", type=prop_type)
        repr_str = repr(prop)

        assert "name='test'" in repr_str
        assert "type=" in repr_str

    def test_repr_with_defaults(self):
        """Test repr includes default when set."""
        prop_type = TypeMetadata(type="int")
        prop = Property(name="count", type=prop_type, default=5)
        repr_str = repr(prop)

        assert "default=5" in repr_str

    def test_repr_with_min_max(self):
        """Test repr includes min and max when set."""
        prop_type = TypeMetadata(type="float")
        prop = Property(name="value", type=prop_type, min=0.0, max=1.0)
        repr_str = repr(prop)

        assert "min=0.0" in repr_str
        assert "max=1.0" in repr_str

    def test_repr_with_title(self):
        """Test repr includes title when set."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="test", type=prop_type, title="Test Property")
        repr_str = repr(prop)

        assert "title='Test Property'" in repr_str

    def test_repr_with_required(self):
        """Test repr includes required when True."""
        prop_type = TypeMetadata(type="bool")
        prop = Property(name="enabled", type=prop_type, required=True)
        repr_str = repr(prop)

        assert "required=True" in repr_str

    def test_repr_truncates_long_description(self):
        """Test repr truncates descriptions longer than 50 chars."""
        prop_type = TypeMetadata(type="str")
        long_desc = "a" * 100
        prop = Property(name="test", type=prop_type, description=long_desc)
        repr_str = repr(prop)

        assert "..." in repr_str
        assert next(len(p) for p in repr_str.split(", ") if "description=" in p) < 100


class TestPropertySerialization:
    """Test Property serialization methods."""

    def test_serialize_default_none(self):
        """Test serializing None default value."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="test", type=prop_type, default=None)
        serialized = prop.serialize_default(None)
        assert serialized is None

    def test_serialize_default_primitive(self):
        """Test serializing primitive default values."""
        prop_type = TypeMetadata(type="int")
        prop = Property(name="count", type=prop_type, default=42)
        serialized = prop.serialize_default(42)
        assert serialized == 42

    def test_serialize_default_string(self):
        """Test serializing string default value."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="text", type=prop_type, default="hello")
        serialized = prop.serialize_default("hello")
        assert serialized == "hello"


class TestPropertyJsonSchema:
    """Test Property JSON schema generation."""

    def test_get_json_schema_basic(self):
        """Test generating JSON schema for basic property."""
        prop_type = TypeMetadata(type="str")
        prop = Property(name="text", type=prop_type)
        schema = prop.get_json_schema()

        assert "type" in schema or "anyOf" in schema
        assert schema.get("description") is None

    def test_get_json_schema_with_description(self):
        """Test JSON schema includes description."""
        prop_type = TypeMetadata(type="int")
        prop = Property(name="count", type=prop_type, description="Number of items")
        schema = prop.get_json_schema()

        assert schema["description"] == "Number of items"

    def test_get_json_schema_with_min_float(self):
        """Test JSON schema includes minimum for float type."""
        prop_type = TypeMetadata(type="float")
        prop = Property(name="value", type=prop_type, min=0.5)
        schema = prop.get_json_schema()

        assert schema["minimum"] == 0.5

    def test_get_json_schema_with_min_int(self):
        """Test JSON schema converts min to int for int type."""
        prop_type = TypeMetadata(type="int")
        prop = Property(name="count", type=prop_type, min=1)
        schema = prop.get_json_schema()

        assert schema["minimum"] == 1


class TestPropertyFromField:
    """Test Property.from_field static method."""

    def test_from_field_basic(self):
        """Test creating Property from basic Pydantic field."""
        field = Field(description="A test field")
        prop_type = TypeMetadata(type="str")
        prop = Property.from_field("test_field", prop_type, field)

        assert prop.name == "test_field"
        assert prop.description == "A test field"
        assert prop.title == "Test Field"  # Auto-generated from name

    def test_from_field_with_custom_title(self):
        """Test from_field uses custom title if provided."""
        field = Field(title="Custom Title")
        prop_type = TypeMetadata(type="str")
        prop = Property.from_field("test_field", prop_type, field)

        assert prop.title == "Custom Title"

    def test_from_field_with_default(self):
        """Test from_field with default value."""
        field = Field(default="default_value")
        prop_type = TypeMetadata(type="str")
        prop = Property.from_field("text", prop_type, field)

        assert prop.default == "default_value"
        assert prop.required is False

    def test_from_field_required(self):
        """Test from_field marks required when no default."""
        field = Field()  # No default means required
        prop_type = TypeMetadata(type="str")
        prop = Property.from_field("required_field", prop_type, field)

        assert prop.default is None
        assert prop.required is True

    def test_from_field_with_json_schema_extra(self):
        """Test from_field preserves json_schema_extra."""
        extra = {"exclusiveMinimum": True, "pattern": "^[a-z]+$"}
        field = Field(json_schema_extra=extra)
        prop_type = TypeMetadata(type="str")
        prop = Property.from_field("pattern_field", prop_type, field)

        assert prop.json_schema_extra == extra
