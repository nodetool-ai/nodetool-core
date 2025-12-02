from pydantic import Field

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import AudioRef, HFLoraSD, ImageRef
from nodetool.workflows.property import Property


def test_property_repr_and_schema():
    prop = Property(
        name="age",
        type=TypeMetadata(type="int"),
        default=5,
        title="Age",
        description="A" * 60,
        min=1,
        max=10,
    )
    r = repr(prop)
    assert "Property(" in r
    assert "age" in r
    schema = prop.get_json_schema()
    assert schema["type"] == "integer"
    assert schema["minimum"] == 1
    assert schema["maximum"] == 10
    assert "description" in schema


def test_property_from_field():
    field = Field(1, title="Number", description="desc", ge=0, le=5)
    prop = Property.from_field("number", TypeMetadata(type="int"), field)
    assert prop.name == "number"
    assert prop.title == "Number"
    assert prop.default == 1
    assert prop.min == 0
    assert prop.max == 5


def test_property_serialization_with_basetype_default():
    """Test that BaseType instances in default field are properly serialized with type field."""
    # Test with ImageRef
    image_ref = ImageRef(uri="test://image.jpg", asset_id="123")
    prop = Property(
        name="image",
        type=TypeMetadata(type="image"),
        default=image_ref,
        title="Image",
        description="An image reference",
    )

    # Serialize to dict
    serialized = prop.model_dump()

    # Verify the default contains the type field
    assert serialized["default"] is not None
    assert isinstance(serialized["default"], dict)
    assert "type" in serialized["default"]
    assert serialized["default"]["type"] == "image"
    assert serialized["default"]["uri"] == "test://image.jpg"
    assert serialized["default"]["asset_id"] == "123"


def test_property_serialization_with_multiple_basetype_defaults():
    """Test serialization with various BaseType subclasses."""
    # Test with AudioRef
    audio_ref = AudioRef(uri="test://audio.mp3")
    prop1 = Property(
        name="audio",
        type=TypeMetadata(type="audio"),
        default=audio_ref,
    )
    serialized1 = prop1.model_dump()
    assert serialized1["default"]["type"] == "audio"
    assert serialized1["default"]["uri"] == "test://audio.mp3"

    # Test with HFLoraSD
    lora = HFLoraSD(repo_id="test/lora-model", path="/path/to/model")
    prop2 = Property(
        name="lora",
        type=TypeMetadata(type="hf.lora_sd"),
        default=lora,
    )
    serialized2 = prop2.model_dump()
    assert serialized2["default"]["type"] == "hf.lora_sd"
    assert serialized2["default"]["repo_id"] == "test/lora-model"
    assert serialized2["default"]["path"] == "/path/to/model"


def test_property_serialization_with_none_default():
    """Test that None defaults serialize correctly."""
    prop = Property(
        name="optional_value",
        type=TypeMetadata(type="int"),
        default=None,
    )
    serialized = prop.model_dump()
    assert serialized["default"] is None


def test_property_serialization_with_primitive_default():
    """Test that primitive defaults still work correctly."""
    # Test with int
    prop1 = Property(
        name="count",
        type=TypeMetadata(type="int"),
        default=42,
    )
    serialized1 = prop1.model_dump()
    assert serialized1["default"] == 42

    # Test with string
    prop2 = Property(
        name="name",
        type=TypeMetadata(type="str"),
        default="test",
    )
    serialized2 = prop2.model_dump()
    assert serialized2["default"] == "test"

    # Test with bool
    prop3 = Property(
        name="enabled",
        type=TypeMetadata(type="bool"),
        default=True,
    )
    serialized3 = prop3.model_dump()
    assert serialized3["default"] is True
