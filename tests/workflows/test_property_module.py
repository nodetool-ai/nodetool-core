import json
from pydantic import Field
from nodetool.workflows.property import Property
from nodetool.metadata.type_metadata import TypeMetadata


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
