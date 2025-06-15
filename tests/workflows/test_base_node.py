import enum
import os
from typing import Optional, Union
import pytest
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.property import Property
from nodetool.metadata.type_metadata import TypeMetadata

from nodetool.workflows.base_node import (
    NODE_BY_TYPE,
    BaseNode,
    add_node_type,
    get_node_class,
    type_metadata,
)
from nodetool.metadata.types import OutputSlot, DataframeRef, ColumnDef


current_dir = os.path.dirname(os.path.realpath(__file__))
test_file = os.path.join(current_dir, "test.jpg")


class DummyClass(BaseNode):
    prop: int = 123

    def process(self, context: ProcessingContext) -> int:
        return self.prop


class StringNode(BaseNode):
    value: str = "test"

    def process(self, context: ProcessingContext) -> str:
        return self.value


def test_node_creation():
    node = BaseNode(id="")
    assert node._id == ""


def test_node_metadata_method():
    node = DummyClass()
    assert isinstance(node.get_metadata(), NodeMetadata)


def test_node_find_property_method():
    node = DummyClass(prop=123)
    assert isinstance(node.find_property("prop"), Property)


def test_node_find_property_fail():
    node = DummyClass(prop=123)
    assert node.find_property("non_existent_prop") is None


def test_node_find_output_method():
    node = DummyClass()
    assert isinstance(node.find_output("output"), OutputSlot)


def test_node_find_output_fail():
    node = DummyClass()
    assert node.find_output("non_existent_output") is None


def test_node_assign_property_method():
    node = DummyClass()
    node.assign_property("prop", 456)
    assert node.prop == 456


def test_node_assign_property_fail():
    node = DummyClass()
    node.assign_property("prop", "test")
    assert node.prop == 123


def test_node_is_assignable_method():
    node = DummyClass()
    assert node.is_assignable("prop", 456) is True


def test_node_output_type():
    node = DummyClass()
    assert node.outputs() == [OutputSlot(type=TypeMetadata(type="int"), name="output")]


def test_string_node_output_type():
    node = StringNode(_id="")
    assert node.outputs() == [OutputSlot(type=TypeMetadata(type="str"), name="output")]


def test_node_set_node_properties():
    node = DummyClass()
    node.set_node_properties({"prop": 789})
    assert node.prop == 789


def test_node_set_node_properties_skip_errors():
    node = DummyClass()
    node.set_node_properties({"prop": "test"}, skip_errors=True)
    assert node.prop == 123


def test_node_properties_dict():
    node = DummyClass()
    assert "prop" in node.properties_dict()


def test_node_properties():
    node = DummyClass()
    assert any(prop.name == "prop" for prop in node.properties())


def test_node_node_properties():
    node = DummyClass(prop=123)
    assert node.node_properties() == {"prop": 123}


@pytest.mark.asyncio
async def test_node_convert_output_value(context: ProcessingContext):
    node = DummyClass()
    output = 123
    assert await node.convert_output(context, output) == {"output": 123}


def test_type_metadata_basic_types():
    assert type_metadata(int) == TypeMetadata(type="int")
    assert type_metadata(str) == TypeMetadata(type="str")
    assert type_metadata(float) == TypeMetadata(type="float")
    assert type_metadata(bool) == TypeMetadata(type="bool")


def test_type_metadata_list():
    assert type_metadata(list[int]) == TypeMetadata(
        type="list", type_args=[TypeMetadata(type="int")]
    )


def test_type_metadata_dict():
    assert type_metadata(dict[str, int]) == TypeMetadata(
        type="dict", type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")]
    )


def test_type_metadata_union():
    assert type_metadata(int | str) == TypeMetadata(
        type="union", type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")]
    )


def test_type_metadata_optional():
    assert type_metadata(Optional[int]) == TypeMetadata(type="int", optional=True)


def test_type_metadata_enum():
    class TestEnum(enum.Enum):
        A = "a"
        B = "b"

    metadata = type_metadata(TestEnum)
    assert metadata.type == "enum"
    assert metadata.type_name == "test_base_node.TestEnum"
    assert metadata.values is not None
    assert set(metadata.values) == {"a", "b"}


def test_type_metadata_nested():
    assert type_metadata(list[dict[str, Union[int, str]]]) == TypeMetadata(
        type="list",
        type_args=[
            TypeMetadata(
                type="dict",
                type_args=[
                    TypeMetadata(type="str"),
                    TypeMetadata(
                        type="union",
                        type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")],
                    ),
                ],
            )
        ],
    )


def test_type_metadata_unknown_type():
    class CustomClass:
        pass

    with pytest.raises(ValueError, match="Unknown type"):
        type_metadata(CustomClass)


def test_add_node_type_and_classname():
    class TestNode(BaseNode):
        pass

    add_node_type(TestNode)
    assert TestNode.get_node_type() in NODE_BY_TYPE


def test_get_node_class_and_by_name():
    class TestNode(BaseNode):
        pass

    add_node_type(TestNode)
    assert get_node_class(TestNode.get_node_type()) == TestNode


def test_base_node_from_dict():
    node_dict = {
        "type": DummyClass.get_node_type(),
        "id": "test_id",
        "parent_id": "parent_id",
        "ui_properties": {"x": 100, "y": 200},
        "data": {"prop": 456},
    }
    node = DummyClass.from_dict(node_dict)
    assert isinstance(node, DummyClass)
    assert node.id == "test_id"
    assert node.parent_id == "parent_id"
    assert node._ui_properties == {"x": 100, "y": 200}
    assert node.prop == 456


def test_base_node_get_json_schema():
    schema = DummyClass.get_json_schema()
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "prop" in schema["properties"]


class DataframeNode(BaseNode):
    dataframe: DataframeRef = DataframeRef()

    def process(self, context: ProcessingContext) -> DataframeRef:
        return self.dataframe


def test_node_assign_property_with_dataframe_dict():
    """Test that a dict with 'type' field is parsed into complex types like DataframeRef"""
    node = DataframeNode()

    # Create a dict representation of a DataframeRef
    dataframe_dict = {
        "type": "dataframe",
        "uri": "test://dataframe.csv",
        "columns": [
            {"name": "col1", "data_type": "int"},
            {"name": "col2", "data_type": "string"},
        ],
        "data": [[1, "a"], [2, "b"]],
    }

    # This should parse the dict into a DataframeRef object
    node.assign_property("dataframe", dataframe_dict)

    # Verify it was parsed correctly
    assert isinstance(node.dataframe, DataframeRef)
    assert node.dataframe.uri == "test://dataframe.csv"
    assert len(node.dataframe.columns) == 2
    assert node.dataframe.columns[0].name == "col1"
    assert node.dataframe.columns[0].data_type == "int"
    assert node.dataframe.data == [[1, "a"], [2, "b"]]


def test_node_set_properties_with_complex_types():
    """Test set_node_properties with complex types from dicts"""
    node = DataframeNode()

    properties = {
        "dataframe": {
            "type": "dataframe",
            "uri": "test://df.parquet",
            "columns": [{"name": "id", "data_type": "int"}],
        }
    }

    node.set_node_properties(properties)

    assert isinstance(node.dataframe, DataframeRef)
    assert node.dataframe.uri == "test://df.parquet"
    assert node.dataframe.columns[0].name == "id"


def test_node_assign_property_uses_from_dict_for_base_types():
    """Test that dicts with 'type' field are parsed using BaseType.from_dict"""
    from nodetool.metadata.types import ImageRef

    class ImageNode(BaseNode):
        image: ImageRef = ImageRef()

        def process(self, context: ProcessingContext) -> ImageRef:
            return self.image

    node = ImageNode()

    # Create a dict representation of an ImageRef
    image_dict = {"type": "image", "uri": "test://image.png", "asset_id": "12345"}

    # This should use from_dict, not model_validate
    node.assign_property("image", image_dict)

    # Verify it was parsed correctly
    assert isinstance(node.image, ImageRef)
    assert node.image.uri == "test://image.png"
    assert node.image.asset_id == "12345"


def test_node_from_dict_with_base_type_properties():
    """Test that node deserialization handles BaseType properties correctly"""
    node_dict = {
        "type": DataframeNode.get_node_type(),
        "id": "test_id",
        "data": {
            "dataframe": {
                "type": "dataframe",
                "uri": "test://df.csv",
                "columns": [
                    {"name": "id", "data_type": "int"},
                    {"name": "name", "data_type": "string"},
                ],
            }
        },
    }

    node = BaseNode.from_dict(node_dict)
    assert isinstance(node, DataframeNode)
    assert isinstance(node.dataframe, DataframeRef)
    assert node.dataframe.uri == "test://df.csv"
    assert len(node.dataframe.columns) == 2


def test_node_assign_property_list_of_base_types():
    """Test that lists of dicts with 'type' field are parsed correctly"""
    from nodetool.metadata.types import ImageRef

    class MultiImageNode(BaseNode):
        images: list[ImageRef] = []

        def process(self, context: ProcessingContext) -> list[ImageRef]:
            return self.images

    node = MultiImageNode()

    # Create a list of dict representations
    images_list = [
        {"type": "image", "uri": "test://img1.png"},
        {"type": "image", "uri": "test://img2.png"},
    ]

    node.assign_property("images", images_list)

    # Verify they were parsed correctly
    assert len(node.images) == 2
    assert all(isinstance(img, ImageRef) for img in node.images)
    assert node.images[0].uri == "test://img1.png"
    assert node.images[1].uri == "test://img2.png"
