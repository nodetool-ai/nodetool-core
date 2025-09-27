import asyncio
import enum
import os
from typing import AsyncGenerator, ClassVar, Optional, Union, TypedDict
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
from nodetool.metadata.types import OutputSlot, DataframeRef, ColumnDef, ImageRef
from nodetool.workflows.inbox import NodeInbox


current_dir = os.path.dirname(os.path.realpath(__file__))
test_file = os.path.join(current_dir, "test.jpg")


class DummyClass(BaseNode):
    prop: int = 123

    async def process(self, context: ProcessingContext) -> int:
        return self.prop


class StringNode(BaseNode):
    value: str = "test"

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class ImageReturnNode(BaseNode):
    image: ImageRef = ImageRef()

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.image


class DictReturnNode(BaseNode):
    async def process(self, context: ProcessingContext) -> dict[str, int]:
        return {"value": 1}


class TypedDictProcessNode(BaseNode):
    class OutputType(TypedDict):
        text: str
        count: int

    async def process(
        self, context: ProcessingContext
    ) -> "TypedDictProcessNode.OutputType":
        return {"text": "hello", "count": 0}


class TypedDictStreamNode(BaseNode):
    class OutputType(TypedDict):
        text: str
        count: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator["TypedDictStreamNode.OutputType", None]:
        yield {"text": "hello", "count": 0}


class StreamNode(BaseNode):
    class OutputType(TypedDict):
        stream: int

    async def gen_process(
        self, context: ProcessingContext
    ) -> AsyncGenerator[OutputType, None]:
        yield {"stream": 1}


class ConfigurableDynamicNode(BaseNode):
    _expose_as_tool: ClassVar[bool] = True
    _supports_dynamic_outputs: ClassVar[bool] = True
    _is_dynamic: ClassVar[bool] = True
    _layout: ClassVar[str] = "custom"

    value: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return 0


class StreamingInputNode(BaseNode):
    calls: list[str] = []

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    async def process(self, context: ProcessingContext) -> int:
        self.calls.append("process")
        return 1


class CustomRoutingNode(BaseNode):
    suppressed_outputs: set[str] = {"meta"}

    def should_route_output(self, output_name: str) -> bool:
        return output_name not in self.suppressed_outputs


def test_node_creation():
    node = BaseNode(id="")  # type: ignore[call-arg]
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
    node = StringNode(id="")  # type: ignore[call-arg]
    assert node.outputs() == [OutputSlot(type=TypeMetadata(type="str"), name="output")]


def test_image_return_node_outputs():
    node = ImageReturnNode()
    assert node.outputs() == [
        OutputSlot(type=TypeMetadata(type="image"), name="output")
    ]


def test_dict_return_node_outputs():
    node = DictReturnNode()
    assert node.outputs() == [
        OutputSlot(
            type=TypeMetadata(
                type="dict",
                type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")],
            ),
            name="output",
        )
    ]


def test_typed_dict_process_node_outputs():
    node = TypedDictProcessNode()
    assert node.outputs() == [
        OutputSlot(type=TypeMetadata(type="str"), name="text"),
        OutputSlot(type=TypeMetadata(type="int"), name="count"),
    ]


def test_typed_dict_stream_node_outputs():
    node = TypedDictStreamNode()
    assert node.outputs() == [
        OutputSlot(type=TypeMetadata(type="str"), name="text"),
        OutputSlot(type=TypeMetadata(type="int"), name="count"),
    ]


def test_return_type_streaming_node():
    assert StreamNode.return_type() == StreamNode.OutputType


def test_return_type_process_variants():
    assert DummyClass.return_type() == int
    assert ImageReturnNode.return_type() is ImageRef
    assert DictReturnNode.return_type() == dict[str, int]


def test_return_type_typed_dict_process():
    assert TypedDictProcessNode.return_type() is TypedDictProcessNode.OutputType


def test_return_type_typed_dict_async_generator():
    assert TypedDictStreamNode.return_type() is TypedDictStreamNode.OutputType


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
    node, _ = DummyClass.from_dict(node_dict)
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

    async def process(self, context: ProcessingContext) -> DataframeRef:
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
    assert node.dataframe.columns is not None
    columns = node.dataframe.columns
    assert len(columns) == 2
    assert columns[0].name == "col1"
    assert columns[0].data_type == "int"
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
    assert node.dataframe.columns is not None
    assert node.dataframe.columns[0].name == "id"


def test_node_assign_property_uses_from_dict_for_base_types():
    """Test that dicts with 'type' field are parsed using BaseType.from_dict"""
    from nodetool.metadata.types import ImageRef

    class ImageNode(BaseNode):
        image: ImageRef = ImageRef()

        async def process(self, context: ProcessingContext) -> ImageRef:
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

    node, _ = BaseNode.from_dict(node_dict)
    assert isinstance(node, DataframeNode)
    assert isinstance(node.dataframe, DataframeRef)
    assert node.dataframe.uri == "test://df.csv"
    assert node.dataframe.columns is not None
    assert len(node.dataframe.columns) == 2


def test_node_assign_property_list_of_base_types():
    """Test that lists of dicts with 'type' field are parsed correctly"""
    from nodetool.metadata.types import ImageRef

    class MultiImageNode(BaseNode):
        images: list[ImageRef] = []

        async def process(self, context: ProcessingContext) -> list[ImageRef]:
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


def test_configurable_dynamic_node_flags():
    node = ConfigurableDynamicNode()
    assert node.expose_as_tool() is True
    assert node.supports_dynamic_outputs() is True
    assert node.is_dynamic() is True
    assert node.layout() == "custom"


def test_configurable_dynamic_node_id_parent_helpers():
    node = ConfigurableDynamicNode(id="abc", parent_id="parent")
    assert node.id == "abc"
    assert node.parent_id == "parent"
    assert node.has_parent() is True
    as_dict = node.to_dict()
    assert as_dict["id"] == "abc"
    assert as_dict["parent_id"] == "parent"
    assert as_dict["type"] == ConfigurableDynamicNode.get_node_type()


def test_configurable_dynamic_node_dynamic_outputs_management():
    node = ConfigurableDynamicNode(value=42)
    assert node.get_dynamic_output_slots() == []
    node.add_output("summary", int)
    outputs = node.get_dynamic_output_slots()
    assert outputs == [OutputSlot(type=TypeMetadata(type="int"), name="summary")]
    node.add_output("raw")
    assert len(node.get_dynamic_output_slots()) == 2
    node.remove_output("raw")
    assert len(node.get_dynamic_output_slots()) == 1


def test_configurable_dynamic_node_properties_and_dynamic_values():
    node = ConfigurableDynamicNode()
    node.assign_property("value", 3)
    assert node.value == 3
    error = node.assign_property("missing", 1)
    assert error is None  # dynamic property stored
    assert node.read_property("missing") == 1
    with pytest.raises(ValueError):
        node.read_property("nonexistent")
    assert node.properties_for_client() == {}


def test_configurable_dynamic_node_outputs_for_instance():
    node = ConfigurableDynamicNode()
    class_outputs = node.outputs_for_instance()
    assert OutputSlot(type=TypeMetadata(type="int"), name="output") in class_outputs
    node.add_output("extra", str)
    instance_outputs = node.outputs_for_instance()
    assert any(o.name == "extra" for o in instance_outputs)


def test_should_route_output_custom_node():
    node = CustomRoutingNode()
    assert node.should_route_output("output") is True
    assert node.should_route_output("meta") is False


@pytest.mark.asyncio
async def test_has_input_iter_and_recv():
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    assert node.has_input() is False
    inbox.put("input", 7)
    await asyncio.sleep(0)
    assert node.has_input() is True
    assert await node.recv("input") == 7


@pytest.mark.asyncio
async def test_iter_input_consumes_all_values():
    inbox = NodeInbox()
    inbox.add_upstream("stream", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    inbox.put("stream", 1)
    inbox.put("stream", 2)
    inbox.mark_source_done("stream")
    collected = []
    async for item in node.iter_input("stream"):
        collected.append(item)
    assert collected == [1, 2]


@pytest.mark.asyncio
async def test_iter_any_input_respects_order():
    inbox = NodeInbox()
    inbox.add_upstream("a", 1)
    inbox.add_upstream("b", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    inbox.put("a", 1)
    inbox.put("b", 2)
    inbox.mark_source_done("a")
    inbox.mark_source_done("b")
    collected = []
    async for name, item in node.iter_any_input():
        collected.append((name, item))
    assert collected == [("a", 1), ("b", 2)]


@pytest.mark.asyncio
async def test_recv_without_inbox_raises():
    node = DummyClass()
    with pytest.raises(RuntimeError):
        await node.recv("missing")


@pytest.mark.asyncio
async def test_iter_input_without_inbox_raises():
    node = DummyClass()
    with pytest.raises(RuntimeError):
        async for _ in node.iter_input("missing"):
            pass


@pytest.mark.asyncio
async def test_iter_any_without_inbox_raises():
    node = DummyClass()
    with pytest.raises(RuntimeError):
        async for _ in node.iter_any_input():
            pass


@pytest.mark.asyncio
async def test_recv_handles_end_of_stream():
    inbox = NodeInbox()
    inbox.add_upstream("input", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    inbox.mark_source_done("input")
    with pytest.raises(StopAsyncIteration):
        await node.recv("input")


@pytest.mark.asyncio
async def test_iter_input_handles_end_of_stream():
    inbox = NodeInbox()
    inbox.add_upstream("stream", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    inbox.mark_source_done("stream")
    collected = [item async for item in node.iter_input("stream")]
    assert collected == []


@pytest.mark.asyncio
async def test_iter_any_handles_end_of_stream():
    inbox = NodeInbox()
    inbox.add_upstream("stream", 1)
    node = DummyClass()
    node.attach_inbox(inbox)
    inbox.mark_source_done("stream")
    collected = [item async for item in node.iter_any_input()]
    assert collected == []


def test_required_inputs_default():
    assert DummyClass().required_inputs() == []


def test_expose_visibility_helpers():
    assert ConfigurableDynamicNode.expose_as_tool() is True
    assert ConfigurableDynamicNode.supports_dynamic_outputs() is True
    assert ConfigurableDynamicNode.is_visible() is True


def test_layout_and_namespace_helpers():
    assert ConfigurableDynamicNode.layout() == "custom"
    assert ConfigurableDynamicNode.get_namespace() == ConfigurableDynamicNode.__module__
    assert ConfigurableDynamicNode.get_title() == "Configurable Dynamic"


def test_properties_and_fields_helpers():
    props = ConfigurableDynamicNode.properties()
    names = [p.name for p in props]
    assert "value" in names
    props_dict = ConfigurableDynamicNode.properties_dict()
    assert "value" in props_dict
    fields = ConfigurableDynamicNode.field_types()
    assert "value" in fields
    inherited = ConfigurableDynamicNode.inherited_fields()
    assert "value" in inherited


def test_node_properties_collection_and_dynamic_state():
    node = ConfigurableDynamicNode(value=5)
    assert node.node_properties()["value"] == 5
    node.add_output("extra")
    node_outputs = node.outputs_for_instance()
    assert any(o.name == "extra" for o in node_outputs)


def test_dynamic_properties_read_and_assignment():
    node = ConfigurableDynamicNode()
    node.assign_property("new_prop", 10)
    assert node.read_property("new_prop") == 10
    with pytest.raises(ValueError):
        _ = DummyClass().read_property("missing")
