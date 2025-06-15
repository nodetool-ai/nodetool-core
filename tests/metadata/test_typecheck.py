import pytest
from enum import Enum
from typing import Any

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import (
    typecheck,
    is_assignable,
    TYPE_ENUM_TO_ASSET_TYPE,
)
from nodetool.metadata.types import (
    ImageRef,
    VideoRef,
    AudioRef,
    FolderRef,
    TextRef,
    ModelRef,
    NPArray,
    DataframeRef,
)


class TestEnum(Enum):
    """Test enum for testing enum type checking."""

    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


class TestTypecheck:
    """Test cases for the typecheck function."""

    def test_any_type_compatibility(self):
        """Test that 'any' type is compatible with all types."""
        any_type = TypeMetadata(type="any")
        int_type = TypeMetadata(type="int")
        str_type = TypeMetadata(type="str")
        list_type = TypeMetadata(type="list", type_args=[TypeMetadata(type="int")])

        assert typecheck(any_type, int_type) is True
        assert typecheck(int_type, any_type) is True
        assert typecheck(any_type, str_type) is True
        assert typecheck(any_type, list_type) is True

    def test_same_type_compatibility(self):
        """Test that same types are compatible."""
        assert typecheck(TypeMetadata(type="int"), TypeMetadata(type="int")) is True
        assert typecheck(TypeMetadata(type="str"), TypeMetadata(type="str")) is True
        assert typecheck(TypeMetadata(type="float"), TypeMetadata(type="float")) is True
        assert typecheck(TypeMetadata(type="bool"), TypeMetadata(type="bool")) is True

    def test_different_type_incompatibility(self):
        """Test that different types are incompatible."""
        assert typecheck(TypeMetadata(type="int"), TypeMetadata(type="str")) is False
        assert typecheck(TypeMetadata(type="float"), TypeMetadata(type="bool")) is False
        assert typecheck(TypeMetadata(type="str"), TypeMetadata(type="list")) is False

    def test_comfy_type_compatibility(self):
        """Test ComfyUI type compatibility."""
        comfy1 = TypeMetadata(type="comfy.model")
        comfy2 = TypeMetadata(type="comfy.model")
        comfy3 = TypeMetadata(type="comfy.vae")

        assert typecheck(comfy1, comfy2) is True
        assert typecheck(comfy1, comfy3) is False

    def test_list_type_compatibility(self):
        """Test list type compatibility with nested types."""
        list_int = TypeMetadata(type="list", type_args=[TypeMetadata(type="int")])
        list_str = TypeMetadata(type="list", type_args=[TypeMetadata(type="str")])
        list_any = TypeMetadata(type="list", type_args=[TypeMetadata(type="any")])

        assert typecheck(list_int, list_int) is True
        assert typecheck(list_int, list_str) is False
        assert typecheck(list_int, list_any) is True
        assert typecheck(list_any, list_str) is True

    def test_dict_type_compatibility(self):
        """Test dict type compatibility."""
        dict_str_int = TypeMetadata(
            type="dict", type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")]
        )
        dict_str_str = TypeMetadata(
            type="dict", type_args=[TypeMetadata(type="str"), TypeMetadata(type="str")]
        )
        dict_any_any = TypeMetadata(
            type="dict", type_args=[TypeMetadata(type="any"), TypeMetadata(type="any")]
        )
        dict_no_args = TypeMetadata(type="dict", type_args=[])

        assert typecheck(dict_str_int, dict_str_int) is True
        assert typecheck(dict_str_int, dict_str_str) is False
        assert typecheck(dict_str_int, dict_any_any) is True
        assert typecheck(dict_no_args, dict_str_int) is True  # No args means any dict

    def test_union_type_compatibility(self):
        """Test union type compatibility."""
        union1 = TypeMetadata(
            type="union", type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")]
        )
        union2 = TypeMetadata(
            type="union", type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")]
        )
        union3 = TypeMetadata(
            type="union",
            type_args=[TypeMetadata(type="int"), TypeMetadata(type="float")],
        )

        # Unions with same types (order doesn't matter) should be compatible
        assert typecheck(union1, union2) is True
        # All types in union1 must be compatible with at least one type in union3
        assert (
            typecheck(union1, union3) is False
        )  # str is not compatible with int or float

    def test_enum_type_compatibility(self):
        """Test enum type compatibility."""
        enum1 = TypeMetadata(type="enum", values=["a", "b", "c"])
        enum2 = TypeMetadata(type="enum", values=["a", "b", "c"])
        enum3 = TypeMetadata(type="enum", values=["a", "b"])
        enum4 = TypeMetadata(type="enum", values=["a", "b", "c", "d"])

        assert typecheck(enum1, enum2) is True
        assert typecheck(enum1, enum3) is False
        assert typecheck(enum1, enum4) is False

    def test_nested_complex_types(self):
        """Test typecheck with deeply nested complex types."""
        # List of lists
        list_of_lists_int = TypeMetadata(
            type="list",
            type_args=[TypeMetadata(type="list", type_args=[TypeMetadata(type="int")])],
        )
        list_of_lists_str = TypeMetadata(
            type="list",
            type_args=[TypeMetadata(type="list", type_args=[TypeMetadata(type="str")])],
        )
        assert typecheck(list_of_lists_int, list_of_lists_int) is True
        assert typecheck(list_of_lists_int, list_of_lists_str) is False

        # Dict of lists
        dict_of_lists = TypeMetadata(
            type="dict",
            type_args=[
                TypeMetadata(type="str"),
                TypeMetadata(type="list", type_args=[TypeMetadata(type="int")]),
            ],
        )
        dict_of_lists_same = TypeMetadata(
            type="dict",
            type_args=[
                TypeMetadata(type="str"),
                TypeMetadata(type="list", type_args=[TypeMetadata(type="int")]),
            ],
        )
        assert typecheck(dict_of_lists, dict_of_lists_same) is True

        # List of unions
        list_of_unions = TypeMetadata(
            type="list",
            type_args=[
                TypeMetadata(
                    type="union",
                    type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")],
                )
            ],
        )
        list_of_unions_same = TypeMetadata(
            type="list",
            type_args=[
                TypeMetadata(
                    type="union",
                    type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")],
                )
            ],
        )
        assert typecheck(list_of_unions, list_of_unions_same) is True

    def test_edge_case_compatibility(self):
        """Test edge cases in typecheck function."""
        # Dict with no type args should be compatible with any dict
        dict_no_args = TypeMetadata(type="dict", type_args=[])
        dict_with_args = TypeMetadata(
            type="dict", type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")]
        )
        assert typecheck(dict_no_args, dict_with_args) is True
        assert typecheck(dict_with_args, dict_no_args) is True

        # Dict with wrong number of type args (not 2)
        dict_one_arg = TypeMetadata(type="dict", type_args=[TypeMetadata(type="str")])
        dict_three_args = TypeMetadata(
            type="dict",
            type_args=[
                TypeMetadata(type="str"),
                TypeMetadata(type="int"),
                TypeMetadata(type="bool"),
            ],
        )
        assert typecheck(dict_one_arg, dict_with_args) is True
        assert typecheck(dict_three_args, dict_with_args) is True


class TestIsAssignable:
    """Test cases for the is_assignable function."""

    def test_any_type_assignability(self):
        """Test that any value is assignable to 'any' type."""
        any_type = TypeMetadata(type="any")

        assert is_assignable(any_type, 42) is True
        assert is_assignable(any_type, "hello") is True
        assert is_assignable(any_type, [1, 2, 3]) is True
        assert is_assignable(any_type, {"key": "value"}) is True
        assert is_assignable(any_type, None) is True

    def test_object_type_assignability(self):
        """Test object type assignability (non-primitives)."""
        object_type = TypeMetadata(type="object")

        # Primitives should not be assignable
        assert is_assignable(object_type, 42) is False
        assert is_assignable(object_type, "hello") is False
        assert is_assignable(object_type, True) is False
        assert is_assignable(object_type, None) is False
        assert is_assignable(object_type, [1, 2, 3]) is False
        assert is_assignable(object_type, {"key": "value"}) is False

        # Non-primitives should be assignable
        class CustomClass:
            pass

        assert is_assignable(object_type, CustomClass()) is True

    def test_primitive_type_assignability(self):
        """Test primitive type assignability."""
        assert is_assignable(TypeMetadata(type="int"), 42) is True
        assert is_assignable(TypeMetadata(type="int"), "42") is False
        assert is_assignable(TypeMetadata(type="str"), "hello") is True
        assert is_assignable(TypeMetadata(type="str"), 42) is False
        assert is_assignable(TypeMetadata(type="bool"), True) is True
        assert is_assignable(TypeMetadata(type="bool"), 1) is False

        # Float should accept both float and int
        assert is_assignable(TypeMetadata(type="float"), 3.14) is True
        assert is_assignable(TypeMetadata(type="float"), 42) is True

    def test_dict_value_with_type_field(self):
        """Test dict values with 'type' field."""
        image_type = TypeMetadata(type="image")
        assert is_assignable(image_type, {"type": "image", "uri": "test.jpg"}) is True
        assert is_assignable(image_type, {"type": "video", "uri": "test.mp4"}) is False

    def test_list_type_assignability(self):
        """Test list type assignability."""
        list_int = TypeMetadata(type="list", type_args=[TypeMetadata(type="int")])
        list_str = TypeMetadata(type="list", type_args=[TypeMetadata(type="str")])
        list_any = TypeMetadata(type="list", type_args=[TypeMetadata(type="any")])
        list_no_args = TypeMetadata(type="list", type_args=[])

        assert is_assignable(list_int, [1, 2, 3]) is True
        assert is_assignable(list_int, [1, "2", 3]) is False
        assert is_assignable(list_str, ["a", "b", "c"]) is True
        assert is_assignable(list_any, [1, "2", True]) is True
        assert is_assignable(list_no_args, [1, "2", True]) is True

        # Test ImageRef containing a list
        image_ref_list = ImageRef(data=[1, 2, 3])
        assert is_assignable(list_int, image_ref_list) is True

    def test_dict_type_assignability(self):
        """Test dict type assignability."""
        dict_str_int = TypeMetadata(
            type="dict", type_args=[TypeMetadata(type="str"), TypeMetadata(type="int")]
        )
        dict_no_args = TypeMetadata(type="dict", type_args=[])

        assert is_assignable(dict_str_int, {"a": 1, "b": 2}) is True
        assert is_assignable(dict_str_int, {"a": "1", "b": 2}) is False
        assert is_assignable(dict_str_int, {1: 1, 2: 2}) is False
        assert is_assignable(dict_no_args, {"any": "dict"}) is True

    def test_asset_type_assignability(self):
        """Test asset type assignability."""
        # Test with dict representation
        assert (
            is_assignable(
                TypeMetadata(type="image"), {"type": "image", "uri": "test.jpg"}
            )
            is True
        )
        assert (
            is_assignable(
                TypeMetadata(type="video"), {"type": "video", "uri": "test.mp4"}
            )
            is True
        )
        assert (
            is_assignable(
                TypeMetadata(type="audio"), {"type": "audio", "uri": "test.mp3"}
            )
            is True
        )

        # Test with actual asset objects
        assert (
            is_assignable(TypeMetadata(type="image"), ImageRef(uri="test.jpg")) is True
        )
        assert (
            is_assignable(TypeMetadata(type="video"), VideoRef(uri="test.mp4")) is True
        )
        assert (
            is_assignable(TypeMetadata(type="audio"), AudioRef(uri="test.mp3")) is True
        )
        assert (
            is_assignable(TypeMetadata(type="folder"), FolderRef(uri="/path")) is True
        )
        assert is_assignable(TypeMetadata(type="file"), TextRef(uri="test.txt")) is True
        assert (
            is_assignable(TypeMetadata(type="model"), ModelRef(uri="model.bin")) is True
        )
        assert is_assignable(TypeMetadata(type="dataframe"), DataframeRef()) is True

        # Test wrong asset type
        assert (
            is_assignable(TypeMetadata(type="image"), VideoRef(uri="test.mp4")) is False
        )

    def test_tensor_type_assignability(self):
        """Test tensor (NPArray) type assignability."""
        tensor_int = TypeMetadata(type="tensor", type_args=[TypeMetadata(type="int")])
        tensor_float = TypeMetadata(
            type="tensor", type_args=[TypeMetadata(type="float")]
        )
        tensor_no_args = TypeMetadata(type="tensor", type_args=[])

        np_array_int = NPArray.from_list([1, 2, 3])
        np_array_float = NPArray.from_list([1.0, 2.0, 3.0])
        np_array_mixed = NPArray.from_list([1, 2.0, "3"])

        # Note: The implementation seems to check each element
        assert is_assignable(tensor_int, np_array_int) is True
        assert is_assignable(tensor_float, np_array_float) is True
        assert is_assignable(tensor_no_args, np_array_mixed) is True

    def test_union_type_assignability(self):
        """Test union type assignability."""
        union_int_str = TypeMetadata(
            type="union", type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")]
        )

        assert is_assignable(union_int_str, 42) is True
        assert is_assignable(union_int_str, "hello") is True
        assert is_assignable(union_int_str, 3.14) is False
        assert is_assignable(union_int_str, True) is False

    def test_enum_type_assignability(self):
        """Test enum type assignability."""
        enum_type = TypeMetadata(type="enum", values=["value1", "value2", "value3"])

        # Test with enum instance
        assert is_assignable(enum_type, TestEnum.VALUE1) is True
        assert is_assignable(enum_type, TestEnum.VALUE2) is True

        # Test with raw values
        assert is_assignable(enum_type, "value1") is True
        assert is_assignable(enum_type, "value2") is True
        assert is_assignable(enum_type, "invalid") is False

        # Test with integer enum values
        enum_int = TypeMetadata(type="enum", values=[1, 2, 3])
        assert is_assignable(enum_int, 1) is True
        assert is_assignable(enum_int, 2) is True
        assert is_assignable(enum_int, 4) is False

    def test_comfy_type_assignability(self):
        """Test ComfyUI type assignability (always True for now)."""
        comfy_type = TypeMetadata(type="comfy.model")

        # According to the implementation, comfy types are always assignable
        assert is_assignable(comfy_type, {"id": "123"}) is True
        assert is_assignable(comfy_type, "anything") is True
        assert is_assignable(comfy_type, 42) is True

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test None values with enum
        enum_type = TypeMetadata(type="enum", values=None)
        with pytest.raises(AssertionError):
            is_assignable(enum_type, "value")

        # Test unknown type in NameToType
        unknown_type = TypeMetadata(type="unknown_type")
        with pytest.raises(KeyError):
            is_assignable(unknown_type, "value")

        # Test ImageRef with non-list data
        image_type = TypeMetadata(type="image")
        image_ref_with_str = ImageRef(data="not a list")
        assert is_assignable(image_type, image_ref_with_str) is True

        # Test NPArray with mixed types when expecting specific type
        tensor_int = TypeMetadata(type="tensor", type_args=[TypeMetadata(type="int")])
        np_array_with_str = NPArray.from_list([1, 2, "three"])
        assert is_assignable(tensor_int, np_array_with_str) is False

    def test_nested_assignability(self):
        """Test assignability with nested complex types."""
        # List of lists
        list_of_lists_int = TypeMetadata(
            type="list",
            type_args=[TypeMetadata(type="list", type_args=[TypeMetadata(type="int")])],
        )
        assert is_assignable(list_of_lists_int, [[1, 2], [3, 4]]) is True
        assert is_assignable(list_of_lists_int, [[1, 2], ["3", "4"]]) is False
        assert is_assignable(list_of_lists_int, [1, 2, 3]) is False

        # Dict of lists
        dict_of_lists = TypeMetadata(
            type="dict",
            type_args=[
                TypeMetadata(type="str"),
                TypeMetadata(type="list", type_args=[TypeMetadata(type="int")]),
            ],
        )
        assert is_assignable(dict_of_lists, {"a": [1, 2], "b": [3, 4]}) is True
        assert is_assignable(dict_of_lists, {"a": [1, 2], "b": ["3", "4"]}) is False
        assert is_assignable(dict_of_lists, {"a": 1, "b": 2}) is False

        # Union in list
        list_of_unions = TypeMetadata(
            type="list",
            type_args=[
                TypeMetadata(
                    type="union",
                    type_args=[TypeMetadata(type="int"), TypeMetadata(type="str")],
                )
            ],
        )
        assert is_assignable(list_of_unions, [1, "two", 3, "four"]) is True
        assert is_assignable(list_of_unions, [1, 2, 3]) is True
        assert is_assignable(list_of_unions, ["one", "two"]) is True
        assert is_assignable(list_of_unions, [1, 2.5, 3]) is False


class TestTypeEnumToAssetType:
    """Test the TYPE_ENUM_TO_ASSET_TYPE mapping."""

    def test_asset_type_mapping(self):
        """Test that all expected asset types are in the mapping."""
        expected_mappings = {
            "audio": "audio",
            "video": "video",
            "image": "image",
            "tensor": "tensor",
            "folder": "folder",
            "file": "file",
            "dataframe": "dataframe",
            "model": "model",
        }

        # Check that all expected mappings exist
        for key, value in expected_mappings.items():
            assert key in TYPE_ENUM_TO_ASSET_TYPE
            assert TYPE_ENUM_TO_ASSET_TYPE[key] == value
