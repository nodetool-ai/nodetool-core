from enum import Enum
from nodetool.metadata.types import ImageRef, NameToType

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import NPArray
from typing import Any


TYPE_ENUM_TO_ASSET_TYPE = {
    "audio": "audio",
    "video": "video",
    "image": "image",
    "tensor": "tensor",
    "folder": "folder",
    "file": "file",
    "dataframe": "dataframe",
    "model": "model",
    "thread": "thread",
    "thread_message": "thread_message",
}


def is_assignable(type_meta: TypeMetadata, value: Any) -> bool:
    """Checks if a value is assignable to a given type metadata.

    This function recursively checks if the provided value conforms to the
    type specification defined in type_meta. It handles various types including
    primitives, lists, dictionaries, enums, unions, assets (like image, video),
    tensors, and ComfyUI types (currently treated as always assignable).

    Args:
        type_meta: The metadata defining the expected type.
        value: The value to check.

    Returns:
        True if the value is assignable to the type, False otherwise.
    """
    python_type = type(value)

    # Handle the 'any' type, which accepts any value.
    if type_meta.type == "any":
        return True

    # Handle the 'object' type, which accepts non-primitive values.
    if type_meta.type == "object":
        primitive_types = (int, float, str, bool, type(None), Enum, list, dict, tuple)
        return not isinstance(value, primitive_types)

    # TODO: implement type checking for comfy types
    # Currently, ComfyUI types are always considered assignable.
    if type_meta.is_comfy_type():
        return True

    # Handle dictionary values
    if python_type is dict and "type" in value:
        return value["type"] == type_meta.type

    # Handle list types.
    if type_meta.type == "list":
        if python_type is list:
            t = (
                type_meta.type_args[0]
                if len(type_meta.type_args) > 0
                else TypeMetadata(type="any")
            )
            return all(is_assignable(t, v) for v in value)
        # Handle ImageRef containing a list.
        if python_type == ImageRef:
            assert isinstance(value, ImageRef)
            return type(value.data) is list
    # Handle dictionary types.
    if type_meta.type == "dict" and python_type is dict:
        if len(type_meta.type_args) != 2:
            # If type args aren't specified, assume any dict is valid.
            return True
        t = type_meta.type_args[0]  # Should handle potential Any type here if needed
        u = type_meta.type_args[1]  # Should handle potential Any type here if needed
        return all(
            is_assignable(t, k) and is_assignable(u, v) for k, v in value.items()
        )
    # Handle float types, allowing integers as well.
    if type_meta.type == "float" and (
        isinstance(value, float) or isinstance(value, int)
    ):
        return True
    # Handle asset types defined in TYPE_ENUM_TO_ASSET_TYPE.
    if type_meta.type in TYPE_ENUM_TO_ASSET_TYPE:
        asset_type = TYPE_ENUM_TO_ASSET_TYPE[type_meta.type]
        python_class = NameToType[type_meta.type]
        if isinstance(value, dict):
            # Check if the dictionary represents the correct asset type.
            return "type" in value and value["type"] == asset_type
        else:
            # Check if the value is an instance of the expected asset class.
            return isinstance(value, python_class)
    # Handle tensor types (NPArray).
    if type_meta.type == "tensor" and python_type == NPArray:
        t = (
            type_meta.type_args[0]
            if len(type_meta.type_args) > 0
            else TypeMetadata(type="any")
        )  # Use TypeMetadata for consistency
        return all(is_assignable(t, v) for v in value)
    # Handle union types.
    if type_meta.type == "union":
        return any(is_assignable(t, value) for t in type_meta.type_args)
    # Handle enum types.
    if type_meta.type == "enum":
        if isinstance(value, Enum):
            # Check if the enum value exists in the defined enum values.
            return value.value in type_meta.values
        else:
            # Check if the raw value exists in the defined enum values.
            assert type_meta.values is not None
            return value in type_meta.values

    # Default case: check if the Python type matches the expected type.
    return python_type == NameToType[type_meta.type]


def typecheck(type1: TypeMetadata, type2: TypeMetadata) -> bool:
    """Checks if two TypeMetadata instances are compatible.

    This function determines if two type metadata definitions represent
    compatible types. It handles 'any', ComfyUI types, lists, dicts,
    tensors, unions, and enums.

    Args:
        type1: The first type metadata.
        type2: The second type metadata.

    Returns:
        True if the types are compatible, False otherwise.
    """
    if type1.type == "any" or type2.type == "any":
        return True

    if type1.type != type2.type:
        return False

    if type1.is_comfy_type() and type2.is_comfy_type():
        return type1.type == type2.type

    if type1.type == "list" and type2.type == "list":
        return typecheck(type1.type_args[0], type2.type_args[0])

    if type1.type == "dict" and type2.type == "dict":
        if len(type1.type_args) != 2 or len(type2.type_args) != 2:
            return True
        return typecheck(type1.type_args[0], type2.type_args[0]) and typecheck(
            type1.type_args[1], type2.type_args[1]
        )

    if type1.type == "tensor" and type2.type == "tensor":
        return typecheck(type1.type_args[0], type2.type_args[0])

    if type1.type == "union" and type2.type == "union":
        return all(
            any(typecheck(t1, t2) for t2 in type2.type_args) for t1 in type1.type_args
        )

    if type1.type == "enum" and type2.type == "enum":
        return set(type1.values or []) == set(type2.values or [])

    return type1.type == type2.type
