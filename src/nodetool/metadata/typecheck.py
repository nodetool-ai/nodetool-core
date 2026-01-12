from enum import Enum
from typing import Any

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import ImageRef, NameToType, NPArray


def typecheck(type1: TypeMetadata, type2: TypeMetadata) -> bool:
    """Checks if type1 is a subtype of type2.

    This function determines if type1 is a subtype of type2, meaning that
    any value of type1 can be used where type2 is expected. It handles 'any',
    ComfyUI types, lists, dicts, tensors, unions, and enums.

    Examples of subtype relationships:
    - list[str] is a subtype of list[any]
    - str is a subtype of union[str, int]
    - enum with values [1, 2] is a subtype of enum with values [1, 2, 3]
    - float is a subtype of list[float] (single value can be wrapped into list)
    - int is a subtype of list[int] (single value can be wrapped into list)

    Args:
        type1: The subtype to check.
        type2: The supertype to check against.

    Returns:
        True if type1 is a subtype of type2, False otherwise.
    """
    # Any type can be used where 'any' is expected (any is a supertype of everything)
    if type2.type == "any":
        return True

    # 'any' can be used where 'any' is expected, but also 'any' is a subtype of everything
    if type1.type == "any":
        return True

    # Integer can be converted to float
    if type1.type == "int" and type2.type == "float":
        return True

    # Float cannot be converted to integer
    if type1.type == "float" and type2.type == "int":
        # Allow float to int conversion (loss of precision acceptable for workflow edge validation)
        return True

    # Handle union types - check both type1 and type2 cases
    if type1.type == "union" and type2.type == "union":
        # For union subtyping: type1 is a subtype of type2 if every member of type1
        # is a subtype of at least one member of type2
        return all(any(typecheck(t1, t2) for t2 in type2.type_args) for t1 in type1.type_args)

    # Handle union types in type2 - type1 is a subtype if it's a subtype of any member
    if type2.type == "union":
        return any(typecheck(type1, t2) for t2 in type2.type_args)

    # Handle union types in type1 - accept if any member matches
    if type1.type == "union":
        return any(typecheck(t1, type2) for t1 in type1.type_args)

    # Special case: T -> list[T] (single value can be wrapped into a list)
    # This allows edges where source outputs T but target expects list[T]
    # Only apply when list has specific type args (not list[Any])
    if type2.type == "list" and type1.type != "list" and len(type2.type_args) > 0:
        element_type = type2.type_args[0]
        return typecheck(type1, element_type)

    # From here on, we need the base types to match for most cases
    if type1.type != type2.type:
        return False

    # ComfyUI types - exact match required
    if type1.is_comfy_type() and type2.is_comfy_type():
        return type1.type == type2.type

    # List types - element type must be subtype
    if type1.type == "list" and type2.type == "list":
        if len(type1.type_args) == 0 and len(type2.type_args) == 0:
            return True
        if len(type1.type_args) == 0 or len(type2.type_args) == 0:
            # If one has no type args, treat it as list[any]
            element_type1 = type1.type_args[0] if len(type1.type_args) > 0 else TypeMetadata(type="any")
            element_type2 = type2.type_args[0] if len(type2.type_args) > 0 else TypeMetadata(type="any")
            return typecheck(element_type1, element_type2)
        return typecheck(type1.type_args[0], type2.type_args[0])

    # Dictionary types - both key and value types must be subtypes
    if type1.type == "dict" and type2.type == "dict":
        if len(type1.type_args) != 2 or len(type2.type_args) != 2:
            return True  # If type args aren't fully specified, assume compatible
        return typecheck(type1.type_args[0], type2.type_args[0]) and typecheck(type1.type_args[1], type2.type_args[1])

    # Enum types - type1 values must be equal to type2 values for exact compatibility
    if type1.type == "enum" and type2.type == "enum":
        if type1.values is None or type2.values is None:
            return True  # If values aren't specified, assume compatible
        return set(type1.values) == set(type2.values)

    # For other types, require exact match
    return type1.type == type2.type


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
    "np_array": "np_array",
}


def is_empty(value: Any) -> bool:
    """Checks if a value is empty."""
    return value is None or (isinstance(value, list | dict) and len(value) == 0)


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
            t = type_meta.type_args[0] if len(type_meta.type_args) > 0 else TypeMetadata(type="any")
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
        return all(is_assignable(t, k) and is_assignable(u, v) for k, v in value.items())
    # Handle float types, allowing integers as well.
    if type_meta.type == "float" and isinstance(value, float | int):
        return True
    # Handle tensor types (NPArray) - must come before asset type check
    if (type_meta.type == "tensor" or type_meta.type == "np_array") and python_type == NPArray:
        t = (
            type_meta.type_args[0] if len(type_meta.type_args) > 0 else TypeMetadata(type="any")
        )  # Use TypeMetadata for consistency
        # Convert NPArray to list to check element types
        data = value.to_list() if hasattr(value, "to_list") else list(value)
        return all(is_assignable(t, v) for v in data)
    # Handle asset types defined in TYPE_ENUM_TO_ASSET_TYPE.
    if type_meta.type in TYPE_ENUM_TO_ASSET_TYPE:
        asset_type = TYPE_ENUM_TO_ASSET_TYPE[type_meta.type]
        # Special case for file type which maps to text
        if type_meta.type == "file" and type_meta.type not in NameToType:
            python_class = NameToType.get("text")
        # Special case for tensor which maps to np_array
        elif type_meta.type == "tensor" and type_meta.type not in NameToType:
            python_class = NameToType.get("np_array", NPArray)
        # Special case for model which maps to model_ref
        elif type_meta.type == "model" and type_meta.type not in NameToType:
            python_class = NameToType.get("model_ref")
        else:
            python_class = NameToType.get(type_meta.type)

        if python_class is None:
            return False

        if isinstance(value, dict):
            # Check if the dictionary represents the correct asset type.
            return "type" in value and value["type"] == asset_type
        else:
            # Check if the value is an instance of the expected asset class.
            # Special case for ModelRef which has type "model_ref" but maps to "model"
            if hasattr(value, "type") and type_meta.type == "model" and value.type == "model_ref":
                return isinstance(value, python_class)
            return isinstance(value, python_class)
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
