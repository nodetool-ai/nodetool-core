from pydantic import BaseModel
from typing import Any, Optional


ALL_TYPES = [
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "tuple",
    "union",
    "enum",
    "any",
]


class TypeMetadata(BaseModel):
    """
    Metadata for a type.
    """

    type: str
    optional: bool = False
    values: Optional[list[str | int]] = None
    type_args: list["TypeMetadata"] = []
    type_name: Optional[str] = None

    def __repr__(self):
        result = ""

        if self.type == "list":
            item_type = self.type_args[0].__repr__() if self.type_args else "Any"
            result = f"List[{item_type}]"
        elif self.type == "dict":
            if len(self.type_args) >= 2:
                key_type = self.type_args[0].__repr__()
                val_type = self.type_args[1].__repr__()
                result = f"Dict[{key_type}, {val_type}]"
            else:
                result = "Dict[Any, Any]"
        elif self.type == "tuple":
            types = ", ".join(arg.__repr__() for arg in self.type_args)
            result = f"Tuple[{types}]"
        elif self.type == "union":
            types = " | ".join(arg.__repr__() for arg in self.type_args)
            result = f"({types})"
        elif self.type == "enum":
            values = str(self.values) if self.values else "[]"
            result = f"Enum{values}"
        else:
            # For primitive and other types
            result = self.type

            if self.type_args:
                args = ", ".join(arg.__repr__() for arg in self.type_args)
                result += f"[{args}]"

        # Handle optional types
        if self.optional:
            result = f"Optional[{result}]"

        return result

    def is_asset_type(self, recursive: bool = False):
        from nodetool.metadata.types import asset_types

        if recursive and self.is_union_type():
            return any(t.is_asset_type(recursive=True) for t in self.type_args)
        return self.type in asset_types

    def is_cacheable_type(self):
        if self.is_list_type() or self.is_union_type() or self.is_dict_type():
            return all(t.is_cacheable_type() for t in self.type_args)
        if self.is_comfy_data_type():
            return True
        if self.is_comfy_type():
            return False
        return True

    def is_serializable_type(self):
        if self.is_list_type() or self.is_union_type() or self.is_dict_type():
            return all(t.is_serializable_type() for t in self.type_args)
        if self.is_comfy_data_type():
            return True
        if self.is_comfy_type():
            return False
        return True

    def is_comfy_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_comfy_type(recursive=True) for t in self.type_args)
        return self.type.startswith("comfy.")

    def is_comfy_model(self, recursive: bool = False):
        from nodetool.metadata.types import comfy_model_types

        if recursive and self.is_union_type():
            return any(t.is_comfy_model(recursive=True) for t in self.type_args)
        return self.type in comfy_model_types

    def is_model_file_type(self, recursive: bool = False):
        from nodetool.metadata.types import model_file_types

        if recursive and self.is_union_type():
            return any(t.is_model_file_type(recursive=True) for t in self.type_args)
        return self.type in model_file_types

    def is_comfy_data_type(self, recursive: bool = False):
        from nodetool.metadata.types import comfy_data_types

        if recursive and self.is_union_type():
            return any(t.is_comfy_data_type(recursive=True) for t in self.type_args)
        return self.type in comfy_data_types

    def is_primitive_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_primitive_type(recursive=True) for t in self.type_args)
        return self.type in ["int", "float", "bool", "str", "text"]

    def is_enum_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_enum_type(recursive=True) for t in self.type_args)
        return self.type == "enum"

    def is_list_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_list_type(recursive=True) for t in self.type_args)
        return self.type == "list"

    def is_tuple_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_tuple_type(recursive=True) for t in self.type_args)
        return self.type == "tuple"

    def is_dict_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_dict_type(recursive=True) for t in self.type_args)
        return self.type == "dict"

    def is_image_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_image_type(recursive=True) for t in self.type_args)
        return self.type == "image"

    def is_audio_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_audio_type(recursive=True) for t in self.type_args)
        return self.type == "audio"

    def is_video_type(self, recursive: bool = False):
        if recursive and self.is_union_type():
            return any(t.is_video_type(recursive=True) for t in self.type_args)
        return self.type == "video"

    def is_union_type(self):
        return self.type == "union"

    def get_python_type(self):
        from nodetool.metadata.types import NameToType

        if self.is_enum_type():
            if self.type_name not in NameToType:
                raise ValueError(
                    f"Unknown enum type: {self.type_name}. Types must derive from BaseType"
                )
            return NameToType[self.type_name]
        else:
            if self.type not in NameToType:
                raise ValueError(
                    f"Unknown type: {self.type}. Types must derive from BaseType"
                )
            return NameToType[self.type]

    def get_json_schema(self) -> dict[str, Any]:
        """
        Returns a JSON schema for the type.
        """
        if self.type == "any":
            return {}
        if self.is_comfy_type():
            return {"type": "object", "properties": {"id": {"type": "string"}}}
        if self.is_image_type():
            return {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "format": "uri"},
                    "type": {"type": "string", "enum": ["image"]},
                },
            }
        if self.is_audio_type():
            return {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "format": "uri"},
                    "type": {"type": "string", "enum": ["audio"]},
                },
            }
        if self.is_video_type():
            return {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "format": "uri"},
                    "type": {"type": "string", "enum": ["video"]},
                },
            }
        if self.is_asset_type():  # fallback for other asset types
            return {
                "type": "object",
                "properties": {
                    "uri": {"type": "string", "format": "uri"},
                    "type": {"type": "string"},
                },
            }
        if self.type == "none":
            return {"type": "null"}
        if self.type == "int":
            return {"type": "integer"}
        if self.type == "float":
            return {"type": "number"}
        if self.type == "bool":
            return {"type": "boolean"}
        if self.type == "str":
            return {"type": "string"}
        if self.type == "bytes":
            return {"type": "string", "format": "binary"}
        if self.type == "text":
            return {"type": "string"}
        if self.type == "np_array":
            return {"type": "array", "items": {"type": "number"}}
        if self.type == "list":
            if not self.type_args:
                return {"type": "array"}
            else:
                return {
                    "type": "array",
                    "items": self.type_args[0].get_json_schema(),
                }
        if self.type == "dict":
            if not self.type_args:
                return {"type": "object"}
            return {
                "type": "object",
                "properties": {
                    f"key_{i}": t.get_json_schema()
                    for i, t in enumerate(self.type_args)
                },
            }
        if self.type == "union":
            return {
                "anyOf": [t.get_json_schema() for t in self.type_args],
            }
        if self.type == "enum":
            if self.values is None:
                return {"type": "string"}
            else:
                return {
                    "type": "string",
                    "enum": self.values,
                }
        if self.type == "tuple":
            return {
                "type": "array",
                "items": [t.get_json_schema() for t in self.type_args],
            }

        if self.type == "object":
            return {
                "type": "object",
            }

        python_type = self.get_python_type()
        if issubclass(python_type, BaseModel):
            return python_type.model_json_schema()

        raise ValueError(f"Unknown type: {self.type}")
