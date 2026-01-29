import inspect
from enum import Enum
from pydantic import BaseModel
from typing import get_args, get_origin, Union
import nodetool.metadata.types as nt


def py_type_to_cs(py_t):
    mapping = {
        str: "string",
        int: "int",
        float: "double",
        bool: "bool",
        bytes: "byte[]",
    }
    if py_t is None or py_t is type(None):
        return "object"
    origin = get_origin(py_t)
    if origin is Union:
        args = [a for a in get_args(py_t) if a is not type(None)]
        if len(args) == 1:
            return py_type_to_cs(args[0]) + "?"
        return "object"
    if origin in (list, set, tuple):
        args = get_args(py_t)
        inner = py_type_to_cs(args[0]) if args else "object"
        return f"List<{inner}>"
    if origin is dict:
        args = get_args(py_t)
        key = py_type_to_cs(args[0]) if args else "string"
        value = py_type_to_cs(args[1]) if len(args) > 1 else "object"
        return f"Dictionary<{key}, {value}>"
    if inspect.isclass(py_t):
        if issubclass(py_t, Enum):
            return py_t.__name__
        if issubclass(py_t, BaseModel):
            return py_t.__name__
    return mapping.get(py_t, "object")


def generate_enum(enum_cls):
    lines = [f"public enum {enum_cls.__name__}", "{"]
    for name in enum_cls.__members__:
        lines.append(f"    {name},")
    lines.append("}")
    return "\n".join(lines)


def generate_class(cls):
    lines = ["[MessagePack.MessagePackObject(keyAsPropertyName: true)]"]
    lines.append(f"public partial class {cls.__name__}")
    lines.append("{")
    for field_name, field in cls.model_fields.items():
        cs_type = py_type_to_cs(field.annotation)
        lines.append(f"    public {cs_type} {field_name} {{ get; set; }}")
    lines.append("}")
    return "\n".join(lines)


classes = []
enums = []
for name, obj in inspect.getmembers(nt):
    if inspect.isclass(obj) and obj.__module__ == nt.__name__:
        if issubclass(obj, BaseModel):
            classes.append(obj)
        elif issubclass(obj, Enum):
            enums.append(obj)

cs_lines = ["using MessagePack;", "using System.Collections.Generic;", "namespace Nodetool.Metadata {"]
for enum_cls in enums:
    cs_lines.append(generate_enum(enum_cls))
    cs_lines.append("")
for cls in classes:
    cs_lines.append(generate_class(cls))
    cs_lines.append("")
cs_lines.append("}")

with open("csharp/Nodetool/Metadata/Types.cs", "w") as f:
    f.write("\n".join(cs_lines))
