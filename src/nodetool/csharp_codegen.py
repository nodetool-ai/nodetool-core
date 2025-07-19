"""Generate C# classes for Nodetool ``BaseType`` subclasses."""

import os
import pkgutil
import importlib
import inspect
from typing import Any, get_origin, get_args

from pydantic import BaseModel

from .metadata.types import BaseType


_PRIMITIVE_TYPE_MAP = {
    str: "string",
    int: "int",
    float: "double",
    bool: "bool",
    bytes: "byte[]",
}


def _python_type_to_csharp(tp: Any) -> str:
    """Convert a Python type annotation to a C# type string."""
    origin = get_origin(tp)
    if origin is None:
        if isinstance(tp, type):
            if tp in _PRIMITIVE_TYPE_MAP:
                return _PRIMITIVE_TYPE_MAP[tp]
            if issubclass(tp, BaseType):
                return tp.__name__
            if issubclass(tp, BaseModel):
                return tp.__name__
        # typing.Any or unknown
        if tp is Any:
            return "object"
        # typing.Literal values
        if str(tp).startswith("typing.Literal"):
            args = get_args(tp)
            if args:
                return _python_type_to_csharp(type(args[0]))
        return "object"

    if origin is list or origin is list:
        args = get_args(tp)
        inner = _python_type_to_csharp(args[0]) if args else "object"
        return f"List<{inner}>"
    if origin is dict:
        args = get_args(tp)
        key = _python_type_to_csharp(args[0]) if args else "object"
        val = _python_type_to_csharp(args[1]) if len(args) > 1 else "object"
        return f"Dictionary<{key}, {val}>"
    if origin is set:
        args = get_args(tp)
        inner = _python_type_to_csharp(args[0]) if args else "object"
        return f"List<{inner}>"
    if origin is tuple:
        args = get_args(tp)
        inner = _python_type_to_csharp(args[0]) if args else "object"
        return f"List<{inner}>"
    if origin is type(None):
        return "object"
    if str(origin).startswith("typing.Union"):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_csharp(args[0]) + "?"
        return "object"

    return "object"


def _default_value_to_csharp(value: Any) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _generate_class_source(cls: type[BaseType], namespace: str) -> str:
    lines = ["using MessagePack;", "using System.Collections.Generic;", "", f"namespace {namespace};", ""]
    lines.append("[MessagePackObject]")
    lines.append(f"public class {cls.__name__}")
    lines.append("{")
    index = 0
    for name, field in cls.model_fields.items():
        csharp_type = _python_type_to_csharp(field.annotation)
        default = _default_value_to_csharp(field.default)
        lines.append(f"    [Key({index})]")
        if default is not None:
            lines.append(f"    public {csharp_type} {name} {{ get; set; }} = {default};")
        else:
            lines.append(f"    public {csharp_type} {name} {{ get; set; }}")
        index += 1
    lines.append("}")
    return "\n".join(lines) + "\n"


def _discover_base_types() -> list[type[BaseType]]:
    import nodetool

    classes: list[type[BaseType]] = []
    for mod in pkgutil.walk_packages(nodetool.__path__, nodetool.__name__ + "."):
        try:
            module = importlib.import_module(mod.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseType) and obj is not BaseType:
                classes.append(obj)
    # unique and sort by name
    unique = {c.__name__: c for c in classes}
    return [unique[n] for n in sorted(unique.keys())]


def generate_csharp_types(output_dir: str, namespace: str = "Nodetool.Types") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for cls in _discover_base_types():
        src = _generate_class_source(cls, namespace)
        with open(os.path.join(output_dir, f"{cls.__name__}.cs"), "w", encoding="utf-8") as f:
            f.write(src)

