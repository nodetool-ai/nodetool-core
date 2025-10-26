from enum import Enum
import os
import sys
import inspect
import pkgutil
from types import GenericAlias, UnionType
from typing import Any, Union
import importlib
import shutil
import typing
from pydantic import BaseModel
import subprocess

from nodetool.metadata.types import BaseType
from nodetool.metadata.utils import is_enum_type
from nodetool.workflows.base_node import BaseNode


"""
Code Generation Module for NodeTool DSL

This module provides functionality to automatically generate Domain Specific Language (DSL)
modules based on node definitions in the source module. It creates Python classes that act as
wrappers around the base node classes, providing a more intuitive and user-friendly interface
for connecting nodes in a graph.

The generated DSL classes maintain the same structure and field definitions as the original
node classes but add additional type hints and functionality to make them work seamlessly
in a graph-based workflow environment.

Typical usage:
    create_dsl_modules("nodetool.nodes", "nodetool.dsl")
"""


def create_python_module_file(filename: str, content: str) -> None:
    """
    Create a Python module file with the given filename and content.

    This function creates the necessary directory structure if it doesn't exist
    and writes the provided content to the specified file.

    Args:
        filename (str): The path and name of the file to be created.
        content (str): The content to be written to the file.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        file.write(content)
    try:
        subprocess.run(["black", filename], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(
            f"Error running black on {filename}: {e.stderr.decode()}", file=sys.stderr
        )
    except FileNotFoundError:
        print(
            "black formatter not found. Please ensure it is installed and in your PATH.",
            file=sys.stderr,
        )


def type_to_string(field_type: type | GenericAlias | UnionType) -> str:
    """
    Converts a Python type to its string representation.

    This function handles various Python types including Union types, generic types
    (like List, Dict, etc.), and regular types, converting them to their string
    representation for use in generated code.

    Args:
        field_type (type): The Python type to convert.

    Returns:
        str: The string representation of the type that can be used in generated code.

    Examples:
        >>> type_to_string(list[str])
        'list[str]'
        >>> type_to_string(Union[int, str])
        'int | str'
    """
    if isinstance(field_type, UnionType):
        return str(field_type)
    if isinstance(field_type, GenericAlias):
        args = [type_to_string(arg) for arg in field_type.__args__]
        if field_type.__origin__ is list:
            return f"list[{args[0]}]"
        elif field_type.__origin__ is dict:
            return f"dict[{args[0]}, {args[1]}]"
        elif field_type.__origin__ is set:
            return f"set[{args[0]}]"
        elif field_type.__origin__ is tuple:
            return f"tuple[{', '.join(args)}]"

    if isinstance(field_type, typing._UnionGenericAlias):  # type: ignore
        return f"{type_to_string(field_type.__args__[0])} | None"

    assert isinstance(
        field_type, type
    ), f"Field type is not a type: {field_type}, generic aliases like List[str] are not supported, use list[str] instead"

    if issubclass(field_type, BaseType):
        return f"types.{field_type.__name__}"
    if isinstance(field_type, str):
        return field_type
    if isinstance(field_type, dict):
        return "dict"
    if field_type.__name__ == "Optional":
        return f"{type_to_string(field_type.__args__[0])} | None"
    if field_type == Union:
        return " | ".join(field_type.__args__)
    return field_type.__name__


def field_default(default_value: Any) -> str:
    """
    Returns a string representation of the default value for a field.

    This function handles different types of default values, including None, Enums,
    Pydantic BaseModels (and nodetool's BaseType), lists, dicts, and other
    primitive types, converting them to their string representation for use in
    generated code.

    Args:
        default_value (Any): The default value of the field.

    Returns:
        str: The string representation of the default value that can be used in
             generated code.
    """
    if default_value is None:
        return "None"

    # 1. Handle Enum members
    if isinstance(default_value, Enum):
        enum_cls = type(default_value)
        # Enums should be rendered as module.qualname.membername
        # This relies on the necessary modules being imported in the generated file.
        if enum_cls.__module__ == "__main__":
            # Fallback for enums defined in __main__, may need specific handling
            # or ensuring they are aliased/imported directly in generated code.
            # For now, using qualname assuming it's available in the scope.
            return f"{enum_cls.__qualname__}.{default_value.name}"
        return f"{enum_cls.__module__}.{enum_cls.__qualname__}.{default_value.name}"

    # 2. Handle Pydantic BaseModel / nodetool's BaseType
    if isinstance(default_value, BaseType):
        model_cls = type(default_value)
        args_strs = []

        # Introspect fields if it's a Pydantic model
        if isinstance(default_value, BaseModel):  # Check if it's a Pydantic BaseModel
            for field_name in default_value.model_fields.keys():
                value = getattr(default_value, field_name)
                value_str = field_default(value)  # Recursive call
                args_strs.append(f"{field_name}={value_str}")
            # BaseType instances are prefixed with "types." in generated code
            return f"types.{model_cls.__name__}({', '.join(args_strs)})"

    # 3. Handle basic Python types
    if isinstance(default_value, str):
        return repr(default_value)  # Handles quotes and escapes

    if isinstance(default_value, (int, float, bool)):
        return str(default_value)

    # 4. Handle collections
    if isinstance(default_value, list):
        return f"[{', '.join(field_default(item) for item in default_value)}]"

    if isinstance(default_value, tuple):
        if not default_value:  # Empty tuple
            return "()"
        # Tuples with one item need a trailing comma
        elements_str = ", ".join(field_default(item) for item in default_value)
        return f"({elements_str}{',' if len(default_value) == 1 else ''})"

    if isinstance(default_value, dict):
        return f"{{{', '.join(f'{field_default(k)}: {field_default(v)}' for k, v in default_value.items())}}}"

    if isinstance(default_value, set):
        if not default_value:  # Empty set
            return "set()"
        return f"{{{', '.join(field_default(item) for item in default_value)}}}"

    # 5. Fallback for any other types
    return repr(default_value)


def _is_optional_annotation(field_type: type | GenericAlias | UnionType) -> bool:
    """
    Detect whether an annotation includes ``None`` (Optional/Union with None).
    """
    if isinstance(field_type, UnionType):
        return any(arg is type(None) for arg in field_type.__args__)

    origin = typing.get_origin(field_type)
    if origin is Union:
        return any(arg is type(None) for arg in typing.get_args(field_type))

    return False


def _strip_optional_from_str(type_str: str) -> str:
    """
    Remove ``None`` members from a pipe-separated union string representation.
    """
    parts = [part.strip() for part in type_str.split("|")]
    without_none = [part for part in parts if part != "None"]
    return " | ".join(without_none) if without_none else "None"


def _connect_annotation(field_type: type | GenericAlias | UnionType, field) -> str:
    """
    Render the Connect[...] annotation for a field, preserving optionality.
    """
    base_str = type_to_string(field_type)
    optional = _is_optional_annotation(field_type)

    if not optional and getattr(field, "is_required", True) is False and field.default is None:
        optional = True

    if optional:
        base_str = _strip_optional_from_str(base_str)

    connect_str = f"Connect[{base_str}]"
    if optional:
        connect_str += " | None"
    return connect_str


def generate_class_source(node_cls: type[BaseNode]) -> str:
    """
    Generate the source code for a graph node class based on the provided node class.

    This function creates a DSL wrapper class that inherits from GraphNode and maintains
    the same structure and field definitions as the original node class. It handles
    special cases like enum types and adds appropriate type hints and field definitions.

    The generated class acts as a DSL wrapper for easier connection between nodes in a
    graph-based workflow.

    Args:
        node_cls (type[BaseNode]): The node class to generate the source code for.

    Returns:
        str: The generated source code for the class as a string.

    Note:
        The generated class will have the same name as the original class but will
        inherit from GraphNode instead of BaseNode.
    """
    return_type = None
    typed_output_annotations: dict[str, str] = {}

    try:
        return_type = node_cls.return_type()
        if return_type is not None:
            annotations = getattr(return_type, "__annotations__", None)
            if annotations and not (
                isinstance(return_type, type) and issubclass(return_type, BaseModel)
            ):
                for field_name, field_type in annotations.items():
                    try:
                        typed_output_annotations[field_name] = type_to_string(field_type)
                    except Exception:
                        typed_output_annotations[field_name] = "typing.Any"
    except Exception:
        return_type = None

    imports = (
        "import typing\n"
        "from pydantic import Field\n"
        "from nodetool.dsl.handles import Connect, OutputHandle, OutputsProxy, connect_field\n"
    )

    output_annotation = "typing.Any"
    if return_type is not None:
        try:
            output_annotation = type_to_string(return_type)
        except Exception:
            output_annotation = "typing.Any"

    class_body = f"class {node_cls.__name__}(GraphNode[{output_annotation}]):\n"

    # Add class docstring if it exists
    if node_cls.__doc__:
        class_body += f'    """{node_cls.__doc__}"""\n\n'

    fields = node_cls.inherited_fields()
    node_type = node_cls.get_node_type()

    # First, add enum types as class attributes
    for field_name, field_type in node_cls.field_types().items():
        if field_name not in fields:
            continue
        if is_enum_type(field_type):
            imports += f"import {field_type.__module__}\n"
            class_body += f"    {field_type.__name__}: typing.ClassVar[type] = {field_type.__module__}.{field_type.__qualname__}\n"

    # Then add the fields
    for field_name, field_type in node_cls.field_types().items():
        try:
            if field_name not in fields:
                continue
            field = fields[field_name]
            if is_enum_type(field_type):
                enum_full_path = f"{field_type.__module__}.{field_type.__qualname__}"
                if isinstance(field.default, Enum):
                    field_def = f"Field(default={enum_full_path}.{field.default.name}, description={repr(field.description)})"
                else:
                    field_def = f"Field(default={enum_full_path}({repr(field.default)}), description={repr(field.description)})"
                class_body += f"    {field_name}: {enum_full_path} = {field_def}\n"
            else:
                new_field_type = _connect_annotation(field_type, field)
                field_def = f"connect_field(default={field_default(field.default)}, description={repr(field.description)})"
                class_body += f"    {field_name}: {new_field_type} = {field_def}\n"
        except Exception as e:
            print(f"Error generating field {field_name} for {node_cls.__name__}: {e}")
            raise e

    if typed_output_annotations:
        class_body += "\n    @property\n"
        class_body += (
            f'    def out(self) -> "{node_cls.__name__}Outputs":\n'
            f"        return {node_cls.__name__}Outputs(self)\n"
        )

    class_body += "\n    @classmethod\n"
    class_body += f'    def get_node_type(cls): return "{node_type}"\n'

    outputs_class = ""
    if typed_output_annotations:
        outputs_class = (
            f"\n\nclass {node_cls.__name__}Outputs(OutputsProxy[{output_annotation}]):\n"
        )
        for slot_name, slot_type in typed_output_annotations.items():
            outputs_class += "    @property\n"
            outputs_class += (
                f"    def {slot_name}(self) -> OutputHandle[{slot_type}]:\n"
                f"        return typing.cast(OutputHandle[{slot_type}], self['{slot_name}'])\n\n"
            )
        if outputs_class.endswith("\n\n"):
            outputs_class = outputs_class[:-1]

    rebuild = f"\n{node_cls.__name__}.model_rebuild(force=True)\n"
    return imports + "\n" + class_body + outputs_class + rebuild


def create_dsl_modules(source_path: str, target_path: str):
    """
    Generate DSL modules based on node classes found in the source module.

    This function walks through all packages and modules in the source root,
    finds all node classes (subclasses of BaseNode), and generates corresponding
    DSL wrapper classes in the target directory. It maintains the same module
    hierarchy as the source.

    Args:
        source_path (str): The source path to the node classes (e.g., "nodetool/nodes/package").
        target_path (str): The filesystem path where the target modules should be generated.

    Example:
        >>> create_dsl_modules("nodetool/nodes/package", "/path/to/output")
        # This will generate DSL modules in the specified output directory
    """
    source_module = source_path.replace("src/", "").replace("/", ".")
    source_root_module = importlib.import_module(source_module)

    # Get the absolute path of the source directory
    source_abs_path = os.path.abspath(source_path)

    for _, module_name, _ in pkgutil.walk_packages(
        source_root_module.__path__, prefix=source_root_module.__name__ + "."
    ):
        # Import the module to get its file path
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "__file__") or module.__file__ is None:
                continue

            # Check if the module file is actually within the source_path
            module_abs_path = os.path.abspath(module.__file__)
            if not module_abs_path.startswith(source_abs_path):
                continue

        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            continue

        # Get the relative part of the module path
        relative_module = module_name[len(source_module) + 1 :]
        # Convert to filesystem path
        relative_path = relative_module.replace(".", "/")

        # Create the full target path
        full_target_path = os.path.join(target_path, relative_path)
        print(f"Processing {relative_module} -> {full_target_path}")

        if os.path.exists(full_target_path):
            if os.path.isdir(full_target_path):
                shutil.rmtree(full_target_path)
            else:
                os.remove(full_target_path)

        source_code = ""
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseNode) and obj is not BaseNode and obj.is_visible():
                source_code += generate_class_source(obj)
                source_code += "\n\n"

        if source_code == "":
            continue

        # Check if the original module is an __init__.py file
        if module.__file__.endswith("__init__.py"):  # type: ignore
            target_file_path = os.path.join(full_target_path, "__init__.py")
        else:
            target_file_path = full_target_path + ".py"

        disclaimer = (
            "# This file is auto-generated by nodetool.dsl.codegen.\n"
            "# Please do not edit this file manually.\n\n"
            "# Instead, edit the node class in the source module and run the following commands to regenerate the DSL:\n"
            "# nodetool package scan\n"
            "# nodetool codegen\n\n"
        )

        source_code = (
            disclaimer
            + "from pydantic import BaseModel, Field\n"
            + "import typing\n"
            + "from typing import Any\n"
            + "import nodetool.metadata.types\n"
            + "import nodetool.metadata.types as types\n"
            + "from nodetool.dsl.graph import GraphNode\n\n"
            + source_code
        )

        print(f"Writing {target_file_path}")
        create_python_module_file(target_file_path, source_code)
