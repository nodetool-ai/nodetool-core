"""Generate C# classes for Nodetool ``BaseType`` subclasses."""

import os
import sys
import shutil
import pkgutil
import importlib
import inspect
from typing import Any, get_origin, get_args

from pydantic import BaseModel

from .metadata.types import BaseType
from .workflows.base_node import BaseNode
from .packages.discovery import walk_source_modules

# Import these at module level to ensure they're available
try:
    import nodetool.nodes
    import nodetool.workflows
except ImportError:
    pass


_PRIMITIVE_TYPE_MAP = {
    str: "string",
    int: "int",
    float: "double",
    bool: "bool",
    bytes: "byte[]",
    # Additional common Python types
    type(None): "object",
    object: "object",
}


def _python_type_to_csharp(tp: Any) -> str:
    """Convert a Python type annotation to a C# type string."""
    origin = get_origin(tp)
    if origin is None:
        if isinstance(tp, type):
            if tp in _PRIMITIVE_TYPE_MAP:
                return _PRIMITIVE_TYPE_MAP[tp]
            # Handle BaseType subclasses - use Nodetool.Types namespace
            try:
                if issubclass(tp, BaseType):
                    return f"Nodetool.Types.{tp.__name__}"
            except TypeError:
                pass
            # Handle other BaseModel subclasses
            try:
                if issubclass(tp, BaseModel):
                    return tp.__name__
            except TypeError:
                pass
        # typing.Any or unknown
        if tp is Any:
            return "object"
        # typing.Literal values
        if str(tp).startswith("typing.Literal"):
            args = get_args(tp)
            if args:
                return _python_type_to_csharp(type(args[0]))
        return "object"

    if origin is list:
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
        return f"HashSet<{inner}>"
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is ...:
            # Variable length tuple like tuple[int, ...]
            inner = _python_type_to_csharp(args[0])
            return f"List<{inner}>"
        else:
            # Fixed length tuple - use List for simplicity
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
    if isinstance(value, list):
        if not value:  # Empty list
            return "new List<object>()"
        # For non-empty lists, we'd need more sophisticated handling
        return "new List<object>()"
    if isinstance(value, dict):
        if not value:  # Empty dict
            return "new Dictionary<string, object>()"
        return "new Dictionary<string, object>()"
    # Handle BaseType instances
    if isinstance(value, BaseType):
        return f"new Nodetool.Types.{type(value).__name__}()"
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
    for mod in pkgutil.walk_packages(nodetool.__path__, nodetool.__name__ + "."): # type: ignore
        try:
            module = importlib.import_module(mod.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseType) and obj is not BaseType and
                    obj.__module__.startswith("nodetool")):
                    classes.append(obj)
            except Exception:
                continue
    # unique and sort by name
    unique = {c.__name__: c for c in classes}
    return [unique[n] for n in sorted(unique.keys())]


def _discover_base_nodes_from_source(source_path: str) -> list[tuple[str, type[BaseNode]]]:
    """Discover all BaseNode subclasses from the specified source path."""
    source_module = source_path.replace("src/", "").replace("/", ".")
    try:
        source_root_module = importlib.import_module(source_module)
    except ImportError as e:
        print(f"Could not import source module {source_module}: {e}")
        return []
    
    # Get the absolute path of the source directory
    source_abs_path = os.path.abspath(source_path)
    
    nodes_with_modules = []
    
    for _, module_name, _ in pkgutil.walk_packages(
        source_root_module.__path__, prefix=source_root_module.__name__ + "."
    ):
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
            
        # Find BaseNode classes in this module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if (inspect.isclass(obj) and
                    issubclass(obj, BaseNode) and 
                    obj is not BaseNode and 
                    hasattr(obj, 'is_visible') and obj.is_visible()):
                    # Get the relative module path for directory structure
                    relative_module = module_name[len(source_module) + 1:]
                    nodes_with_modules.append((relative_module, obj))
            except Exception:
                continue
    
    return nodes_with_modules


def _generate_csharp_for_namespace(source_path: str, output_path: str, namespace: str) -> int:
    """Generate C# classes for a specific namespace using shared discovery logic."""
    classes_generated = 0

    for relative_module, module, relative_path in walk_source_modules(source_path):
        # Create the full target path
        full_target_path = os.path.join(output_path, relative_path)
        print(f"Processing {relative_module} -> {full_target_path}")

        if os.path.exists(full_target_path):
            if os.path.isdir(full_target_path):
                shutil.rmtree(full_target_path)
            else:
                os.remove(full_target_path)

        # Find BaseNode classes in this module (same logic as DSL)
        node_classes = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseNode) and obj is not BaseNode and obj.is_visible():
                node_classes.append(obj)

        if not node_classes:
            continue

        # Generate C# source for all classes in this module
        # Get namespace from first class
        first_metadata = node_classes[0].get_metadata()
        namespace_parts = first_metadata.namespace.split(".")
        csharp_namespace = ".".join(part.capitalize() for part in namespace_parts)
        
        # Common using statements and namespace
        source_code = f"""using MessagePack;
using System.Collections.Generic;
using Nodetool.Types;

namespace {csharp_namespace};

"""
        
        # Generate classes without their individual using/namespace blocks
        for node_cls in node_classes:
            try:
                class_source = _generate_node_class_body(node_cls)
                source_code += class_source + "\n"
                classes_generated += 1
            except Exception as e:
                print(f"Error generating C# class for {node_cls.__name__}: {e}")
                continue

        if source_code.strip() == "":
            continue

        # Check if the original module is an __init__.py file
        if module.__file__.endswith("__init__.py"):  # type: ignore
            target_file_path = os.path.join(full_target_path, "__init__.cs")
        else:
            target_file_path = full_target_path + ".cs"

        # Create directory if needed
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

        print(f"Writing {target_file_path}")
        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write(source_code)

    return classes_generated


def _generate_node_class_source(node_cls: type[BaseNode]) -> str:
    """Generate C# class source for a BaseNode subclass using its metadata."""
    try:
        metadata = node_cls.get_metadata()
        namespace_parts = metadata.namespace.split(".")
        # Convert namespace like "nodetool.nodes.openai" to "Nodetool.Nodes.OpenAI"
        csharp_namespace = ".".join(part.capitalize() for part in namespace_parts)
        
        lines = [
            "using MessagePack;",
            "using System.Collections.Generic;",
            "using Nodetool.Types;",
            "",
            f"namespace {csharp_namespace};",
            "",
            "[MessagePackObject]",
            f"public class {node_cls.__name__}",
            "{"
        ]
        
        # Add properties from node metadata
        index = 0
        
        # Add input properties
        for prop in metadata.properties:
            python_type = prop.type.get_python_type()
            csharp_type = _python_type_to_csharp(python_type)
            default = _default_value_to_csharp(prop.default)
            lines.append(f"    [Key({index})]")
            if default is not None:
                lines.append(f"    public {csharp_type} {prop.name} {{ get; set; }} = {default};")
            else:
                lines.append(f"    public {csharp_type} {prop.name} {{ get; set; }}")
            index += 1
        
        # Check if we need a return type class for multiple outputs
        if len(metadata.outputs) > 1:
            # Generate a return type class
            return_class_name = f"{node_cls.__name__}Output"
            lines.extend([
                "",
                "    [MessagePackObject]",
                f"    public class {return_class_name}",
                "    {"
            ])
            
            output_index = 0
            for output in metadata.outputs:
                python_type = output.type.get_python_type()
                output_type = _python_type_to_csharp(python_type)
                lines.append(f"        [Key({output_index})]")
                lines.append(f"        public {output_type} {output.name} {{ get; set; }}")
                output_index += 1
            
            lines.append("    }")
            
            # Add a method to get the return type
            lines.extend([
                "",
                f"    public {return_class_name} Process()",
                "    {",
                "        // Implementation would be generated based on node logic",
                f"        return new {return_class_name}();",
                "    }"
            ])
        else:
            # Single output or no output
            if metadata.outputs:
                python_type = metadata.outputs[0].type.get_python_type()
                output_type = _python_type_to_csharp(python_type)
                lines.extend([
                    "",
                    f"    public {output_type} Process()",
                    "    {",
                    "        // Implementation would be generated based on node logic",
                    f"        return default({output_type});",
                    "    }"
                ])
            else:
                lines.extend([
                    "",
                    "    public void Process()",
                    "    {",
                    "        // Implementation would be generated based on node logic",
                    "    }"
                ])
        
        lines.append("}")
        return "\n".join(lines) + "\n"
        
    except Exception as e:
        # Fall back to generating from class fields if metadata fails
        print(f"Warning: Could not get metadata for {node_cls.__name__}: {e}")
        return _generate_fallback_node_class(node_cls)


def _generate_node_class_body(node_cls: type[BaseNode]) -> str:
    """Generate C# class body without using statements and namespace."""
    try:
        metadata = node_cls.get_metadata()
        
        lines = [
            "[MessagePackObject]",
            f"public class {node_cls.__name__}",
            "{"
        ]
        
        # Add properties from node metadata
        index = 0
        
        # Add input properties
        for prop in metadata.properties:
            python_type = prop.type.get_python_type()
            csharp_type = _python_type_to_csharp(python_type)
            default = _default_value_to_csharp(prop.default)
            lines.append(f"    [Key({index})]")
            if default is not None:
                lines.append(f"    public {csharp_type} {prop.name} {{ get; set; }} = {default};")
            else:
                lines.append(f"    public {csharp_type} {prop.name} {{ get; set; }}")
            index += 1
        
        # Check if we need a return type class for multiple outputs
        if len(metadata.outputs) > 1:
            # Generate a return type class
            return_class_name = f"{node_cls.__name__}Output"
            lines.extend([
                "",
                "    [MessagePackObject]",
                f"    public class {return_class_name}",
                "    {"
            ])
            
            output_index = 0
            for output in metadata.outputs:
                python_type = output.type.get_python_type()
                output_type = _python_type_to_csharp(python_type)
                lines.append(f"        [Key({output_index})]")
                lines.append(f"        public {output_type} {output.name} {{ get; set; }}")
                output_index += 1
            
            lines.append("    }")
            
            # Add a method to get the return type
            lines.extend([
                "",
                f"    public {return_class_name} Process()",
                "    {",
                "        // Implementation would be generated based on node logic",
                f"        return new {return_class_name}();",
                "    }"
            ])
        else:
            # Single output or no output
            if metadata.outputs:
                python_type = metadata.outputs[0].type.get_python_type()
                output_type = _python_type_to_csharp(python_type)
                lines.extend([
                    "",
                    f"    public {output_type} Process()",
                    "    {",
                    "        // Implementation would be generated based on node logic",
                    f"        return default({output_type});",
                    "    }"
                ])
            else:
                lines.extend([
                    "",
                    "    public void Process()",
                    "    {",
                    "        // Implementation would be generated based on node logic",
                    "    }"
                ])
        
        lines.append("}")
        return "\n".join(lines)
        
    except Exception as e:
        # Fall back to generating from class fields if metadata fails
        print(f"Warning: Could not get metadata for {node_cls.__name__}: {e}")
        return _generate_fallback_node_class_body(node_cls)


def _generate_fallback_node_class(node_cls: type[BaseNode]) -> str:
    """Generate C# class from BaseNode fields as fallback when metadata fails."""
    namespace = "Nodetool.Nodes"
    lines = [
        "using MessagePack;",
        "using System.Collections.Generic;", 
        "using Nodetool.Types;",
        "",
        f"namespace {namespace};",
        "",
        "[MessagePackObject]",
        f"public class {node_cls.__name__}",
        "{"
    ]
    
    index = 0
    for name, field in node_cls.model_fields.items():
        if name.startswith("_"):  # Skip private fields
            continue
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


def _generate_fallback_node_class_body(node_cls: type[BaseNode]) -> str:
    """Generate C# class body from BaseNode fields as fallback when metadata fails."""
    lines = [
        "[MessagePackObject]",
        f"public class {node_cls.__name__}",
        "{"
    ]
    
    index = 0
    for name, field in node_cls.model_fields.items():
        if name.startswith("_"):  # Skip private fields
            continue
        csharp_type = _python_type_to_csharp(field.annotation)
        default = _default_value_to_csharp(field.default)
        lines.append(f"    [Key({index})]")
        if default is not None:
            lines.append(f"    public {csharp_type} {name} {{ get; set; }} = {default};")
        else:
            lines.append(f"    public {csharp_type} {name} {{ get; set; }}")
        index += 1
    
    lines.append("}")
    return "\n".join(lines)


def generate_csharp_types(output_dir: str, namespace: str = "Nodetool.Types") -> None:
    """Generate C# classes for BaseType subclasses."""
    os.makedirs(output_dir, exist_ok=True)
    for cls in _discover_base_types():
        src = _generate_class_source(cls, namespace)
        with open(os.path.join(output_dir, f"{cls.__name__}.cs"), "w", encoding="utf-8") as f:
            f.write(src)


def generate_csharp_nodes(output_dir: str) -> None:
    """Generate C# classes for BaseNode subclasses using DSL discovery pattern."""
    # Add the src directory to the Python path to allow relative imports
    src_dir = os.path.abspath("src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    base_nodes_path = os.path.join("src", "nodetool", "nodes")
    
    if not os.path.isdir(base_nodes_path):
        print(f"Error: Nodes directory not found at {base_nodes_path}")
        return

    namespaces = [
        d
        for d in os.listdir(base_nodes_path)
        if os.path.isdir(os.path.join(base_nodes_path, d))
    ]

    if not namespaces:
        print(f"No subdirectories found in {base_nodes_path} to treat as namespaces.")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_classes = 0

    for namespace in namespaces:
        source_path = os.path.join(base_nodes_path, namespace)
        output_path = os.path.join(output_dir, namespace)

        print(f"Generating C# classes from {source_path} to {output_path} for namespace '{namespace}'...")
        
        # Use the exact same discovery logic as DSL codegen
        nodes_found = _generate_csharp_for_namespace(source_path, output_path, namespace)
        total_classes += nodes_found
        
        if nodes_found > 0:
            print(f"✅ Generated {nodes_found} C# classes for namespace '{namespace}'!")

    print(f"✅ Generated {total_classes} total C# classes!")

