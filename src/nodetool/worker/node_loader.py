import os
import sys
from typing import Any

from enum import Enum

from pydantic_core import PydanticUndefined

from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode


def _discover_namespaces() -> list[str]:
    """Auto-discover namespaces from installed nodetool-* packages."""
    try:
        from nodetool.packages.registry import discover_node_packages
        namespaces = set()
        for pkg in discover_node_packages():
            for ns in pkg.namespaces:
                namespaces.add(ns)
        return sorted(namespaces)
    except Exception as e:
        print(f"Warning: auto-discovery failed: {e}", file=sys.stderr)
        return []


def node_to_metadata(node_class: type[BaseNode]) -> dict[str, Any]:
    """Extract metadata dict from a BaseNode subclass.

    Returns a dict matching the format expected by the TS NodeRegistry's
    loadPythonMetadata(), with node_type, title, description, properties,
    outputs, and required_settings.
    """
    node_type = node_class.get_node_type()
    title = getattr(node_class, "title", "") or node_type.split(".")[-1]
    description = (node_class.__doc__ or "").strip()

    # Extract properties from Pydantic fields
    properties = []
    for name, field_info in node_class.model_fields.items():
        if name.startswith("_"):
            continue
        prop = {
            "name": name,
            "type": {"type": _field_type_name(field_info.annotation)},
        }
        if field_info.default is not PydanticUndefined and field_info.default is not None:
            from pydantic import BaseModel
            default = field_info.default
            if isinstance(default, BaseModel):
                default = default.model_dump()
            elif isinstance(default, Enum):
                default = default.value
            prop["default"] = default
        if field_info.description:
            prop["description"] = field_info.description
        properties.append(prop)

    # Extract outputs from return type annotation of process()
    outputs = _extract_outputs(node_class)

    # Required settings (API keys etc.) — call as classmethod
    try:
        required_settings = node_class.required_settings() or []
    except (TypeError, AttributeError):
        required_settings = []

    return {
        "node_type": node_type,
        "title": title,
        "description": description,
        "properties": properties,
        "outputs": outputs,
        "required_settings": list(required_settings),
        "is_streaming_output": _call_or_get(node_class, "is_streaming_output"),
        "is_streaming_input": _call_or_get(node_class, "is_streaming_input"),
        "is_dynamic": _call_or_get(node_class, "is_dynamic"),
    }


def _call_or_get(cls: type, name: str) -> bool:
    """Get a class attribute, calling it if it's a method."""
    attr = getattr(cls, name, False)
    if callable(attr):
        try:
            return attr()
        except TypeError:
            return False
    return bool(attr)


def _field_type_name(annotation: Any) -> str:
    """Convert a Python type annotation to a type name string."""
    if annotation is None:
        return "any"
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        args = getattr(annotation, "__args__", ())
        arg_names = ", ".join(_field_type_name(a) for a in args)
        return f"{_simple_name(origin)}[{arg_names}]" if args else _simple_name(origin)
    return _simple_name(annotation)


def _simple_name(t: Any) -> str:
    name = getattr(t, "__name__", None) or str(t)
    type_map = {
        "ImageRef": "ImageRef",
        "AudioRef": "AudioRef",
        "VideoRef": "VideoRef",
        "TextRef": "TextRef",
    }
    return type_map.get(name, name)


def _extract_outputs(node_class: type[BaseNode]) -> list[dict]:
    """Extract output slot metadata from node's return type."""
    from typing import get_type_hints

    try:
        hints = get_type_hints(node_class.process)
        ret = hints.get("return")
        if ret is None:
            return [{"name": "output", "type": {"type": "any"}}]
        return [{"name": "output", "type": {"type": _field_type_name(ret)}}]
    except Exception:
        return [{"name": "output", "type": {"type": "any"}}]


def _import_node_packages(namespaces: list[str]) -> None:
    """Import node packages so their classes register in NODE_BY_TYPE."""
    import importlib
    import pkgutil

    for ns in namespaces:
        module_name = f"nodetool.nodes.{ns}"
        try:
            pkg = importlib.import_module(module_name)
        except ImportError:
            print(f"Warning: could not import {module_name}", file=sys.stderr)
            continue
        # Walk submodules to trigger registration
        for _importer, modname, _ispkg in pkgutil.walk_packages(
            path=pkg.__path__, prefix=pkg.__name__ + "."
        ):
            try:
                importlib.import_module(modname)
            except Exception as e:
                print(f"Warning: failed to import {modname}: {e}", file=sys.stderr)


def load_nodes(
    namespaces: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load all registered nodes filtered by namespace allowlist.

    Fallback chain:
    1. Explicit `namespaces` argument (from --namespaces CLI)
    2. NODETOOL_WORKER_NAMESPACES env var
    3. _discover_namespaces() auto-discovery
    4. Empty list (graceful — no Python nodes)
    """
    if namespaces is None:
        ns_env = os.environ.get("NODETOOL_WORKER_NAMESPACES")
        if ns_env:
            namespaces = ns_env.split(",")
        else:
            namespaces = _discover_namespaces()

    _import_node_packages(namespaces)

    result = []
    for node_type, node_class in NODE_BY_TYPE.items():
        top_ns = node_type.split(".")[0]
        if top_ns in namespaces:
            try:
                result.append(node_to_metadata(node_class))
            except Exception as e:
                print(f"Warning: failed to extract metadata for {node_type}: {e}", file=sys.stderr)
    return result
