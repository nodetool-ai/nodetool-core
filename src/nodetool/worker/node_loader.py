import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic_core import PydanticUndefined

from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode


def _discover_namespaces() -> list[str]:
    """Auto-discover namespaces from installed nodetool node packages.

    Uses Python entry_points (group='nodetool.namespaces') for discovery.
    Each entry point value is a comma-separated list of namespaces.
    Falls back to scanning for nodetool.nodes.* packages.
    """
    namespaces: set[str] = set()

    # Method 1: entry_points (group kwarg supported since Python 3.9+)
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="nodetool.namespaces")

        for ep in eps:
            for ns in str(ep.value).split(","):
                ns = ns.strip()
                if ns:
                    namespaces.add(ns)
    except Exception as e:
        print(f"Warning: entry_points discovery failed: {e}", file=sys.stderr)

    # Method 2: scan for nodetool.nodes.* subpackages
    if not namespaces:
        try:
            import importlib
            nodes_pkg = importlib.import_module("nodetool.nodes")
            if hasattr(nodes_pkg, "__path__"):
                for package_path in nodes_pkg.__path__:
                    path = Path(package_path)
                    if not path.exists():
                        continue
                    for child in path.iterdir():
                        if child.is_dir() and not child.name.startswith("_"):
                            namespaces.add(child.name)
        except ImportError:
            pass

    return sorted(namespaces)


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
            prop["default"] = _normalize_default(field_info.default)
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
        # Realtime capability flags — keys match the TS-side NodeDescriptor
        # serialization in packages/node-sdk/src/node-metadata.ts so the
        # NodeRegistry sees Python and TS realtime nodes uniformly.
        "is_realtime_capable": _call_or_get(node_class, "is_realtime_capable"),
        "owns_warm_state": _call_or_get(node_class, "owns_warm_state"),
        "is_media_adapter": _call_or_get(node_class, "is_media_adapter"),
        "realtime_profile": _dict_call_or_get(node_class, "realtime_profile"),
    }


def _call_or_get(cls: type, name: str) -> bool:
    """Get a class attribute, calling it if it's a method."""
    attr = getattr(cls, name, False)
    if callable(attr):
        try:
            return bool(attr())
        except TypeError:
            return False
    return bool(attr)


def _dict_call_or_get(cls: type, name: str) -> dict[str, Any]:
    """Get a dict-valued class attribute/method and return a shallow copy."""
    attr = getattr(cls, name, {})
    if callable(attr):
        try:
            value = attr()
        except TypeError:
            return {}
    else:
        value = attr
    if isinstance(value, dict):
        return dict(value)
    return {}


def _normalize_default(value: Any) -> Any:
    """Convert field defaults into msgpack-safe primitives."""
    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return {
            key: _normalize_default(item)
            for key, item in value.model_dump().items()
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_normalize_default(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_default(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_default(item)
            for key, item in value.items()
        }
    return value


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
    """Extract output slot metadata using BaseNode's canonical output inference."""
    try:
        return [
            {
                "name": slot.name,
                "type": slot.type.model_dump(exclude_none=True),
            }
            for slot in node_class.outputs()
        ]
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
        namespaces = ns_env.split(",") if ns_env else _discover_namespaces()

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
