#!/usr/bin/env python3
"""
Export Python-side metadata for systematic parity checking against the
TypeScript port.

Outputs a JSON document on stdout with sections:
  - models   : DB model schemas (table, columns, indexes)
  - api      : FastAPI route definitions (method, path, name)
  - cli      : Click / Typer CLI command tree
  - library  : Selected library function signatures

Usage:
    python scripts/export_parity_snapshot.py              # full snapshot
    python scripts/export_parity_snapshot.py models        # models only
    python scripts/export_parity_snapshot.py api           # api only
    python scripts/export_parity_snapshot.py cli           # cli only
    python scripts/export_parity_snapshot.py library       # library only
"""
from __future__ import annotations

import datetime
import inspect
import json
import sys
import types
import typing


# ── Type Mapping ──────────────────────────────────────────────────────


def _python_type_to_parity(annotation: type) -> str:
    """Map a Python type annotation to one of the canonical parity types
    used by the TS ``FieldDef.type`` union:
    ``"string" | "number" | "boolean" | "json" | "datetime"``
    """
    if annotation is type(None):
        return "none"
    if annotation is str:
        return "string"
    if annotation in (int, float):
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is datetime.datetime:
        return "datetime"
    if annotation is dict or annotation is list:
        return "json"

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    # Handle Optional[X]  /  X | None  (Union with None)
    if isinstance(annotation, types.UnionType) or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_parity(non_none[0])
        return "json"

    # Literal[…] → string (all our Literals are string enums)
    if origin is typing.Literal:
        return "string"

    # Generic dict / list
    if origin is dict:
        return "json"
    if origin is list:
        return "json"

    return "json"


def _is_optional(annotation: type) -> bool:
    """Return True when the annotation admits None."""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())
    if isinstance(annotation, types.UnionType) or origin is typing.Union:
        return type(None) in args
    return False


# ── Model Export ──────────────────────────────────────────────────────


def export_models() -> dict:
    from nodetool.models.asset import Asset
    from nodetool.models.job import Job
    from nodetool.models.message import Message
    from nodetool.models.oauth_credential import OAuthCredential
    from nodetool.models.prediction import Prediction
    from nodetool.models.run_event import RunEvent
    from nodetool.models.run_lease import RunLease
    from nodetool.models.run_node_state import RunNodeState
    from nodetool.models.secret import Secret
    from nodetool.models.thread import Thread
    from nodetool.models.workflow import Workflow
    from nodetool.models.workflow_version import WorkflowVersion
    from nodetool.models.workspace import Workspace

    model_classes = [
        Asset,
        Job,
        Workflow,
        WorkflowVersion,
        Message,
        Thread,
        Prediction,
        Secret,
        OAuthCredential,
        RunEvent,
        RunNodeState,
        RunLease,
        Workspace,
    ]

    result: dict[str, dict] = {}
    for model_cls in model_classes:
        schema = model_cls.get_table_schema()
        db_fields = model_cls.db_fields()

        columns: dict[str, dict] = {}
        for field_name, field_info in db_fields.items():
            columns[field_name] = {
                "type": _python_type_to_parity(field_info.annotation),
                "optional": _is_optional(field_info.annotation),
            }

        indexes: list[dict] = []
        if hasattr(model_cls, "_indexes"):
            for idx in model_cls._indexes:
                indexes.append(
                    {
                        "name": idx.get("name", ""),
                        "columns": idx.get("columns", []),
                        "unique": idx.get("unique", False),
                    }
                )

        result[model_cls.__name__] = {
            "table_name": schema.get("table_name"),
            "primary_key": schema.get("primary_key", "id"),
            "columns": columns,
            "indexes": indexes,
        }

    return result


# ── API Export ────────────────────────────────────────────────────────


def export_api() -> list[dict]:
    from nodetool.api.app import create_app

    app = create_app()
    routes: list[dict] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        name = getattr(route, "name", None)
        if path and methods:
            for method in sorted(methods):
                routes.append({"method": method, "path": path, "name": name or ""})
    routes.sort(key=lambda r: (r["path"], r["method"]))
    return routes


# ── CLI Export ────────────────────────────────────────────────────────


def _walk_click_group(group, prefix: str = "") -> list[dict]:
    """Recursively walk a Click/Typer group and collect commands."""
    import click

    entries: list[dict] = []
    if isinstance(group, click.Group):
        for cmd_name, cmd in sorted(group.commands.items()):
            full_name = f"{prefix} {cmd_name}".strip()
            params = []
            for p in cmd.params:
                params.append(
                    {
                        "name": p.name,
                        "type": p.type.name if hasattr(p.type, "name") else str(p.type),
                        "required": p.required,
                    }
                )
            entries.append(
                {
                    "name": full_name,
                    "is_group": isinstance(cmd, click.Group),
                    "params": params,
                }
            )
            if isinstance(cmd, click.Group):
                entries.extend(_walk_click_group(cmd, full_name))
    return entries


def export_cli() -> list[dict]:
    try:
        from nodetool.cli import cli

        return _walk_click_group(cli)
    except Exception as exc:
        return [{"error": str(exc)}]


# ── Library Function Export ───────────────────────────────────────────


def export_library() -> list[dict]:
    """Export signatures of key library functions that TS should mirror."""
    targets: list[tuple[str, str]] = [
        ("nodetool.models.base_model", "DBModel"),
        ("nodetool.workflows.graph", "Graph"),
    ]

    entries: list[dict] = []
    for module_path, class_name in targets:
        try:
            mod = __import__(module_path, fromlist=[class_name])
            cls = getattr(mod, class_name)
            methods: list[dict] = []
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith("_"):
                    continue
                sig = inspect.signature(method)
                params = []
                for pname, param in sig.parameters.items():
                    if pname == "self" or pname == "cls":
                        continue
                    params.append(
                        {
                            "name": pname,
                            "kind": str(param.kind.name),
                            "has_default": param.default is not inspect.Parameter.empty,
                        }
                    )
                methods.append({"name": name, "params": params})
            entries.append(
                {
                    "module": module_path,
                    "class": class_name,
                    "methods": methods,
                }
            )
        except Exception as exc:
            entries.append(
                {
                    "module": module_path,
                    "class": class_name,
                    "error": str(exc),
                }
            )
    return entries


# ── Main ──────────────────────────────────────────────────────────────

EXPORTERS = {
    "models": export_models,
    "api": export_api,
    "cli": export_cli,
    "library": export_library,
}


def main() -> None:
    sections = sys.argv[1:] if len(sys.argv) > 1 else list(EXPORTERS.keys())
    snapshot: dict = {}
    for section in sections:
        fn = EXPORTERS.get(section)
        if fn is None:
            print(f"Unknown section: {section}", file=sys.stderr)
            sys.exit(1)
        snapshot[section] = fn()
    json.dump(snapshot, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
