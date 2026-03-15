"""Tests for the parity snapshot export script."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

SCRIPT = "scripts/export_parity_snapshot.py"


def _run_export(*sections: str) -> dict:
    """Run the export script and return the parsed JSON output."""
    cmd = [sys.executable, SCRIPT, *list(sections)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Export failed:\n{result.stderr}"
    return json.loads(result.stdout)


class TestModelExport:
    def test_exports_all_models(self) -> None:
        data = _run_export("models")
        models = data["models"]
        expected_models = [
            "Asset",
            "Job",
            "Workflow",
            "WorkflowVersion",
            "Message",
            "Thread",
            "Prediction",
            "Secret",
            "OAuthCredential",
            "RunEvent",
            "RunNodeState",
            "RunLease",
            "Workspace",
        ]
        for name in expected_models:
            assert name in models, f"Missing model: {name}"

    def test_model_has_table_name(self) -> None:
        data = _run_export("models")
        for name, schema in data["models"].items():
            assert "table_name" in schema, f"{name} missing table_name"
            assert isinstance(schema["table_name"], str)
            assert len(schema["table_name"]) > 0

    def test_model_has_columns(self) -> None:
        data = _run_export("models")
        for name, schema in data["models"].items():
            assert "columns" in schema, f"{name} missing columns"
            assert len(schema["columns"]) > 0, f"{name} has no columns"

    def test_column_types_are_valid(self) -> None:
        valid_types = {"string", "number", "boolean", "json", "datetime", "none"}
        data = _run_export("models")
        for name, schema in data["models"].items():
            for col_name, col_def in schema["columns"].items():
                assert col_def["type"] in valid_types, (
                    f"{name}.{col_name} has invalid type '{col_def['type']}'"
                )

    def test_asset_model_structure(self) -> None:
        data = _run_export("models")
        asset = data["models"]["Asset"]
        assert asset["table_name"] == "nodetool_assets"
        assert asset["primary_key"] == "id"
        assert "id" in asset["columns"]
        assert "user_id" in asset["columns"]
        assert "name" in asset["columns"]
        assert "created_at" in asset["columns"]
        assert asset["columns"]["id"]["type"] == "string"
        assert asset["columns"]["created_at"]["type"] == "datetime"


class TestCliExport:
    def test_exports_cli_commands(self) -> None:
        data = _run_export("cli")
        commands = data["cli"]
        assert len(commands) > 0, "No CLI commands exported"

    def test_has_expected_top_level_commands(self) -> None:
        data = _run_export("cli")
        names = {cmd["name"] for cmd in data["cli"]}
        for expected in ["serve", "info", "workflows", "assets", "jobs"]:
            assert expected in names, f"Missing CLI command: {expected}"


class TestLibraryExport:
    def test_exports_library_classes(self) -> None:
        data = _run_export("library")
        classes = data["library"]
        assert len(classes) > 0, "No library classes exported"

    def test_dbmodel_methods(self) -> None:
        data = _run_export("library")
        dbmodel = next(
            (c for c in data["library"] if c["class"] == "DBModel"), None
        )
        assert dbmodel is not None, "DBModel not found in library export"
        method_names = {m["name"] for m in dbmodel["methods"]}
        assert "save" in method_names
        assert "delete" in method_names


class TestFullSnapshot:
    def test_all_sections_present(self) -> None:
        data = _run_export()
        assert "models" in data
        assert "api" in data
        assert "cli" in data
        assert "library" in data
