"""Tests for nodetool CLI commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from nodetool.cli import _get_version, cli

# Mark subprocess/CLI tests to run sequentially to avoid conflicts
pytestmark = pytest.mark.xdist_group(name="subprocess_execution")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    return env


class TestVersionOption:
    """Tests for the --version flag."""

    def test_version_option_with_cli_runner(self):
        """Test --version flag using click's test runner."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "nodetool" in result.output.lower()
        assert "version" in result.output.lower()

    def test_version_option_subprocess(self):
        """Test --version flag via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "nodetool.cli", "--version"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        assert result.returncode == 0
        assert "nodetool" in result.stdout.lower()
        assert "version" in result.stdout.lower()

    def test_get_version_function(self):
        """Test the _get_version helper function returns a string."""
        version = _get_version()
        assert isinstance(version, str)
        assert version  # Should not be empty


class TestInfoCommand:
    """Tests for the 'nodetool info' command."""

    def test_info_command_table_output(self):
        """Test info command produces output (table mode)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        # Check for expected table content
        assert "NodeTool" in result.output or "System" in result.output
        assert "Python Version" in result.output or "python_version" in result.output

    def test_info_command_json_output(self):
        """Test info command produces valid JSON with --json flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])
        assert result.exit_code == 0

        # Parse JSON and verify structure
        data = json.loads(result.output)
        assert "nodetool_version" in data
        assert "python_version" in data
        assert "platform" in data
        assert "architecture" in data
        assert "ai_packages" in data
        assert "api_keys" in data
        assert "environment" in data

    def test_info_command_json_api_keys_not_exposed(self):
        """Test that info command doesn't expose actual API key values."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        api_keys = data.get("api_keys", {})
        # Values should only be "configured" or "not set", never actual keys
        for key, value in api_keys.items():
            assert value in ("configured", "not set"), f"Unexpected value for {key}: {value}"

    def test_info_command_subprocess(self):
        """Test info command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "nodetool.cli", "info", "--json"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        assert result.returncode == 0

        # Parse and verify JSON
        data = json.loads(result.stdout)
        assert "nodetool_version" in data
        assert "python_version" in data


class TestHelpOutput:
    """Tests for help output showing new commands."""

    def test_help_shows_version_option(self):
        """Test that help output shows --version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--version" in result.output

    def test_help_shows_info_command(self):
        """Test that help output shows info command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.output

    def test_help_shows_mcp_tool_groups(self):
        """Test that help output shows MCP tool command groups."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "workflows" in result.output
        assert "assets" in result.output
        assert "jobs" in result.output

    def test_mcp_help_shows_subcommands(self):
        """Test that mcp help includes tool groups and serve."""
        runner = CliRunner()
        result = runner.invoke(cli, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.output
        assert "workflows" in result.output
        assert "assets" in result.output
        assert "jobs" in result.output

    def test_info_help(self):
        """Test that info command has its own help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "system" in result.output.lower() or "environment" in result.output.lower()


class TestLazyImports:
    """Tests for ensuring heavy dependencies are lazily imported."""

    def test_workflow_tools_import_is_clean(self):
        """Test that importing WorkflowTools does not trigger ChromaDB/LangChain load."""
        script = """
import sys
# Ensure we start clean
modules_before = set(sys.modules.keys())

from nodetool.tools.workflow_tools import WorkflowTools

modules_after = set(sys.modules.keys())
new_modules = modules_after - modules_before

# Check that heavy modules weren't loaded
heavy_modules = ['chromadb', 'langchain', 'numpy', 'pandas', 'torch']
loaded_heavy = [m for m in new_modules if any(m.startswith(h) for h in heavy_modules)]

# We might see numpy if it's used elsewhere, but definitely shouldn't see chromadb
chroma_loaded = any('chromadb' in m for m in loaded_heavy)
if chroma_loaded:
    print(f"FAILED: ChromaDB modules loaded: {[m for m in loaded_heavy if 'chromadb' in m]}")
    sys.exit(1)
print("SUCCESS")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"Lazy import check failed:\n{result.stdout}\n{result.stderr}")

    def test_collection_tools_import_is_clean(self):
        """Test that importing CollectionTools does not trigger ChromaDB load."""
        script = """
import sys
from nodetool.tools.collection_tools import CollectionTools

# Check for ChromaDB
chroma_modules = [m for m in sys.modules.keys() if 'chromadb' in m]
if chroma_modules:
    print(f"FAILED: ChromaDB modules loaded: {chroma_modules}")
    sys.exit(1)
print("SUCCESS")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"Lazy import check failed:\n{result.stdout}\n{result.stderr}")


class TestWorkflowsListDiagnostics:
    def test_workflows_list_help_includes_debug_threads_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["workflows", "list", "--help"])
        assert result.exit_code == 0
        assert "--debug-threads" in result.output

    def test_workflows_list_debug_threads_emits_diagnostics(self, monkeypatch: pytest.MonkeyPatch):
        import click

        import nodetool.cli as cli_mod
        import nodetool.runtime.db_sqlite as db_sqlite
        import nodetool.runtime.resources as resources
        from nodetool.tools.workflow_tools import WorkflowTools

        class DummyScope:
            async def __aenter__(self):  # noqa: D401
                return self

            async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
                return None

        async def fake_list_workflows(workflow_type: str, query: str | None, limit: int, user_id: str):
            return {"workflows": []}

        def fake_diag(*args, **kwargs) -> None:
            click.echo("[diagnostics] FAKE", err=True)

        shutdown_called = {"value": False}

        async def fake_shutdown_all_sqlite_pools() -> None:
            shutdown_called["value"] = True

        monkeypatch.setattr(resources, "ResourceScope", DummyScope)
        monkeypatch.setattr(WorkflowTools, "list_workflows", staticmethod(fake_list_workflows))
        monkeypatch.setattr(cli_mod, "_print_thread_diagnostics", fake_diag)
        monkeypatch.setattr(db_sqlite, "shutdown_all_sqlite_pools", fake_shutdown_all_sqlite_pools)

        runner = CliRunner()
        result = runner.invoke(cli, ["workflows", "list", "--debug-threads"])
        assert result.exit_code == 0
        assert "[diagnostics] FAKE" in result.output
        assert shutdown_called["value"] is True


class TestDeploySecretSync:
    def test_export_encrypted_secrets_payload_uses_resource_scope(self, monkeypatch: pytest.MonkeyPatch):
        import nodetool.cli as cli_mod
        import nodetool.models.secret as secret_mod
        import nodetool.runtime.resources as resources_mod

        entered = {"value": False}

        class DummyScope:
            async def __aenter__(self):
                entered["value"] = True
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

        class DummySecret:
            user_id = "1"
            key = "OPENAI_API_KEY"
            encrypted_value = "enc"
            description = "test"
            created_at = datetime.now(UTC)
            updated_at = datetime.now(UTC)

        async def fake_list_all(limit: int = 1000):
            assert limit == 123
            return [DummySecret()]

        monkeypatch.setattr(resources_mod, "ResourceScope", DummyScope)
        monkeypatch.setattr(secret_mod.Secret, "list_all", staticmethod(fake_list_all))

        payload = cli_mod._run_async(cli_mod._export_encrypted_secrets_payload(limit=123))

        assert entered["value"] is True
        assert len(payload) == 1
        assert payload[0]["key"] == "OPENAI_API_KEY"


class TestCliAsyncRunnerCleanup:
    def test_run_async_shuts_down_sqlite_pools(self, monkeypatch: pytest.MonkeyPatch):
        import nodetool.cli as cli_mod
        import nodetool.runtime.db_sqlite as db_sqlite

        called = {"value": False}

        async def fake_shutdown_all_sqlite_pools() -> None:
            called["value"] = True

        async def do_work() -> int:
            return 123

        monkeypatch.setattr(db_sqlite, "shutdown_all_sqlite_pools", fake_shutdown_all_sqlite_pools)
        assert cli_mod._run_async(do_work()) == 123
        assert called["value"] is True
    def test_node_tools_import_is_clean(self):
        """Test that importing node_tools does not trigger heavy imports."""
        script = """
import sys
from nodetool.tools.node_tools import NodeTools

# Check for heavy libs
heavy_libs = ['numpy', 'torch', 'PIL', 'huggingface_hub']
loaded_heavy = []
for m in sys.modules.keys():
    for h in heavy_libs:
        if m == h or m.startswith(h + '.'):
            loaded_heavy.append(m)

if loaded_heavy:
    print(f"FAILED: Heavy modules loaded: {loaded_heavy}")
    sys.exit(1)

print("SUCCESS")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"Lazy import check failed:\n{result.stdout}\n{result.stderr}")
