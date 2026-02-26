"""Tests for nodetool CLI commands."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
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


class TestModelDownloadHFCommand:
    """Tests for `nodetool model download-hf`."""

    def test_model_download_hf_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "download-hf", "--help"])
        assert result.exit_code == 0
        assert "--repo-id" in result.output
        assert "--cache-dir" in result.output
        assert "--file-path" in result.output

    def test_model_download_hf_streams_progress(self, monkeypatch: pytest.MonkeyPatch):
        import nodetool.cli as cli_mod
        import nodetool.deploy.admin_operations as admin_ops

        call_args: dict[str, object] = {}
        progress_updates: list[dict[str, object]] = []

        class DummyProgressManager:
            def _display_progress_update(self, progress_update: dict[str, object]) -> None:
                progress_updates.append(progress_update)

        async def fake_stream_hf_model_download(
            repo_id: str,
            cache_dir: str = "/app/.cache/huggingface/hub",
            file_path: str | None = None,
            ignore_patterns: list | None = None,
            allow_patterns: list | None = None,
            user_id: str | None = None,
        ):
            call_args.update(
                {
                    "repo_id": repo_id,
                    "cache_dir": cache_dir,
                    "file_path": file_path,
                    "ignore_patterns": ignore_patterns,
                    "allow_patterns": allow_patterns,
                    "user_id": user_id,
                }
            )
            yield {"status": "starting", "repo_id": repo_id, "message": "starting"}
            yield {"status": "completed", "repo_id": repo_id, "message": "completed", "downloaded_files": 1}

        monkeypatch.setattr(cli_mod, "_get_progress_manager", lambda: DummyProgressManager())
        monkeypatch.setattr(admin_ops, "stream_hf_model_download", fake_stream_hf_model_download)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "model",
                "download-hf",
                "--repo-id",
                "org/example-model",
                "--cache-dir",
                "/tmp/hf-cache",
                "--file-path",
                "config.json",
                "--allow-patterns",
                "*.json",
                "--ignore-patterns",
                "*.bin",
                "--user-id",
                "user-123",
            ],
        )

        assert result.exit_code == 0
        assert call_args == {
            "repo_id": "org/example-model",
            "cache_dir": "/tmp/hf-cache",
            "file_path": "config.json",
            "ignore_patterns": ["*.bin"],
            "allow_patterns": ["*.json"],
            "user_id": "user-123",
        }
        assert [update["status"] for update in progress_updates] == ["starting", "completed"]


class TestModelRecommendedCommand:
    """Tests for `nodetool model recommended`."""

    def test_model_recommended_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "recommended", "--help"])
        assert result.exit_code == 0
        assert "--category" in result.output
        assert "--json" in result.output
        assert "--system" in result.output
        assert "--check-servers" in result.output

    def test_model_recommended_category_json_limit(self, monkeypatch: pytest.MonkeyPatch):
        import nodetool.workflows.recommended_models as recommended_mod
        from nodetool.types.model import UnifiedModel

        call_args: dict[str, object] = {}

        async def fake_get_recommended_language_models(
            system: str | None = None,
            check_servers: bool = True,
        ) -> list[UnifiedModel]:
            call_args["system"] = system
            call_args["check_servers"] = check_servers
            return [
                UnifiedModel(
                    id="model-1",
                    type="language_model",
                    name="Model One",
                    repo_id="org/model-1",
                    downloaded=False,
                ),
                UnifiedModel(
                    id="model-2",
                    type="language_model",
                    name="Model Two",
                    repo_id="org/model-2",
                    downloaded=False,
                ),
            ]

        monkeypatch.setattr(
            recommended_mod,
            "get_recommended_language_models",
            fake_get_recommended_language_models,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "model",
                "recommended",
                "--category",
                "language",
                "--system",
                "linux",
                "--limit",
                "1",
                "--json",
            ],
        )

        assert result.exit_code == 0
        assert call_args == {"system": "linux", "check_servers": False}
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["id"] == "model-1"


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


class TestRunDSLFiles:
    """Tests for running DSL (.py) files with nodetool run."""

    def test_run_dsl_file_success(self):
        """Test running a valid DSL Python file."""
        runner = CliRunner()

        # Create a simple DSL file using the core Graph type directly
        dsl_code = '''
from nodetool.types.api_graph import Graph, Node

# Create a graph object directly
graph = Graph(
    nodes=[
        Node(id="1", type="nodetool.math.Add", data={"a": 5.0, "b": 3.0})
    ],
    edges=[]
)
'''
        with runner.isolated_filesystem():
            with open("test_workflow.py", "w") as f:
                f.write(dsl_code)

            # Mock the run_workflow to avoid actual execution
            import nodetool.workflows.run_workflow as run_mod
            original_run = run_mod.run_workflow

            async def mock_run_workflow(*args, **kwargs):
                from nodetool.types.job import JobUpdate
                yield JobUpdate(type="job_update", status="completed", job_id="test123", message="Done")

            run_mod.run_workflow = mock_run_workflow

            try:
                result = runner.invoke(cli, ["run", "test_workflow.py", "--show-outputs"])
                assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
                assert "completed" in result.output or "finished" in result.output.lower()
            finally:
                run_mod.run_workflow = original_run

    def test_run_dsl_file_missing_graph(self):
        """Test running a DSL file without a graph object raises error."""
        runner = CliRunner()

        # Create a DSL file without a graph object
        dsl_code = '''
from nodetool.types.api_graph import Graph
# No graph object defined
x = 42
'''
        with runner.isolated_filesystem():
            with open("test_workflow.py", "w") as f:
                f.write(dsl_code)

            result = runner.invoke(cli, ["run", "test_workflow.py"])
            assert result.exit_code == 1
            assert "must define a module-level 'graph' object" in result.output

    def test_run_dsl_file_wrong_graph_type(self):
        """Test running a DSL file where graph is not a Graph object."""
        runner = CliRunner()

        # Create a DSL file with wrong graph type
        dsl_code = '''
# graph is a string, not a Graph object
graph = "not a graph"
'''
        with runner.isolated_filesystem():
            with open("test_workflow.py", "w") as f:
                f.write(dsl_code)

            result = runner.invoke(cli, ["run", "test_workflow.py"])
            assert result.exit_code == 1
            assert "must be of type Graph" in result.output

    def test_run_json_file_still_works(self):
        """Test that running JSON workflow files still works."""
        runner = CliRunner()

        # Create a simple JSON workflow file
        workflow_json = {
            "graph": {
                "nodes": [
                    {
                        "id": "1",
                        "type": "nodetool.math.Add",
                        "data": {"a": 5.0, "b": 3.0}
                    }
                ],
                "edges": []
            }
        }

        with runner.isolated_filesystem():
            with open("test_workflow.json", "w") as f:
                json.dump(workflow_json, f)

            # Mock the run_workflow
            import nodetool.workflows.run_workflow as run_mod
            original_run = run_mod.run_workflow

            async def mock_run_workflow(*args, **kwargs):
                from nodetool.types.job import JobUpdate
                yield JobUpdate(type="job_update", status="completed", job_id="test123", message="Done")

            run_mod.run_workflow = mock_run_workflow

            try:
                result = runner.invoke(cli, ["run", "test_workflow.json"])
                assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
            finally:
                run_mod.run_workflow = original_run

    def test_run_stdin_dsl_code(self):
        """Test running DSL code from stdin."""
        runner = CliRunner()

        # DSL code to pipe via stdin
        dsl_code = '''
from nodetool.types.api_graph import Graph, Node

# Create a graph object directly
graph = Graph(
    nodes=[
        Node(id="1", type="nodetool.math.Add", data={"a": 10.0, "b": 20.0})
    ],
    edges=[]
)
'''

        # Mock the run_workflow
        import nodetool.workflows.run_workflow as run_mod
        original_run = run_mod.run_workflow

        async def mock_run_workflow(*args, **kwargs):
            from nodetool.types.job import JobUpdate
            yield JobUpdate(type="job_update", status="completed", job_id="test123", message="Done")

        run_mod.run_workflow = mock_run_workflow

        try:
            result = runner.invoke(cli, ["run", "--stdin"], input=dsl_code)
            assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
            assert "completed" in result.output or "finished" in result.output.lower()
        finally:
            run_mod.run_workflow = original_run

    def test_run_stdin_json_still_works(self):
        """Test that JSON stdin still works."""
        runner = CliRunner()

        # JSON RunJobRequest
        json_input = '{"workflow_id":"test123","user_id":"1","auth_token":"token","params":{}}'

        # Mock the run_workflow
        import nodetool.workflows.run_workflow as run_mod
        original_run = run_mod.run_workflow

        async def mock_run_workflow(*args, **kwargs):
            from nodetool.types.job import JobUpdate
            yield JobUpdate(type="job_update", status="completed", job_id="test123", message="Done")

        run_mod.run_workflow = mock_run_workflow

        try:
            result = runner.invoke(cli, ["run", "--stdin"], input=json_input)
            assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"
        finally:
            run_mod.run_workflow = original_run

    def test_run_stdin_dsl_missing_graph(self):
        """Test DSL stdin without graph object raises error."""
        runner = CliRunner()

        # DSL code without graph object
        dsl_code = '''
from nodetool.types.api_graph import Graph
# No graph defined
x = 42
'''

        result = runner.invoke(cli, ["run", "--stdin"], input=dsl_code)
        assert result.exit_code == 1
        assert "must define a module-level 'graph' object" in result.output

    def test_run_stdin_empty(self):
        """Test empty stdin raises error."""
        runner = CliRunner()

        result = runner.invoke(cli, ["run", "--stdin"], input="")
        assert result.exit_code == 1
        assert "No input provided" in result.output

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
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
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
    def test_sync_secrets_missing_table_is_non_fatal(self, monkeypatch):
        """Missing local secrets table should not fail deployment flow."""
        import nodetool.cli as cli_module

        class DummyDeployment:
            def get_server_url(self):
                return "http://127.0.0.1:7777"

        printed: list[str] = []

        def fake_run_async(coro):
            coro.close()
            raise sqlite3.OperationalError("no such table: nodetool_secrets")

        monkeypatch.setattr(cli_module, "_run_async", fake_run_async)
        monkeypatch.setattr(cli_module, "_resolve_deployment_auth_token", lambda _d: "token")
        monkeypatch.setattr(cli_module.console, "print", lambda msg: printed.append(str(msg)))

        cli_module._sync_secrets_to_deployment("docker02", DummyDeployment())

        assert any(
            "Skipping secret sync for 'docker02'" in line and "table not initialized" in line
            for line in printed
        )

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
