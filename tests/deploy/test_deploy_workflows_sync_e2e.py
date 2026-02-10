"""Real E2E integration tests for deploy workflow sync against a live container."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_container_runtime() -> str | None:
    for runtime in ("docker", "podman"):
        try:
            result = subprocess.run([runtime, "version"], capture_output=True, timeout=5, check=False)
            if result.returncode == 0:
                return runtime
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _run_runtime(runtime: str, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [runtime, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _container_image_available(runtime: str, image: str) -> bool:
    inspect_result = _run_runtime(runtime, ["image", "inspect", image], timeout=20)
    if inspect_result.returncode == 0:
        return True

    # Some Docker setups can run/tag local images but fail inspect-by-name.
    list_result = _run_runtime(
        runtime,
        ["image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        timeout=20,
    )
    if list_result.returncode != 0:
        return False
    return image in {line.strip() for line in list_result.stdout.splitlines() if line.strip()}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_s: int = 90) -> None:
    start = time.time()
    last_error = ""
    while time.time() - start < timeout_s:
        try:
            response = requests.get(f"{base_url}/health", timeout=3)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code} body={response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(f"Container API did not become healthy at {base_url}: {last_error}")


def _remove_container(runtime: str, container_name: str) -> None:
    _run_runtime(runtime, ["rm", "-f", container_name], timeout=20)


def _start_live_container(
    runtime: str,
    image: str,
    container_name: str,
    token: str,
    container_workspace: Path,
    container_hf_cache: Path,
) -> str:
    host_port = _find_free_port()
    base_url = f"http://127.0.0.1:{host_port}"

    run_args = [
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{host_port}:7777",
        "-v",
        f"{REPO_ROOT}:/app/nodetool-core:ro",
        "-v",
        f"{container_workspace}:/workspace:rw",
        "-v",
        f"{container_hf_cache}:/hf-cache:rw",
        "-e",
        "ENV=test",
        "-e",
        "PORT=7777",
        "-e",
        "NODETOOL_API_URL=http://localhost:7777",
        "-e",
        "DB_PATH=/workspace/nodetool.db",
        "-e",
        "HF_HOME=/hf-cache",
        "-e",
        "NODETOOL_SERVER_MODE=private",
        "-e",
        "AUTH_PROVIDER=static",
        "-e",
        f"SERVER_AUTH_TOKEN={token}",
        image,
    ]
    run_result = _run_runtime(runtime, run_args, timeout=90)
    assert run_result.returncode == 0, (
        "Failed to start test container:\n"
        f"runtime={runtime}\nstdout={run_result.stdout}\nstderr={run_result.stderr}"
    )

    try:
        _wait_for_health(base_url)
    except Exception as exc:
        logs_result = _run_runtime(runtime, ["logs", "--tail", "200", container_name], timeout=20)
        logs = (logs_result.stdout or "") + (logs_result.stderr or "")
        raise AssertionError(f"Container failed health check at {base_url}: {exc}\nContainer logs:\n{logs}") from exc
    return base_url


def _contains_value(payload: Any, expected: Any) -> bool:
    if payload == expected:
        return True
    if isinstance(payload, dict):
        return any(_contains_value(value, expected) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_value(item, expected) for item in payload)
    return False


def _pythonpath_env() -> str:
    src_path = str(Path.cwd() / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    return f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path


def _subprocess_env(home_path: Path, db_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home_path)
    env["DB_PATH"] = str(db_path)
    env["PYTHONPATH"] = _pythonpath_env()
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


@pytest.mark.integration
def test_deploy_workflows_sync_real_e2e():
    """Start a real container, sync a local workflow via CLI, then execute it via REST."""
    runtime = _resolve_container_runtime()
    if runtime is None:
        pytest.skip("Docker or Podman is not available")

    image = "nodetool:local"
    if not _container_image_available(runtime, image):
        pytest.skip("nodetool:local image not found in active runtime context")

    token = f"sync-e2e-token-{uuid.uuid4().hex[:8]}"
    container_name = f"nodetool-sync-e2e-{uuid.uuid4().hex[:8]}"
    deployment_name = "sync-e2e"
    workflow_id = f"sync_workflow_{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="nodetool-sync-home-") as tmp_home, tempfile.TemporaryDirectory(
        prefix="nodetool-sync-container-"
    ) as tmp_container:
        home_path = Path(tmp_home)
        container_workspace = Path(tmp_container) / "workspace"
        container_hf_cache = Path(tmp_container) / "hf-cache"
        container_workspace.mkdir(parents=True, exist_ok=True)
        container_hf_cache.mkdir(parents=True, exist_ok=True)

        try:
            base_url = _start_live_container(
                runtime=runtime,
                image=image,
                container_name=container_name,
                token=token,
                container_workspace=container_workspace,
                container_hf_cache=container_hf_cache,
            )

            deployment_dir = home_path / ".config" / "nodetool"
            deployment_dir.mkdir(parents=True, exist_ok=True)
            deployment_yaml = deployment_dir / "deployment.yaml"
            deployment_config = {
                "version": "1.0",
                "defaults": {
                    "chat_provider": "llama_cpp",
                    "default_model": "",
                    "log_level": "INFO",
                    "auth_provider": "local",
                    "extra": {},
                },
                "deployments": {
                    deployment_name: {
                        "type": "docker",
                        "enabled": True,
                        "host": "127.0.0.1",
                        "use_proxy": False,
                        "image": {"name": "nodetool", "tag": "local", "registry": "docker.io"},
                        "container": {"name": "sync-e2e", "port": int(base_url.rsplit(":", 1)[1])},
                        "server_auth_token": token,
                    }
                },
            }
            deployment_yaml.write_text(yaml.safe_dump(deployment_config, sort_keys=False), encoding="utf-8")

            local_db_path = home_path / "local-workflows.sqlite3"
            seed_env = _subprocess_env(home_path, local_db_path)

            seed_script = f"""
import asyncio
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import ResourceScope
from nodetool.models.migrations import run_startup_migrations

async def main():
    await run_startup_migrations()
    async with ResourceScope():
        workflow = Workflow(
            id={workflow_id!r},
            user_id="1",
            name="sync-e2e-output",
            access="private",
            graph={{
                "nodes": [
                    {{
                        "id": "out",
                        "type": "nodetool.workflows.test_helper.StringOutput",
                        "data": {{"name": "result", "value": "ok"}},
                    }}
                ],
                "edges": [],
            }},
        )
        await workflow.save()

asyncio.run(main())
"""
            seed_result = subprocess.run(
                [sys.executable, "-c", seed_script],
                capture_output=True,
                text=True,
                env=seed_env,
                check=False,
            )
            assert seed_result.returncode == 0, (
                f"Failed to seed local workflow:\nSTDOUT:\n{seed_result.stdout}\nSTDERR:\n{seed_result.stderr}"
            )

            env = _subprocess_env(home_path, local_db_path)
            sync_cmd = [
                sys.executable,
                "-m",
                "nodetool.cli",
                "deploy",
                "workflows",
                "sync",
                deployment_name,
                workflow_id,
            ]
            sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, env=env, check=False)
            assert sync_result.returncode == 0, (
                f"CLI sync failed:\nSTDOUT:\n{sync_result.stdout}\nSTDERR:\n{sync_result.stderr}"
            )

            headers = {"Authorization": f"Bearer {token}"}
            run_response = requests.post(
                f"{base_url}/api/workflows/{workflow_id}/run",
                headers=headers,
                json={},
                timeout=30,
            )
            assert run_response.status_code == 200, run_response.text
            run_payload = run_response.json()
            assert isinstance(run_payload, dict), run_payload
        finally:
            _remove_container(runtime, container_name)


@pytest.mark.integration
def test_deploy_workflows_list_real_e2e():
    """Start a real container and verify CLI workflow listing against the deployed server."""
    runtime = _resolve_container_runtime()
    if runtime is None:
        pytest.skip("Docker or Podman is not available")

    image = "nodetool:local"
    if not _container_image_available(runtime, image):
        pytest.skip("nodetool:local image not found in active runtime context")

    token = f"sync-list-token-{uuid.uuid4().hex[:8]}"
    container_name = f"nodetool-sync-list-e2e-{uuid.uuid4().hex[:8]}"
    deployment_name = "sync-list-e2e"
    workflow_id = f"sync_list_workflow_{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="nodetool-sync-list-home-") as tmp_home, tempfile.TemporaryDirectory(
        prefix="nodetool-sync-list-container-"
    ) as tmp_container:
        home_path = Path(tmp_home)
        container_workspace = Path(tmp_container) / "workspace"
        container_hf_cache = Path(tmp_container) / "hf-cache"
        container_workspace.mkdir(parents=True, exist_ok=True)
        container_hf_cache.mkdir(parents=True, exist_ok=True)

        try:
            base_url = _start_live_container(
                runtime=runtime,
                image=image,
                container_name=container_name,
                token=token,
                container_workspace=container_workspace,
                container_hf_cache=container_hf_cache,
            )

            host_port = int(base_url.rsplit(":", 1)[1])
            deployment_dir = home_path / ".config" / "nodetool"
            deployment_dir.mkdir(parents=True, exist_ok=True)
            deployment_yaml = deployment_dir / "deployment.yaml"
            deployment_yaml.write_text(
                yaml.safe_dump(
                    {
                        "version": "1.0",
                        "defaults": {
                            "chat_provider": "llama_cpp",
                            "default_model": "",
                            "log_level": "INFO",
                            "auth_provider": "local",
                            "extra": {},
                        },
                        "deployments": {
                            deployment_name: {
                                "type": "docker",
                                "enabled": True,
                                "host": "127.0.0.1",
                                "use_proxy": False,
                                "image": {"name": "nodetool", "tag": "local", "registry": "docker.io"},
                                "container": {"name": "sync-list-e2e", "port": host_port},
                                "server_auth_token": token,
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            headers = {"Authorization": f"Bearer {token}"}
            create_response = requests.put(
                f"{base_url}/api/workflows/{workflow_id}",
                headers=headers,
                json={
                    "name": "sync-list-workflow",
                    "description": "sync list e2e",
                    "access": "private",
                    "graph": {
                        "nodes": [
                            {
                                "id": "out",
                                "type": "nodetool.workflows.test_helper.StringOutput",
                                "data": {"name": "result", "value": "ok"},
                            }
                        ],
                        "edges": [],
                    },
                    "tags": [],
                    "package_name": "",
                    "settings": {},
                    "run_mode": None,
                },
                timeout=20,
            )
            assert create_response.status_code == 200, create_response.text

            env = _subprocess_env(home_path, home_path / "unused.sqlite3")
            list_cmd = [
                sys.executable,
                "-m",
                "nodetool.cli",
                "deploy",
                "workflows",
                "list",
                deployment_name,
            ]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True, env=env, check=False)
            assert list_result.returncode == 0, (
                f"CLI list failed:\nSTDOUT:\n{list_result.stdout}\nSTDERR:\n{list_result.stderr}"
            )
            assert workflow_id in list_result.stdout, list_result.stdout
        finally:
            _remove_container(runtime, container_name)


@pytest.mark.integration
def test_deploy_workflows_sync_missing_local_workflow_e2e():
    """CLI sync should fail cleanly when local workflow does not exist."""
    runtime = _resolve_container_runtime()
    if runtime is None:
        pytest.skip("Docker or Podman is not available")

    image = "nodetool:local"
    if not _container_image_available(runtime, image):
        pytest.skip("nodetool:local image not found in active runtime context")

    token = f"sync-missing-token-{uuid.uuid4().hex[:8]}"
    container_name = f"nodetool-sync-missing-e2e-{uuid.uuid4().hex[:8]}"
    deployment_name = "sync-missing-e2e"
    missing_workflow_id = f"missing_workflow_{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="nodetool-sync-missing-home-") as tmp_home, tempfile.TemporaryDirectory(
        prefix="nodetool-sync-missing-container-"
    ) as tmp_container:
        home_path = Path(tmp_home)
        container_workspace = Path(tmp_container) / "workspace"
        container_hf_cache = Path(tmp_container) / "hf-cache"
        container_workspace.mkdir(parents=True, exist_ok=True)
        container_hf_cache.mkdir(parents=True, exist_ok=True)

        try:
            base_url = _start_live_container(
                runtime=runtime,
                image=image,
                container_name=container_name,
                token=token,
                container_workspace=container_workspace,
                container_hf_cache=container_hf_cache,
            )

            host_port = int(base_url.rsplit(":", 1)[1])
            deployment_dir = home_path / ".config" / "nodetool"
            deployment_dir.mkdir(parents=True, exist_ok=True)
            deployment_yaml = deployment_dir / "deployment.yaml"
            deployment_yaml.write_text(
                yaml.safe_dump(
                    {
                        "version": "1.0",
                        "defaults": {
                            "chat_provider": "llama_cpp",
                            "default_model": "",
                            "log_level": "INFO",
                            "auth_provider": "local",
                            "extra": {},
                        },
                        "deployments": {
                            deployment_name: {
                                "type": "docker",
                                "enabled": True,
                                "host": "127.0.0.1",
                                "use_proxy": False,
                                "image": {"name": "nodetool", "tag": "local", "registry": "docker.io"},
                                "container": {"name": "sync-missing-e2e", "port": host_port},
                                "server_auth_token": token,
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            local_db_path = home_path / "missing-local-workflows.sqlite3"
            env = _subprocess_env(home_path, local_db_path)
            migrate_script = """
import asyncio
from nodetool.models.migrations import run_startup_migrations

asyncio.run(run_startup_migrations())
"""
            migrate_result = subprocess.run(
                [sys.executable, "-c", migrate_script],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            assert migrate_result.returncode == 0, (
                f"Failed to initialize local DB:\nSTDOUT:\n{migrate_result.stdout}\nSTDERR:\n{migrate_result.stderr}"
            )

            sync_cmd = [
                sys.executable,
                "-m",
                "nodetool.cli",
                "deploy",
                "workflows",
                "sync",
                deployment_name,
                missing_workflow_id,
            ]
            sync_result = subprocess.run(sync_cmd, capture_output=True, text=True, env=env, check=False)
            assert sync_result.returncode != 0
            assert "Workflow not found locally" in sync_result.stdout, (
                f"Unexpected sync output:\nSTDOUT:\n{sync_result.stdout}\nSTDERR:\n{sync_result.stderr}"
            )
        finally:
            _remove_container(runtime, container_name)
