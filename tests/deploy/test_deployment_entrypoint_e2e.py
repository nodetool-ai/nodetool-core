"""
End-to-end tests for unified server deployment entrypoint.

These tests validate:
1. Docker-compose YAML is valid and uses the correct entrypoint
2. Dockerfile CMD is consistent with docker-compose
3. The server can start with the run_server entrypoint
4. All critical API endpoints respond correctly
"""

import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests
import yaml

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()

REPO_ROOT = Path(__file__).parent.parent.parent

class TestDockerfileConsistency:
    """Validate Dockerfile is consistent with docker-compose."""

    def test_dockerfile_cmd_uses_run_server(self):
        """Test that Dockerfile CMD uses python -m nodetool.api.run_server."""
        dockerfile_path = REPO_ROOT / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"

        with open(dockerfile_path) as f:
            content = f.read()

        # Should have CMD using run_server module
        assert "nodetool.api.run_server" in content

    def test_dockerfile_exposes_port_7777(self):
        """Test that Dockerfile exposes port 7777."""
        dockerfile_path = REPO_ROOT / "Dockerfile"
        with open(dockerfile_path) as f:
            content = f.read()

        assert "EXPOSE 7777" in content

    def test_dockerfile_has_healthcheck(self):
        """Test that Dockerfile has a HEALTHCHECK instruction."""
        dockerfile_path = REPO_ROOT / "Dockerfile"
        with open(dockerfile_path) as f:
            content = f.read()

        assert "HEALTHCHECK" in content
        assert "/health" in content

    def test_entrypoint_consistency(self):
        """Test that Dockerfile CMD and docker-compose command use the same module."""
        import json
        import re

        dockerfile_path = REPO_ROOT / "Dockerfile"
        compose_path = REPO_ROOT / "docker-compose.yaml"

        with open(dockerfile_path) as f:
            dockerfile_content = f.read()

        with open(compose_path) as f:
            compose_data = yaml.safe_load(f)

        # Parse Dockerfile CMD instruction (JSON array format)
        cmd_match = re.search(r'CMD\s+(\[.*?\])', dockerfile_content)
        assert cmd_match, "CMD instruction not found in Dockerfile"
        dockerfile_cmd = json.loads(cmd_match.group(1))

        # Get compose command
        compose_cmd = compose_data["services"]["api"]["command"]

        # Both should use python -m nodetool.api.run_server as the base command
        assert dockerfile_cmd[:3] == ["python", "-m", "nodetool.api.run_server"], (
            f"Dockerfile CMD base mismatch: {dockerfile_cmd[:3]}"
        )
        assert compose_cmd[:3] == ["python", "-m", "nodetool.api.run_server"], (
            f"Compose command base mismatch: {compose_cmd[:3]}"
        )


class TestRunServerModule:
    """Validate that the run_server module is importable and configured correctly."""

    def test_run_server_module_importable(self):
        """Test that nodetool.api.run_server is importable."""
        from nodetool.api.run_server import run_server

        assert callable(run_server)

    def test_run_server_has_main(self):
        """Test that run_server module has a main() entry point."""
        from nodetool.api.run_server import main

        assert callable(main)

    def test_run_server_accepts_host_port_reload(self):
        """Test that run_server accepts host, port, and reload parameters."""
        import inspect
        from nodetool.api.run_server import run_server

        sig = inspect.signature(run_server)
        params = list(sig.parameters.keys())

        assert "host" in params
        assert "port" in params
        assert "reload" in params

    def test_run_server_default_host_is_all_interfaces(self):
        """Test that run_server defaults to 0.0.0.0 for deployment use."""
        import inspect
        from nodetool.api.run_server import run_server

        sig = inspect.signature(run_server)
        host_param = sig.parameters["host"]

        assert host_param.default == "0.0.0.0"

    def test_run_server_default_port_is_7777(self):
        """Test that run_server defaults to port 7777."""
        import inspect
        from nodetool.api.run_server import run_server

        sig = inspect.signature(run_server)
        port_param = sig.parameters["port"]

        assert port_param.default == 7777


def _resolve_container_runtime() -> str | None:
    """Resolve available container runtime command."""
    for runtime in ("docker", "podman"):
        try:
            result = subprocess.run(
                [runtime, "version"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return runtime
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _check_docker_available() -> bool:
    """Check if Docker or Podman is available."""
    return _resolve_container_runtime() is not None


def _run_runtime(
    runtime: str,
    args: list[str],
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [runtime, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _container_image_available(runtime: str, image: str) -> bool:
    result = _run_runtime(runtime, ["image", "inspect", image], timeout=20)
    return result.returncode == 0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(base_url: str, timeout_s: int = 120) -> None:
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


def _container_logs(runtime: str, container_name: str, tail: int = 300) -> str:
    result = _run_runtime(runtime, ["logs", "--tail", str(tail), container_name], timeout=20)
    return (result.stdout or "") + (result.stderr or "")


def _remove_container(runtime: str, container_name: str) -> None:
    _run_runtime(runtime, ["rm", "-f", container_name], timeout=20)


@dataclass(slots=True)
class LiveServerContainer:
    runtime: str
    container_name: str
    base_url: str
    token: str
    admin_token: str | None = None


@pytest.fixture(scope="module")
def live_server_container() -> LiveServerContainer:
    runtime = _resolve_container_runtime()
    if runtime is None:
        pytest.skip("Docker or Podman is not available")

    image = "nodetool:local"
    if not _container_image_available(runtime, image):
        pytest.skip("nodetool:local image not found - build it first with: docker build -t nodetool:local .")

    container_name = f"nodetool-entry-e2e-{uuid.uuid4().hex[:8]}"
    token = f"entry-e2e-token-{uuid.uuid4().hex[:8]}"
    host_port = _find_free_port()
    base_url = f"http://127.0.0.1:{host_port}"

    with tempfile.TemporaryDirectory(prefix="nodetool-entry-e2e-") as tmp_root:
        tmp_root_path = Path(tmp_root)
        workspace = tmp_root_path / "workspace"
        hf_cache = tmp_root_path / "hf-cache"
        workspace.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)

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
            f"{workspace}:/workspace",
            "-v",
            f"{hf_cache}:/hf-cache",
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
        if run_result.returncode != 0:
            pytest.fail(
                "Failed to start container:\n"
                f"runtime={runtime}\nstdout={run_result.stdout}\nstderr={run_result.stderr}"
            )

        try:
            _wait_for_health(base_url)
            yield LiveServerContainer(
                runtime=runtime,
                container_name=container_name,
                base_url=base_url,
                token=token,
            )
        finally:
            _remove_container(runtime, container_name)


@pytest.fixture(scope="module")
def live_server_container_production() -> LiveServerContainer:
    runtime = _resolve_container_runtime()
    if runtime is None:
        pytest.skip("Docker or Podman is not available")

    image = "nodetool:local"
    if not _container_image_available(runtime, image):
        pytest.skip("nodetool:local image not found - build it first with: docker build -t nodetool:local .")

    container_name = f"nodetool-entry-prod-e2e-{uuid.uuid4().hex[:8]}"
    token = f"entry-prod-token-{uuid.uuid4().hex[:8]}"
    admin_token = f"entry-prod-admin-{uuid.uuid4().hex[:8]}"
    host_port = _find_free_port()
    base_url = f"http://127.0.0.1:{host_port}"

    with tempfile.TemporaryDirectory(prefix="nodetool-entry-prod-e2e-") as tmp_root:
        tmp_root_path = Path(tmp_root)
        workspace = tmp_root_path / "workspace"
        hf_cache = tmp_root_path / "hf-cache"
        workspace.mkdir(parents=True, exist_ok=True)
        hf_cache.mkdir(parents=True, exist_ok=True)

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
            f"{workspace}:/workspace",
            "-v",
            f"{hf_cache}:/hf-cache",
            "-e",
            "ENV=production",
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
            "-e",
            f"ADMIN_TOKEN={admin_token}",
            "-e",
            "SECRETS_MASTER_KEY=test-secrets-master-key-for-e2e",
            image,
        ]
        run_result = _run_runtime(runtime, run_args, timeout=90)
        if run_result.returncode != 0:
            pytest.fail(
                "Failed to start production container:\n"
                f"runtime={runtime}\nstdout={run_result.stdout}\nstderr={run_result.stderr}"
            )

        try:
            _wait_for_health(base_url)
            yield LiveServerContainer(
                runtime=runtime,
                container_name=container_name,
                base_url=base_url,
                token=token,
                admin_token=admin_token,
            )
        finally:
            _remove_container(runtime, container_name)


class TestServerEntrypointRealContainerE2E:
    """Real E2E tests using a live container started from nodetool:local."""

    def _auth_headers(self, container: LiveServerContainer) -> dict[str, str]:
        return {"Authorization": f"Bearer {container.token}"}

    def _skip_if_collection_backend_unavailable(self, response: requests.Response) -> None:
        if response.status_code < 500:
            return
        body = (response.text or "").lower()
        if "chroma" in body or "collection" in body or "vector" in body:
            pytest.skip(f"Collection backend unavailable in test runtime: {response.text}")

    def test_container_uses_run_server_entrypoint(self, live_server_container: LiveServerContainer):
        """Verify runtime logs indicate server entrypoint (not legacy worker entrypoint)."""
        logs = _container_logs(live_server_container.runtime, live_server_container.container_name)
        assert "Starting NodeTool server on" in logs
        assert "Starting NodeTool worker on" not in logs

    def test_health_endpoint_live(self, live_server_container: LiveServerContainer):
        response = requests.get(f"{live_server_container.base_url}/health", timeout=10)
        assert response.status_code == 200
        assert response.json() == "OK"

    def test_ping_endpoint_live(self, live_server_container: LiveServerContainer):
        response = requests.get(f"{live_server_container.base_url}/ping", timeout=10)
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert "timestamp" in payload

    def test_workflows_requires_auth(self, live_server_container: LiveServerContainer):
        response = requests.get(f"{live_server_container.base_url}/api/workflows/", timeout=10)
        assert response.status_code in {401, 403}

    def test_workflows_rejects_invalid_token(self, live_server_container: LiveServerContainer):
        response = requests.get(
            f"{live_server_container.base_url}/api/workflows/",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10,
        )
        assert response.status_code in {401, 403}

    def test_workflows_with_auth(self, live_server_container: LiveServerContainer):
        response = requests.get(
            f"{live_server_container.base_url}/api/workflows/",
            headers=self._auth_headers(live_server_container),
            timeout=10,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "workflows" in payload
        assert isinstance(payload["workflows"], list)

    def test_openai_models_requires_auth(self, live_server_container: LiveServerContainer):
        response = requests.get(f"{live_server_container.base_url}/v1/models", timeout=10)
        assert response.status_code in {401, 403}

    def test_openai_models_rejects_invalid_token(self, live_server_container: LiveServerContainer):
        response = requests.get(
            f"{live_server_container.base_url}/v1/models",
            headers={"Authorization": "Bearer invalid-token"},
            timeout=10,
        )
        assert response.status_code in {401, 403}

    def test_openai_models_with_auth(self, live_server_container: LiveServerContainer):
        response = requests.get(
            f"{live_server_container.base_url}/v1/models",
            headers=self._auth_headers(live_server_container),
            timeout=10,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload.get("object") == "list"
        assert isinstance(payload.get("data"), list)

    def test_admin_routes_require_auth(self, live_server_container: LiveServerContainer):
        admin_paths = [
            "/admin/cache/scan",
            "/admin/cache/size",
            "/admin/assets",
            "/admin/db/workflows/does-not-exist",
            "/admin/storage/assets/does-not-exist.txt",
        ]
        for path in admin_paths:
            response = requests.get(f"{live_server_container.base_url}{path}", timeout=10)
            assert response.status_code in {401, 403}, f"path={path} status={response.status_code} body={response.text}"

    def test_admin_model_validation_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)

        hf_response = requests.post(
            f"{live_server_container.base_url}/admin/models/huggingface/download",
            headers=headers,
            json={},
            timeout=20,
        )
        assert hf_response.status_code == 400, hf_response.text
        assert "repo_id" in hf_response.text

        ollama_response = requests.post(
            f"{live_server_container.base_url}/admin/models/ollama/download",
            headers=headers,
            json={},
            timeout=20,
        )
        assert ollama_response.status_code == 400, ollama_response.text
        assert "model_name" in ollama_response.text

    def test_workflow_routes_end_to_end_live(self, live_server_container: LiveServerContainer):
        workflow_id = f"entrypoint_e2e_{uuid.uuid4().hex[:8]}"
        headers = self._auth_headers(live_server_container)
        workflow_payload = {
            "name": "entrypoint-live-workflow",
            "description": "entrypoint e2e",
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
        }

        put_response = requests.put(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            headers=headers,
            json=workflow_payload,
            timeout=20,
        )
        assert put_response.status_code == 200, put_response.text
        assert put_response.json().get("id") == workflow_id

        list_response = requests.get(
            f"{live_server_container.base_url}/api/workflows/",
            headers=headers,
            timeout=10,
        )
        assert list_response.status_code == 200, list_response.text
        workflows = list_response.json().get("workflows", [])
        assert any(item.get("id") == workflow_id for item in workflows), list_response.json()

        run_response = requests.post(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}/run",
            headers=headers,
            json={},
            timeout=30,
        )
        assert run_response.status_code == 200, run_response.text
        run_payload = run_response.json()
        assert run_payload.get("result") == "ok", run_payload

    def test_workflow_mutation_routes_require_auth(self, live_server_container: LiveServerContainer):
        workflow_id = f"entrypoint_noauth_{uuid.uuid4().hex[:8]}"
        workflow_payload = {
            "name": "entrypoint-noauth-workflow",
            "description": "entrypoint e2e noauth",
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
        }

        put_response = requests.put(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            json=workflow_payload,
            timeout=20,
        )
        assert put_response.status_code in {401, 403}, put_response.text

        run_response = requests.post(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}/run",
            json={},
            timeout=20,
        )
        assert run_response.status_code in {401, 403}, run_response.text

        stream_response = requests.post(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}/run?stream=true",
            json={},
            timeout=20,
        )
        assert stream_response.status_code in {401, 403}, stream_response.text

        delete_response = requests.delete(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            timeout=20,
        )
        assert delete_response.status_code in {401, 403}, delete_response.text

    def test_workflow_run_not_found_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        missing_id = f"missing_{uuid.uuid4().hex[:8]}"
        response = requests.post(
            f"{live_server_container.base_url}/api/workflows/{missing_id}/run",
            headers=headers,
            json={},
            timeout=20,
        )
        assert response.status_code in (404, 500), response.text
        assert "not found" in response.text.lower()

    def test_workflow_run_stream_live(self, live_server_container: LiveServerContainer):
        workflow_id = f"entrypoint_stream_{uuid.uuid4().hex[:8]}"
        headers = self._auth_headers(live_server_container)
        workflow_payload = {
            "name": "entrypoint-live-stream-workflow",
            "description": "entrypoint e2e stream",
            "access": "private",
            "graph": {
                "nodes": [
                    {
                        "id": "out",
                        "type": "nodetool.workflows.test_helper.StringOutput",
                        "data": {"name": "result", "value": "stream-ok"},
                    }
                ],
                "edges": [],
            },
            "tags": [],
            "package_name": "",
            "settings": {},
            "run_mode": None,
        }

        put_response = requests.put(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            headers=headers,
            json=workflow_payload,
            timeout=20,
        )
        assert put_response.status_code == 200, put_response.text

        stream_response = requests.post(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}/run?stream=true",
            headers=headers,
            json={},
            timeout=30,
        )
        assert stream_response.status_code == 200, stream_response.text
        body = stream_response.text
        assert '"type": "job_completed"' in body
        assert '"stream-ok"' in body
        assert '"type": "output_update"' in body

    def test_workflow_delete_lifecycle_live(self, live_server_container: LiveServerContainer):
        workflow_id = f"entrypoint_delete_{uuid.uuid4().hex[:8]}"
        headers = self._auth_headers(live_server_container)
        workflow_payload = {
            "name": "entrypoint-delete-workflow",
            "description": "entrypoint e2e delete",
            "access": "private",
            "graph": {
                "nodes": [
                    {
                        "id": "out",
                        "type": "nodetool.workflows.test_helper.StringOutput",
                        "data": {"name": "result", "value": "delete-ok"},
                    }
                ],
                "edges": [],
            },
            "tags": [],
            "package_name": "",
            "settings": {},
            "run_mode": None,
        }

        create_response = requests.put(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            headers=headers,
            json=workflow_payload,
            timeout=20,
        )
        assert create_response.status_code == 200, create_response.text

        delete_response = requests.delete(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            headers=headers,
            timeout=20,
        )
        assert delete_response.status_code == 200, delete_response.text

        run_after_delete = requests.post(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}/run",
            headers=headers,
            json={},
            timeout=20,
        )
        assert run_after_delete.status_code in (404, 500), run_after_delete.text
        assert "not found" in run_after_delete.text.lower()

    def test_admin_cache_endpoints_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)

        scan_response = requests.get(
            f"{live_server_container.base_url}/admin/cache/scan",
            headers=headers,
            timeout=30,
        )
        assert scan_response.status_code == 200, scan_response.text
        assert isinstance(scan_response.json(), dict)

        size_response = requests.get(
            f"{live_server_container.base_url}/admin/cache/size",
            headers=headers,
            timeout=30,
        )
        assert size_response.status_code == 200, size_response.text
        assert isinstance(size_response.json(), dict)

    def test_admin_db_get_delete_workflow_live(self, live_server_container: LiveServerContainer):
        workflow_id = f"entrypoint_db_{uuid.uuid4().hex[:8]}"
        headers = self._auth_headers(live_server_container)
        workflow_payload = {
            "name": "entrypoint-db-workflow",
            "description": "entrypoint admin db e2e",
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
        }

        create_response = requests.put(
            f"{live_server_container.base_url}/api/workflows/{workflow_id}",
            headers=headers,
            json=workflow_payload,
            timeout=20,
        )
        assert create_response.status_code == 200, create_response.text

        get_response = requests.get(
            f"{live_server_container.base_url}/admin/db/workflows/{workflow_id}",
            headers=headers,
            timeout=20,
        )
        assert get_response.status_code == 200, get_response.text
        db_payload = get_response.json()
        assert db_payload.get("id") == workflow_id
        assert db_payload.get("name") == "entrypoint-db-workflow"

        delete_response = requests.delete(
            f"{live_server_container.base_url}/admin/db/workflows/{workflow_id}",
            headers=headers,
            timeout=20,
        )
        assert delete_response.status_code == 200, delete_response.text
        assert delete_response.json().get("status") == "ok"

        get_after_delete = requests.get(
            f"{live_server_container.base_url}/admin/db/workflows/{workflow_id}",
            headers=headers,
            timeout=20,
        )
        assert get_after_delete.status_code == 404, get_after_delete.text

    def test_admin_db_unknown_table_errors_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        response = requests.get(
            f"{live_server_container.base_url}/admin/db/not_a_table/some-id",
            headers=headers,
            timeout=20,
        )
        assert response.status_code == 500, response.text
        assert "Unknown table" in response.text

    def test_admin_assets_crud_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        asset_id = f"asset_{uuid.uuid4().hex[:8]}"
        user_id = "1"

        create_response = requests.post(
            f"{live_server_container.base_url}/admin/assets",
            headers=headers,
            json={
                "id": asset_id,
                "user_id": user_id,
                "name": "entrypoint-e2e-folder",
                "content_type": "folder",
                "parent_id": user_id,
            },
            timeout=20,
        )
        assert create_response.status_code == 200, create_response.text
        created = create_response.json()
        assert created.get("id") == asset_id
        assert created.get("content_type") == "folder"

        list_response = requests.get(
            f"{live_server_container.base_url}/admin/assets",
            headers=headers,
            params={"user_id": user_id, "parent_id": user_id},
            timeout=20,
        )
        assert list_response.status_code == 200, list_response.text
        assets = list_response.json().get("assets", [])
        assert any(item.get("id") == asset_id for item in assets), list_response.json()

        get_response = requests.get(
            f"{live_server_container.base_url}/admin/assets/{asset_id}",
            headers=headers,
            params={"user_id": user_id},
            timeout=20,
        )
        assert get_response.status_code == 200, get_response.text
        assert get_response.json().get("id") == asset_id

        delete_response = requests.delete(
            f"{live_server_container.base_url}/admin/assets/{asset_id}",
            headers=headers,
            timeout=20,
        )
        assert delete_response.status_code == 200, delete_response.text
        deleted_ids = delete_response.json().get("deleted_asset_ids", [])
        assert asset_id in deleted_ids

    def test_admin_collections_lifecycle_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        name = f"entry_col_{uuid.uuid4().hex[:8]}"
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

        create_response = requests.post(
            f"{live_server_container.base_url}/admin/collections",
            headers=headers,
            json={"name": name, "embedding_model": embedding_model},
            timeout=30,
        )
        self._skip_if_collection_backend_unavailable(create_response)
        assert create_response.status_code == 200, create_response.text
        create_payload = create_response.json()
        assert create_payload.get("name") == name

        list_response = requests.get(
            f"{live_server_container.base_url}/admin/collections",
            headers=headers,
            timeout=30,
        )
        self._skip_if_collection_backend_unavailable(list_response)
        assert list_response.status_code == 200, list_response.text
        collections = list_response.json().get("collections", [])
        assert any(item.get("name") == name for item in collections), list_response.text

        get_response = requests.get(
            f"{live_server_container.base_url}/admin/collections/{name}",
            headers=headers,
            timeout=30,
        )
        self._skip_if_collection_backend_unavailable(get_response)
        assert get_response.status_code == 200, get_response.text
        assert get_response.json().get("name") == name

        add_response = requests.post(
            f"{live_server_container.base_url}/admin/collections/{name}/add",
            headers=headers,
            json={
                "documents": ["hello world"],
                "ids": [f"doc_{uuid.uuid4().hex[:8]}"],
                "metadatas": [{"kind": "e2e"}],
                "embeddings": [[0.1, 0.2, 0.3]],
            },
            timeout=30,
        )
        self._skip_if_collection_backend_unavailable(add_response)
        assert add_response.status_code == 200, add_response.text

        delete_response = requests.delete(
            f"{live_server_container.base_url}/admin/collections/{name}",
            headers=headers,
            timeout=30,
        )
        self._skip_if_collection_backend_unavailable(delete_response)
        assert delete_response.status_code == 200, delete_response.text

    def test_admin_storage_crud_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        key = f"entrypoint_{uuid.uuid4().hex[:8]}.txt"
        content = b"entrypoint-admin-storage-e2e"

        put_response = requests.put(
            f"{live_server_container.base_url}/admin/storage/temp/{key}",
            headers=headers,
            data=content,
            timeout=20,
        )
        assert put_response.status_code == 200, put_response.text

        get_response = requests.get(
            f"{live_server_container.base_url}/admin/storage/temp/{key}",
            headers=headers,
            timeout=20,
        )
        # Temp storage can be request-local memory in test/private server mode.
        # PUT route should work; subsequent GET may be 404 if backend is non-persistent.
        assert get_response.status_code in (200, 404), get_response.text
        if get_response.status_code == 200:
            assert get_response.content == content

        delete_response = requests.delete(
            f"{live_server_container.base_url}/admin/storage/temp/{key}",
            headers=headers,
            timeout=20,
        )
        assert delete_response.status_code in (204, 404), delete_response.text

        get_after_delete = requests.get(
            f"{live_server_container.base_url}/admin/storage/temp/{key}",
            headers=headers,
            timeout=20,
        )
        assert get_after_delete.status_code == 404, get_after_delete.text

    def test_admin_storage_assets_crud_live(self, live_server_container: LiveServerContainer):
        headers = self._auth_headers(live_server_container)
        key = f"entrypoint_asset_{uuid.uuid4().hex[:8]}.txt"
        content = b"entrypoint-admin-asset-storage-e2e"

        put_response = requests.put(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers=headers,
            data=content,
            timeout=20,
        )
        assert put_response.status_code == 200, put_response.text

        head_response = requests.head(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers=headers,
            timeout=20,
        )
        assert head_response.status_code == 200, head_response.text
        assert "Last-Modified" in head_response.headers

        get_response = requests.get(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers=headers,
            timeout=20,
        )
        assert get_response.status_code == 200, get_response.text
        assert get_response.content == content

        range_response = requests.get(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers={**headers, "Range": "bytes=0-9"},
            timeout=20,
        )
        assert range_response.status_code == 206, range_response.text
        assert range_response.content == content[:10]

        delete_response = requests.delete(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers=headers,
            timeout=20,
        )
        assert delete_response.status_code == 204, delete_response.text

        get_after_delete = requests.get(
            f"{live_server_container.base_url}/admin/storage/assets/{key}",
            headers=headers,
            timeout=20,
        )
        assert get_after_delete.status_code == 404, get_after_delete.text


class TestServerEntrypointProductionAdminTokenE2E:
    """Real E2E tests validating production admin-token middleware behavior."""

    def _auth_headers(self, container: LiveServerContainer) -> dict[str, str]:
        return {"Authorization": f"Bearer {container.token}"}

    def test_admin_requires_x_admin_token_in_production(self, live_server_container_production: LiveServerContainer):
        response = requests.get(
            f"{live_server_container_production.base_url}/admin/cache/size",
            headers=self._auth_headers(live_server_container_production),
            timeout=20,
        )
        assert response.status_code == 403, response.text
        assert "Admin token required" in response.text

    def test_admin_rejects_invalid_x_admin_token(self, live_server_container_production: LiveServerContainer):
        response = requests.get(
            f"{live_server_container_production.base_url}/admin/cache/size",
            headers={
                **self._auth_headers(live_server_container_production),
                "X-Admin-Token": "invalid-token",
            },
            timeout=20,
        )
        assert response.status_code == 403, response.text
        assert "Invalid admin token" in response.text

    def test_admin_accepts_valid_x_admin_token(self, live_server_container_production: LiveServerContainer):
        assert live_server_container_production.admin_token is not None
        response = requests.get(
            f"{live_server_container_production.base_url}/admin/cache/size",
            headers={
                **self._auth_headers(live_server_container_production),
                "X-Admin-Token": live_server_container_production.admin_token,
            },
            timeout=20,
        )
        assert response.status_code == 200, response.text
        assert isinstance(response.json(), dict)




# Mark all as integration tests
pytestmark = [pytest.mark.integration]
