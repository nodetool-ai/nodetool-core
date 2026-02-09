"""
End-to-end tests for unified server deployment entrypoint.

These tests validate:
1. Docker-compose YAML is valid and uses the correct entrypoint
2. Dockerfile CMD is consistent with docker-compose
3. The server can start with the run_server entrypoint
4. All critical API endpoints respond correctly
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()

REPO_ROOT = Path(__file__).parent.parent.parent


class TestDockerComposeConfiguration:
    """Validate docker-compose.yaml configuration."""

    def test_docker_compose_is_valid_yaml(self):
        """Test that docker-compose.yaml is valid YAML."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        assert compose_path.exists(), "docker-compose.yaml not found"

        with open(compose_path) as f:
            data = yaml.safe_load(f)

        assert data is not None
        assert "services" in data

    def test_api_service_uses_run_server_entrypoint(self):
        """Test that the api service uses python -m nodetool.api.run_server."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        api_service = data["services"]["api"]
        command = api_service["command"]

        # Should use python -m nodetool.api.run_server (not nodetool serve)
        assert command[0] == "python", f"Expected 'python' as first arg, got '{command[0]}'"
        assert command[1] == "-m", f"Expected '-m' as second arg, got '{command[1]}'"
        assert command[2] == "nodetool.api.run_server", (
            f"Expected 'nodetool.api.run_server' as module, got '{command[2]}'"
        )

    def test_api_service_binds_to_all_interfaces(self):
        """Test that the api service binds to 0.0.0.0 for container accessibility."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        api_service = data["services"]["api"]
        command = api_service["command"]

        # Should bind to 0.0.0.0
        assert "--host" in command
        host_idx = command.index("--host")
        assert command[host_idx + 1] == "0.0.0.0"

    def test_api_service_uses_port_7777(self):
        """Test that the api service uses port 7777."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        api_service = data["services"]["api"]
        command = api_service["command"]

        assert "--port" in command
        port_idx = command.index("--port")
        assert command[port_idx + 1] == "7777"

    def test_api_service_has_healthcheck(self):
        """Test that the api service has a health check configured."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        api_service = data["services"]["api"]
        assert "healthcheck" in api_service

        healthcheck = api_service["healthcheck"]
        assert "test" in healthcheck
        # Health check should hit /health endpoint
        test_cmd = " ".join(healthcheck["test"])
        assert "/health" in test_cmd

    def test_api_service_exposes_port(self):
        """Test that the api service exposes port 7777."""
        compose_path = REPO_ROOT / "docker-compose.yaml"
        with open(compose_path) as f:
            data = yaml.safe_load(f)

        api_service = data["services"]["api"]
        assert "expose" in api_service
        assert "7777" in api_service["expose"]


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


class TestServerEntrypointE2E:
    """E2E tests for the server using TestClient."""

    def test_health_endpoint_via_run_server_config(self):
        """Test /health endpoint works when app is created via run_server path."""
        from fastapi.testclient import TestClient
        from nodetool.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == "OK"

    def test_ping_endpoint_via_run_server_config(self):
        """Test /ping endpoint returns structured health info."""
        from fastapi.testclient import TestClient
        from nodetool.api.server import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_server_modes_all_create_successfully(self):
        """Test that all server modes can create an app without errors."""
        from nodetool.api.server import create_app

        # Desktop mode (default)
        app = create_app(mode="desktop")
        assert app is not None

        # Private mode with static auth
        app = create_app(mode="private", auth_provider="static")
        assert app is not None

    def test_deploy_routes_available(self):
        """Test that deploy routes are mounted on the server."""
        from fastapi.testclient import TestClient
        from nodetool.api.server import create_app

        app = create_app()
        routes = [r.path for r in app.routes]

        # Deploy admin routes should be available
        assert any("/admin" in r for r in routes)
        # Storage routes should be available
        assert any("/storage" in r for r in routes)
        # Workflow deploy routes should be available
        assert any("/workflows" in r for r in routes)

    def test_websocket_endpoints_available(self):
        """Test that WebSocket endpoints are available."""
        from nodetool.api.server import create_app

        app = create_app()
        routes = [r.path for r in app.routes if hasattr(r, "path")]

        # Main ws endpoint
        assert "/ws" in routes

    def test_openai_compatible_endpoints_available(self):
        """Test that OpenAI-compatible endpoints are available."""
        from nodetool.api.server import create_app

        app = create_app()
        routes = [r.path for r in app.routes]

        assert any("/v1" in r for r in routes)


def _check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestDockerValidation:
    """Test Docker-specific validation."""

    @pytest.mark.skipif(
        not _check_docker_available(),
        reason="Docker is not available",
    )
    def test_docker_compose_config_valid(self):
        """Test that docker-compose config is valid by running docker compose config."""
        result = subprocess.run(
            ["docker", "compose", "-f", str(REPO_ROOT / "docker-compose.yaml"), "config"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # docker compose config will fail if the config is invalid
        # Note: it may warn about missing env vars, but shouldn't error
        assert result.returncode == 0, f"docker compose config failed: {result.stderr}"


# Mark all as integration tests
pytestmark = [pytest.mark.integration]
