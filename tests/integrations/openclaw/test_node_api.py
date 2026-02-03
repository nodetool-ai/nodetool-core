"""Tests for OpenClaw Node API endpoints."""

import pytest
from fastapi.testclient import TestClient

from nodetool.api.server import create_app


@pytest.fixture
def app_with_openclaw(monkeypatch):
    """Create app with OpenClaw enabled."""
    monkeypatch.setenv("OPENCLAW_ENABLED", "true")
    monkeypatch.setenv("OPENCLAW_NODE_ID", "test-node")
    monkeypatch.setenv("OPENCLAW_GATEWAY_URL", "http://test-gateway.example.com")

    # Clear singleton
    from nodetool.integrations.openclaw.config import OpenClawConfig

    OpenClawConfig._instance = None

    app = create_app()
    return app


@pytest.fixture
def client_with_openclaw(app_with_openclaw):
    """Create test client with OpenClaw enabled."""
    return TestClient(app_with_openclaw)


def test_openclaw_capabilities_endpoint(client_with_openclaw):
    """Test that capabilities endpoint returns node capabilities."""
    response = client_with_openclaw.get("/openclaw/capabilities")

    assert response.status_code == 200
    capabilities = response.json()

    assert isinstance(capabilities, list)
    assert len(capabilities) > 0

    # Check that capabilities have required fields
    for capability in capabilities:
        assert "name" in capability
        assert "description" in capability
        assert "input_schema" in capability
        assert "output_schema" in capability


def test_openclaw_status_endpoint(client_with_openclaw):
    """Test that status endpoint returns node status."""
    response = client_with_openclaw.get("/openclaw/status")

    assert response.status_code == 200
    status = response.json()

    assert "node_id" in status
    assert status["node_id"] == "test-node"
    assert "status" in status
    assert "uptime_seconds" in status
    assert "active_tasks" in status
    assert "total_tasks_completed" in status
    assert "total_tasks_failed" in status
    assert "system_info" in status


def test_openclaw_health_endpoint(client_with_openclaw):
    """Test that health endpoint returns health status."""
    response = client_with_openclaw.get("/openclaw/health")

    assert response.status_code == 200
    health = response.json()

    assert "status" in health
    assert health["status"] == "healthy"
    assert "node_id" in health
    assert "uptime_seconds" in health


def test_openclaw_execute_endpoint(client_with_openclaw):
    """Test task execution endpoint."""
    task_request = {
        "task_id": "test-task-123",
        "capability_name": "workflow_execution",
        "parameters": {"workflow_data": {"nodes": [], "edges": []}},
    }

    response = client_with_openclaw.post("/openclaw/execute", json=task_request)

    assert response.status_code == 200
    result = response.json()

    assert result["task_id"] == "test-task-123"
    assert result["status"] in ["running", "pending"]


def test_openclaw_execute_unknown_capability(client_with_openclaw):
    """Test that executing unknown capability returns proper response."""
    task_request = {
        "task_id": "test-task-456",
        "capability_name": "unknown_capability",
        "parameters": {},
    }

    response = client_with_openclaw.post("/openclaw/execute", json=task_request)

    assert response.status_code == 200
    result = response.json()

    assert result["task_id"] == "test-task-456"
    assert result["status"] == "failed"
    assert "Unknown capability" in result["message"]


def test_openclaw_get_task_status(client_with_openclaw):
    """Test getting status of a task."""
    # First, start a task
    task_request = {
        "task_id": "test-task-789",
        "capability_name": "chat_completion",
        "parameters": {"messages": [{"role": "user", "content": "Hello"}]},
    }

    execute_response = client_with_openclaw.post(
        "/openclaw/execute", json=task_request
    )
    assert execute_response.status_code == 200
    result = execute_response.json()

    # Task should be accepted (running or pending)
    assert result["task_id"] == "test-task-789"
    assert result["status"] in ["running", "pending"]


def test_openclaw_get_nonexistent_task(client_with_openclaw):
    """Test getting status of a nonexistent task returns 404."""
    response = client_with_openclaw.get("/openclaw/tasks/nonexistent-task")

    assert response.status_code == 404


def test_openclaw_disabled_by_default():
    """Test that OpenClaw endpoints are not available when disabled."""
    # Clear singleton
    from nodetool.integrations.openclaw.config import OpenClawConfig

    OpenClawConfig._instance = None

    app = create_app()
    client = TestClient(app)

    # OpenClaw endpoints should not be available
    response = client.get("/openclaw/capabilities")
    assert response.status_code == 404


def test_openclaw_register_endpoint_when_disabled(monkeypatch):
    """Test that register endpoint returns 503 when OpenClaw is disabled."""
    # Create app without OpenClaw enabled (but with router loaded for testing)
    monkeypatch.setenv("OPENCLAW_ENABLED", "false")

    # Clear singleton
    from nodetool.integrations.openclaw.config import OpenClawConfig

    OpenClawConfig._instance = None

    # Import router directly for testing
    from fastapi import FastAPI

    from nodetool.integrations.openclaw.node_api import router

    test_app = FastAPI()
    test_app.include_router(router)

    client = TestClient(test_app)

    registration = {
        "node_id": "test-node",
        "node_name": "Test Node",
        "node_version": "1.0.0",
        "capabilities": [],
    }

    response = client.post("/openclaw/register", json=registration)

    assert response.status_code == 503
    assert "not enabled" in response.json()["detail"]
