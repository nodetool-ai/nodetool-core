import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nodetool.deploy.workflow_routes import create_workflow_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(create_workflow_router())
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestWorkflowRoutes:
    def test_list_workflows(self, client):
        mock_registry = {
            "workflow1": MagicMock(name="Workflow One"),
            "workflow2": MagicMock(name="Workflow Two"),
        }
        with patch("nodetool.deploy.workflow_routes._workflow_registry", mock_registry):
            response = client.get("/workflows")
            assert response.status_code == 200
            data = response.json()
            assert "workflows" in data
            ids = {w["id"] for w in data["workflows"]}
            assert ids == {"workflow1", "workflow2"}

    def test_execute_workflow_success(self, client):
        request_data = {"workflow_id": "wf", "params": {"x": 1}}

        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()

        with patch("nodetool.deploy.workflow_routes.get_workflow_by_id") as mock_get, \
            patch("nodetool.deploy.workflow_routes.run_workflow") as mock_run, \
            patch("nodetool.deploy.workflow_routes.ProcessingContext") as mock_ctx_cls:

            mock_get.return_value = mock_workflow

            # Configure context to pass values through
            mock_ctx = MagicMock()
            mock_ctx.encode_assets_as_uri.side_effect = lambda v: v
            mock_ctx_cls.return_value = mock_ctx

            async def mock_gen(*args, **kwargs):
                from nodetool.workflows.types import OutputUpdate

                yield OutputUpdate(
                    node_id="n1",
                    node_name="out",
                    output_name="result",
                    value="ok",
                    output_type="string",
                )

            mock_run.return_value = mock_gen()

            response = client.post("/workflows/execute", json=request_data)
            assert response.status_code == 200
            data = response.json()
            assert data["results"]["out"] == "ok"

    def test_execute_workflow_missing_id(self, client):
        response = client.post("/workflows/execute", json={"params": {}})
        assert response.status_code == 400
        assert "workflow_id is required" in response.json()["detail"]

    def test_execute_workflow_not_found(self, client):
        with patch("nodetool.deploy.workflow_routes.get_workflow_by_id") as mock_get:
            mock_get.side_effect = ValueError("not found")
            response = client.post("/workflows/execute", json={"workflow_id": "bad"})
            assert response.status_code == 404

    def test_execute_workflow_stream(self, client):
        request_data = {"workflow_id": "wf", "params": {}}

        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()

        with patch("nodetool.deploy.workflow_routes.get_workflow_by_id") as mock_get, \
            patch("nodetool.deploy.workflow_routes.run_workflow") as mock_run, \
            patch("nodetool.deploy.workflow_routes.ProcessingContext"):

            mock_get.return_value = mock_workflow

            async def mock_gen(*args, **kwargs):
                from nodetool.types.job import JobUpdate
                from nodetool.workflows.types import OutputUpdate
                yield JobUpdate(status="running")
                yield OutputUpdate(
                    node_id="n1",
                    node_name="out",
                    output_name="result",
                    value="ok",
                    output_type="string",
                )

            mock_run.return_value = mock_gen()

            with client.stream("POST", "/workflows/execute/stream", json=request_data) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                content = resp.read().decode()
                lines = [l for l in content.strip().split("\n") if l.startswith("data: ")]
                # should include output_update and complete
                assert len(lines) >= 2

