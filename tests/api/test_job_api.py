import json
import os

import pytest
from nodetool.types.graph import Edge
from nodetool.types.graph import Graph
from nodetool.models.job import Job
from nodetool.workflows.run_job_request import RunJobRequest
from conftest import make_job
from fastapi.testclient import TestClient
from nodetool.models.workflow import Workflow

current_dir = os.path.dirname(os.path.realpath(__file__))
test_file = os.path.join(current_dir, "test.jpg")


def test_get(client: TestClient, headers: dict[str, str], user_id: str):
    job = make_job(user_id)
    job.save()
    response = client.get(f"/api/jobs/{job.id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data["id"] == job.id
    assert data["status"] == job.status


def test_put(client: TestClient, headers: dict[str, str], user_id: str):
    job = make_job(user_id)
    job.save()
    response = client.put(
        f"/api/jobs/{job.id}", headers=headers, json={"status": "completed"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert data["id"] == job.id
    assert data["status"] == "completed"

    job_reloaded = Job.get(job.id)
    assert job_reloaded is not None
    assert job_reloaded.status == "completed"


def test_index(client: TestClient, headers: dict[str, str], user_id: str):
    make_job(user_id)
    make_job(user_id)
    response = client.get("/api/jobs", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data["jobs"]) == 2


def test_index_limit(client: TestClient, headers: dict[str, str], user_id: str):
    make_job(user_id)
    make_job(user_id)
    response = client.get("/api/jobs", params={"page_size": 1}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert len(data["jobs"]) == 1


@pytest.mark.asyncio
async def test_create(
    client: TestClient,
    workflow: Workflow,
    user_id: str,
    headers: dict[str, str],
):

    req = RunJobRequest(
        workflow_id=workflow.id,
        graph=Graph(nodes=[], edges=[]),
        params={},
    )

    response = client.post(
        "/api/jobs/",
        json=req.model_dump(),
        headers=headers,
    )
    assert response.status_code == 200
    job_data = response.json()

    assert job_data["status"] == "running"
    assert job_data["workflow_id"] == workflow.id
    assert job_data["user_id"] == user_id
    assert job_data["graph"] == {"nodes": [], "edges": []}
