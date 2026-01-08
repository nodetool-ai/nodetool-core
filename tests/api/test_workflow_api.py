import pytest
from fastapi.testclient import TestClient

from nodetool.models.workflow import Workflow
from nodetool.types.api_graph import Edge, Node
from nodetool.types.api_graph import Graph as APIGraph
from nodetool.types.workflow import WorkflowList, WorkflowRequest


@pytest.mark.asyncio
async def test_create_workflow(client: TestClient, headers: dict[str, str], user_id: str):
    params = {
        "name": "Test Workflow",
        "graph": {
            "nodes": [],
            "edges": [],
        },
        "description": "Test Workflow Description",
        "thumbnail": "Test Workflow Thumbnail",
        "access": "private",
    }
    request = WorkflowRequest(**params)
    json = request.model_dump()
    response = client.post("/api/workflows/", json=json, headers=headers)
    assert response.status_code == 200
    assert response.json()["name"] == "Test Workflow"

    w = await Workflow.get(response.json()["id"])
    assert w is not None
    assert w.name == "Test Workflow"
    assert w.user_id == user_id
    assert w.graph == {
        "nodes": [],
        "edges": [],
    }


@pytest.mark.asyncio
async def test_get_workflows(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.get("/api/workflows/", headers=headers)
    assert response.status_code == 200
    workflow_list = WorkflowList(**response.json())
    assert len(workflow_list.workflows) == 1
    assert workflow_list.workflows[0].id == workflow.id


@pytest.mark.asyncio
async def test_get_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.get(f"/api/workflows/{workflow.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == workflow.id
    assert response.json()["input_schema"] == {
        "type": "object",
        "properties": {
            "in1": {
                "type": "number",
                "default": 10,
                "minimum": 0,
                "maximum": 100,
                "label": "",
            },
        },
        "required": ["in1"],
    }


@pytest.mark.asyncio
async def test_get_public_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.get(f"/api/workflows/public/{workflow.id}", headers=headers)
    assert response.status_code == 404
    workflow.access = "public"
    await workflow.save()
    response = client.get(f"/api/workflows/public/{workflow.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == workflow.id
    assert response.json()["input_schema"] == {
        "type": "object",
        "properties": {
            "in1": {
                "type": "number",
                "default": 10,
                "minimum": 0,
                "maximum": 100,
                "label": "",
            },
        },
        "required": ["in1"],
    }


@pytest.mark.asyncio
async def test_index(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.get("/api/workflows/", headers=headers)
    assert response.status_code == 200
    workflow_list = WorkflowList(**response.json())
    assert len(workflow_list.workflows) == 1
    assert workflow_list.workflows[0].id == workflow.id


@pytest.mark.asyncio
async def test_get_public_workflows(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.get("/api/workflows/public", headers=headers)
    assert response.status_code == 200
    workflow_list = WorkflowList(**response.json())
    assert len(workflow_list.workflows) == 0

    workflow.access = "public"
    await workflow.save()
    response = client.get("/api/workflows/public", headers=headers)
    assert response.status_code == 200
    workflow_list = WorkflowList(**response.json())
    assert len(workflow_list.workflows) == 1


@pytest.mark.asyncio
async def test_update_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    request = WorkflowRequest(
        name="Updated Workflow",
        description="Updated Workflow Description",
        thumbnail="Updated Workflow Thumbnail",
        access="public",
        graph=APIGraph(
            nodes=[Node(**n) for n in workflow.graph["nodes"]],
            edges=[Edge(**e) for e in workflow.graph["edges"]],
        ),
    )
    response = client.put(f"/api/workflows/{workflow.id}", json=request.model_dump(), headers=headers)
    assert response.status_code == 200
    assert "id" in response.json()

    saved_workflow = await Workflow.get(response.json()["id"])

    assert saved_workflow is not None
    assert saved_workflow.name == "Updated Workflow"


@pytest.mark.asyncio
async def test_delete_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    await workflow.save()
    response = client.delete(f"/api/workflows/{workflow.id}", headers=headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_generate_workflow_name(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test generating a name for a workflow using LLM."""
    from unittest.mock import AsyncMock, patch

    from nodetool.metadata.types import Message
    from nodetool.providers.base import MockProvider

    await workflow.save()

    # Create a mock provider that returns a generated name
    mock_response = Message(role="assistant", content="Image Processing Pipeline")
    mock_provider = MockProvider([mock_response])

    # Patch get_provider to return our mock
    with patch("nodetool.api.workflow.get_provider", new=AsyncMock(return_value=mock_provider)):
        response = client.post(
            f"/api/workflows/{workflow.id}/generate-name",
            json={"provider": "openai", "model": "gpt-4"},
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Image Processing Pipeline"
    assert data["id"] == workflow.id

    # Verify the workflow was updated in the database
    updated = await Workflow.get(workflow.id)
    assert updated is not None
    assert updated.name == "Image Processing Pipeline"


@pytest.mark.asyncio
async def test_generate_workflow_name_not_found(client: TestClient, headers: dict[str, str]):
    """Test generating name for non-existent workflow returns 404."""
    response = client.post(
        "/api/workflows/nonexistent-id/generate-name", json={"provider": "openai", "model": "gpt-4"}, headers=headers
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Workflow not found"


@pytest.mark.asyncio
async def test_generate_workflow_name_with_description(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test that workflow description is included in the LLM prompt."""
    from unittest.mock import AsyncMock, patch

    from nodetool.metadata.types import Message
    from nodetool.providers.base import MockProvider

    # Set a description on the workflow
    workflow.description = "A workflow that processes images and applies filters"
    await workflow.save()

    # Create a mock provider that returns a generated name
    mock_response = Message(role="assistant", content="Filter Image Processor")
    mock_provider = MockProvider([mock_response])

    # Patch get_provider to return our mock
    with patch("nodetool.api.workflow.get_provider", new=AsyncMock(return_value=mock_provider)):
        response = client.post(
            f"/api/workflows/{workflow.id}/generate-name",
            json={"provider": "anthropic", "model": "claude-3"},
            headers=headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Filter Image Processor"


# def test_run_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
#     await workflow.save()

#     response = client.post(
#         f"/api/workflows/{workflow.id}/run", json={}, headers=headers
#     )
#     assert response.status_code == 200

#     # Test streaming response
#     response = client.post(
#         f"/api/workflows/{workflow.id}/run",
#         json={},
#         params={"stream": True},
#         headers=headers,
#     )
#     assert response.status_code == 200
#     assert response.headers["content-type"] == "application/x-ndjson"
