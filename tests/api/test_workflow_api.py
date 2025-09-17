from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from nodetool.types.workflow import WorkflowRequest, WorkflowList
from nodetool.types.graph import Edge, Graph as APIGraph, Node
from nodetool.models.workflow import Workflow


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
async def test_get_public_workflow(
    client: TestClient, workflow: Workflow, headers: dict[str, str]
):
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
async def test_get_public_workflows(
    client: TestClient, workflow: Workflow, headers: dict[str, str]
):
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
async def test_update_workflow(
    client: TestClient, workflow: Workflow, headers: dict[str, str]
):
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
    response = client.put(
        f"/api/workflows/{workflow.id}", json=request.model_dump(), headers=headers
    )
    assert response.status_code == 200
    assert "id" in response.json()

    saved_workflow = await Workflow.get(response.json()["id"])

    assert saved_workflow is not None
    assert saved_workflow.name == "Updated Workflow"


@pytest.mark.asyncio
async def test_delete_workflow(
    client: TestClient, workflow: Workflow, headers: dict[str, str]
):
    await workflow.save()
    response = client.delete(f"/api/workflows/{workflow.id}", headers=headers)
    assert response.status_code == 200


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


@pytest.mark.asyncio
async def test_get_image_generation_workflows(
    client: TestClient, headers: dict[str, str], user_id: str
):
    def image_workflow_graph() -> dict[str, Any]:
        return {
            "nodes": [
                Node(
                    id="input", type="nodetool.input.StringInput", data={"name": "prompt"}
                ).model_dump(),
                Node(
                    id="output",
                    type="nodetool.output.ImageOutput",
                    data={"name": "image"},
                ).model_dump(),
            ],
            "edges": [],
        }

    def text_workflow_graph() -> dict[str, Any]:
        return {
            "nodes": [
                Node(
                    id="text_input",
                    type="nodetool.input.StringInput",
                    data={"name": "prompt"},
                ).model_dump(),
                Node(
                    id="text_output",
                    type="nodetool.output.StringOutput",
                    data={"name": "response"},
                ).model_dump(),
            ],
            "edges": [],
        }

    user_image_workflow = await Workflow.create(
        user_id=user_id,
        name="user-image",
        access="private",
        graph=image_workflow_graph(),
    )

    public_image_workflow = await Workflow.create(
        user_id="someone-else",
        name="public-image",
        access="public",
        graph=image_workflow_graph(),
    )

    non_image_workflow = await Workflow.create(
        user_id=user_id,
        name="text-workflow",
        access="private",
        graph=text_workflow_graph(),
    )

    response = client.get("/api/workflows/image-generation", headers=headers)
    assert response.status_code == 200

    workflow_list = WorkflowList(**response.json())
    returned_ids = {wf.id for wf in workflow_list.workflows}

    assert user_image_workflow.id in returned_ids
    assert public_image_workflow.id in returned_ids
    assert non_image_workflow.id not in returned_ids

    for wf in workflow_list.workflows:
        assert wf.input_schema is not None
        assert wf.output_schema is not None

        input_properties = wf.input_schema.get("properties", {})
        assert any(
            isinstance(prop, dict)
            and (
                prop.get("type") == "string"
                or (
                    isinstance(prop.get("type"), list)
                    and "string" in prop.get("type")
                )
            )
            for prop in input_properties.values()
        )

        output_properties = wf.output_schema.get("properties", {})

        def has_image_value(prop: dict[str, Any]) -> bool:
            if not isinstance(prop, dict):
                return False
            if prop.get("type") == "object":
                type_field = prop.get("properties", {}).get("type", {})
                const = type_field.get("const")
                enum = type_field.get("enum", [])
                if const == "image" or (isinstance(enum, list) and "image" in enum):
                    return True
            if prop.get("type") == "array":
                return has_image_value(prop.get("items", {}))
            return False

        assert any(has_image_value(prop) for prop in output_properties.values())
