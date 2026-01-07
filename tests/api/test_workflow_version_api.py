import pytest
from fastapi.testclient import TestClient

from nodetool.models.workflow import Workflow
from nodetool.models.workflow_version import WorkflowVersion
from nodetool.types.graph import Edge, Node
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.workflow import (
    AutosaveWorkflowRequest,
    CreateWorkflowVersionRequest,
    WorkflowRequest,
    WorkflowVersionList,
)
from nodetool.types.workflow import WorkflowVersion as WorkflowVersionType


@pytest.mark.asyncio
async def test_create_workflow_version(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test creating a new version of a workflow."""
    await workflow.save()

    request = CreateWorkflowVersionRequest(
        name="Initial Version",
        description="First version of the workflow",
    )

    response = client.post(
        f"/api/workflows/{workflow.id}/versions",
        json=request.model_dump(),
        headers=headers,
    )
    assert response.status_code == 200
    version = WorkflowVersionType(**response.json())
    assert version.workflow_id == workflow.id
    assert version.version == 1
    assert version.name == "Initial Version"
    assert version.description == "First version of the workflow"


@pytest.mark.asyncio
async def test_create_multiple_versions(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test creating multiple versions increments the version number."""
    await workflow.save()

    # Create first version
    request1 = CreateWorkflowVersionRequest(name="Version 1")
    response1 = client.post(
        f"/api/workflows/{workflow.id}/versions",
        json=request1.model_dump(),
        headers=headers,
    )
    assert response1.status_code == 200
    version1 = WorkflowVersionType(**response1.json())
    assert version1.version == 1

    # Create second version
    request2 = CreateWorkflowVersionRequest(name="Version 2")
    response2 = client.post(
        f"/api/workflows/{workflow.id}/versions",
        json=request2.model_dump(),
        headers=headers,
    )
    assert response2.status_code == 200
    version2 = WorkflowVersionType(**response2.json())
    assert version2.version == 2


@pytest.mark.asyncio
async def test_list_workflow_versions(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test listing all versions of a workflow."""
    await workflow.save()

    # Create some versions
    for i in range(3):
        request = CreateWorkflowVersionRequest(name=f"Version {i + 1}")
        client.post(
            f"/api/workflows/{workflow.id}/versions",
            json=request.model_dump(),
            headers=headers,
        )

    response = client.get(
        f"/api/workflows/{workflow.id}/versions",
        headers=headers,
    )
    assert response.status_code == 200
    version_list = WorkflowVersionList(**response.json())
    assert len(version_list.versions) == 3


@pytest.mark.asyncio
async def test_get_specific_version(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test getting a specific version of a workflow."""
    await workflow.save()

    # Create a version
    request = CreateWorkflowVersionRequest(
        name="Test Version",
        description="A test version",
    )
    client.post(
        f"/api/workflows/{workflow.id}/versions",
        json=request.model_dump(),
        headers=headers,
    )

    response = client.get(
        f"/api/workflows/{workflow.id}/versions/1",
        headers=headers,
    )
    assert response.status_code == 200
    version = WorkflowVersionType(**response.json())
    assert version.version == 1
    assert version.name == "Test Version"


@pytest.mark.asyncio
async def test_get_nonexistent_version(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test getting a version that doesn't exist returns 404."""
    await workflow.save()

    response = client.get(
        f"/api/workflows/{workflow.id}/versions/999",
        headers=headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_restore_workflow_version(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test restoring a workflow to a previous version."""
    await workflow.save()

    # Save original graph
    original_graph = workflow.graph.copy()

    # Create a version
    request = CreateWorkflowVersionRequest(name="Original Version")
    client.post(
        f"/api/workflows/{workflow.id}/versions",
        json=request.model_dump(),
        headers=headers,
    )

    # Modify the workflow
    workflow.graph = {"nodes": [], "edges": []}
    await workflow.save()

    # Verify the workflow was modified
    modified_workflow = await Workflow.get(workflow.id)
    assert modified_workflow is not None
    assert modified_workflow.graph == {"nodes": [], "edges": []}

    # Restore to version 1
    response = client.post(
        f"/api/workflows/{workflow.id}/versions/1/restore",
        headers=headers,
    )
    assert response.status_code == 200

    # Verify the workflow was restored
    restored_workflow = await Workflow.get(workflow.id)
    assert restored_workflow is not None
    assert restored_workflow.graph == original_graph


@pytest.mark.asyncio
async def test_create_version_for_nonexistent_workflow(client: TestClient, headers: dict[str, str]):
    """Test creating a version for a workflow that doesn't exist."""
    request = CreateWorkflowVersionRequest(name="Test")
    response = client.post(
        "/api/workflows/nonexistent-id/versions",
        json=request.model_dump(),
        headers=headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_workflow_version_model_get_next_version(
    workflow: Workflow,
):
    """Test the get_next_version method."""
    await workflow.save()

    # First version should be 1
    next_version = await WorkflowVersion.get_next_version(workflow.id)
    assert next_version == 1

    # Create a version
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Test Version",
    )

    # Next version should be 2
    next_version = await WorkflowVersion.get_next_version(workflow.id)
    assert next_version == 2


@pytest.mark.asyncio
async def test_workflow_version_model_get_latest_version(
    workflow: Workflow,
):
    """Test the get_latest_version method."""
    await workflow.save()

    # No versions yet
    latest = await WorkflowVersion.get_latest_version(workflow.id)
    assert latest is None

    # Create versions
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Version 1",
    )
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Version 2",
    )

    latest = await WorkflowVersion.get_latest_version(workflow.id)
    assert latest is not None
    assert latest.version == 2
    assert latest.name == "Version 2"


@pytest.mark.asyncio
async def test_workflow_version_model_get_by_version(
    workflow: Workflow,
):
    """Test the get_by_version method."""
    await workflow.save()

    # Create versions
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Version 1",
    )
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Version 2",
    )

    # Get version 1
    v1 = await WorkflowVersion.get_by_version(workflow.id, 1)
    assert v1 is not None
    assert v1.version == 1
    assert v1.name == "Version 1"

    # Get version 2
    v2 = await WorkflowVersion.get_by_version(workflow.id, 2)
    assert v2 is not None
    assert v2.version == 2
    assert v2.name == "Version 2"

    # Get nonexistent version
    v99 = await WorkflowVersion.get_by_version(workflow.id, 99)
    assert v99 is None


@pytest.mark.asyncio
async def test_autosave_workflow(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test autosave endpoint creates an autosave version."""
    await workflow.save()

    response = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["skipped"] is False
    assert data["message"] == "autosaved"
    assert data["version"] is not None
    assert data["version"]["save_type"] == "autosave"
    assert "Autosave" in data["version"]["name"]


@pytest.mark.asyncio
async def test_autosave_rate_limiting(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test autosave respects rate limiting (skips if too soon)."""
    await workflow.save()

    # First autosave should succeed
    response1 = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave"},
        headers=headers,
    )
    assert response1.status_code == 200
    assert response1.json()["skipped"] is False

    # Second autosave immediately should be skipped
    response2 = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave"},
        headers=headers,
    )
    assert response2.status_code == 200
    assert response2.json()["skipped"] is True
    assert response2.json()["message"] == "skipped (too soon)"


@pytest.mark.asyncio
async def test_autosave_force_bypasses_rate_limit(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test autosave with force=true bypasses rate limiting."""
    await workflow.save()

    # First autosave
    response1 = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave"},
        headers=headers,
    )
    assert response1.status_code == 200
    assert response1.json()["skipped"] is False

    # Force autosave should succeed despite rate limit
    response2 = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave", "force": True},
        headers=headers,
    )
    assert response2.status_code == 200
    assert response2.json()["skipped"] is False


@pytest.mark.asyncio
async def test_autosave_max_versions_limit(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test autosave respects max versions per workflow."""
    await workflow.save()

    # Create autosaves with force to bypass rate limiting
    for _i in range(25):  # More than the default max of 20
        response = client.post(
            f"/api/workflows/{workflow.id}/autosave",
            json={"save_type": "autosave", "force": True},
            headers=headers,
        )
        if response.json().get("skipped"):
            break

    # The last one should be skipped due to max versions
    last_response = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "autosave", "force": True},
        headers=headers,
    )
    assert last_response.json()["skipped"] is True
    assert "max versions" in last_response.json()["message"]


@pytest.mark.asyncio
async def test_autosave_creates_autosave_metadata(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test autosave includes autosave_metadata with client_id and trigger_reason."""
    await workflow.save()

    response = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={
            "save_type": "autosave",
            "client_id": "test-client-123",
            "description": "Test autosave",
        },
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["version"]["autosave_metadata"]["client_id"] == "test-client-123"
    assert data["version"]["autosave_metadata"]["trigger_reason"] == "autosave"


@pytest.mark.asyncio
async def test_checkpoint_save_type(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test checkpoint save_type creates version with correct save_type."""
    await workflow.save()

    response = client.post(
        f"/api/workflows/{workflow.id}/autosave",
        json={"save_type": "checkpoint", "description": "Pre-run checkpoint"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["version"]["save_type"] == "checkpoint"
    assert data["version"]["description"] == "Pre-run checkpoint"


@pytest.mark.asyncio
async def test_workflow_update_does_not_create_version(client: TestClient, workflow: Workflow, headers: dict[str, str]):
    """Test that PUT /workflows/{id} does NOT automatically create a version."""
    await workflow.save()

    # Count versions before update
    response = client.get(f"/api/workflows/{workflow.id}/versions", headers=headers)
    initial_count = len(response.json()["versions"])

    # Update workflow
    graph = APIGraph(nodes=[Node(id="1", type="nodetool.input.TextInput", data={"label": "Input"})], edges=[])
    request = WorkflowRequest(
        name="Updated Name",
        access="private",
        graph=graph,
    )
    response = client.put(
        f"/api/workflows/{workflow.id}",
        json=request.model_dump(),
        headers=headers,
    )
    assert response.status_code == 200

    # Count versions after update - should be the same
    response = client.get(f"/api/workflows/{workflow.id}/versions", headers=headers)
    final_count = len(response.json()["versions"])
    assert final_count == initial_count


@pytest.mark.asyncio
async def test_workflow_version_model_get_latest_autosave(
    workflow: Workflow,
):
    """Test get_latest_autosave method returns only autosave versions."""
    await workflow.save()

    # Create a manual version
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Manual Version",
        save_type="manual",
    )

    # Create an autosave
    autosave = await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Autosave 1",
        save_type="autosave",
    )

    latest_autosave = await WorkflowVersion.get_latest_autosave(workflow.id)
    assert latest_autosave is not None
    assert latest_autosave.id == autosave.id
    assert latest_autosave.save_type == "autosave"


@pytest.mark.asyncio
async def test_workflow_version_model_count_autosaves(
    workflow: Workflow,
):
    """Test count_autosaves method."""
    await workflow.save()

    # No autosaves yet
    count = await WorkflowVersion.count_autosaves(workflow.id)
    assert count == 0

    # Create autosaves
    for i in range(3):
        await WorkflowVersion.create(
            workflow_id=workflow.id,
            user_id=workflow.user_id,
            graph=workflow.graph,
            name=f"Autosave {i + 1}",
            save_type="autosave",
        )

    # Create a manual version (should not count)
    await WorkflowVersion.create(
        workflow_id=workflow.id,
        user_id=workflow.user_id,
        graph=workflow.graph,
        name="Manual Version",
        save_type="manual",
    )

    count = await WorkflowVersion.count_autosaves(workflow.id)
    assert count == 3
