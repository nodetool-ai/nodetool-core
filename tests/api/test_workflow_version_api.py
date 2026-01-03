import pytest
from fastapi.testclient import TestClient

from nodetool.models.workflow import Workflow
from nodetool.models.workflow_version import WorkflowVersion
from nodetool.types.graph import Edge, Node
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.workflow import (
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
