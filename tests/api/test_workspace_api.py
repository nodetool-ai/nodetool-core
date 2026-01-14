"""
Tests for the Workspace API endpoints.
"""

import os
import tempfile

import pytest


class TestWorkspaceAPI:
    """Test cases for the Workspace API endpoints."""

    def test_list_workspaces_empty(self, client, headers):
        """Test listing workspaces when none exist."""
        response = client.get("/api/workspaces", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "workspaces" in data
        assert isinstance(data["workspaces"], list)
        # May be empty or have workspaces from other tests

    def test_create_workspace_success(self, client, headers, tmp_path):
        """Test creating a workspace with valid data."""
        workspace_path = str(tmp_path)

        response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Test Workspace",
                "path": workspace_path,
                "is_default": False,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Workspace"
        assert data["path"] == workspace_path
        assert data["is_default"] is False
        assert data["is_accessible"] is True

    def test_create_workspace_relative_path_rejected(self, client, headers):
        """Test that relative paths are rejected."""
        response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Test Workspace",
                "path": "relative/path",
                "is_default": False,
            },
        )
        assert response.status_code == 400
        assert "absolute" in response.json()["detail"].lower()

    def test_create_workspace_nonexistent_path_rejected(self, client, headers):
        """Test that non-existent paths are rejected."""
        response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Test Workspace",
                "path": "/nonexistent/path/that/does/not/exist",
                "is_default": False,
            },
        )
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]

    def test_get_workspace(self, client, headers, tmp_path):
        """Test getting a specific workspace."""
        # First create a workspace
        workspace_path = str(tmp_path)
        create_response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Get Test Workspace",
                "path": workspace_path,
            },
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Then get it
        get_response = client.get(f"/api/workspaces/{workspace_id}", headers=headers)
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["name"] == "Get Test Workspace"
        assert data["path"] == workspace_path

    def test_get_workspace_not_found(self, client, headers):
        """Test getting a non-existent workspace returns 404."""
        response = client.get("/api/workspaces/nonexistent-id", headers=headers)
        assert response.status_code == 404

    def test_delete_workspace(self, client, headers, tmp_path):
        """Test deleting a workspace."""
        # First create a workspace
        workspace_path = str(tmp_path)
        create_response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Delete Test Workspace",
                "path": workspace_path,
            },
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Then delete it
        delete_response = client.delete(f"/api/workspaces/{workspace_id}", headers=headers)
        assert delete_response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/api/workspaces/{workspace_id}", headers=headers)
        assert get_response.status_code == 404

    def test_update_workspace(self, client, headers, tmp_path):
        """Test updating a workspace."""
        # Create a workspace
        workspace_path = str(tmp_path)
        create_response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "Original Name",
                "path": workspace_path,
                "is_default": False,
            },
        )
        assert create_response.status_code == 201
        workspace_id = create_response.json()["id"]

        # Update name and set as default
        update_response = client.put(
            f"/api/workspaces/{workspace_id}",
            headers=headers,
            json={
                "name": "Updated Name",
                "is_default": True,
            },
        )
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["name"] == "Updated Name"
        assert data["is_default"] is True
        assert data["path"] == workspace_path


class TestWorkflowFileEndpoints:
    """Test cases for the workflow-based file endpoints."""

    @pytest.fixture
    def workspace_and_workflow(self, client, headers, tmp_path):
        """Create a workspace and workflow for testing file operations."""
        # Create workspace
        workspace_path = str(tmp_path)
        ws_response = client.post(
            "/api/workspaces",
            headers=headers,
            json={
                "name": "File Test Workspace",
                "path": workspace_path,
            },
        )
        assert ws_response.status_code == 201
        workspace_id = ws_response.json()["id"]

        # Create workflow with workspace_id
        wf_response = client.post(
            "/api/workflows",
            headers=headers,
            json={
                "name": "File Test Workflow",
                "graph": {"nodes": [], "edges": []},
                "workspace_id": workspace_id,
                "access": "private",
            },
        )
        assert wf_response.status_code == 200
        workflow_id = wf_response.json()["id"]

        return {
            "workspace_id": workspace_id,
            "workflow_id": workflow_id,
            "workspace_path": workspace_path,
        }

    def test_list_workflow_files(self, client, headers, workspace_and_workflow):
        """Test listing files in a workflow's workspace."""
        workflow_id = workspace_and_workflow["workflow_id"]
        workspace_path = workspace_and_workflow["workspace_path"]

        # Create a test file in the workspace
        test_file = os.path.join(workspace_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/files",
            headers=headers,
        )
        assert response.status_code == 200
        files = response.json()
        names = [f["name"] for f in files]
        assert "test.txt" in names

    def test_list_workflow_files_with_subpath(self, client, headers, workspace_and_workflow):
        """Test listing files in a subdirectory of a workflow's workspace."""
        workflow_id = workspace_and_workflow["workflow_id"]
        workspace_path = workspace_and_workflow["workspace_path"]

        # Create a subdirectory with a file
        subdir = os.path.join(workspace_path, "subdir")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "nested.txt"), "w") as f:
            f.write("nested content")

        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/files",
            params={"path": "subdir"},
            headers=headers,
        )
        assert response.status_code == 200
        files = response.json()
        names = [f["name"] for f in files]
        assert "nested.txt" in names

    def test_list_workflow_files_workflow_not_found(self, client, headers):
        """Test listing files with non-existent workflow returns 404."""
        response = client.get(
            "/api/workspaces/workflow/nonexistent-workflow-id/files",
            headers=headers,
        )
        assert response.status_code == 404
        assert "Workflow not found" in response.json()["detail"]

    def test_list_workflow_files_no_workspace(self, client, headers):
        """Test listing files for workflow without workspace returns 404."""
        # Create workflow without workspace_id
        wf_response = client.post(
            "/api/workflows",
            headers=headers,
            json={
                "name": "No Workspace Workflow",
                "graph": {"nodes": [], "edges": []},
                "access": "private",
            },
        )
        assert wf_response.status_code == 200
        workflow_id = wf_response.json()["id"]

        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/files",
            headers=headers,
        )
        assert response.status_code == 404
        assert "does not have an associated workspace" in response.json()["detail"]

    def test_upload_workflow_file(self, client, headers, workspace_and_workflow):
        """Test uploading a file to a workflow's workspace."""
        workflow_id = workspace_and_workflow["workflow_id"]
        workspace_path = workspace_and_workflow["workspace_path"]

        content = b"uploaded content"
        response = client.post(
            f"/api/workspaces/workflow/{workflow_id}/upload/uploaded.txt",
            files={"file": ("uploaded.txt", content, "text/plain")},
            headers=headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "uploaded.txt"

        # Verify file exists
        uploaded_path = os.path.join(workspace_path, "uploaded.txt")
        assert os.path.exists(uploaded_path)
        with open(uploaded_path, "rb") as f:
            assert f.read() == content

    def test_upload_workflow_file_nested_path(self, client, headers, workspace_and_workflow):
        """Test uploading a file to a nested path in a workflow's workspace."""
        workflow_id = workspace_and_workflow["workflow_id"]
        workspace_path = workspace_and_workflow["workspace_path"]

        content = b"nested uploaded content"
        response = client.post(
            f"/api/workspaces/workflow/{workflow_id}/upload/nested/dir/file.txt",
            files={"file": ("file.txt", content, "text/plain")},
            headers=headers,
        )
        assert response.status_code == 200

        # Verify file exists in nested path
        uploaded_path = os.path.join(workspace_path, "nested", "dir", "file.txt")
        assert os.path.exists(uploaded_path)

    def test_download_workflow_file(self, client, headers, workspace_and_workflow):
        """Test downloading a file from a workflow's workspace."""
        workflow_id = workspace_and_workflow["workflow_id"]
        workspace_path = workspace_and_workflow["workspace_path"]

        # Create a test file
        content = b"download test content"
        test_file = os.path.join(workspace_path, "download.txt")
        with open(test_file, "wb") as f:
            f.write(content)

        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/download/download.txt",
            headers=headers,
        )
        assert response.status_code == 200
        assert response.content == content

    def test_download_workflow_file_not_found(self, client, headers, workspace_and_workflow):
        """Test downloading non-existent file returns 404."""
        workflow_id = workspace_and_workflow["workflow_id"]

        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/download/nonexistent.txt",
            headers=headers,
        )
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    def test_path_traversal_blocked(self, client, headers, workspace_and_workflow):
        """Test that path traversal attempts are blocked."""
        workflow_id = workspace_and_workflow["workflow_id"]

        # Try to access parent directory
        response = client.get(
            f"/api/workspaces/workflow/{workflow_id}/files",
            params={"path": "../"},
            headers=headers,
        )
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]
