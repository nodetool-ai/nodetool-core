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
