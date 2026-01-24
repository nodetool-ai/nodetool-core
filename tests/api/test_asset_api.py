import os
import sys
import zipfile
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from nodetool.models.asset import Asset
from nodetool.runtime.resources import require_scope
from nodetool.types.asset import AssetCreateRequest, AssetUpdateRequest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from conftest import make_image, make_text

test_jpg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")

# Ensure all tests in this module run in the same xdist worker to prevent database race conditions
pytestmark = pytest.mark.xdist_group(name="database")


@pytest.mark.asyncio
async def test_index(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the GET /api/assets endpoint.
    Verifies that the endpoint returns a list of assets for the authenticated user.
    """
    image = await make_image(user_id)
    response = client.get("/api/assets", headers=headers)
    json_response = response.json()
    assert response.status_code == 200
    assert len(json_response["assets"]) == 1
    assert json_response["assets"][0]["id"] == image.id


@pytest.mark.asyncio
async def test_delete(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the DELETE /api/assets/{id} endpoint.
    Verifies that the asset is deleted from both the database and storage.
    """
    image = await make_image(user_id)
    response = client.delete(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert await Asset.find(user_id, image.id) is None
    storage = require_scope().get_asset_storage()
    assert not await storage.file_exists(image.file_name)


@pytest.mark.asyncio
async def test_pagination(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test pagination functionality of the GET /api/assets endpoint.
    Verifies that assets are properly paginated with correct page sizes and cursor behavior.
    """
    for _ in range(5):
        await make_image(user_id)
    response = client.get("/api/assets", headers=headers, params={"page_size": 3})
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 3
    next_cursor = response.json()["next"]
    response = client.get("/api/assets", params={"cursor": next_cursor, "page_size": 3}, headers=headers)
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 2
    assert response.json()["next"] == ""


@pytest.mark.asyncio
async def test_get(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the GET /api/assets/{id} endpoint.
    Verifies that a single asset can be retrieved by its ID.
    """
    image = await make_image(user_id)
    response = client.get(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == image.id


@pytest.mark.asyncio
async def test_put(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the PUT /api/assets/{id} endpoint.
    Verifies that an asset's metadata can be updated successfully.
    """
    image = await make_image(user_id)
    response = client.put(
        f"/api/assets/{image.id}",
        json=AssetUpdateRequest(parent_id=user_id, name="bild.jpeg", content_type="image/jpeg").model_dump(),
        headers=headers,
    )
    assert response.status_code == 200
    image_reloaded = await Asset.find(user_id, image.id)
    assert image_reloaded is not None


@pytest.mark.asyncio
async def test_create(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the POST /api/assets endpoint.
    Verifies that a new asset can be created with file upload and metadata.
    """
    with open(test_jpg, "rb") as file_handle:
        response = client.post(
            "/api/assets",
            files={"file": ("test.jpg", file_handle, "image/jpeg")},
            data={
                "json": AssetCreateRequest(
                    parent_id=user_id, name="bild.jpeg", content_type="image/jpeg"
                ).model_dump_json()
            },
            headers=headers,
        )
    assert response.status_code == 200
    image_reloaded = await Asset.find(user_id, response.json()["id"])
    assert image_reloaded is not None
    assert image_reloaded.name == "bild.jpeg"


@pytest.mark.asyncio
async def test_storage_stream_content_length(client: TestClient, headers: dict[str, str], user_id: str):
    image = await make_image(user_id)
    storage = require_scope().get_asset_storage()
    expected_size = len(storage.storage[image.file_name])

    response = client.get(f"/api/storage/{image.file_name}", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("Content-Length") == str(expected_size)


@pytest.mark.asyncio
async def test_download_deduplicated_names(client: TestClient, headers: dict[str, str], user_id: str):
    """Ensure duplicate asset names are uniquified in zip downloads."""
    img1 = await make_image(user_id)
    img2 = await make_image(user_id)

    response = client.post(
        "/api/assets/download",
        json={"asset_ids": [img1.id, img2.id]},
        headers=headers,
    )
    assert response.status_code == 200

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        names = zf.namelist()

    assert len(names) == 2
    assert len(set(names)) == 2


# Search functionality tests
@pytest.mark.asyncio
async def test_search_basic_functionality(client: TestClient, headers: dict[str, str], user_id: str):
    """Test basic search functionality with valid queries."""
    # Create test assets with different names
    img = await make_image(user_id)
    await img.update(name="sunset_photo")
    img = await make_image(user_id)
    await img.update(name="beach_vacation")
    txt = await make_text(user_id, "content")
    await txt.update(name="document.txt")

    # Test basic search
    response = client.get("/api/assets/search", params={"query": "photo"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_global_search"] is True
    assert len(data["assets"]) == 1
    assert data["assets"][0]["name"] == "sunset_photo"


@pytest.mark.asyncio
async def test_search_query_validation(client: TestClient, headers: dict[str, str], user_id: str):
    """Test search query length validation."""
    # Test query too short
    response = client.get("/api/assets/search", params={"query": "a"}, headers=headers)
    assert response.status_code == 400
    assert "at least 2 characters" in response.json()["detail"]

    # Test minimum valid length
    response = client.get("/api/assets/search", params={"query": "ab"}, headers=headers)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_with_content_type_filter(client: TestClient, headers: dict[str, str], user_id: str):
    """Test search with content type filtering."""
    # Create assets of different types
    img = await make_image(user_id)
    await img.update(name="test_image")
    txt = await make_text(user_id, "content")
    await txt.update(name="test_document")

    # Search for images only
    response = client.get(
        "/api/assets/search",
        params={"query": "test", "content_type": "image"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["content_type"].startswith("image")


@pytest.mark.asyncio
async def test_search_pagination(client: TestClient, headers: dict[str, str], user_id: str):
    """Test search pagination functionality."""
    # Create multiple matching assets
    for i in range(5):
        img = await make_image(user_id)
        await img.update(name=f"photo_{i}")

    # Test first page
    response = client.get("/api/assets/search", params={"query": "photo", "page_size": 2}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 2
    assert data["total_count"] == 2

    # Test with cursor if provided
    if data["next_cursor"]:
        response = client.get(
            "/api/assets/search",
            params={"query": "photo", "page_size": 2, "cursor": data["next_cursor"]},
            headers=headers,
        )
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_folder_path_information(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that search returns folder path information."""
    # Create folder structure
    folder = await Asset.create(user_id=user_id, name="My Folder", content_type="folder")
    subfolder = await Asset.create(user_id=user_id, name="Sub Folder", content_type="folder", parent_id=folder.id)

    # Create asset in subfolder
    image = await make_image(user_id, parent_id=subfolder.id)
    await image.update(name="nested_photo")

    response = client.get("/api/assets/search", params={"query": "photo"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1

    asset = data["assets"][0]
    assert "folder_name" in asset
    assert "folder_path" in asset
    assert "folder_id" in asset
    # Should include breadcrumb path
    assert "Sub Folder" in asset["folder_path"]


@pytest.mark.asyncio
async def test_search_empty_results(client: TestClient, headers: dict[str, str], user_id: str):
    """Test search with no matching results."""
    img = await make_image(user_id)
    await img.update(name="sunset")

    response = client.get("/api/assets/search", params={"query": "nonexistent"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 0
    assert data["total_count"] == 0


@pytest.mark.asyncio
async def test_search_case_insensitive(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that search is case insensitive."""
    img = await make_image(user_id)
    await img.update(name="SUNSET_Photo")

    # Test lowercase search
    response = client.get("/api/assets/search", params={"query": "sunset"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1


@pytest.mark.asyncio
async def test_search_special_characters(client: TestClient, headers: dict[str, str], user_id: str):
    """Test search with special characters that could cause SQL injection."""
    img = await make_image(user_id)
    await img.update(name="test_file")

    # Test SQL injection attempts
    malicious_queries = ["'; DROP TABLE assets; --", "test%", "test_", "test\\"]

    for query in malicious_queries:
        response = client.get("/api/assets/search", params={"query": query}, headers=headers)
        # Should not crash and should return proper response
        assert response.status_code == 200
        # Response should be well-formed JSON
        data = response.json()
        assert "assets" in data


@pytest.mark.asyncio
async def test_search_user_isolation(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that search only returns current user's assets."""
    # Create asset for current user
    img = await make_image(user_id)
    await img.update(name="my_photo")

    # Create asset for different user
    other_user_id = "other_user"
    Asset.create(user_id=other_user_id, name="other_photo", content_type="image/jpeg")

    # Search should only return current user's assets
    response = client.get("/api/assets/search", params={"query": "photo"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["name"] == "my_photo"
    assert data["assets"][0]["user_id"] == user_id


@pytest.mark.asyncio
async def test_search_contains_behavior(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that search finds matches anywhere in the filename."""
    # Create assets with query in different positions
    img = await make_image(user_id)
    await img.update(name="photo_sunset")  # at beginning
    img = await make_image(user_id)
    await img.update(name="beautiful_photo_vacation")  # in middle
    img = await make_image(user_id)
    await img.update(name="vacation_photo")  # at end

    response = client.get("/api/assets/search", params={"query": "photo"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    # Should find all three
    assert len(data["assets"]) == 3


@pytest.mark.asyncio
async def test_create_asset_with_node_and_job_id(client: TestClient, headers: dict[str, str], user_id: str):
    """Test creating an asset with node_id and job_id via API."""
    node_id = "test_node_api_123"
    job_id = "test_job_api_456"
    
    with open(test_jpg, "rb") as file_handle:
        response = client.post(
            "/api/assets",
            files={"file": ("test.jpg", file_handle, "image/jpeg")},
            data={
                "json": AssetCreateRequest(
                    parent_id=user_id,
                    name="test_with_ids.jpeg",
                    content_type="image/jpeg",
                    node_id=node_id,
                    job_id=job_id,
                ).model_dump_json()
            },
            headers=headers,
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["node_id"] == node_id
    assert data["job_id"] == job_id
    
    # Verify in database
    asset = await Asset.find(user_id, data["id"])
    assert asset is not None
    assert asset.node_id == node_id
    assert asset.job_id == job_id


@pytest.mark.asyncio
async def test_filter_assets_by_node_id(client: TestClient, headers: dict[str, str], user_id: str):
    """Test filtering assets by node_id via API."""
    node_id = "test_node_filter"
    
    # Create asset with node_id
    asset_with_node = await Asset.create(
        user_id=user_id,
        name="asset_with_node",
        content_type="image/jpeg",
        node_id=node_id,
    )
    
    # Create asset without node_id
    await Asset.create(
        user_id=user_id,
        name="asset_without_node",
        content_type="image/jpeg",
    )
    
    # Filter by node_id
    response = client.get(
        "/api/assets",
        params={"node_id": node_id},
        headers=headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["id"] == asset_with_node.id
    assert data["assets"][0]["node_id"] == node_id


@pytest.mark.asyncio
async def test_filter_assets_by_job_id(client: TestClient, headers: dict[str, str], user_id: str):
    """Test filtering assets by job_id via API."""
    job_id = "test_job_filter"
    
    # Create asset with job_id
    asset_with_job = await Asset.create(
        user_id=user_id,
        name="asset_with_job",
        content_type="image/jpeg",
        job_id=job_id,
    )
    
    # Create asset without job_id
    await Asset.create(
        user_id=user_id,
        name="asset_without_job",
        content_type="image/jpeg",
    )
    
    # Filter by job_id
    response = client.get(
        "/api/assets",
        params={"job_id": job_id},
        headers=headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["id"] == asset_with_job.id
    assert data["assets"][0]["job_id"] == job_id


@pytest.mark.asyncio
async def test_filter_assets_by_workflow_id(client: TestClient, headers: dict[str, str], user_id: str):
    """Test filtering assets by workflow_id via API."""
    workflow_id = "test_workflow_filter"
    
    # Create asset with workflow_id
    asset_with_workflow = await Asset.create(
        user_id=user_id,
        name="asset_with_workflow",
        content_type="image/jpeg",
        workflow_id=workflow_id,
    )
    
    # Create asset without workflow_id
    await Asset.create(
        user_id=user_id,
        name="asset_without_workflow",
        content_type="image/jpeg",
    )
    
    # Filter by workflow_id
    response = client.get(
        "/api/assets",
        params={"workflow_id": workflow_id},
        headers=headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["id"] == asset_with_workflow.id
    assert data["assets"][0]["workflow_id"] == workflow_id


@pytest.mark.asyncio
async def test_filter_assets_by_multiple_criteria(client: TestClient, headers: dict[str, str], user_id: str):
    """Test filtering assets by multiple criteria simultaneously."""
    workflow_id = "test_workflow_multi"
    node_id = "test_node_multi"
    job_id = "test_job_multi"
    
    # Create asset matching all criteria
    asset_match = await Asset.create(
        user_id=user_id,
        name="asset_match_all",
        content_type="image/jpeg",
        workflow_id=workflow_id,
        node_id=node_id,
        job_id=job_id,
    )
    
    # Create assets matching only some criteria
    await Asset.create(
        user_id=user_id,
        name="asset_partial",
        content_type="image/jpeg",
        workflow_id=workflow_id,
        node_id=node_id,
    )
    
    # Filter by all criteria
    response = client.get(
        "/api/assets",
        params={
            "workflow_id": workflow_id,
            "node_id": node_id,
            "job_id": job_id,
        },
        headers=headers,
    )
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["assets"]) == 1
    assert data["assets"][0]["id"] == asset_match.id
