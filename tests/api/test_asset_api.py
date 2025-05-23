import os
from fastapi.testclient import TestClient
import pytest
from nodetool.common.environment import Environment
from nodetool.models.asset import Asset
from nodetool.types.asset import AssetCreateRequest, AssetUpdateRequest
from conftest import make_image


test_jpg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")


def test_index(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the GET /api/assets endpoint.
    Verifies that the endpoint returns a list of assets for the authenticated user.
    """
    image = make_image(user_id)
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
    image = make_image(user_id)
    response = client.delete(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert Asset.find(user_id, image.id) is None
    assert not await Environment.get_asset_storage().file_exists(image.file_name)


def test_pagination(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test pagination functionality of the GET /api/assets endpoint.
    Verifies that assets are properly paginated with correct page sizes and cursor behavior.
    """
    for _ in range(5):
        make_image(user_id)
    response = client.get("/api/assets", headers=headers, params={"page_size": 3})
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 3
    next_cursor = response.json()["next"]
    response = client.get(
        "/api/assets", params={"cursor": next_cursor, "page_size": 3}, headers=headers
    )
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 2
    assert response.json()["next"] == ""


def test_get(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the GET /api/assets/{id} endpoint.
    Verifies that a single asset can be retrieved by its ID.
    """
    image = make_image(user_id)
    response = client.get(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == image.id


def test_put(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the PUT /api/assets/{id} endpoint.
    Verifies that an asset's metadata can be updated successfully.
    """
    image = make_image(user_id)
    response = client.put(
        f"/api/assets/{image.id}",
        json=AssetUpdateRequest(
            parent_id=user_id, name="bild.jpeg", content_type="image/jpeg"
        ).model_dump(),
        headers=headers,
    )
    assert response.status_code == 200
    image_reloaded = Asset.find(user_id, image.id)
    assert image_reloaded is not None


def test_create(client: TestClient, headers: dict[str, str], user_id: str):
    """
    Test the POST /api/assets endpoint.
    Verifies that a new asset can be created with file upload and metadata.
    """
    response = client.post(
        "/api/assets",
        files={"file": ("test.jpg", open(test_jpg, "rb"), "image/jpeg")},
        data={
            "json": AssetCreateRequest(
                parent_id=user_id, name="bild.jpeg", content_type="image/jpeg"
            ).model_dump_json()
        },
        headers=headers,
    )
    assert response.status_code == 200
    image_reloaded = Asset.find(user_id, response.json()["id"])
    assert image_reloaded is not None
    assert image_reloaded.name == "bild.jpeg"


def test_storage_stream_content_length(client: TestClient, user_id: str):
    image = make_image(user_id)
    storage = Environment.get_asset_storage()
    expected_size = len(storage.storage[image.file_name])

    response = client.get(f"/api/storage/{image.file_name}")
    assert response.status_code == 200
    assert response.headers.get("Content-Length") == str(expected_size)
