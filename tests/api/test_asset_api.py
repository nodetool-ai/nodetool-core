import json
import os
from fastapi.testclient import TestClient
import pytest
from nodetool.common.environment import Environment
from nodetool.models.asset import Asset
from nodetool.models.user import User
from nodetool.types.asset import AssetCreateRequest, AssetUpdateRequest
from conftest import make_image


test_jpg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.jpg")


def test_index(client: TestClient, headers: dict[str, str], user: User):
    """
    Test the GET /api/assets endpoint.
    Verifies that the endpoint returns a list of assets for the authenticated user.
    """
    image = make_image(user)
    response = client.get("/api/assets", headers=headers)
    json = response.json()
    assert response.status_code == 200
    assert len(json["assets"]) == 1
    assert json["assets"][0]["id"] == image.id


@pytest.mark.asyncio
async def test_delete(client: TestClient, headers: dict[str, str], user: User):
    """
    Test the DELETE /api/assets/{id} endpoint.
    Verifies that the asset is deleted from both the database and storage.
    """
    image = make_image(user)
    response = client.delete(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert Asset.find(user.id, image.id) is None
    assert not await Environment.get_asset_storage().file_exists(image.file_name)


def test_pagination(client: TestClient, headers: dict[str, str], user: User):
    """
    Test pagination functionality of the GET /api/assets endpoint.
    Verifies that assets are properly paginated with correct page sizes and cursor behavior.
    """
    for _ in range(5):
        make_image(user)
    response = client.get("/api/assets", headers=headers, params={"page_size": 3})
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 3
    next_cursor = response.json()["next"]
    response = client.get(
        f"/api/assets", params={"cursor": next_cursor, "page_size": 3}, headers=headers
    )
    assert response.status_code == 200
    assert len(response.json()["assets"]) == 2
    assert response.json()["next"] == ""


def test_get(client: TestClient, headers: dict[str, str], user: User):
    """
    Test the GET /api/assets/{id} endpoint.
    Verifies that a single asset can be retrieved by its ID.
    """
    image = make_image(user)
    response = client.get(f"/api/assets/{image.id}", headers=headers)
    assert response.status_code == 200
    assert response.json()["id"] == image.id


def test_put(client: TestClient, headers: dict[str, str], user: User):
    """
    Test the PUT /api/assets/{id} endpoint.
    Verifies that an asset's metadata can be updated successfully.
    """
    image = make_image(user)
    response = client.put(
        f"/api/assets/{image.id}",
        json=AssetUpdateRequest(
            parent_id=user.id, name="bild.jpeg", content_type="image/jpeg"
        ).model_dump(),
        headers=headers,
    )
    assert response.status_code == 200
    image = Asset.find(user.id, image.id)
    assert image is not None


def test_create(client: TestClient, headers: dict[str, str], user: User):
    """
    Test the POST /api/assets endpoint.
    Verifies that a new asset can be created with file upload and metadata.
    """
    response = client.post(
        "/api/assets",
        files={"file": ("test.jpg", open(test_jpg, "rb"), "image/jpeg")},
        data={
            "json": AssetCreateRequest(
                parent_id=user.id, name="bild.jpeg", content_type="image/jpeg"
            ).model_dump_json()
        },
        headers=headers,
    )
    assert response.status_code == 200
    image = Asset.find(user.id, response.json()["id"])
    assert image is not None
    assert image.name == "bild.jpeg"
