from datetime import datetime

import pytest

from nodetool.models.asset import (
    Asset,
)
from nodetool.runtime.resources import require_scope
from tests.conftest import make_image

# Ensure all tests in this module run in the same xdist worker to prevent database race conditions
pytestmark = pytest.mark.xdist_group(name="database")


@pytest.mark.asyncio
async def test_asset_find(user_id: str):
    asset = await make_image(user_id)

    found_asset = await Asset.find(user_id, asset.id)

    if found_asset:
        assert asset.id == found_asset.id
    else:
        pytest.fail("Asset not found")

    # Test finding an asset that does not exist in the database
    not_found_asset = await Asset.find(user_id, "invalid_id")
    assert not_found_asset is None


@pytest.mark.asyncio
async def test_paginate_assets(user_id: str):
    for _i in range(5):
        await Asset.create(
            user_id=user_id,
            name="test_image",
            content_type="image/jpeg",
        )

    assets, _last_key = await Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0

    assets, _last_key = await Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0


@pytest.mark.asyncio
async def test_paginate_assets_by_parent(user_id: str):
    # Assuming parent_id should also be user_id based on previous code?
    # This test logic might need review as parent_id is usually another asset ID.
    # Keeping user_id for now as per direct refactoring.
    parent_id_to_use = user_id

    for i in range(5):
        # Pass user_id to make_image
        await make_image(user_id, parent_id=parent_id_to_use if i == 0 else None)

    assets, _last_key = await Asset.paginate(user_id=user_id, parent_id=parent_id_to_use, limit=4)
    assert len(assets) > 0

    # This paginate call seems unrelated to the parent_id logic above, keeping as is with user_id
    assets, _last_key = await Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0


@pytest.mark.asyncio
async def test_create_asset(user_id: str):
    asset = await Asset.create(
        user_id=user_id,
        name="test_asset",
        content_type="image/jpeg",
    )

    assert asset.name == "test_asset"
    assert asset.content_type == "image/jpeg"


@pytest.mark.asyncio
async def test_created_at(user_id: str):
    asset = await Asset.create(
        user_id=user_id,
        name="test_asset",
        content_type="image/jpeg",
    )

    assert asset.created_at is not None
    assert isinstance(asset.created_at, datetime)


@pytest.mark.asyncio
async def test_create_asset_with_node_and_job_id(user_id: str):
    """Test creating an asset with node_id and job_id."""
    node_id = "test_node_123"
    job_id = "test_job_456"
    
    asset = await Asset.create(
        user_id=user_id,
        name="test_asset_with_ids",
        content_type="image/jpeg",
        node_id=node_id,
        job_id=job_id,
    )

    assert asset.name == "test_asset_with_ids"
    assert asset.node_id == node_id
    assert asset.job_id == job_id


@pytest.mark.asyncio
async def test_paginate_assets_by_node_id(user_id: str):
    """Test filtering assets by node_id."""
    node_id = "test_node_789"
    
    # Create assets with and without node_id
    await Asset.create(
        user_id=user_id,
        name="asset_with_node",
        content_type="image/jpeg",
        node_id=node_id,
    )
    await Asset.create(
        user_id=user_id,
        name="asset_without_node",
        content_type="image/jpeg",
    )

    # Query by node_id
    assets, _last_key = await Asset.paginate(user_id=user_id, node_id=node_id, limit=10)
    
    assert len(assets) == 1
    assert assets[0].name == "asset_with_node"
    assert assets[0].node_id == node_id


@pytest.mark.asyncio
async def test_paginate_assets_by_job_id(user_id: str):
    """Test filtering assets by job_id."""
    job_id = "test_job_999"
    
    # Create assets with and without job_id
    await Asset.create(
        user_id=user_id,
        name="asset_with_job",
        content_type="image/jpeg",
        job_id=job_id,
    )
    await Asset.create(
        user_id=user_id,
        name="asset_without_job",
        content_type="image/jpeg",
    )

    # Query by job_id
    assets, _last_key = await Asset.paginate(user_id=user_id, job_id=job_id, limit=10)
    
    assert len(assets) == 1
    assert assets[0].name == "asset_with_job"
    assert assets[0].job_id == job_id


@pytest.mark.asyncio
async def test_paginate_assets_by_node_and_job_id(user_id: str):
    """Test filtering assets by both node_id and job_id."""
    node_id = "test_node_combined"
    job_id = "test_job_combined"
    
    # Create various combinations
    await Asset.create(
        user_id=user_id,
        name="asset_both_ids",
        content_type="image/jpeg",
        node_id=node_id,
        job_id=job_id,
    )
    await Asset.create(
        user_id=user_id,
        name="asset_only_node",
        content_type="image/jpeg",
        node_id=node_id,
    )
    await Asset.create(
        user_id=user_id,
        name="asset_only_job",
        content_type="image/jpeg",
        job_id=job_id,
    )

    # Query by both node_id and job_id
    assets, _last_key = await Asset.paginate(
        user_id=user_id, 
        node_id=node_id, 
        job_id=job_id, 
        limit=10
    )
    
    assert len(assets) == 1
    assert assets[0].name == "asset_both_ids"
    assert assets[0].node_id == node_id
    assert assets[0].job_id == job_id


# Search functionality model tests
@pytest.mark.asyncio
async def test_search_assets_global_basic(user_id: str):
    """Test basic search functionality in the model layer."""
    # Create test assets
    await Asset.create(user_id=user_id, name="sunset_photo", content_type="image/jpeg")
    await Asset.create(user_id=user_id, name="beach_vacation", content_type="image/jpeg")
    await Asset.create(user_id=user_id, name="document.txt", content_type="text/plain")

    # Test search
    assets, _next_cursor, folder_paths = await Asset.search_assets_global(user_id=user_id, query="photo")

    assert len(assets) == 1
    assert assets[0].name == "sunset_photo"
    assert len(folder_paths) == 1
    assert folder_paths[0]["folder_name"] == "Home"


@pytest.mark.asyncio
async def test_search_assets_global_contains_search(user_id: str):
    """Test that search finds matches anywhere in filename."""
    # Create assets with query in different positions
    await Asset.create(user_id=user_id, name="photo_sunset", content_type="image/jpeg")
    await Asset.create(user_id=user_id, name="beautiful_photo_vacation", content_type="image/jpeg")
    await Asset.create(user_id=user_id, name="vacation_photo", content_type="image/jpeg")

    assets, _, _ = await Asset.search_assets_global(user_id=user_id, query="photo")

    # Should find all three
    assert len(assets) == 3
    names = [asset.name for asset in assets]
    assert "photo_sunset" in names
    assert "beautiful_photo_vacation" in names
    assert "vacation_photo" in names


@pytest.mark.asyncio
async def test_search_assets_global_content_type_filter(user_id: str):
    """Test search with content type filtering."""
    await Asset.create(user_id=user_id, name="test_image", content_type="image/jpeg")
    await Asset.create(user_id=user_id, name="test_document", content_type="text/plain")

    # Search for images only
    assets, _, _ = await Asset.search_assets_global(user_id=user_id, query="test", content_type="image")

    assert len(assets) == 1
    assert assets[0].content_type == "image/jpeg"


@pytest.mark.asyncio
async def test_search_assets_global_user_isolation(user_id: str):
    """Test that search only returns current user's assets."""
    # Create asset for current user
    await Asset.create(user_id=user_id, name="my_photo", content_type="image/jpeg")

    # Create asset for different user
    other_user_id = "other_user"
    await Asset.create(user_id=other_user_id, name="other_photo", content_type="image/jpeg")

    assets, _, _ = await Asset.search_assets_global(user_id=user_id, query="photo")

    assert len(assets) == 1
    assert assets[0].name == "my_photo"
    assert assets[0].user_id == user_id


@pytest.mark.asyncio
async def test_search_assets_global_sql_injection_protection(user_id: str):
    """Test that search properly sanitizes input to prevent SQL injection."""
    await Asset.create(user_id=user_id, name="test_file", content_type="image/jpeg")

    # Test various injection attempts
    malicious_queries = ["'; DROP TABLE assets; --", "test%", "test_", "test\\"]

    for query in malicious_queries:
        # Should not crash
        assets, _next_cursor, folder_paths = await Asset.search_assets_global(user_id=user_id, query=query)
        # Should return well-formed results
        assert isinstance(assets, list)
        assert isinstance(folder_paths, list)


@pytest.mark.asyncio
async def test_search_assets_global_empty_results(user_id: str):
    """Test search with no matching results."""
    await Asset.create(user_id=user_id, name="sunset", content_type="image/jpeg")

    assets, _next_cursor, folder_paths = await Asset.search_assets_global(user_id=user_id, query="nonexistent")

    assert len(assets) == 0
    assert len(folder_paths) == 0


@pytest.mark.asyncio
async def test_search_assets_global_pagination(user_id: str):
    """Test search pagination."""
    # Create multiple matching assets
    for i in range(5):
        await Asset.create(user_id=user_id, name=f"photo_{i}", content_type="image/jpeg")

    # Test limited results
    assets, _next_cursor, folder_paths = await Asset.search_assets_global(user_id=user_id, query="photo", limit=2)

    assert len(assets) == 2
    assert len(folder_paths) == 2


@pytest.mark.asyncio
async def test_get_asset_path_info_batch_performance(user_id: str):
    """Test that get_asset_path_info handles multiple assets efficiently."""
    # Create folder structure
    folder = await Asset.create(user_id=user_id, name="My Folder", content_type="folder")
    subfolder = await Asset.create(user_id=user_id, name="Sub Folder", content_type="folder", parent_id=folder.id)

    # Create assets in different locations
    asset1 = await Asset.create(user_id=user_id, name="root_asset", content_type="image/jpeg")
    asset2 = await Asset.create(
        user_id=user_id,
        name="folder_asset",
        content_type="image/jpeg",
        parent_id=folder.id,
    )
    asset3 = await Asset.create(
        user_id=user_id,
        name="subfolder_asset",
        content_type="image/jpeg",
        parent_id=subfolder.id,
    )

    # Test batch path info retrieval
    asset_ids = [asset1.id, asset2.id, asset3.id]
    path_info = await Asset.get_asset_path_info(user_id, asset_ids)

    assert len(path_info) == 3

    # Check root asset
    assert path_info[asset1.id]["folder_name"] == "Home"
    assert path_info[asset1.id]["folder_path"] == "Home"

    # Check folder asset
    assert path_info[asset2.id]["folder_name"] == "My Folder"
    assert "My Folder" in path_info[asset2.id]["folder_path"]

    # Check subfolder asset
    assert path_info[asset3.id]["folder_name"] == "Sub Folder"
    assert "Sub Folder" in path_info[asset3.id]["folder_path"]


@pytest.mark.asyncio
async def test_get_asset_path_info_fallback_handling(user_id: str):
    """Test fallback behavior for get_asset_path_info."""
    # Test with empty list
    path_info = await Asset.get_asset_path_info(user_id, [])
    assert path_info == {}

    # Test with non-existent asset
    path_info = await Asset.get_asset_path_info(user_id, ["nonexistent_id"])
    assert path_info == {}


@pytest.mark.asyncio
async def test_get_asset_path_info_nested_folders(user_id: str):
    """Test path info for deeply nested folder structures."""
    # Create deep folder structure
    folder1 = await Asset.create(user_id=user_id, name="Folder1", content_type="folder")
    folder2 = await Asset.create(user_id=user_id, name="Folder2", content_type="folder", parent_id=folder1.id)
    folder3 = await Asset.create(user_id=user_id, name="Folder3", content_type="folder", parent_id=folder2.id)

    # Create asset in deeply nested folder
    asset = await Asset.create(
        user_id=user_id,
        name="deep_asset",
        content_type="image/jpeg",
        parent_id=folder3.id,
    )

    path_info = await Asset.get_asset_path_info(user_id, [asset.id])

    assert len(path_info) == 1
    assert asset.id in path_info

    # Should include full path breadcrumb
    path = path_info[asset.id]["folder_path"]
    assert "Home" in path
    assert "Folder1" in path
    assert "Folder2" in path
    assert "Folder3" in path

    # Should show immediate parent
    assert path_info[asset.id]["folder_name"] == "Folder3"
