from datetime import datetime
import pytest
from nodetool.models.asset import (
    Asset,
)
from conftest import make_image


def test_asset_find(user_id: str):
    asset = make_image(user_id)

    found_asset = Asset.find(user_id, asset.id)

    if found_asset:
        assert asset.id == found_asset.id
    else:
        pytest.fail("Asset not found")

    # Test finding an asset that does not exist in the database
    not_found_asset = Asset.find(user_id, "invalid_id")
    assert not_found_asset is None


def test_paginate_assets(user_id: str):
    for i in range(5):
        Asset.create(
            user_id=user_id,
            name="test_image",
            content_type="image/jpeg",
        )

    assets, last_key = Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0

    assets, last_key = Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0


def test_paginate_assets_by_parent(user_id: str):
    # Assuming parent_id should also be user_id based on previous code?
    # This test logic might need review as parent_id is usually another asset ID.
    # Keeping user_id for now as per direct refactoring.
    parent_id_to_use = user_id

    for i in range(5):
        # Pass user_id to make_image
        make_image(user_id, parent_id=parent_id_to_use if i == 0 else None)

    assets, last_key = Asset.paginate(
        user_id=user_id, parent_id=parent_id_to_use, limit=4
    )
    assert len(assets) > 0

    # This paginate call seems unrelated to the parent_id logic above, keeping as is with user_id
    assets, last_key = Asset.paginate(user_id=user_id, content_type="image", limit=3)
    assert len(assets) > 0


def test_create_asset(user_id: str):
    asset = Asset.create(
        user_id=user_id,
        name="test_asset",
        content_type="image/jpeg",
    )

    assert asset.name == "test_asset"
    assert asset.content_type == "image/jpeg"


def test_created_at(user_id: str):
    asset = Asset.create(
        user_id=user_id,
        name="test_asset",
        content_type="image/jpeg",
    )

    assert asset.created_at is not None
    assert isinstance(asset.created_at, datetime)
