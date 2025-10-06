import pytest
from nodetool.models.thread import Thread


@pytest.mark.asyncio
async def test_find_thread(user_id: str):
    thread = await Thread.create(
        user_id=user_id,
    )

    found_thread = await Thread.get(thread.id)

    if found_thread:
        assert thread.id == found_thread.id
    else:
        pytest.fail("Thread not found")

    # Test finding a thread that does not exist in the database
    not_found_thread = await Thread.get("invalid_id")
    assert not_found_thread is None


@pytest.mark.asyncio
async def test_paginate_threads(user_id: str):
    await Thread.create(user_id=user_id)

    threads, last_key = await Thread.paginate(user_id=user_id, limit=10)
    assert len(threads) > 0
    assert last_key == ""


@pytest.mark.asyncio
async def test_create_thread(user_id: str):
    thread = await Thread.create(
        user_id=user_id,
    )

    assert await Thread.get(thread.id) is not None


@pytest.mark.asyncio
async def test_create_thread_with_custom_id(user_id: str):
    custom_id = "custom-thread-id-123"
    thread = await Thread.create(
        user_id=user_id,
        id=custom_id,
    )

    assert thread.id == custom_id
    found_thread = await Thread.get(custom_id)
    assert found_thread is not None
    assert found_thread.id == custom_id
    assert found_thread.user_id == user_id
