"""Tests for mock data generation."""

import pytest

from nodetool.api.mock_data import (
    generate_mock_assets,
    generate_mock_messages,
    generate_mock_threads,
    generate_mock_workflows,
    populate_mock_data,
)
from nodetool.models.asset import Asset
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow


@pytest.mark.asyncio
async def test_generate_mock_threads():
    """Test generating mock threads."""
    threads = await generate_mock_threads(user_id="test_user", count=3)
    
    assert len(threads) == 3
    for thread in threads:
        assert thread.user_id == "test_user"
        assert thread.title is not None
        assert thread.id is not None
    
    # Verify threads were saved to database
    saved_thread = await Thread.get(threads[0].id)
    assert saved_thread is not None
    assert saved_thread.title == threads[0].title


@pytest.mark.asyncio
async def test_generate_mock_messages():
    """Test generating mock messages."""
    # First create a thread
    thread = await Thread.create(user_id="test_user", title="Test Thread")
    
    # Generate messages
    messages = await generate_mock_messages(
        thread_id=thread.id, user_id="test_user", count=6
    )
    
    assert len(messages) == 6
    
    # Check alternating roles
    for i, message in enumerate(messages):
        if i % 2 == 0:
            assert message.role == "user"
        else:
            assert message.role == "assistant"
            assert message.provider is not None
            assert message.model is not None
    
    # Verify messages were saved
    saved_message = await Message.get(messages[0].id)
    assert saved_message is not None


@pytest.mark.asyncio
async def test_generate_mock_workflows():
    """Test generating mock workflows."""
    workflows = await generate_mock_workflows(user_id="test_user", count=3)
    
    assert len(workflows) == 3
    for workflow in workflows:
        assert workflow.user_id == "test_user"
        assert workflow.name is not None
        assert workflow.description is not None
        assert "mock" in workflow.tags
        assert "example" in workflow.tags
        assert "nodes" in workflow.graph
        assert "edges" in workflow.graph
    
    # Verify workflows were saved
    saved_workflow = await Workflow.get(workflows[0].id)
    assert saved_workflow is not None
    assert saved_workflow.name == workflows[0].name


@pytest.mark.asyncio
async def test_generate_mock_assets():
    """Test generating mock assets."""
    assets = await generate_mock_assets(user_id="test_user", count=5)
    
    # Should have folder + images (if PIL available) + text files
    assert len(assets) > 1
    
    # First asset should be folder
    assert assets[0].content_type == "folder"
    assert assets[0].name == "Mock Data"
    
    # Verify assets were saved
    saved_asset = await Asset.get(assets[0].id)
    assert saved_asset is not None
    assert saved_asset.name == assets[0].name


@pytest.mark.asyncio
async def test_populate_mock_data():
    """Test populating all mock data."""
    result = await populate_mock_data(user_id="test_user")
    
    assert result["threads"] > 0
    assert result["messages"] > 0
    assert result["workflows"] > 0
    assert result["assets"] > 0
    # Collections might be 0 if ChromaDB is not available
    assert result["collections"] >= 0
    
    # Verify data exists in database
    threads = await Thread.query(limit=10)
    assert len(threads[0]) > 0
    
    workflows = await Workflow.query(limit=10)
    assert len(workflows[0]) > 0
