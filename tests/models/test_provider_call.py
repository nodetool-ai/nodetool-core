import pytest
from datetime import datetime

from nodetool.models.provider_call import ProviderCall


@pytest.mark.asyncio
async def test_create_provider_call(user_id: str):
    """Test creating a provider call record."""
    call = await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    assert call.id is not None
    assert call.user_id == user_id
    assert call.provider == "openai"
    assert call.model_id == "gpt-4o-mini"
    assert call.cost == 0.05
    assert call.input_tokens == 100
    assert call.output_tokens == 50
    assert call.total_tokens == 150
    assert call.created_at is not None


@pytest.mark.asyncio
async def test_get_provider_call(user_id: str):
    """Test retrieving a provider call record."""
    call = await ProviderCall.create(
        user_id=user_id,
        provider="anthropic",
        model_id="claude-3-opus",
        cost=0.10,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
    )

    retrieved = await ProviderCall.get(call.id)
    assert retrieved is not None
    assert retrieved.id == call.id
    assert retrieved.user_id == user_id
    assert retrieved.provider == "anthropic"
    assert retrieved.model_id == "claude-3-opus"


@pytest.mark.asyncio
async def test_paginate_provider_calls(user_id: str):
    """Test paginating provider calls."""
    # Create multiple calls
    for i in range(5):
        await ProviderCall.create(
            user_id=user_id,
            provider="openai",
            model_id=f"gpt-4o-mini-{i}",
            cost=0.01 * i,
            input_tokens=10 * i,
            output_tokens=5 * i,
            total_tokens=15 * i,
        )

    calls, next_key = await ProviderCall.paginate(user_id=user_id, limit=3)
    assert len(calls) >= 3  # At least 3 calls
    assert all(call.user_id == user_id for call in calls)


@pytest.mark.asyncio
async def test_paginate_with_provider_filter(user_id: str):
    """Test filtering provider calls by provider."""
    # Create calls for different providers
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    await ProviderCall.create(
        user_id=user_id,
        provider="anthropic",
        model_id="claude-3-opus",
        cost=0.10,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
    )

    # Filter by OpenAI
    calls, _ = await ProviderCall.paginate(user_id=user_id, provider="openai", limit=10)
    assert len(calls) >= 1
    assert all(call.provider == "openai" for call in calls)


@pytest.mark.asyncio
async def test_paginate_with_model_filter(user_id: str):
    """Test filtering provider calls by model."""
    # Create calls for different models
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o",
        cost=0.15,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    # Filter by model
    calls, _ = await ProviderCall.paginate(user_id=user_id, model_id="gpt-4o-mini", limit=10)
    assert len(calls) >= 1
    assert all(call.model_id == "gpt-4o-mini" for call in calls)


@pytest.mark.asyncio
async def test_aggregate_by_user(user_id: str):
    """Test aggregating costs by user."""
    # Create multiple calls
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.03,
        input_tokens=60,
        output_tokens=30,
        total_tokens=90,
    )

    # Aggregate
    result = await ProviderCall.aggregate_by_user(user_id=user_id)
    assert result["user_id"] == user_id
    assert result["total_cost"] >= 0.08  # At least the two calls we created
    assert result["total_tokens"] >= 240  # At least 150 + 90
    assert result["call_count"] >= 2


@pytest.mark.asyncio
async def test_aggregate_by_provider(user_id: str):
    """Test aggregating costs by provider."""
    # Create calls for different providers
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    await ProviderCall.create(
        user_id=user_id,
        provider="anthropic",
        model_id="claude-3-opus",
        cost=0.10,
        input_tokens=200,
        output_tokens=100,
        total_tokens=300,
    )

    # Aggregate by provider
    results = await ProviderCall.aggregate_by_provider(user_id=user_id)
    assert len(results) >= 2
    
    # Find OpenAI and Anthropic results
    openai_result = next((r for r in results if r["provider"] == "openai"), None)
    anthropic_result = next((r for r in results if r["provider"] == "anthropic"), None)
    
    assert openai_result is not None
    assert anthropic_result is not None
    assert openai_result["total_cost"] >= 0.05
    assert anthropic_result["total_cost"] >= 0.10


@pytest.mark.asyncio
async def test_aggregate_by_model(user_id: str):
    """Test aggregating costs by model."""
    # Create calls for different models
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o",
        cost=0.15,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    # Aggregate by model
    results = await ProviderCall.aggregate_by_model(user_id=user_id)
    assert len(results) >= 2
    
    # Find specific model results
    mini_result = next((r for r in results if r["model_id"] == "gpt-4o-mini"), None)
    regular_result = next((r for r in results if r["model_id"] == "gpt-4o"), None)
    
    assert mini_result is not None
    assert regular_result is not None
    assert mini_result["total_cost"] >= 0.05
    assert regular_result["total_cost"] >= 0.15


@pytest.mark.asyncio
async def test_create_with_metadata(user_id: str):
    """Test creating a provider call with metadata."""
    metadata = {
        "workflow_id": "wf123",
        "node_id": "node456",
        "custom_field": "value",
    }

    call = await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        metadata=metadata,
    )

    assert call.metadata == metadata
    
    # Retrieve and verify
    retrieved = await ProviderCall.get(call.id)
    assert retrieved is not None
    assert retrieved.metadata == metadata


@pytest.mark.asyncio
async def test_create_with_optional_tokens(user_id: str):
    """Test creating a provider call with optional token fields."""
    call = await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o",
        cost=0.10,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        cached_tokens=20,
        reasoning_tokens=10,
    )

    assert call.cached_tokens == 20
    assert call.reasoning_tokens == 10
    
    # Retrieve and verify
    retrieved = await ProviderCall.get(call.id)
    assert retrieved is not None
    assert retrieved.cached_tokens == 20
    assert retrieved.reasoning_tokens == 10
