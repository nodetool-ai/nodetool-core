import pytest
from fastapi.testclient import TestClient

from nodetool.models.provider_call import ProviderCall


@pytest.mark.asyncio
async def test_list_provider_calls(client: TestClient, headers: dict[str, str], user_id: str):
    """Test listing provider calls."""
    # Create some test data
    await ProviderCall.create(
        user_id=user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        cost=0.05,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    response = client.get("/api/costs/", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "calls" in data
    assert len(data["calls"]) >= 1
    assert data["calls"][0]["provider"] == "openai"
    assert data["calls"][0]["model_id"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_list_provider_calls_with_provider_filter(
    client: TestClient, headers: dict[str, str], user_id: str
):
    """Test listing provider calls with provider filter."""
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

    response = client.get("/api/costs/?provider=openai", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "calls" in data
    # All returned calls should be from OpenAI
    for call in data["calls"]:
        assert call["provider"] == "openai"


@pytest.mark.asyncio
async def test_list_provider_calls_with_model_filter(
    client: TestClient, headers: dict[str, str], user_id: str
):
    """Test listing provider calls with model filter."""
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

    response = client.get("/api/costs/?model_id=gpt-4o-mini", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "calls" in data
    # All returned calls should use gpt-4o-mini
    for call in data["calls"]:
        assert call["model_id"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_list_provider_calls_with_limit(
    client: TestClient, headers: dict[str, str], user_id: str
):
    """Test listing provider calls with limit."""
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

    response = client.get("/api/costs/?limit=3", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "calls" in data
    assert len(data["calls"]) <= 3


@pytest.mark.asyncio
async def test_aggregate_costs(client: TestClient, headers: dict[str, str], user_id: str):
    """Test aggregating costs."""
    # Create some test data
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

    response = client.get("/api/costs/aggregate", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == user_id
    assert data["total_cost"] >= 0.08
    assert data["total_tokens"] >= 240
    assert data["call_count"] >= 2


@pytest.mark.asyncio
async def test_aggregate_costs_with_provider_filter(
    client: TestClient, headers: dict[str, str], user_id: str
):
    """Test aggregating costs with provider filter."""
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

    response = client.get("/api/costs/aggregate?provider=openai", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == user_id
    assert data["provider"] == "openai"
    assert data["total_cost"] >= 0.05


@pytest.mark.asyncio
async def test_aggregate_costs_by_provider(
    client: TestClient, headers: dict[str, str], user_id: str
):
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

    response = client.get("/api/costs/aggregate/by-provider", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 2
    
    # Find OpenAI and Anthropic results
    openai_result = next((r for r in data if r["provider"] == "openai"), None)
    anthropic_result = next((r for r in data if r["provider"] == "anthropic"), None)
    
    assert openai_result is not None
    assert anthropic_result is not None
    assert openai_result["total_cost"] >= 0.05
    assert anthropic_result["total_cost"] >= 0.10


@pytest.mark.asyncio
async def test_aggregate_costs_by_model(
    client: TestClient, headers: dict[str, str], user_id: str
):
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

    response = client.get("/api/costs/aggregate/by-model", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data) >= 2
    
    # Find specific model results
    mini_result = next((r for r in data if r["model_id"] == "gpt-4o-mini"), None)
    regular_result = next((r for r in data if r["model_id"] == "gpt-4o"), None)
    
    assert mini_result is not None
    assert regular_result is not None
    assert mini_result["total_cost"] >= 0.05
    assert regular_result["total_cost"] >= 0.15


@pytest.mark.asyncio
async def test_aggregate_costs_by_model_with_provider_filter(
    client: TestClient, headers: dict[str, str], user_id: str
):
    """Test aggregating costs by model with provider filter."""
    # Create calls for different providers and models
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

    response = client.get("/api/costs/aggregate/by-model?provider=openai", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    # All returned models should be from OpenAI
    for result in data:
        assert result["provider"] == "openai"


@pytest.mark.asyncio
async def test_get_cost_summary(client: TestClient, headers: dict[str, str], user_id: str):
    """Test getting comprehensive cost summary."""
    # Create some test data
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

    response = client.get("/api/costs/summary", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "overall" in data
    assert "by_provider" in data
    assert "by_model" in data
    assert "recent_calls" in data
    
    # Check overall aggregation
    assert data["overall"]["user_id"] == user_id
    assert data["overall"]["total_cost"] >= 0.15
    
    # Check provider breakdown
    assert len(data["by_provider"]) >= 2
    
    # Check model breakdown
    assert len(data["by_model"]) >= 2
    
    # Check recent calls
    assert len(data["recent_calls"]) >= 2
    assert "provider" in data["recent_calls"][0]
    assert "model_id" in data["recent_calls"][0]
    assert "cost" in data["recent_calls"][0]


@pytest.mark.asyncio
async def test_list_provider_calls_empty(client: TestClient, headers: dict[str, str], user_id: str):
    """Test listing provider calls when there are none."""
    response = client.get("/api/costs/", headers=headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "calls" in data
    # May have calls from other tests, just check structure
    assert isinstance(data["calls"], list)
