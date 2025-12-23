from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.provider_call import ProviderCall

log = get_logger(__name__)

router = APIRouter(prefix="/api/costs", tags=["costs"])


class ProviderCallResponse(BaseModel):
    """Response model for a single provider call record."""

    id: str
    user_id: str
    provider: str
    model_id: str
    cost: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    created_at: str
    metadata: Optional[dict[str, Any]] = None

    class Config:
        from_attributes = True


class ProviderCallListResponse(BaseModel):
    """Response model for a list of provider calls."""

    calls: List[ProviderCallResponse]
    next_start_key: Optional[str] = None


class AggregateResponse(BaseModel):
    """Response model for aggregated cost data."""

    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    call_count: int


class UserAggregateResponse(AggregateResponse):
    """Response model for user-level aggregation."""

    user_id: str
    provider: Optional[str] = None
    model_id: Optional[str] = None


class ProviderAggregateResponse(AggregateResponse):
    """Response model for provider-level aggregation."""

    provider: str


class ModelAggregateResponse(AggregateResponse):
    """Response model for model-level aggregation."""

    provider: str
    model_id: str


@router.get("/", response_model=ProviderCallListResponse)
async def list_provider_calls(
    user_id: str = Depends(current_user),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    start_key: Optional[str] = Query(None, description="Pagination start key"),
):
    """
    List provider API calls for the current user with optional filtering.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name
        model_id: Optional filter by model ID
        limit: Maximum number of records to return (1-1000)
        start_key: Pagination start key

    Returns:
        List of provider call records
    """
    calls, next_start_key = await ProviderCall.paginate(
        user_id=user_id,
        provider=provider,
        model_id=model_id,
        limit=limit,
        start_key=start_key,
        reverse=True,  # Most recent first
    )

    log.info(
        "Costs API list_provider_calls",
        extra={
            "user_id": user_id,
            "provider": provider,
            "model_id": model_id,
            "limit": limit,
            "call_count": len(calls),
        },
    )

    return ProviderCallListResponse(
        calls=[
            ProviderCallResponse(
                id=call.id,
                user_id=call.user_id,
                provider=call.provider,
                model_id=call.model_id,
                cost=call.cost,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                total_tokens=call.total_tokens,
                cached_tokens=call.cached_tokens,
                reasoning_tokens=call.reasoning_tokens,
                created_at=call.created_at.isoformat() if call.created_at else "",
                metadata=call.metadata,
            )
            for call in calls
        ],
        next_start_key=next_start_key,
    )


@router.get("/aggregate", response_model=UserAggregateResponse)
async def aggregate_costs(
    user_id: str = Depends(current_user),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
):
    """
    Get aggregated cost statistics for the current user.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name
        model_id: Optional filter by model ID

    Returns:
        Aggregated cost and usage statistics
    """
    aggregation = await ProviderCall.aggregate_by_user(
        user_id=user_id,
        provider=provider,
        model_id=model_id,
    )

    log.info(
        "Costs API aggregate_costs",
        extra={
            "user_id": user_id,
            "provider": provider,
            "model_id": model_id,
            "total_cost": aggregation["total_cost"],
        },
    )

    return UserAggregateResponse(**aggregation)


@router.get("/aggregate/by-provider", response_model=List[ProviderAggregateResponse])
async def aggregate_costs_by_provider(
    user_id: str = Depends(current_user),
):
    """
    Get cost statistics aggregated by provider for the current user.

    Args:
        user_id: Current authenticated user ID

    Returns:
        List of aggregations, one per provider
    """
    aggregations = await ProviderCall.aggregate_by_provider(user_id=user_id)

    log.info(
        "Costs API aggregate_costs_by_provider",
        extra={
            "user_id": user_id,
            "provider_count": len(aggregations),
        },
    )

    return [ProviderAggregateResponse(**agg) for agg in aggregations]


@router.get("/aggregate/by-model", response_model=List[ModelAggregateResponse])
async def aggregate_costs_by_model(
    user_id: str = Depends(current_user),
    provider: Optional[str] = Query(None, description="Filter by provider"),
):
    """
    Get cost statistics aggregated by model for the current user.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name

    Returns:
        List of aggregations, one per model
    """
    aggregations = await ProviderCall.aggregate_by_model(
        user_id=user_id,
        provider=provider,
    )

    log.info(
        "Costs API aggregate_costs_by_model",
        extra={
            "user_id": user_id,
            "provider": provider,
            "model_count": len(aggregations),
        },
    )

    return [ModelAggregateResponse(**agg) for agg in aggregations]


@router.get("/summary", response_model=dict[str, Any])
async def get_cost_summary(
    user_id: str = Depends(current_user),
):
    """
    Get a comprehensive cost summary for the current user.

    Includes:
    - Overall totals
    - Breakdown by provider
    - Breakdown by model
    - Recent activity

    Args:
        user_id: Current authenticated user ID

    Returns:
        Comprehensive cost summary
    """
    # Get overall aggregation
    overall = await ProviderCall.aggregate_by_user(user_id=user_id)

    # Get provider breakdown
    by_provider = await ProviderCall.aggregate_by_provider(user_id=user_id)

    # Get model breakdown
    by_model = await ProviderCall.aggregate_by_model(user_id=user_id)

    # Get recent calls (last 10)
    recent_calls, _ = await ProviderCall.paginate(
        user_id=user_id,
        limit=10,
        reverse=True,
    )

    log.info(
        "Costs API get_cost_summary",
        extra={
            "user_id": user_id,
            "total_cost": overall["total_cost"],
            "provider_count": len(by_provider),
            "model_count": len(by_model),
        },
    )

    return {
        "overall": overall,
        "by_provider": by_provider,
        "by_model": by_model,
        "recent_calls": [
            {
                "id": call.id,
                "provider": call.provider,
                "model_id": call.model_id,
                "cost": call.cost,
                "total_tokens": call.total_tokens,
                "created_at": call.created_at.isoformat() if call.created_at else "",
            }
            for call in recent_calls
        ],
    }
