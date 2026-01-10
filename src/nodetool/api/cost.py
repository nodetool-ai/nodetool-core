from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.prediction import Prediction

log = get_logger(__name__)

router = APIRouter(prefix="/api/costs", tags=["costs"])


class PredictionResponse(BaseModel):
    """Response model for a single prediction/cost record."""

    id: str
    user_id: str
    node_id: str
    provider: str
    model: str
    workflow_id: str | None = None
    cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    created_at: str | None = None
    metadata: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """Response model for a list of predictions."""

    calls: list[PredictionResponse]
    next_start_key: str | None = None


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
    provider: str | None = None
    model: str | None = None


class ProviderAggregateResponse(AggregateResponse):
    """Response model for provider-level aggregation."""

    provider: str


class ModelAggregateResponse(AggregateResponse):
    """Response model for model-level aggregation."""

    provider: str
    model: str


@router.get("/", response_model=PredictionListResponse)
async def list_provider_calls(
    user_id: str = Depends(current_user),
    provider: str | None = Query(None, description="Filter by provider"),
    model: str | None = Query(None, description="Filter by model"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    start_key: str | None = Query(None, description="Pagination start key"),
):
    """
    List provider API calls for the current user with optional filtering.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name
        model: Optional filter by model name
        limit: Maximum number of records to return (1-1000)
        start_key: Pagination start key

    Returns:
        List of prediction/cost records
    """
    predictions, next_start_key = await Prediction.paginate(
        user_id=user_id,
        provider=provider,
        model=model,
        limit=limit,
        start_key=start_key,
        reverse=True,  # Most recent first
    )

    log.info(
        "Costs API list_provider_calls",
        extra={
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "limit": limit,
            "call_count": len(predictions),
        },
    )

    return PredictionListResponse(
        calls=[
            PredictionResponse(
                id=pred.id,
                user_id=pred.user_id,
                node_id=pred.node_id,
                provider=pred.provider,
                model=pred.model,
                workflow_id=pred.workflow_id,
                cost=pred.cost,
                input_tokens=pred.input_tokens,
                output_tokens=pred.output_tokens,
                total_tokens=pred.total_tokens,
                cached_tokens=pred.cached_tokens,
                reasoning_tokens=pred.reasoning_tokens,
                created_at=pred.created_at.isoformat() if pred.created_at else None,
                metadata=pred.metadata,
            )
            for pred in predictions
        ],
        next_start_key=next_start_key,
    )


@router.get("/aggregate", response_model=UserAggregateResponse)
async def aggregate_costs(
    user_id: str = Depends(current_user),
    provider: str | None = Query(None, description="Filter by provider"),
    model: str | None = Query(None, description="Filter by model"),
):
    """
    Get aggregated cost statistics for the current user.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name
        model: Optional filter by model name

    Returns:
        Aggregated cost and usage statistics
    """
    aggregation = await Prediction.aggregate_by_user(
        user_id=user_id,
        provider=provider,
        model=model,
    )

    log.info(
        "Costs API aggregate_costs",
        extra={
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "total_cost": aggregation["total_cost"],
        },
    )

    return UserAggregateResponse(**aggregation)


@router.get("/aggregate/by-provider", response_model=list[ProviderAggregateResponse])
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
    aggregations = await Prediction.aggregate_by_provider(user_id=user_id)

    log.info(
        "Costs API aggregate_costs_by_provider",
        extra={
            "user_id": user_id,
            "provider_count": len(aggregations),
        },
    )

    return [ProviderAggregateResponse(**agg) for agg in aggregations]


@router.get("/aggregate/by-model", response_model=list[ModelAggregateResponse])
async def aggregate_costs_by_model(
    user_id: str = Depends(current_user),
    provider: str | None = Query(None, description="Filter by provider"),
):
    """
    Get cost statistics aggregated by model for the current user.

    Args:
        user_id: Current authenticated user ID
        provider: Optional filter by provider name

    Returns:
        List of aggregations, one per model
    """
    aggregations = await Prediction.aggregate_by_model(
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
    overall = await Prediction.aggregate_by_user(user_id=user_id)

    # Get provider breakdown
    by_provider = await Prediction.aggregate_by_provider(user_id=user_id)

    # Get model breakdown
    by_model = await Prediction.aggregate_by_model(user_id=user_id)

    # Get recent calls (last 10)
    recent_predictions, _ = await Prediction.paginate(
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
                "id": pred.id,
                "provider": pred.provider,
                "model": pred.model,
                "cost": pred.cost,
                "total_tokens": pred.total_tokens,
                "created_at": pred.created_at.isoformat() if pred.created_at else "",
            }
            for pred in recent_predictions
        ],
    }
