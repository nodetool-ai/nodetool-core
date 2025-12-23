from datetime import datetime
from typing import Any

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field


@DBIndex(columns=["user_id"], name="idx_provider_call_user_id")
@DBIndex(columns=["provider"], name="idx_provider_call_provider")
@DBIndex(columns=["model_id"], name="idx_provider_call_model_id")
@DBIndex(columns=["user_id", "provider"], name="idx_provider_call_user_provider")
@DBIndex(columns=["user_id", "model_id"], name="idx_provider_call_user_model")
@DBIndex(columns=["created_at"], name="idx_provider_call_created_at")
class ProviderCall(DBModel):
    """
    Model for tracking individual API calls to AI providers.
    
    This model logs granular information about each API call including:
    - User making the call
    - Provider and model used
    - Cost and token usage
    - Timestamps
    - Additional metadata
    
    Used for cost aggregation, usage analytics, and billing.
    """

    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "nodetool_provider_calls",
        }

    id: str = DBField()
    user_id: str = DBField()
    provider: str = DBField()
    model_id: str = DBField()
    cost: float = DBField(default=0.0)
    input_tokens: int = DBField(default=0)
    output_tokens: int = DBField(default=0)
    total_tokens: int = DBField(default=0)
    cached_tokens: int | None = DBField(default=None)
    reasoning_tokens: int | None = DBField(default=None)
    created_at: datetime = DBField(default_factory=datetime.now)
    metadata: dict[str, Any] | None = DBField(default=None)

    @classmethod
    async def create(cls, user_id: str, provider: str, model_id: str, **kwargs) -> "ProviderCall":
        """
        Create a new provider call record.
        
        Args:
            user_id: ID of the user making the call
            provider: Provider name (e.g., "openai", "anthropic")
            model_id: Model identifier (e.g., "gpt-4o-mini", "claude-3-opus")
            **kwargs: Additional fields (cost, tokens, metadata, etc.)
        
        Returns:
            Created ProviderCall instance
        """
        return await super().create(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            provider=provider,
            model_id=model_id,
            **kwargs,
        )

    @classmethod
    async def paginate(
        cls,
        user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        reverse: bool = True,
    ):
        """
        Paginate provider call records with optional filtering.
        
        Args:
            user_id: Filter by user ID
            provider: Filter by provider name
            model_id: Filter by model ID
            limit: Maximum number of records to return
            start_key: Pagination start key
            reverse: Sort in reverse chronological order
        
        Returns:
            Tuple of (list of ProviderCall instances, next pagination key)
        """
        # Build condition
        condition = Field("id").greater_than(start_key or "")
        
        if user_id:
            condition = condition.and_(Field("user_id").equals(user_id))
        if provider:
            condition = condition.and_(Field("provider").equals(provider))
        if model_id:
            condition = condition.and_(Field("model_id").equals(model_id))

        return await cls.query(
            condition=condition,
            limit=limit,
            reverse=reverse,
        )

    @classmethod
    async def aggregate_by_user(
        cls,
        user_id: str,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Aggregate cost and usage statistics for a user.
        
        Args:
            user_id: User ID to aggregate for
            provider: Optional provider filter
            model_id: Optional model filter
        
        Returns:
            Dictionary with aggregated totals
        """
        # Fetch all records for the user with filters
        calls, _ = await cls.paginate(
            user_id=user_id,
            provider=provider,
            model_id=model_id,
            limit=10000,  # High limit to get all records
            reverse=False,
        )

        total_cost = sum(call.cost for call in calls)
        total_input_tokens = sum(call.input_tokens for call in calls)
        total_output_tokens = sum(call.output_tokens for call in calls)
        total_tokens = sum(call.total_tokens for call in calls)
        call_count = len(calls)

        return {
            "user_id": user_id,
            "provider": provider,
            "model_id": model_id,
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "call_count": call_count,
        }

    @classmethod
    async def aggregate_by_provider(
        cls,
        user_id: str,
    ) -> list[dict[str, Any]]:
        """
        Aggregate cost and usage by provider for a user.
        
        Args:
            user_id: User ID to aggregate for
        
        Returns:
            List of aggregations, one per provider
        """
        # Fetch all records for the user
        calls, _ = await cls.paginate(
            user_id=user_id,
            limit=10000,
            reverse=False,
        )

        # Group by provider
        provider_stats: dict[str, dict[str, Any]] = {}
        for call in calls:
            if call.provider not in provider_stats:
                provider_stats[call.provider] = {
                    "provider": call.provider,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }
            
            stats = provider_stats[call.provider]
            stats["total_cost"] += call.cost
            stats["total_input_tokens"] += call.input_tokens
            stats["total_output_tokens"] += call.output_tokens
            stats["total_tokens"] += call.total_tokens
            stats["call_count"] += 1

        return list(provider_stats.values())

    @classmethod
    async def aggregate_by_model(
        cls,
        user_id: str,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Aggregate cost and usage by model for a user.
        
        Args:
            user_id: User ID to aggregate for
            provider: Optional provider filter
        
        Returns:
            List of aggregations, one per model
        """
        # Fetch all records for the user
        calls, _ = await cls.paginate(
            user_id=user_id,
            provider=provider,
            limit=10000,
            reverse=False,
        )

        # Group by model
        model_stats: dict[str, dict[str, Any]] = {}
        for call in calls:
            key = f"{call.provider}:{call.model_id}"
            if key not in model_stats:
                model_stats[key] = {
                    "provider": call.provider,
                    "model_id": call.model_id,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }
            
            stats = model_stats[key]
            stats["total_cost"] += call.cost
            stats["total_input_tokens"] += call.input_tokens
            stats["total_output_tokens"] += call.output_tokens
            stats["total_tokens"] += call.total_tokens
            stats["call_count"] += 1

        return list(model_stats.values())
