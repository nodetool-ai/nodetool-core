from datetime import datetime
from typing import Any

from nodetool.models.base_model import DBField, DBIndex, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

"""
Defines the Prediction database model.

Represents a prediction or job execution within the nodetool system.
Stores details about the execution, including the user, workflow (optional),
node involved, provider, model used, status, timing, cost, and any logs or errors.

This model also serves for cost tracking and usage analytics for AI provider calls.
"""


@DBIndex(columns=["user_id", "provider"], name="idx_prediction_user_provider")
@DBIndex(columns=["user_id", "model"], name="idx_prediction_user_model")
@DBIndex(columns=["created_at"], name="idx_prediction_created_at")
class Prediction(DBModel):
    """Database model representing a prediction or job execution."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for predictions."""
        return {
            "table_name": "nodetool_predictions",
        }

    id: str = DBField(hash_key=True)
    user_id: str = DBField()
    node_id: str = DBField(default="")
    provider: str = DBField(default="")
    model: str = DBField()
    workflow_id: str | None = DBField(default=None)
    error: str | None = DBField(default=None)
    logs: str | None = DBField(default=None)
    status: str = DBField(default="starting")
    created_at: datetime | None = DBField(default_factory=datetime.now)
    started_at: datetime | None = DBField(default=None)
    completed_at: datetime | None = DBField(default=None)
    cost: float | None = DBField(default=None)
    duration: float | None = DBField(default=None)
    hardware: str | None = DBField(default=None)

    # Token-based usage (for text/chat/embedding models)
    input_tokens: int | None = DBField(default=None)
    output_tokens: int | None = DBField(default=None)
    total_tokens: int | None = DBField(default=None)
    cached_tokens: int | None = DBField(default=None)
    reasoning_tokens: int | None = DBField(default=None)

    # Size-based usage (for image/audio/video models)
    input_size: int | None = DBField(default=None)  # Input data size in bytes
    output_size: int | None = DBField(default=None)  # Output data size in bytes

    # Model parameters (resolution, quality, voice, etc.)
    parameters: dict[str, Any] | None = DBField(default=None)

    # Additional metadata
    metadata: dict[str, Any] | None = DBField(default=None)

    @classmethod
    async def create(
        cls,
        **kwargs,
    ):
        """Creates a new prediction record in the database.

        Args:
            user_id: The ID of the user initiating the prediction.
            node_id: The ID of the node performing the prediction.
            provider: The name of the prediction provider (e.g., 'openai', 'anthropic').
            model: The specific model used for the prediction.
            workflow_id: Optional ID of the workflow this prediction belongs to.
            status: Initial status of the prediction (default: 'starting').
            cost: Optional estimated or actual cost of the prediction.
            duration: Optional duration of the prediction in seconds.
            hardware: Optional identifier for the hardware used.
            created_at: Optional timestamp when the prediction was created.
            started_at: Optional timestamp when the prediction started.
            completed_at: Optional timestamp when the prediction completed.
            input_tokens: Optional number of input/prompt tokens used.
            output_tokens: Optional number of output/completion tokens used.
            total_tokens: Optional total number of tokens used.
            cached_tokens: Optional number of cached tokens (for providers that support it).
            reasoning_tokens: Optional number of reasoning tokens (for providers that support it).
            input_size: Optional input data size in bytes (for image/audio/video models).
            output_size: Optional output data size in bytes (for image/audio/video models).
            parameters: Optional model-specific parameters (resolution, quality, voice, etc.).
            metadata: Optional additional metadata about the prediction.

        Returns:
            The newly created and saved Prediction instance.
        """
        created_at = kwargs.get('created_at')
        if created_at is None:
            created_at = datetime.now()
            kwargs['created_at'] = created_at
        
        if 'id' not in kwargs:
            kwargs['id'] = create_time_ordered_uuid()

        prediction = cls(**kwargs)
        await prediction.save()
        return prediction

    @classmethod
    async def find(cls, user_id: str, id: str):
        """Finds a prediction by its ID, ensuring it belongs to the specified user.

        Args:
            user_id: The ID of the user who should own the prediction.
            id: The ID of the prediction to find.

        Returns:
            The Prediction object if found and owned by the user, otherwise None.
        """
        prediction = await cls.get(id)
        if prediction is None or prediction.user_id != user_id:
            return None
        return prediction

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        workflow_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        """Paginates through predictions for a user with optional filtering.

        Args:
            user_id: The ID of the user whose predictions to fetch.
            workflow_id: Optional workflow ID to filter predictions by.
            provider: Optional provider name to filter predictions by.
            model: Optional model name to filter predictions by.
            limit: Maximum number of predictions to return.
            start_key: The ID of the prediction to start pagination after (exclusive).
            reverse: Whether to return results in reverse chronological order.

        Returns:
            A tuple containing a list of Prediction objects and the ID of the
            last evaluated prediction (or an empty string if it's the last page).
        """
        # Build condition
        condition = Field("user_id").equals(user_id).and_(Field("id").greater_than(start_key or ""))

        if workflow_id:
            condition = condition.and_(Field("workflow_id").equals(workflow_id))
        if provider:
            condition = condition.and_(Field("provider").equals(provider))
        if model:
            condition = condition.and_(Field("model").equals(model))

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
        model: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate cost and usage statistics for a user.

        Args:
            user_id: User ID to aggregate for
            provider: Optional provider filter
            model: Optional model filter

        Returns:
            Dictionary with aggregated totals

        Note:
            For production use with large datasets, consider implementing
            database-level aggregation queries instead of in-memory aggregation.
        """
        # Fetch records for the user with filters
        predictions, _ = await cls.paginate(
            user_id=user_id,
            provider=provider,
            model=model,
            limit=10000,  # High limit for aggregation
            reverse=False,
        )

        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        for p in predictions:
            total_cost += p.cost or 0
            total_input_tokens += p.input_tokens or 0
            total_output_tokens += p.output_tokens or 0
            total_tokens += p.total_tokens or 0
        # If total_tokens wasn't tracked, calculate from input + output
        if total_tokens == 0:
            total_tokens = total_input_tokens + total_output_tokens
        call_count = len(predictions)

        return {
            "user_id": user_id,
            "provider": provider,
            "model": model,
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
        """Aggregate cost and usage by provider for a user.

        Args:
            user_id: User ID to aggregate for

        Returns:
            List of aggregations, one per provider

        Note:
            For production use with large datasets, consider implementing
            database-level GROUP BY queries instead of in-memory aggregation.
        """
        # Fetch all records for the user
        predictions, _ = await cls.paginate(
            user_id=user_id,
            limit=10000,  # High limit for aggregation
            reverse=False,
        )

        # Group by provider
        provider_stats: dict[str, dict[str, Any]] = {}
        for pred in predictions:
            if pred.provider not in provider_stats:
                provider_stats[pred.provider] = {
                    "provider": pred.provider,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }

            stats = provider_stats[pred.provider]
            stats["total_cost"] += pred.cost or 0
            stats["total_input_tokens"] += pred.input_tokens or 0
            stats["total_output_tokens"] += pred.output_tokens or 0
            stats["total_tokens"] += pred.total_tokens or (pred.input_tokens or 0) + (pred.output_tokens or 0)
            stats["call_count"] += 1

        return list(provider_stats.values())

    @classmethod
    async def aggregate_by_model(
        cls,
        user_id: str,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate cost and usage by model for a user.

        Args:
            user_id: User ID to aggregate for
            provider: Optional provider filter

        Returns:
            List of aggregations, one per model

        Note:
            For production use with large datasets, consider implementing
            database-level GROUP BY queries instead of in-memory aggregation.
        """
        # Fetch all records for the user
        predictions, _ = await cls.paginate(
            user_id=user_id,
            provider=provider,
            limit=10000,  # High limit for aggregation
            reverse=False,
        )

        # Group by model using tuple key to avoid collisions
        model_stats: dict[tuple[str, str], dict[str, Any]] = {}
        for pred in predictions:
            key = (pred.provider, pred.model)
            if key not in model_stats:
                model_stats[key] = {
                    "provider": pred.provider,
                    "model": pred.model,
                    "total_cost": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                }

            stats = model_stats[key]
            stats["total_cost"] += pred.cost or 0
            stats["total_input_tokens"] += pred.input_tokens or 0
            stats["total_output_tokens"] += pred.output_tokens or 0
            stats["total_tokens"] += pred.total_tokens or (pred.input_tokens or 0) + (pred.output_tokens or 0)
            stats["call_count"] += 1

        return list(model_stats.values())
