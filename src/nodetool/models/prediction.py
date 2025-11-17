from datetime import datetime

from nodetool.models.base_model import DBField, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field

"""
Defines the Prediction database model.

Represents a prediction or job execution within the nodetool system.
Stores details about the execution, including the user, workflow (optional),
node involved, provider, model used, status, timing, cost, and any logs or errors.
"""


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
    input_tokens: int | None = DBField(default=None)
    output_tokens: int | None = DBField(default=None)

    @classmethod
    async def create(
        cls,
        user_id: str,
        node_id: str,
        provider: str,
        model: str,
        workflow_id: str | None = None,
        status: str = "starting",
        cost: float | None = None,
        duration: float | None = None,
        hardware: str | None = None,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ):
        """Creates a new prediction record in the database.

        Args:
            user_id: The ID of the user initiating the prediction.
            node_id: The ID of the node performing the prediction.
            provider: The name of the prediction provider (e.g., 'replicate').
            model: The specific model used for the prediction.
            workflow_id: Optional ID of the workflow this prediction belongs to.
            status: Initial status of the prediction (default: 'starting').
            cost: Optional estimated or actual cost of the prediction.
            duration: Optional duration of the prediction in seconds.
            hardware: Optional identifier for the hardware used.
            created_at: Optional timestamp when the prediction was created.
            started_at: Optional timestamp when the prediction started.
            completed_at: Optional timestamp when the prediction completed.

        Returns:
            The newly created and saved Prediction instance.
        """
        if created_at is None:
            created_at = datetime.now()

        prediction = cls(
            id=create_time_ordered_uuid(),
            user_id=user_id,
            node_id=node_id,
            provider=provider,
            model=model,
            workflow_id=workflow_id,
            status=status,
            cost=cost,
            hardware=hardware,
            created_at=created_at,
            started_at=started_at,
            completed_at=completed_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
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
        limit: int = 100,
        start_key: str | None = None,
    ):
        """Paginates through predictions for a user, optionally filtering by workflow.

        Args:
            user_id: The ID of the user whose predictions to fetch.
            workflow_id: Optional workflow ID to filter predictions by.
            limit: Maximum number of predictions to return.
            start_key: The ID of the prediction to start pagination after (exclusive).

        Returns:
            A tuple containing a list of Prediction objects and the ID of the
            last evaluated prediction (or an empty string if it's the last page).
        """
        if workflow_id is None:
            return await cls.query(
                condition=Field("user_id")
                .equals(user_id)
                .and_(Field("id").greater_than(start_key or "")),
                limit=limit,
            )
        else:
            return await cls.query(
                condition=Field("user_id")
                .equals(user_id)
                .and_(Field("workflow_id").equals(workflow_id))
                .and_(Field("id").greater_than(start_key or "")),
                limit=limit,
            )
