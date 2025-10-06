from datetime import datetime
from nodetool.models.base_model import DBModel, DBField, create_time_ordered_uuid
from nodetool.models.condition_builder import Field


"""
Defines the Thread database model.

Represents a conversation thread within the nodetool system, typically used for chatbots
or interactive sessions. Primarily links a user to a sequence of messages.
"""


class Thread(DBModel):
    """Database model representing a conversation thread."""

    @classmethod
    def get_table_schema(cls):
        """Returns the database table schema for threads."""
        return {"table_name": "nodetool_threads"}

    id: str = DBField(hash_key=True)
    user_id: str = DBField(default="")
    title: str | None = DBField(default=None)
    created_at: datetime = DBField(default_factory=datetime.now)
    updated_at: datetime = DBField(default_factory=datetime.now)

    @classmethod
    async def find(cls, user_id: str, id: str):
        """Finds a thread by its ID, ensuring it belongs to the specified user.

        Args:
            user_id: The ID of the user who should own the thread.
            id: The ID of the thread to find.

        Returns:
            The Thread object if found and owned by the user, otherwise None.
        """
        thread = await cls.get(id)
        if thread and thread.user_id == user_id:
            return thread
        return None

    @classmethod
    async def create(cls, user_id: str, id: str | None = None, **kwargs) -> "Thread":
        """Creates a new thread record in the database for a given user.

        Args:
            user_id: The ID of the user creating the thread.
            id: Optional custom ID for the thread. If not provided, generates a time-ordered UUID.
            **kwargs: Additional fields to set on the model.

        Returns:
            The newly created and saved Thread instance.
        """
        return await super().create(
            id=id or create_time_ordered_uuid(),
            user_id=user_id,
            **kwargs,
        )

    @classmethod
    async def paginate(
        cls,
        user_id: str,
        limit: int = 10,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        """Paginates through threads for a specific user.

        Args:
            user_id: The ID of the user whose threads to fetch.
            limit: Maximum number of threads to return.
            start_key: The ID of the thread to start pagination after (exclusive).
            reverse: Whether to reverse the sort order (typically by creation time).

        Returns:
            A tuple containing a list of Thread objects and the ID of the
            last evaluated thread (or an empty string if it's the last page).
        """
        return await cls.query(
            condition=Field("user_id")
            .equals(user_id)
            .and_(Field("id").greater_than(start_key or "")),
            limit=limit,
            reverse=reverse,
        )
