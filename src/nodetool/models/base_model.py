import hashlib
from enum import Enum
from random import randint
from typing import Any, Callable
from uuid import uuid1

from pydantic import BaseModel, Field

from nodetool.config.logging_config import get_logger
from nodetool.models.condition_builder import ConditionBuilder
from nodetool.models.database_adapter import DatabaseAdapter
from nodetool.runtime.resources import maybe_scope

"""
Database Model Base Classes and Utilities

This module provides the core database modeling functionality, including:

- DBModel: A base class for database models that extends Pydantic's BaseModel
- Field decorators and utilities for defining database schemas
- Index management functionality
- ModelObserver: Observer pattern for monitoring model changes

Key Components:
- DBModel: Base class that provides CRUD operations, query capabilities, and index management
- DBField: Field decorator for marking model attributes as database columns
- DBIndex: Decorator for defining database indexes on models
- ModelChangeEvent: Enum for model change event types
- ModelObserver: Global observer registry for model change notifications
"""


log = get_logger(__name__)


class ModelChangeEvent(str, Enum):
    """Types of model change events."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


# Type alias for observer callbacks
ModelObserverCallback = Callable[["DBModel", ModelChangeEvent], Any]


class ModelObserver:
    """
    Global observer registry for model change notifications.

    Observers can subscribe to specific model classes or all models.
    Callbacks receive the model instance and the event type.
    """

    _observers: dict[type | None, list[ModelObserverCallback]] = {}

    @classmethod
    def subscribe(
        cls,
        callback: ModelObserverCallback,
        model_class: type | None = None,
    ) -> None:
        """Subscribe to model changes.

        Args:
            callback: Function called with (model_instance, event_type).
            model_class: If provided, only changes to this model class trigger
                the callback. If None, all model changes are observed.
        """
        if model_class not in cls._observers:
            cls._observers[model_class] = []
        cls._observers[model_class].append(callback)

    @classmethod
    def unsubscribe(
        cls,
        callback: ModelObserverCallback,
        model_class: type | None = None,
    ) -> None:
        """Remove a previously registered observer."""
        if model_class in cls._observers:
            try:
                cls._observers[model_class].remove(callback)
            except ValueError:
                pass

    @classmethod
    def notify(cls, instance: "DBModel", event: ModelChangeEvent) -> None:
        """Notify all relevant observers of a model change."""
        # Notify observers for the specific model class
        for observer in cls._observers.get(type(instance), []):
            try:
                observer(instance, event)
            except Exception as e:
                log.error(f"Error in model observer: {e}")

        # Notify global observers (subscribed with model_class=None)
        for observer in cls._observers.get(None, []):
            try:
                observer(instance, event)
            except Exception as e:
                log.error(f"Error in global model observer: {e}")

    @classmethod
    def clear(cls) -> None:
        """Remove all observers. Primarily useful for testing."""
        cls._observers.clear()


def create_time_ordered_uuid() -> str:
    """
    Create an uuid that is ordered by time.
    """
    return uuid1(randint(0, 2**31)).hex


def DBField(hash_key: bool = False, **kwargs: Any):
    return Field(json_schema_extra={"hash_key": hash_key, "persist": True}, **kwargs)  # type: ignore


def DBIndex(columns: list[str], unique: bool = False, name: str | None = None):
    """
    Decorator to define an index on a model.

    Args:
        columns: List of column names to include in the index
        unique: Whether the index should enforce uniqueness
        name: Optional custom name for the index. If not provided, one will be generated.
    """

    def decorator(cls):
        if not hasattr(cls, "_indexes"):
            cls._indexes = []

        # Generate index name if not provided
        index_name = name or f"idx_{cls.get_table_name()}_{'_'.join(columns)}"

        cls._indexes.append({"name": index_name, "columns": columns, "unique": unique})
        return cls

    return decorator


def compute_etag(data: dict[str, Any]) -> str:
    """Compute an ETag from a model's data dictionary.

    Uses a stable JSON serialization (sorted keys) + MD5 hash to produce a
    short, deterministic fingerprint that changes whenever any field changes.
    """
    import json

    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


class DBModel(BaseModel):
    @classmethod
    def get_table_schema(cls) -> dict[str, Any]:
        """
        Get the name of the table for the model.
        """
        raise NotImplementedError()

    @classmethod
    def get_table_name(cls) -> str:
        """
        Get the name of the table for the model.
        """
        return cls.get_table_schema()["table_name"]

    @classmethod
    async def adapter(cls) -> DatabaseAdapter:
        """Get a database adapter for this model.

        Uses scope-based adapter if a ResourceScope is bound, with fallback
        to Environment.get_database_adapter() for backward compatibility with
        pre-scope operations (e.g., job record creation).

        This prevents class-level adapter caching which can leak adapters
        across loops/threads. Instead, adapters are memoized per-scope or
        per-thread.

        Returns:
            A DatabaseAdapter instance for this model's table
        """
        # Try to get adapter from current ResourceScope if one is bound
        scope = maybe_scope()
        if scope and scope.db:
            return await scope.db.adapter_for_model(cls)

        raise Exception(f"No ResourceScope bound for {cls.__name__}")

    @classmethod
    def has_indexes(cls) -> bool:
        """
        Check if the model has any defined indexes.
        """
        return hasattr(cls, "_indexes")

    @classmethod
    def get_indexes(cls) -> list[dict[str, Any]]:
        """
        Get the list of defined indexes for the model.
        Returns an empty list if no indexes are defined.
        """
        return cls._indexes if cls.has_indexes() else []  # type: ignore

    @classmethod
    async def create_table(cls):
        """
        Create the DB table for the model and its indexes.
        """
        adapter = await cls.adapter()
        await adapter.create_table()

        # Create any defined indexes
        for index in cls.get_indexes():
            await adapter.create_index(
                index_name=index["name"],
                columns=index["columns"],
                unique=index["unique"],
            )

    @classmethod
    async def create_indexes(cls):
        """
        Create all defined indexes for the model.
        """
        adapter = await cls.adapter()
        for index in cls.get_indexes():
            await adapter.create_index(
                index_name=index["name"],
                columns=index["columns"],
                unique=index["unique"],
            )

    @classmethod
    async def drop_indexes(cls):
        """
        Drop all defined indexes for the model.
        """
        adapter = await cls.adapter()
        for index in cls.get_indexes():
            await adapter.drop_index(index["name"])

    @classmethod
    async def drop_table(cls):
        """
        Drop the DB table for the model and its indexes.
        """
        adapter = await cls.adapter()
        # Drop any defined indexes first
        for index in cls.get_indexes():
            await adapter.drop_index(index["name"])

        await adapter.drop_table()

    @classmethod
    async def query(
        cls,
        condition: ConditionBuilder | None = None,
        limit: int = 100,
        order_by: str | None = None,
        reverse: bool = False,
        columns: list[str] | None = None,
    ):
        """
        Query the DB table for the model to retrieve a list of items.
        This method is used for pagination and returns a tuple containing a list of items and the last evaluated key.
        It allows for filtering and sorting the results.

        Args:
            condition: The condition for the query.
            limit: The maximum number of items to retrieve.
            order_by: The column to order the results by.
            reverse: Whether to reverse the order of the results.
            columns: The columns to retrieve.

        Returns:
            A tuple containing a list of items that match the query conditions and the last evaluated key.
        """
        adapter = await cls.adapter()
        items, key = await adapter.query(
            condition=condition,
            limit=limit,
            order_by=order_by,
            reverse=reverse,
            columns=columns,
        )

        def try_load_model(item: dict[str, Any]) -> Any:
            try:
                return cls(**item)
            except Exception as e:
                log.error(f"Error loading model: {e}")
                return None

        def filter_none(items: list[Any]) -> list[Any]:
            return [item for item in items if item is not None]

        return filter_none([try_load_model(item) for item in items]), key

    @classmethod
    async def create(cls, **kwargs):
        """
        Create a model instance from keyword arguments and save it.
        """
        instance = cls(**kwargs)
        await instance.save()
        ModelObserver.notify(instance, ModelChangeEvent.CREATED)
        return instance

    def before_save(self):
        """
        Hook method called before saving the model instance.
        Subclasses can override this method to perform actions before saving.
        """
        pass

    async def save(self):
        """
        Save a model instance and return the instance.
        """
        self.before_save()
        adapter = await self.__class__.adapter()
        await adapter.save(self.model_dump())
        ModelObserver.notify(self, ModelChangeEvent.UPDATED)
        return self

    @classmethod
    def db_fields(cls) -> dict[str, Any]:
        """
        Return a dictionary of fields that should be persisted.
        """
        return {
            field_name: field
            for field_name, field in cls.model_fields.items()
            if field.json_schema_extra and field.json_schema_extra.get("persist", False)
        }

    @classmethod
    async def get(cls, key: str | int):
        """
        Retrieve a model instance from the DB using a key.
        """
        adapter = await cls.adapter()
        item = await adapter.get(key)
        if item is None:
            return None
        return cls(**item)

    async def reload(self):
        """
        Reload the model instance from the DB.
        """
        adapter = await self.__class__.adapter()
        item = await adapter.get(self.partition_value())
        if item is None:
            raise ValueError(f"Item not found: {self.partition_value()}")
        for key, value in item.items():
            setattr(self, key, value)
        return self

    def partition_value(self) -> str:
        """Get the value of the hash key from the table schema."""
        pk = self.__class__.get_table_schema().get("primary_key", "id")
        return getattr(self, pk)

    async def delete(self):
        """
        Delete the model instance from the database.
        """
        adapter = await self.__class__.adapter()
        await adapter.delete(self.partition_value())
        ModelObserver.notify(self, ModelChangeEvent.DELETED)

    async def update(self, **kwargs):
        """
        Update the model instance and save it.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        await self.save()
        return self

    def get_etag(self) -> str:
        """Compute an ETag for this model instance."""
        return compute_etag(self.model_dump())

    @classmethod
    async def list_indexes(cls) -> list[dict[str, Any]]:
        """
        List all indexes defined on the model's table.
        """
        adapter = await cls.adapter()
        return await adapter.list_indexes()
