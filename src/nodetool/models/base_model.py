from typing import Any
from pydantic import BaseModel, Field

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from uuid import uuid1
from random import randint
import atexit
import signal
import asyncio

from nodetool.models.condition_builder import ConditionBuilder
from nodetool.models.database_adapter import DatabaseAdapter

"""
Database Model Base Classes and Utilities

This module provides the core database modeling functionality, including:

- DBModel: A base class for database models that extends Pydantic's BaseModel
- Field decorators and utilities for defining database schemas
- Index management functionality

Key Components:
- DBModel: Base class that provides CRUD operations, query capabilities, and index management
- DBField: Field decorator for marking model attributes as database columns
- DBIndex: Decorator for defining database indexes on models
"""


log = get_logger(__name__)

# Global registry to track all database adapters for cleanup
_global_adapters: list[DatabaseAdapter] = []


async def close_all_database_adapters():
    """Close all registered database adapters and clear the registry."""
    global _global_adapters
    for adapter in _global_adapters:
        try:
            await adapter.close()
        except Exception as e:
            log.warning(f"Error closing database adapter: {e}")
    _global_adapters.clear()


def _shutdown_handler_sync():
    """Synchronous shutdown handler for atexit and signals.

    Attempts to run the async cleanup coroutine. If an event loop
    is already running (e.g., in FastAPI), this will safely skip.

    Note: We suppress logging here because logging may be shut down
    during atexit, causing spurious "I/O operation on closed file" errors.
    """
    import logging

    try:
        # Try to get the running loop (will raise RuntimeError if none exists)
        try:
            loop = asyncio.get_running_loop()
            # If we get here, a loop is running - we can't use asyncio.run()
            # The async cleanup will be handled elsewhere (e.g., FastAPI lifespan)
            return
        except RuntimeError:
            # No running loop, safe to create one
            # Temporarily disable logging to avoid errors during atexit
            logging.disable(logging.CRITICAL)
            try:
                asyncio.run(close_all_database_adapters())
            finally:
                logging.disable(logging.NOTSET)
    except Exception:
        # Suppress exceptions during atexit - logging may already be shut down
        # and we don't want to break the shutdown process
        pass


# Register cleanup handlers to ensure database adapters are closed
# when the Python process exits, regardless of whether explicit
# shutdown methods were called
atexit.register(_shutdown_handler_sync)

# Handle SIGTERM (graceful shutdown)
def _sigterm_handler(signum, frame):
    _shutdown_handler_sync()
    import sys
    sys.exit(0)

# Handle SIGINT (Ctrl+C)
def _sigint_handler(signum, frame):
    _shutdown_handler_sync()
    import sys
    sys.exit(130)  # Standard exit code for SIGINT

try:
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigint_handler)
except (ValueError, RuntimeError):
    # signal.signal() can fail in certain contexts (e.g., non-main thread)
    # This is safe to ignore
    pass


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
        if not hasattr(cls, "__adapter"):
            cls.__adapter = await Environment.get_database_adapter(
                fields=cls.db_fields(),
                table_schema=cls.get_table_schema(),
                indexes=cls.get_indexes(),
            )
            # Register adapter globally for cleanup
            _global_adapters.append(cls.__adapter)
        return cls.__adapter

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
        reverse: bool = False,
    ):
        """
        Query the DB table for the model to retrieve a list of items.
        This method is used for pagination and returns a tuple containing a list of items and the last evaluated key.
        It allows for filtering and sorting the results.

        Args:
            condition: The condition for the query.
            limit: The maximum number of items to retrieve.
            reverse: Whether to reverse the order of the results.

        Returns:
            A tuple containing a list of items that match the query conditions and the last evaluated key.
        """
        adapter = await cls.adapter()
        items, key = await adapter.query(
            condition=condition,
            limit=limit,
            reverse=reverse,
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

    async def update(self, **kwargs):
        """
        Update the model instance and save it.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        await self.save()
        return self

    @classmethod
    async def list_indexes(cls) -> list[dict[str, Any]]:
        """
        List all indexes defined on the model's table.
        """
        adapter = await cls.adapter()
        return await adapter.list_indexes()
