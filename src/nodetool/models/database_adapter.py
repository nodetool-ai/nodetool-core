from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic.fields import FieldInfo

from nodetool.models.condition_builder import ConditionBuilder

"""
Defines the Abstract Base Class (ABC) for database adapters.

This module specifies the interface that all database adapters (like SQLite, PostgreSQL)
must implement to interact with the DBModel base class. It ensures a consistent API
for database operations across different database systems.

Note: Migration functionality has been moved to the dedicated migration system
in nodetool.migrations. The auto_migrate method is deprecated and will be removed.
"""


class DatabaseAdapter(ABC):
    """Abstract Base Class for database adapters.

    Defines the common interface for interacting with different database backends.
    Subclasses must implement all abstract methods to provide concrete database operations.

    Note: Schema migrations are now handled by the migration system in nodetool.migrations.
    The auto_migrate method is deprecated and should not be used for new code.
    """

    fields: Dict[str, FieldInfo]
    table_name: str
    table_schema: Dict[str, Any]

    @abstractmethod
    async def create_table(self) -> None:
        """Creates the database table for the associated model."""
        pass

    @abstractmethod
    async def drop_table(self) -> None:
        """Drops the database table for the associated model."""
        pass

    @abstractmethod
    async def save(self, item: Dict[str, Any]) -> None:
        """Saves (inserts or updates) an item in the database.

        Args:
            item: A dictionary representing the model instance to save.
        """
        pass

    @abstractmethod
    async def get(self, key: Any) -> Dict[str, Any] | None:
        """Retrieves an item from the database by its primary key.

        Args:
            key: The primary key value of the item to retrieve.

        Returns:
            A dictionary representing the item, or None if not found.
        """
        pass

    @abstractmethod
    async def delete(self, primary_key: Any) -> None:
        """Deletes an item from the database by its primary key.

        Args:
            primary_key: The primary key value of the item to delete.
        """
        pass

    @abstractmethod
    async def query(
        self,
        condition: ConditionBuilder | None = None,
        order_by: str | None = None,
        limit: int = 100,
        reverse: bool = False,
        columns: List[str] | None = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Queries the database based on specified conditions.

        Args:
            condition: A ConditionBuilder object defining the query filters.
            limit: The maximum number of items to return.
            reverse: Whether to reverse the sort order (typically by primary key or a defined sort key).
            columns: Optional list of columns to return.
            order_by: The column to order the results by.

        Returns:
            A tuple containing a list of matching items (as dictionaries) and
            a pagination key/token (e.g., the last evaluated key) for fetching the next page.
            The pagination key should be an empty string if there are no more results.
        """
        pass

    def get_primary_key(self) -> str:
        """
        Get the name of the primary key.
        """
        return self.table_schema.get("primary_key", "id")

    @abstractmethod
    async def create_index(self, index_name: str, columns: List[str], unique: bool = False) -> None:
        """
        Create an index on the table with the specified columns.
        :param index_name: The name of the index to create.
        :param columns: A list of column names on which the index will be built.
        :param unique: Whether the index should enforce uniqueness.
        """
        pass

    @abstractmethod
    async def drop_index(self, index_name: str) -> None:
        """
        Drop the index with the given name from the table.
        :param index_name: The name of the index to drop.
        """
        pass

    @abstractmethod
    async def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all the indexes for the table.
        :return: A list of dictionaries representing index metadata.
        """
        pass

    async def auto_migrate(self) -> None:
        """
        Automatically migrate the table to current schema.

        DEPRECATED: This method is deprecated. Schema migrations are now
        handled by the dedicated migration system in nodetool.migrations.
        This method is kept for backward compatibility but does nothing.
        Use 'nodetool migrations upgrade' CLI command instead.
        """
        # Deprecated - migrations now handled by nodetool.migrations
        pass
