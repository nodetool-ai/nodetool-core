import asyncio
import json
import re
import sqlite3
from datetime import datetime
from enum import Enum
from enum import EnumMeta as EnumType
from types import UnionType
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

import aiosqlite
from pydantic.fields import FieldInfo

from nodetool.config.logging_config import get_logger
from nodetool.models.condition_builder import (
    Condition,
    ConditionBuilder,
    ConditionGroup,
    Operator,
)

from .database_adapter import DatabaseAdapter

log = get_logger(__name__)


async def retry_on_locked(func, max_retries=10, initial_delay=0.01):
    """Retry a database operation if it fails due to database lock.

    Uses exponential backoff with jitter to avoid thundering herd.

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts (default 10 for better coverage)
        initial_delay: Initial delay in seconds (doubled each retry)
    """
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await func()
        except (sqlite3.OperationalError, aiosqlite.OperationalError) as e:
            last_exception = e
            error_msg = str(e).lower()

            # Only retry on lock-related errors
            if "locked" not in error_msg and "busy" not in error_msg:
                raise

            if attempt < max_retries - 1:
                # Add jitter to prevent thundering herd
                import random

                jitter = delay * random.uniform(0.5, 1.5)
                log.debug(
                    f"Database locked, retrying in {jitter:.3f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(jitter)
                delay *= 2  # Exponential backoff (max ~5 seconds by final retry)
            else:
                log.error(f"Database operation failed after {max_retries} retries: {e}")
                raise

    raise last_exception or Exception("Unexpected error in retry_on_locked")


def convert_to_sqlite_format(
    value: Any, py_type: Type
) -> int | float | str | bytes | None:
    """
    Convert a Python value to a format suitable for SQLite based on the provided Python type.
    Serialize lists and dicts to JSON strings. Encode bytes using base64.

    :param value: The value to convert, or None.
    :param py_type: The Python type of the value.
    :return: The value converted to a SQLite-compatible format.
    """
    if py_type is None:
        return str(value)

    if value is None:
        return None

    origin = get_origin(py_type)
    if origin is Union or origin is UnionType:
        args = [t for t in py_type.__args__ if t is not type(None)]
        if len(args) == 1:
            return convert_to_sqlite_format(value, args[0])
        else:
            return json.dumps(value)

    if py_type in (str, int, float):
        return value
    elif py_type is set or origin is set:
        return json.dumps(list(value))
    elif py_type in (dict, list) or origin in (dict, list):
        return json.dumps(value)
    elif py_type is bytes:
        return value
    elif py_type is Any:
        return json.dumps(value)
    elif py_type is datetime:
        return value.isoformat()
    elif py_type is bool or (isinstance(py_type, type) and issubclass(py_type, bool)):
        return int(value)
    elif issubclass(py_type, Enum):
        return value.value
    else:
        raise TypeError(f"Unsupported type for SQLite: {py_type}")


def convert_from_sqlite_format(value: Any, py_type: Type) -> Any:
    """
    Convert a value from SQLite to a Python type based on the provided Python type.
    Deserialize JSON strings to lists and dicts.

    :param value: The value to convert, or None.
    :param py_type: The Python type of the value.
    :return: The value converted to a Python type.
    """
    if value is None:
        return None

    origin = get_origin(py_type)
    if origin is Union or origin is UnionType:
        args = [t for t in py_type.__args__ if t is not type(None)]
        if len(args) == 1:
            return convert_from_sqlite_format(value, args[0])
        else:
            return json.loads(value)

    if py_type in (str, int, float):
        return value
    elif py_type is Any:
        return json.loads(value)
    elif py_type is set or origin is set:
        return set(json.loads(value))
    elif py_type in (list, dict) or origin in (list, dict):
        return json.loads(value)
    elif py_type is bytes:
        return value
    elif py_type is datetime:
        return datetime.fromisoformat(value)
    elif py_type is bool or (isinstance(py_type, type) and issubclass(py_type, bool)):
        return bool(value)
    elif issubclass(py_type, Enum):
        return py_type(value)
    else:
        raise TypeError(f"Unsupported type for SQLite: {py_type}")


def convert_from_sqlite_attributes(
    attributes: Dict[str, Any], fields: Dict[str, FieldInfo]
) -> Dict[str, Any]:
    """
    Convert a dictionary of attributes from SQLite to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (
            convert_from_sqlite_format(attributes[key], fields[key].annotation)  # type: ignore
            if key in fields
            else attributes[key]
        )
        for key in attributes
    }


def convert_to_sqlite_attributes(
    attributes: Dict[str, Any], fields: Dict[str, FieldInfo]
) -> Dict[str, Any]:
    """
    Convert a dictionary of attributes from SQLite to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (
            convert_to_sqlite_format(attributes[key], fields[key].annotation)  # type: ignore
            if key in fields
            else attributes[key]
        )
        for key in attributes
    }


def get_sqlite_type(field_type: Any) -> str:
    # Check for Union or Optional types (Optional[X] is just Union[X, None] in typing)
    origin = get_origin(field_type)
    if origin is Union or origin is UnionType:
        # Assume the first non-None type is the desired type for SQLite
        # This works for Optional types as well
        _type = next(t for t in get_args(field_type) if t is not type(None))
        return get_sqlite_type(_type)

    # Direct mapping of Python types to SQLite types
    if field_type is str or field_type is Any or field_type in (list, dict, set) or origin in (list, dict, set):
        return "TEXT"
    elif field_type is int or field_type is bool:  # bool is stored as INTEGER (0 or 1)
        return "INTEGER"
    elif field_type is float:
        return "REAL"
    elif field_type is datetime:
        return "TEXT"
    elif field_type is bytes:  # bytes are stored as BLOB
        return "BLOB"
    elif field_type.__class__ is EnumType:
        return "TEXT"
    elif field_type is None:  # NoneType translates to NULL
        return "NULL"
    else:
        raise Exception(f"Unsupported field type: {field_type}")


def translate_condition_to_sql(condition: str) -> str:
    """
    Translates a condition string with custom syntax into an SQLite-compatible SQL condition string using regex.

    Args:
    - condition (str): The condition string to translate, e.g.,
                       "user_id = :user_id AND begins_with(content_type, :content_type)".

    Returns:
    - str: The translated SQL condition string compatible with SQLite.
    """

    # Define a regex pattern to match the begins_with syntax
    pattern = r"begins_with\((\w+),\s*:(\w+)\)"

    # Function to replace each match with the SQLite LIKE syntax
    def replacement(match):
        column_name, param_name = match.groups()
        return f"{column_name} LIKE :{param_name} || '%'"

    # Use the regex sub function to replace all occurrences of the pattern
    translated_condition = re.sub(pattern, replacement, condition)

    return translated_condition


class SQLiteAdapter(DatabaseAdapter):
    """
    Provides an adapter (`SQLiteAdapter`) to interface Pydantic-based models
    (specifically those using `DBModel` structure, although not directly imported here)
    with an SQLite database.

    Key functionalities include:
    - Automatic table creation and schema migration based on model fields.
    - Type conversion between Python types (including lists, dicts, enums, datetime)
      and SQLite-compatible storage formats (TEXT, INTEGER, REAL, BLOB).
    - Handling of database connections and transactions.
    - CRUD operations (save, get, delete).
    - A query interface that uses the `ConditionBuilder` system for constructing
      complex WHERE clauses.
    - Index creation, deletion, and listing.
    - Query timeout support to prevent long-running operations from blocking shutdown.

    Helper functions (`convert_to_sqlite_format`, `convert_from_sqlite_format`, etc.)
    manage the data type conversions.
    """

    table_name: str
    table_schema: Dict[str, Any]
    indexes: List[Dict[str, Any]]
    _connection: aiosqlite.Connection
    query_timeout: Optional[float]
    _is_shutting_down: bool

    def __init__(
        self,
        fields: Dict[str, FieldInfo],
        table_schema: Dict[str, Any],
        indexes: List[Dict[str, Any]],
        connection: aiosqlite.Connection,
        query_timeout: Optional[float] = None,
    ):
        """Initializes the SQLite adapter with an existing connection.

        Args:
            fields: Dictionary of Pydantic field info.
            table_schema: Dictionary defining the table schema.
            indexes: List of index configurations.
            connection: Existing aiosqlite.Connection instance.
            query_timeout: Optional timeout in seconds for queries. If None, queries
                          will not have a timeout (except during shutdown).
        """
        self.table_name = table_schema["table_name"]
        self.table_schema = table_schema
        self.fields = fields
        self.indexes = indexes
        self._connection = connection
        self.query_timeout = query_timeout
        self._is_shutting_down = False

    async def _execute_with_timeout(self, coro, timeout: Optional[float] = None):
        """Execute a coroutine with optional timeout.

        Uses the adapter's configured query_timeout if no specific timeout is provided.
        During shutdown, applies a short timeout to prevent hanging.

        Args:
            coro: The coroutine to execute.
            timeout: Optional timeout in seconds. If None, uses self.query_timeout.
                    During shutdown, uses a 5-second timeout regardless.

        Returns:
            The result of the coroutine.

        Raises:
            asyncio.TimeoutError: If the operation exceeds the timeout.
        """
        # Determine effective timeout
        effective_timeout = timeout
        if effective_timeout is None and self.query_timeout is not None:
            effective_timeout = self.query_timeout
        if self._is_shutting_down and (effective_timeout is None or effective_timeout > 5.0):
            effective_timeout = 5.0

        if effective_timeout is None:
            return await coro
        else:
            try:
                return await asyncio.wait_for(coro, timeout=effective_timeout)
            except TimeoutError:
                log.error(
                    f"Query timeout ({effective_timeout}s) on table {self.table_name}. "
                    f"Shutting down: {self._is_shutting_down}"
                )
                raise

    async def auto_migrate(
        self,
    ):
        """Run automatic migration for this table.

        Checks if migration has already been completed this session to avoid
        redundant work. Uses per-table locking to prevent concurrent migrations.
        """
        # Perform migration
        if await self.table_exists():
            await self.migrate_table()
        else:
            await self.create_table()
            for index in self.indexes:
                await self.create_index(
                    index["name"], index["columns"], index["unique"]
                )

    @property
    def connection(self) -> aiosqlite.Connection:
        return self._connection

    async def table_exists(self) -> bool:
        """Checks if the table associated with this adapter exists in the database."""
        cursor = await self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (self.table_name,),
        )
        return (await cursor.fetchone()) is not None

    async def get_current_schema(self) -> set[str]:
        """Retrieves the current schema of the table from the database."""
        cursor = await self.connection.execute(f"PRAGMA table_info({self.table_name})")
        rows = await cursor.fetchall()
        current_schema = {row[1] for row in rows}
        return current_schema

    def get_desired_schema(self) -> set[str]:
        """
        Retrieves the desired schema based on the defined fields.
        """
        desired_schema = set(self.fields.keys())
        return desired_schema

    async def create_table(self, suffix="") -> None:
        """Creates the database table based on the model's schema.

        Constructs and executes a CREATE TABLE SQL statement using the defined fields
        and their corresponding SQLite types.

        Args:
            suffix: Optional suffix to append to the table name (used for migrations).
        """
        table_name = self.table_name + suffix
        fields = self.fields
        primary_key = self.get_primary_key()
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        for field_name, field in fields.items():
            field_type = field.annotation
            sql += f"{field_name} {get_sqlite_type(field_type)}, "
        sql += f"PRIMARY KEY ({primary_key}))"

        async def _create():
            await self.connection.execute(sql)
            await self.connection.commit()

        try:
            await retry_on_locked(_create)
        except aiosqlite.Error as e:
            print(f"SQLite error during table creation: {e}")
            raise e

    async def drop_table(self) -> None:
        """Drops the database table associated with this adapter."""
        sql = f"DROP TABLE IF EXISTS {self.table_name}"

        async def _drop():
            await self.connection.execute(sql)
            await self.connection.commit()

        await retry_on_locked(_drop)

    async def migrate_table(self) -> None:
        """Performs schema migration for the table.

        Compares the current schema in the database with the model's defined schema.
        Adds new columns if they exist in the model but not in the database table.
        Handles adding columns by creating a new table, copying data, and replacing the old table.
        Drops columns that are no longer in the model schema (potential data loss).
        """
        current_schema = await self.get_current_schema()
        desired_schema = self.get_desired_schema()

        # Compare current and desired schemas
        fields_to_add = desired_schema - current_schema
        fields_to_remove = current_schema - desired_schema

        # Get current indexes
        current_indexes = {index["name"]: index for index in await self.list_indexes()}
        desired_indexes = {index["name"]: index for index in self.indexes}

        # Compare indexes
        indexes_to_add = set(desired_indexes.keys()) - set(current_indexes.keys())
        indexes_to_update = []

        # Check if existing indexes need updates
        for name in set(current_indexes.keys()) & set(desired_indexes.keys()):
            current = current_indexes[name]
            desired = desired_indexes[name]
            if (
                current["columns"] != desired["columns"]
                or current["unique"] != desired["unique"]
            ):
                indexes_to_update.append(name)

        # If no changes needed, return early
        if not (
            fields_to_add or fields_to_remove or indexes_to_add or indexes_to_update
        ):
            return

        # Drop affected indexes before table modifications
        if fields_to_remove:
            for index in await self.list_indexes():
                await self.drop_index(index["name"])
        else:
            for index_name in indexes_to_update:
                await self.drop_index(index_name)

        # Handle table schema changes
        if fields_to_add:
            for field_name in fields_to_add:
                # Refresh schema each time to avoid races; skip if already present
                try:
                    current_schema = await self.get_current_schema()
                except Exception:
                    current_schema = set()
                if field_name in current_schema:
                    continue

                field_type = get_sqlite_type(self.fields[field_name].annotation)
                try:
                    await self.connection.execute(
                        f"ALTER TABLE {self.table_name} ADD COLUMN {field_name} {field_type}"
                    )
                except sqlite3.OperationalError as e:
                    # If another concurrent migration added the column, ignore
                    if "duplicate column name" in str(e).lower():
                        pass
                    else:
                        raise

        if fields_to_remove:
            # Create new table with desired schema
            await self.create_table(suffix="_new")

            # Copy data
            columns = ", ".join(desired_schema)
            await self.connection.execute(
                f"INSERT INTO {self.table_name}_new ({columns}) SELECT {columns} FROM {self.table_name}"
            )

            await self.connection.execute(f"DROP TABLE {self.table_name}")
            await self.connection.execute(
                f"ALTER TABLE {self.table_name}_new RENAME TO {self.table_name}"
            )

            # Recreate all indexes
            for index in self.indexes:
                await self.create_index(
                    index["name"], index["columns"], index["unique"]
                )
        else:
            # Create new indexes and update modified ones
            for index_name in indexes_to_add | set(indexes_to_update):
                index = desired_indexes[index_name]
                await self.create_index(index_name, index["columns"], index["unique"])

        await self.connection.commit()

    async def save(self, item: Dict[str, Any]) -> None:
        """Saves (inserts or replaces) an item into the database table.

        Converts the item's attributes to SQLite-compatible formats before saving.

        Args:
            item: A dictionary representing the model instance to save.
        """
        valid_keys = [key for key in item if key in self.fields]
        columns = ", ".join(valid_keys)
        placeholders = ", ".join(["?" for _ in valid_keys])
        values = tuple(
            convert_to_sqlite_format(item[key], self.fields[key].annotation)  # type: ignore
            for key in valid_keys
        )
        query = f"INSERT OR REPLACE INTO {self.table_name} ({columns}) VALUES ({placeholders})"

        async def _save():
            await self._execute_with_timeout(
                self.connection.execute(query, values)
            )
            await self._execute_with_timeout(self.connection.commit())

        await retry_on_locked(_save)

    async def get(self, key: Any) -> Dict[str, Any] | None:
        """Retrieves an item from the database table by its primary key.

        Args:
            key: The primary key value of the item to retrieve.

        Returns:
            A dictionary representing the retrieved item, or None if not found.
            Attributes are converted back to their Python types.
        """
        primary_key = self.get_primary_key()
        cols = ", ".join(self.fields.keys())
        query = f"SELECT {cols} FROM {self.table_name} WHERE {primary_key} = ?"

        async def _get():
            cursor = await self._execute_with_timeout(
                self.connection.execute(query, (key,))
            )
            item = await self._execute_with_timeout(cursor.fetchone())
            if item is None:
                return None
            return convert_from_sqlite_attributes(dict(item), self.fields)

        return await _get()

    async def delete(self, primary_key: Any) -> None:
        """Deletes an item from the database table by its primary key.

        Args:
            primary_key: The primary key value of the item to delete.
        """
        pk_column = self.get_primary_key()
        query = f"DELETE FROM {self.table_name} WHERE {pk_column} = ?"

        async def _delete():
            await self._execute_with_timeout(
                self.connection.execute(query, (primary_key,))
            )
            await self._execute_with_timeout(self.connection.commit())

        await retry_on_locked(_delete)

    def _build_condition(
        self, condition: Condition | ConditionGroup
    ) -> tuple[str, list[Any]]:
        """Recursively builds an SQL WHERE clause and parameters from a Condition or ConditionGroup.

        Args:
            condition: The Condition or ConditionGroup object.

        Returns:
            A tuple containing the SQL WHERE clause string and a list of parameters.
        """
        if isinstance(condition, Condition):
            if condition.operator == Operator.IN:
                placeholders = ", ".join(["?" for _ in condition.value])
                sql = f"{condition.field} IN ({placeholders})"
                params = condition.value
            else:
                sql = f"{condition.field} {condition.operator.value} ?"
                params = [condition.value]
            return sql, params
        else:  # ConditionGroup
            sub_conditions = []
            params = []
            for sub_condition in condition.conditions:
                sub_sql, sub_params = self._build_condition(sub_condition)
                sub_conditions.append(sub_sql)
                params.extend(sub_params)
            if len(sub_conditions) == 1:
                return sub_conditions[0], params
            else:
                op = " " + condition.operator.value + " "
                return (
                    op.join(["(" + sub + ")" for sub in sub_conditions]),
                    params,
                )

    async def query(
        self,
        condition: ConditionBuilder | None = None,
        order_by: str | None = None,
        limit: int = 100,
        reverse: bool = False,
        columns: List[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        pk = self.get_primary_key()

        if order_by is None:
            order_by = f"{self.table_name}.{pk}"

        order_by = (
            f"{order_by} DESC"
            if reverse
            else f"{order_by} ASC"
        )

        if columns:
            cols = ", ".join([f"{self.table_name}.{col}" for col in columns])
        else:
            cols = ", ".join([f"{self.table_name}.{col}" for col in self.fields])

        params = []
        where_clause = "1=1"  # Default to select all if no condition
        if condition:  # Check if a condition was provided
            where_clause, params = self._build_condition(
                condition.root
            )  # Pass the root group

        fetch_limit = limit + 1
        query = f"SELECT {cols} FROM {self.table_name} WHERE {where_clause} ORDER BY {order_by} LIMIT {fetch_limit}"

        async def _query():
            cursor = await self._execute_with_timeout(
                self.connection.execute(query, params)
            )
            rows = await self._execute_with_timeout(cursor.fetchall())
            res = [convert_from_sqlite_attributes(dict(row), self.fields) for row in rows]

            if len(res) <= limit:
                return res, ""

            # Pop the extra record used to detect another page
            extra_record = res.pop()
            last_evaluated_key = str(res[-1].get(pk))
            # Guard: if extra record does not advance, fall back to extra key
            if not last_evaluated_key:
                last_evaluated_key = str(extra_record.get(pk))
            return res, last_evaluated_key

        return await _query()

    async def execute_sql(
        self, sql: str, params: Optional[dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executes a given SQL query with parameters and returns the results.

        Args:
            sql: The SQL query string to execute.
            params: A dictionary of parameters to bind to the query.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            returned by the query.
        """
        async def _execute():
            cursor = await self._execute_with_timeout(
                self.connection.execute(sql, params or {})
            )
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                rows = await self._execute_with_timeout(cursor.fetchall())
                return [
                    convert_from_sqlite_attributes(
                        dict(zip(columns, row, strict=False)), self.fields
                    )
                    for row in rows
                ]
            return []

        return await _execute()

    async def create_index(
        self, index_name: str, columns: List[str], unique: bool = False
    ) -> None:
        unique_str = "UNIQUE" if unique else ""
        columns_str = ", ".join(columns)
        sql = f"CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        try:
            await self.connection.execute(sql)
            await self.connection.commit()
        except aiosqlite.Error as e:
            print(f"SQLite error during index creation: {e}")
            raise e

    async def drop_index(self, index_name: str) -> None:
        sql = f"DROP INDEX IF EXISTS {index_name}"

        try:
            await self.connection.execute(sql)
            await self.connection.commit()
        except aiosqlite.Error as e:
            print(f"SQLite error during index deletion: {e}")
            raise e

    async def list_indexes(self) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name=?"

        try:
            cursor = await self.connection.execute(sql, (self.table_name,))
            rows = await cursor.fetchall()
            indexes = []
            for row in rows:
                # Skip system indexes (those starting with sqlite_)
                if row["name"].startswith("sqlite_"):
                    continue

                # Parse the CREATE INDEX statement to extract column names
                create_stmt = row["sql"]
                if not create_stmt:  # Add check for None or empty string
                    continue

                columns = create_stmt.split("(")[-1].split(")")[0].split(",")
                columns = [col.strip() for col in columns]

                indexes.append(
                    {
                        "name": row["name"],
                        "columns": columns,
                        "unique": "UNIQUE" in create_stmt.upper(),
                        "sql": create_stmt,
                    }
                )
            return indexes
        except aiosqlite.Error as e:
            print(f"SQLite error during index listing: {e}")
            raise e
