import asyncio
import base64
import json
import re
import sqlite3
from datetime import datetime
from enum import Enum
from enum import EnumMeta as EnumType
from types import UnionType
from typing import Any, Union, get_args, get_origin

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


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles bytes and other non-serializable types.

    Bytes are encoded as base64 strings with a special marker for decoding.
    Other non-serializable types are converted to their string representation.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            # Encode bytes as base64 with a marker for decoding
            return {"__bytes__": base64.b64encode(o).decode("ascii")}
        if isinstance(o, datetime):
            return {"__datetime__": o.isoformat()}
        if isinstance(o, set):
            return {"__set__": list(o)}
        # For any other non-serializable type, convert to string
        try:
            return super().default(o)
        except TypeError:
            return {"__repr__": repr(o)}


def _decode_special_types(obj: Any) -> Any:
    """
    Object hook for json.loads to decode special types encoded by SafeJSONEncoder.
    """
    if isinstance(obj, dict):
        if "__bytes__" in obj and len(obj) == 1:
            return base64.b64decode(obj["__bytes__"])
        if "__datetime__" in obj and len(obj) == 1:
            return datetime.fromisoformat(obj["__datetime__"])
        if "__set__" in obj and len(obj) == 1:
            return set(obj["__set__"])
        # __repr__ values are left as-is (they're informational)
    return obj


def safe_json_dumps(value: Any) -> str:
    """
    Serialize a value to JSON, handling bytes and other non-serializable types.
    """
    return json.dumps(value, cls=SafeJSONEncoder)


def safe_json_loads(value: str) -> Any:
    """
    Deserialize JSON, decoding special types encoded by safe_json_dumps.
    """
    return json.loads(value, object_hook=_decode_special_types)


async def retry_on_locked(func, max_retries=20, initial_delay=0.02):
    """Retry a database operation if it fails due to database lock.

    Uses exponential backoff with jitter to avoid thundering herd.

    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts (default 20 for high-concurrency scenarios)
        initial_delay: Initial delay in seconds (doubled each retry, starts at 20ms)
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
                log.debug(f"Database locked, retrying in {jitter:.3f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(jitter)
                delay = min(delay * 2, 3.0)  # Exponential backoff capped at 3 seconds
            else:
                log.error(f"Database operation failed after {max_retries} retries: {e}")
                raise

    raise last_exception or Exception("Unexpected error in retry_on_locked")


def convert_to_sqlite_format(value: Any, py_type: type) -> int | float | str | bytes | None:
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
            return safe_json_dumps(value)

    if py_type in (str, int, float):
        return value
    elif py_type is set or origin is set:
        return safe_json_dumps(list(value))
    elif py_type in (dict, list) or origin in (dict, list):
        return safe_json_dumps(value)
    elif py_type is bytes:
        return value
    elif py_type is Any:
        return safe_json_dumps(value)
    elif py_type is datetime:
        return value.isoformat()
    elif py_type is bool or (isinstance(py_type, type) and issubclass(py_type, bool)):
        return int(value)
    elif issubclass(py_type, Enum):
        return value.value
    else:
        raise TypeError(f"Unsupported type for SQLite: {py_type}")


def convert_from_sqlite_format(value: Any, py_type: type) -> Any:
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
            return safe_json_loads(value)

    if py_type in (str, int, float):
        return value
    elif py_type is Any:
        return safe_json_loads(value)
    elif py_type is set or origin is set:
        return set(safe_json_loads(value))
    elif py_type in (list, dict) or origin in (list, dict):
        return safe_json_loads(value)
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


def convert_from_sqlite_attributes(attributes: dict[str, Any], fields: dict[str, FieldInfo]) -> dict[str, Any]:
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


def convert_to_sqlite_attributes(attributes: dict[str, Any], fields: dict[str, FieldInfo]) -> dict[str, Any]:
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


VALID_COLUMN_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_column_name(column_name: str) -> str:
    """Validate column name to prevent SQL injection.

    Args:
        column_name: The column name to validate.

    Returns:
        The validated column name.

    Raises:
        ValueError: If the column name contains invalid characters.
    """
    if not VALID_COLUMN_NAME_RE.match(column_name):
        raise ValueError(f"Invalid column name: {column_name}")
    return column_name


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
    table_schema: dict[str, Any]
    indexes: list[dict[str, Any]]
    _connection: aiosqlite.Connection
    query_timeout: float | None
    _is_shutting_down: bool

    def __init__(
        self,
        fields: dict[str, FieldInfo],
        table_schema: dict[str, Any],
        indexes: list[dict[str, Any]],
        connection: aiosqlite.Connection,
        query_timeout: float | None = None,
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

    async def _execute_with_timeout(self, coro, timeout: float | None = None):
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
    ) -> None:
        """Run automatic migration for this table.

        DEPRECATED: This method is deprecated. Schema migrations are now
        handled by the dedicated migration system in nodetool.migrations.
        This method is kept for backward compatibility but does nothing.
        Use 'nodetool migrations upgrade' CLI command instead.
        """
        # Deprecated - migrations now handled by nodetool.migrations
        # This method intentionally does nothing to avoid conflicts with
        # the new migration system
        pass

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
        log.info(f"Creating table {table_name}")
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
            log.error(f"SQLite error during table creation: {e}")
            raise e

    async def drop_table(self) -> None:
        """Drops the database table associated with this adapter."""
        log.warning(f"Dropping table {self.table_name}")
        sql = f"DROP TABLE IF EXISTS {self.table_name}"

        async def _drop():
            await self.connection.execute(sql)
            await self.connection.commit()

        await retry_on_locked(_drop)

    async def save(self, item: dict[str, Any]) -> None:
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
            await self._execute_with_timeout(self.connection.execute(query, values))
            await self._execute_with_timeout(self.connection.commit())

        await retry_on_locked(_save)

    async def get(self, key: Any) -> dict[str, Any] | None:
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
            cursor = await self._execute_with_timeout(self.connection.execute(query, (key,)))
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
        log.info(f"Deleting record {primary_key} from {self.table_name}")
        query = f"DELETE FROM {self.table_name} WHERE {pk_column} = ?"

        async def _delete():
            await self._execute_with_timeout(self.connection.execute(query, (primary_key,)))
            await self._execute_with_timeout(self.connection.commit())

        await retry_on_locked(_delete)

    def _build_condition(self, condition: Condition | ConditionGroup) -> tuple[str, list[Any]]:
        """Recursively builds an SQL WHERE clause and parameters from a Condition or ConditionGroup.

        Args:
            condition: The Condition or ConditionGroup object.

        Returns:
            A tuple containing the SQL WHERE clause string and a list of parameters.

        Raises:
            ValueError: If a column name contains invalid characters.
        """
        if isinstance(condition, Condition):
            validated_field = _validate_column_name(condition.field)
            if condition.operator == Operator.IN:
                placeholders = ", ".join(["?" for _ in condition.value])
                sql = f"{validated_field} IN ({placeholders})"
                params = condition.value
            else:
                sql = f"{validated_field} {condition.operator.value} ?"
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
        columns: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        pk = self.get_primary_key()

        if order_by is None:
            order_by = f"{self.table_name}.{pk}"

        order_by = f"{order_by} DESC" if reverse else f"{order_by} ASC"

        if columns:
            cols = ", ".join([f"{self.table_name}.{col}" for col in columns])
        else:
            cols = ", ".join([f"{self.table_name}.{col}" for col in self.fields])

        params = []
        where_clause = "1=1"  # Default to select all if no condition
        if condition:  # Check if a condition was provided
            where_clause, params = self._build_condition(condition.root)  # Pass the root group

        fetch_limit = limit + 1
        query = f"SELECT {cols} FROM {self.table_name} WHERE {where_clause} ORDER BY {order_by} LIMIT {fetch_limit}"

        async def _query():
            cursor = await self._execute_with_timeout(self.connection.execute(query, params))
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

    async def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Executes a given SQL query with parameters and returns the results.

        Args:
            sql: The SQL query string to execute.
            params: A dictionary of parameters to bind to the query.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            returned by the query.
        """

        async def _execute():
            cursor = await self._execute_with_timeout(self.connection.execute(sql, params or {}))
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                rows = await self._execute_with_timeout(cursor.fetchall())
                return [
                    convert_from_sqlite_attributes(dict(zip(columns, row, strict=False)), self.fields) for row in rows
                ]
            return []

        return await _execute()

    async def create_index(self, index_name: str, columns: list[str], unique: bool = False) -> None:
        unique_str = "UNIQUE" if unique else ""
        columns_str = ", ".join(columns)
        sql = f"CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        log.info(f"Creating index {index_name} on {self.table_name}")
        try:
            await self.connection.execute(sql)
            await self.connection.commit()
        except aiosqlite.Error as e:
            log.error(f"SQLite error during index creation: {e}")
            raise e

    async def drop_index(self, index_name: str) -> None:
        sql = f"DROP INDEX IF EXISTS {index_name}"

        log.info(f"Dropping index {index_name}")
        try:
            await self.connection.execute(sql)
            await self.connection.commit()
        except aiosqlite.Error as e:
            log.error(f"SQLite error during index deletion: {e}")
            raise e

    async def list_indexes(self) -> list[dict[str, Any]]:
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
            log.error(f"SQLite error during index listing: {e}")
            raise e
