import re
from contextlib import asynccontextmanager
from datetime import datetime
from enum import EnumMeta as EnumType
from types import UnionType
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

# mypy: ignore-errors
import psycopg
from psycopg.rows import dict_row
from psycopg.sql import SQL, Composed, Identifier, Placeholder
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
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


def convert_to_postgres_format(value: Any, py_type: type | None) -> int | float | str | bytes | Jsonb | None:
    """
    Convert a Python value to a format suitable for PostgreSQL based on the provided Python type.
    Serialize lists and dicts to JSON strings. Encode bytes using base64.

    :param value: The value to convert, or None.
    :param py_type: The Python type of the value.
    :return: The value converted to a PostgreSQL-compatible format.
    """
    if value is None:
        return None

    if py_type is None:
        return value

    origin = get_origin(py_type)
    if origin is Union or origin is UnionType:
        args = [t for t in get_args(py_type) if t is not type(None)]
        if len(args) == 1:
            return convert_to_postgres_format(value, args[0])
        else:
            return value

    if py_type in (str, int, float, bool, datetime):
        return value
    elif py_type in (list, dict, set) or origin in (list, dict, set):
        return Jsonb(value)
    elif py_type is bytes:
        return value
    elif py_type is Any:
        return Jsonb(value)
    elif py_type.__class__ is EnumType:
        return value.value
    else:
        raise TypeError(f"Unsupported type for PostgreSQL: {py_type}")


def convert_from_postgres_format(value: Any, py_type: type | None) -> Any:
    """
    Convert a value from PostgreSQL to a Python type based on the provided Python type.
    Deserialize JSON strings to lists and dicts.

    :param value: The value to convert, or None.
    :param py_type: The Python type of the value.
    :return: The value converted to a Python type.
    """
    if value is None:
        return None

    if py_type is None:
        return value

    origin = get_origin(py_type)
    if origin is Union or origin is UnionType:
        args = [t for t in get_args(py_type) if t is not type(None)]
        if len(args) == 1:
            return convert_from_postgres_format(value, args[0])
        else:
            return value

    if py_type in (str, int, float, bool, bytes, dict, list, set) or origin in (dict, list, set) or py_type is datetime:
        return value
    elif py_type.__class__ is EnumType:
        return py_type(value)
    else:
        raise TypeError(f"Unsupported type for PostgreSQL: {py_type}")


def convert_from_postgres_attributes(attributes: dict[str, Any], fields: dict[str, FieldInfo]) -> dict[str, Any]:
    """
    Convert a dictionary of attributes from PostgreSQL to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (
            convert_from_postgres_format(attributes[key], fields[key].annotation) if key in fields else attributes[key]
        )
        for key in attributes
    }


def convert_to_postgres_attributes(attributes: dict[str, Any], fields: dict[str, FieldInfo]) -> dict[str, Any]:
    """
    Convert a dictionary of attributes from PostgreSQL to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (convert_to_postgres_format(attributes[key], fields[key].annotation) if key in fields else attributes[key])
        for key in attributes
    }


def get_postgres_type(field_type: Any) -> str:
    # Check for Union or Optional types (Optional[X] is just Union[X, None] in typing)
    origin = get_origin(field_type)
    if origin is Union or origin is UnionType:
        # Assume the first non-None type is the desired type for PostgreSQL
        # This works for Optional types as well
        _type = next(t for t in get_args(field_type) if t is not type(None))
        return get_postgres_type(_type)

    # Direct mapping of Python types to PostgreSQL types
    if field_type is str:
        return "TEXT"
    elif field_type is Any or field_type in (list, dict, set) or origin in (list, dict, set):
        return "JSONB"
    elif field_type is int:
        return "INTEGER"
    elif field_type is bool:
        return "BOOLEAN"
    elif field_type is float:
        return "REAL"
    elif field_type is datetime:
        return "TIMESTAMP"
    elif field_type is bytes:
        return "BYTEA"
    elif field_type is None:
        return "NULL"
    elif field_type.__class__ is EnumType:
        return "TEXT"
    else:
        raise Exception(f"Unsupported field type: {field_type}")


def translate_condition_to_sql(condition: str) -> str:
    """
    Translates a condition string with custom syntax into a PostgreSQL-compatible SQL condition string using regex.

    Args:
    - condition (str): The condition string to translate, e.g.,
                       "user_id = :user_id AND begins_with(content_type, :content_type)".

    Returns:
    - str: The translated SQL condition string compatible with PostgreSQL.
    """

    # Define a regex pattern to match the begins_with syntax
    pattern = r"begins_with\((\w+),\s*:(\w+)\)"

    # Function to replace each match with the PostgreSQL LIKE syntax
    def replacement(match):
        column_name, param_name = match.groups()
        return f"{column_name} LIKE %({param_name})s || '%'"

    # Use the regex sub function to replace all occurrences of the pattern
    translated_condition = re.sub(pattern, replacement, condition)

    # Replace : with %() for PostgreSQL parameter style
    translated_condition = re.sub(r":(\w+)", r"%({\1})s", translated_condition)

    return translated_condition


def translate_postgres_params(query: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Translate SQLite-style named parameters to PostgreSQL-style parameters.
    """
    translated_query = re.sub(r":(\w+)", r"%(\1)s", query)
    return translated_query, params


class PostgresAdapter(DatabaseAdapter):
    """Adapts DBModel operations to a PostgreSQL database."""

    db_params: dict[str, str]
    table_name: str
    table_schema: dict[str, Any]
    fields: dict[str, FieldInfo]
    indexes: list[dict[str, Any]]
    _pool: AsyncConnectionPool | None

    def __init__(
        self,
        db_params: dict[str, str],
        fields: dict[str, FieldInfo],
        table_schema: dict[str, Any],
        indexes: list[dict[str, Any]],
    ):
        """Initializes the PostgreSQL adapter.

        Establishes connection parameters, checks if the table exists, performs migrations
        or creates the table and indexes as necessary.

        Args:
            db_params: Dictionary containing PostgreSQL connection parameters (database, user, password, host, port).
            fields: Dictionary mapping field names to Pydantic FieldInfo objects.
            table_schema: Schema definition for the table.
            indexes: List of index definitions for the table.
        """
        self.db_params = db_params
        self.table_name = table_schema["table_name"]
        self.table_schema = table_schema
        self.fields = fields
        self.indexes = indexes
        self._pool = None

    async def initialize(self) -> None:
        """Initialize the adapter asynchronously."""
        if await self.table_exists():
            await self.migrate_table()
            for index in self.indexes:
                await self.create_index(index["name"], index["columns"], index["unique"])
        else:
            await self.create_table()
            for index in self.indexes:
                await self.create_index(index["name"], index["columns"], index["unique"])

    async def _get_pool(self) -> AsyncConnectionPool:
        """Provides a lazy-loaded PostgreSQL connection pool using psycopg."""
        if self._pool is None:
            conninfo = f"dbname={self.db_params['database']} user={self.db_params['user']} password={self.db_params['password']} host={self.db_params['host']} port={self.db_params['port']}"
            self._pool = AsyncConnectionPool(conninfo, min_size=1, max_size=10)
            await self._pool.open()
        return self._pool

    async def table_exists(self) -> bool:
        """Checks if the table associated with this adapter exists in the database."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (self.table_name,),
            )
            res = await cursor.fetchone()
            if res is None:
                return False
            return res["exists"]

    async def get_current_schema(self) -> set[str]:
        """Retrieves the current schema (column names) of the table from the database."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (self.table_name,),
            )
            rows = await cursor.fetchall()
            current_schema = {row["column_name"] for row in rows}
        return current_schema

    def get_desired_schema(self) -> set[str]:
        """Gets the desired schema (column names) based on the model's fields."""
        desired_schema = set(self.fields)
        return desired_schema

    async def create_table(self, suffix: str = "") -> None:
        """Creates the database table based on the model's schema.

        Constructs and executes a CREATE TABLE SQL statement using the defined fields
        and their corresponding PostgreSQL types.

        Args:
            suffix: Optional suffix to append to the table name (used for migrations).
        """
        table_name = f"{self.table_name}{suffix}"
        fields = self.fields
        primary_key = self.get_primary_key()
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        for field_name, field in fields.items():
            field_type = field.annotation
            sql += f"{field_name} {get_postgres_type(field_type)}, "
        sql += f"PRIMARY KEY ({primary_key}))"

        try:
            pool = await self._get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql)  # type: ignore[arg-type]
                await conn.commit()
        except psycopg.Error as e:
            print(f"PostgreSQL error during table creation: {e}")
            raise e

    async def drop_table(self) -> None:
        """Drops the database table associated with this adapter."""
        sql = f"DROP TABLE IF EXISTS {self.table_name}"
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql)  # type: ignore[arg-type]
            await conn.commit()

    async def migrate_table(self) -> None:
        """Performs schema migration for the table.

        Compares the current schema in the database with the model's defined schema.
        Adds new columns (using ALTER TABLE ADD COLUMN) if they exist in the model
        but not in the database table.
        Note: Does not currently handle column type changes or removals.
        """
        current_schema = await self.get_current_schema()
        desired_schema = self.get_desired_schema()

        # Compare current and desired schemas
        fields_to_add = desired_schema - current_schema
        fields_to_remove = current_schema - desired_schema

        if len(fields_to_remove) == 0 and len(fields_to_add) == 0:
            return

        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                # Alter table to add new fields
                for field_name in fields_to_add:
                    field_type = get_postgres_type(self.fields[field_name].annotation)
                    await cursor.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {field_name} {field_type}")  # type: ignore[arg-type]

                # Alter table to remove fields
                for field_name in fields_to_remove:
                    await cursor.execute(f"ALTER TABLE {self.table_name} DROP COLUMN {field_name}")  # type: ignore[arg-type]

            await conn.commit()

        # Only create indexes for new fields
        for field_name in fields_to_add:
            for index in self.indexes:
                if field_name in index["columns"]:
                    await self.create_index(index["name"], index["columns"], index["unique"])

    async def save(self, item: dict[str, Any]) -> None:
        """Saves (inserts or updates) an item into the database table.

        Uses an INSERT ... ON CONFLICT (primary_key) DO UPDATE statement.
        Converts the item's attributes to PostgreSQL-compatible formats before saving.

        Args:
            item: A dictionary representing the model instance to save.
        """
        valid_keys = [key for key in item if key in self.fields]
        columns = ", ".join(valid_keys)
        placeholders = ", ".join([f"%({key})s" for key in valid_keys])
        values = {key: convert_to_postgres_format(item[key], self.fields[key].annotation) for key in valid_keys}
        query = (
            f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders}) ON CONFLICT ({self.get_primary_key()}) DO UPDATE SET "
            + ", ".join([f"{key} = EXCLUDED.{key}" for key in valid_keys])
        )
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, values)  # type: ignore[arg-type]
            await conn.commit()

    async def get(self, key: Any) -> dict[str, Any] | None:
        """Retrieves an item from the database table by its primary key.

        Args:
            key: The primary key value of the item to retrieve.

        Returns:
            A dictionary representing the retrieved item, or None if not found.
            Attributes are converted back to their Python types.
        """
        primary_key = self.get_primary_key()
        cols = ", ".join(self.fields)
        query = f"SELECT {cols} FROM {self.table_name} WHERE {primary_key} = %s"
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, (key,))  # type: ignore[arg-type]
            item = await cursor.fetchone()
        if item is None:
            return None
        return convert_from_postgres_attributes(dict(item), self.fields)

    async def delete(self, primary_key: Any) -> None:
        """Deletes an item from the database table by its primary key.

        Args:
            primary_key: The primary key value of the item to delete.
        """
        pk_column = self.get_primary_key()
        query = f"DELETE FROM {self.table_name} WHERE {pk_column} = %s"
        pool = await self._get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (primary_key,))  # type: ignore[arg-type]
            await conn.commit()

    def _build_condition(self, condition: Condition | ConditionGroup) -> tuple[Composed, list[Any]]:
        """Recursively builds a psycopg2 SQL Composed object and parameters for a WHERE clause.

        Args:
            condition: The Condition or ConditionGroup object.

        Returns:
            A tuple containing the SQL Composed object for the WHERE clause and a list of parameters.
        """
        if isinstance(condition, Condition):
            if condition.operator == Operator.IN:
                placeholders = SQL(", ").join([Placeholder()] * len(condition.value))
                sql = SQL("{} IN ({})").format(Identifier(condition.field), placeholders)
                params = condition.value
            else:
                sql = SQL("{} {} {}").format(
                    Identifier(condition.field),
                    SQL(condition.operator.value),
                    Placeholder(),
                )
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
                return (
                    SQL("({})").format(SQL(f" {condition.operator.value} ").join(sub_conditions)),  # type: ignore[arg-type]
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
        if order_by:
            order_clause = SQL("{}.{} {}").format(
                Identifier(self.table_name), Identifier(order_by), SQL("DESC" if reverse else "ASC")
            )
        else:
            order_clause = SQL("{}.{} DESC" if reverse else "{}.{} ASC").format(
                Identifier(self.table_name), Identifier(pk)
            )

        if condition is not None:
            where_clause, params = self._build_condition(condition.build())
        else:
            where_clause = SQL("1=1")
            params = []

        if columns:
            if columns == ["*"]:
                cols = SQL("*")
            else:
                cols = SQL(", ").join(
                    [SQL("{}.{}").format(Identifier(self.table_name), Identifier(col)) for col in columns]
                )
        else:
            cols = SQL(", ").join(
                [SQL("{}.{}").format(Identifier(self.table_name), Identifier(col)) for col in self.fields]
            )

        fetch_limit = limit + 1
        query = SQL("SELECT {} FROM {} WHERE {} ORDER BY {} LIMIT {}").format(
            cols,
            Identifier(self.table_name),
            where_clause,
            order_clause,
            SQL(str(fetch_limit)),  # type: ignore[arg-type]
        )

        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(query, params)
            rows = await cursor.fetchall()
            res = [convert_from_postgres_attributes(dict(row), self.fields) for row in rows]

        if len(res) <= limit:
            return res, ""

        extra_record = res.pop()
        last_evaluated_key = str(res[-1].get(pk))
        if not last_evaluated_key:
            last_evaluated_key = str(extra_record.get(pk))
        return res, last_evaluated_key

    @asynccontextmanager
    async def get_cursor(self):
        """Provides a database cursor within a context manager, handling commit/rollback."""
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            yield cursor

    async def execute_sql(self, sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Executes a given SQL query with parameters and returns the results.

        Uses a RealDictCursor to return rows as dictionaries.

        Args:
            sql: The SQL query string to execute (can be Composed SQL).
            params: A dictionary of parameters to bind to the query.

        Returns:
            A list of dictionaries, where each dictionary represents a row
            returned by the query.
        """
        translated_sql, translated_params = translate_postgres_params(sql, params or {})
        pool = await self._get_pool()
        async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(translated_sql, translated_params)  # type: ignore[arg-type]
            if cursor.description:
                rows = await cursor.fetchall()
                return [convert_from_postgres_attributes(dict(row), self.fields) for row in rows]
            return []

    async def create_index(self, index_name: str, columns: list[str], unique: bool = False) -> None:
        unique_str = "UNIQUE" if unique else ""
        columns_str = ", ".join(columns)
        sql = f"CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        try:
            pool = await self._get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql)  # type: ignore[arg-type]
                await conn.commit()
        except psycopg.Error as e:
            print(f"PostgreSQL error during index creation: {e}")
            raise e

    async def drop_index(self, index_name: str) -> None:
        sql = f"DROP INDEX IF EXISTS {index_name}"

        try:
            pool = await self._get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute(sql)  # type: ignore[arg-type]
                await conn.commit()
        except psycopg.Error as e:
            print(f"PostgreSQL error during index deletion: {e}")
            raise e

    async def list_indexes(self) -> list[dict[str, Any]]:
        sql = """
            SELECT
                i.relname as index_name,
                array_agg(a.attname) as column_names,
                ix.indisunique as is_unique
            FROM
                pg_class t,
                pg_class i,
                pg_index ix,
                pg_attribute a
            WHERE
                t.oid = ix.indrelid
                AND i.oid = ix.indexrelid
                AND a.attrelid = t.oid
                AND a.attnum = ANY(ix.indkey)
                AND t.relkind = 'r'
                AND t.relname = %s
            GROUP BY
                i.relname,
                ix.indisunique
            ORDER BY
                i.relname;
        """

        try:
            pool = await self._get_pool()
            async with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
                await cursor.execute(sql, (self.table_name,))
                rows = await cursor.fetchall()
                indexes = []
                for row in rows:
                    indexes.append(
                        {
                            "name": row["index_name"],
                            "columns": row["column_names"],
                            "unique": row["is_unique"],
                        }
                    )
                return indexes
        except psycopg.Error as e:
            print(f"PostgreSQL error during index listing: {e}")
            raise e

    async def auto_migrate(self) -> None:
        """Run automatic migration for this table.

        DEPRECATED: This method is deprecated. Schema migrations are now
        handled by the dedicated migration system in nodetool.migrations.
        This method is kept for backward compatibility but does nothing.
        Use 'nodetool migrations upgrade' CLI command instead.
        """
        # Deprecated - migrations now handled by nodetool.migrations
        pass

    def __del__(self):
        """Cleanup method when the adapter object is garbage collected."""
        # Note: In async context, close() should be called explicitly
        pass
