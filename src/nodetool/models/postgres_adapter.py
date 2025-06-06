from datetime import datetime
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from types import UnionType
from typing import Any, Dict, List, Optional, get_args
from pydantic.fields import FieldInfo

from nodetool.common.environment import Environment
from nodetool.models.condition_builder import (
    Condition,
    ConditionBuilder,
    ConditionGroup,
    Operator,
)
from contextlib import contextmanager
from .database_adapter import DatabaseAdapter
from typing import Type, Union, get_origin
from psycopg2.extras import Json
from psycopg2.sql import SQL, Identifier, Placeholder, Composed
from enum import EnumMeta as EnumType


log = Environment.get_logger()


def convert_to_postgres_format(
    value: Any, py_type: Type | None
) -> Union[int, float, str, psycopg2.Binary, Json, None]:
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
        args = [t for t in py_type.__args__ if t is not type(None)]
        if len(args) == 1:
            return convert_to_postgres_format(value, args[0])
        else:
            return value

    if py_type in (str, int, float, bool, datetime):
        return value
    elif py_type in (list, dict, set) or origin in (list, dict, set):
        return Json(value)
    elif py_type is bytes:
        return psycopg2.Binary(value)
    elif py_type is Any:
        return Json(value)
    elif py_type.__class__ is EnumType:
        return value.value
    else:
        raise TypeError(f"Unsupported type for PostgreSQL: {py_type}")


def convert_from_postgres_format(value: Any, py_type: Type | None) -> Any:
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
        args = [t for t in py_type.__args__ if t is not type(None)]
        if len(args) == 1:
            return convert_from_postgres_format(value, args[0])
        else:
            return value

    if py_type in (str, int, float, bool, bytes, dict, list, set):
        return value
    elif origin in (dict, list, set):
        return value
    elif py_type is datetime:
        return value
    elif py_type.__class__ is EnumType:
        return py_type(value)
    else:
        raise TypeError(f"Unsupported type for PostgreSQL: {py_type}")


def convert_from_postgres_attributes(
    attributes: Dict[str, Any], fields: Dict[str, FieldInfo]
) -> Dict[str, Any]:
    """
    Convert a dictionary of attributes from PostgreSQL to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (
            convert_from_postgres_format(attributes[key], fields[key].annotation)
            if key in fields
            else attributes[key]
        )
        for key in attributes
    }


def convert_to_postgres_attributes(
    attributes: Dict[str, Any], fields: Dict[str, FieldInfo]
) -> Dict[str, Any]:
    """
    Convert a dictionary of attributes from PostgreSQL to a dictionary of Python types based on the provided fields.
    """
    return {
        key: (
            convert_to_postgres_format(attributes[key], fields[key].annotation)
            if key in fields
            else attributes[key]
        )
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
    elif field_type is Any:
        return "JSONB"
    # Serialized to JSON
    elif field_type in (list, dict, set) or origin in (list, dict, set):
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


def translate_postgres_params(
    query: str, params: Dict[str, Any]
) -> tuple[str, Dict[str, Any]]:
    """
    Translate SQLite-style named parameters to PostgreSQL-style parameters.
    """
    translated_query = re.sub(r":(\w+)", r"%(\1)s", query)
    return translated_query, params


class PostgresAdapter(DatabaseAdapter):
    """Adapts DBModel operations to a PostgreSQL database."""

    db_params: Dict[str, str]
    table_name: str
    table_schema: Dict[str, Any]
    fields: Dict[str, FieldInfo]
    indexes: List[Dict[str, Any]]

    def __init__(
        self,
        db_params: Dict[str, str],
        fields: Dict[str, FieldInfo],
        table_schema: Dict[str, Any],
        indexes: List[Dict[str, Any]],
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
        if self.table_exists():
            self.migrate_table()
            for index in self.indexes:
                self.create_index(index["name"], index["columns"], index["unique"])
        else:
            self.create_table()
            for index in self.indexes:
                self.create_index(index["name"], index["columns"], index["unique"])

    @property
    def connection(self):
        """Provides a lazy-loaded PostgreSQL database connection using psycopg2."""
        if not hasattr(self, "_connection"):
            self._connection = psycopg2.connect(
                database=self.db_params["database"],
                user=self.db_params["user"],
                password=self.db_params["password"],
                host=self.db_params["host"],
                port=self.db_params["port"],
            )
        return self._connection

    def table_exists(self) -> bool:
        """Checks if the table associated with this adapter exists in the database."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (self.table_name,),
            )
            res = cursor.fetchone()
            if res is None:
                return False
            return res[0]

    def get_current_schema(self) -> set[str]:
        """Retrieves the current schema (column names) of the table from the database."""
        with self.connection.cursor() as cursor:
            cursor.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (self.table_name,),
            )
            current_schema = {row[0] for row in cursor.fetchall()}
        return current_schema

    def get_desired_schema(self) -> set[str]:
        """Gets the desired schema (column names) based on the model's fields."""
        desired_schema = set(self.fields.keys())
        return desired_schema

    def create_table(self, suffix: str = "") -> None:
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
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"PostgreSQL error during table creation: {e}")
            raise e

    def drop_table(self) -> None:
        """Drops the database table associated with this adapter."""
        sql = f"DROP TABLE IF EXISTS {self.table_name}"
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
        self.connection.commit()

    def migrate_table(self) -> None:
        """Performs schema migration for the table.

        Compares the current schema in the database with the model's defined schema.
        Adds new columns (using ALTER TABLE ADD COLUMN) if they exist in the model
        but not in the database table.
        Note: Does not currently handle column type changes or removals.
        """
        current_schema = self.get_current_schema()
        desired_schema = self.get_desired_schema()

        # Compare current and desired schemas
        fields_to_add = desired_schema - current_schema
        fields_to_remove = current_schema - desired_schema

        if len(fields_to_remove) == 0 and len(fields_to_add) == 0:
            return

        with self.connection.cursor() as cursor:
            # Alter table to add new fields
            for field_name in fields_to_add:
                field_type = get_postgres_type(self.fields[field_name].annotation)
                cursor.execute(
                    f"ALTER TABLE {self.table_name} ADD COLUMN {field_name} {field_type}"
                )

            # Alter table to remove fields
            for field_name in fields_to_remove:
                cursor.execute(
                    f"ALTER TABLE {self.table_name} DROP COLUMN {field_name}"
                )

        self.connection.commit()

        # Only create indexes for new fields
        for field_name in fields_to_add:
            for index in self.indexes:
                if field_name in index["columns"]:
                    self.create_index(index["name"], index["columns"], index["unique"])

    def save(self, item: Dict[str, Any]) -> None:
        """Saves (inserts or updates) an item into the database table.

        Uses an INSERT ... ON CONFLICT (primary_key) DO UPDATE statement.
        Converts the item's attributes to PostgreSQL-compatible formats before saving.

        Args:
            item: A dictionary representing the model instance to save.
        """
        valid_keys = [key for key in item if key in self.fields]
        columns = ", ".join(valid_keys)
        placeholders = ", ".join([f"%({key})s" for key in valid_keys])
        values = {
            key: convert_to_postgres_format(item[key], self.fields[key].annotation)
            for key in valid_keys
        }
        query = (
            f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders}) ON CONFLICT ({self.get_primary_key()}) DO UPDATE SET "
            + ", ".join([f"{key} = EXCLUDED.{key}" for key in valid_keys])
        )
        with self.connection.cursor() as cursor:
            cursor.execute(query, values)
        self.connection.commit()

    def get(self, key: Any) -> Dict[str, Any] | None:
        """Retrieves an item from the database table by its primary key.

        Args:
            key: The primary key value of the item to retrieve.

        Returns:
            A dictionary representing the retrieved item, or None if not found.
            Attributes are converted back to their Python types.
        """
        primary_key = self.get_primary_key()
        cols = ", ".join(self.fields.keys())
        query = f"SELECT {cols} FROM {self.table_name} WHERE {primary_key} = %s"
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (key,))
            item = cursor.fetchone()
        if item is None:
            return None
        return convert_from_postgres_attributes(dict(item), self.fields)

    def delete(self, primary_key: Any) -> None:
        """Deletes an item from the database table by its primary key.

        Args:
            primary_key: The primary key value of the item to delete.
        """
        pk_column = self.get_primary_key()
        query = f"DELETE FROM {self.table_name} WHERE {pk_column} = %s"
        with self.connection.cursor() as cursor:
            cursor.execute(query, (primary_key,))
        self.connection.commit()

    def _build_condition(
        self, condition: Union[Condition, ConditionGroup]
    ) -> tuple[Composed, list[Any]]:
        """Recursively builds a psycopg2 SQL Composed object and parameters for a WHERE clause.

        Args:
            condition: The Condition or ConditionGroup object.

        Returns:
            A tuple containing the SQL Composed object for the WHERE clause and a list of parameters.
        """
        if isinstance(condition, Condition):
            if condition.operator == Operator.IN:
                placeholders = SQL(", ").join([Placeholder()] * len(condition.value))
                sql = SQL("{} IN ({})").format(
                    Identifier(condition.field), placeholders
                )
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
                    SQL("({})").format(
                        SQL(f" {condition.operator.value} ").join(sub_conditions)
                    ),
                    params,
                )

    def query(
        self,
        condition: ConditionBuilder,
        limit: int = 100,
        reverse: bool = False,
        columns: List[str] | None = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        pk = self.get_primary_key()
        order_by = SQL("{}.{} DESC" if reverse else "{}.{} ASC").format(
            Identifier(self.table_name), Identifier(pk)
        )

        where_clause, params = self._build_condition(condition.build())

        if columns:
            cols = SQL(", ").join(
                [
                    SQL("{}.{}").format(Identifier(self.table_name), Identifier(col))
                    for col in columns
                ]
            )
        else:
            cols = SQL(", ").join(
                [
                    SQL("{}.{}").format(Identifier(self.table_name), Identifier(col))
                    for col in self.fields.keys()
                ]
            )

        query = SQL("SELECT {} FROM {} WHERE {} ORDER BY {} LIMIT {}").format(
            cols,
            Identifier(self.table_name),
            where_clause,
            order_by,
            SQL(str(limit)),
        )

        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            res = [
                convert_from_postgres_attributes(dict(row), self.fields)
                for row in cursor.fetchall()
            ]

        if len(res) == 0 or len(res) < limit:
            return res, ""

        last_evaluated_key = str(res[-1].get(pk))
        return res, last_evaluated_key

    @contextmanager
    def get_cursor(self):
        """Provides a database cursor within a context manager, handling commit/rollback."""
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor

    def execute_sql(
        self, sql: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
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
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(translated_sql, translated_params)
            if cursor.description:
                return [
                    convert_from_postgres_attributes(dict(row), self.fields)
                    for row in cursor.fetchall()
                ]
            return []

    def create_index(
        self, index_name: str, columns: List[str], unique: bool = False
    ) -> None:
        unique_str = "UNIQUE" if unique else ""
        columns_str = ", ".join(columns)
        sql = f"CREATE {unique_str} INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({columns_str})"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"PostgreSQL error during index creation: {e}")
            raise e

    def drop_index(self, index_name: str) -> None:
        sql = f"DROP INDEX IF EXISTS {index_name}"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
            self.connection.commit()
        except psycopg2.Error as e:
            print(f"PostgreSQL error during index deletion: {e}")
            raise e

    def list_indexes(self) -> List[Dict[str, Any]]:
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
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, (self.table_name,))
                indexes = []
                for row in cursor.fetchall():
                    indexes.append(
                        {
                            "name": row["index_name"],
                            "columns": row["column_names"],
                            "unique": row["is_unique"],
                        }
                    )
                return indexes
        except psycopg2.Error as e:
            print(f"PostgreSQL error during index listing: {e}")
            raise e

    def __del__(self):
        """Closes the database connection when the adapter object is garbage collected."""
        if hasattr(self, "_connection") and self._connection:
            self._connection.close()
