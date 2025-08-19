"""
Adapter for interacting with a Supabase backend (PostgreSQL with PostgREST).
"""

import logging

# mypy: ignore-errors
from typing import Any, Dict, List, Type, Union, get_origin, get_args
from pydantic.fields import FieldInfo
from datetime import datetime
from enum import EnumMeta as EnumType

# Assume supabase-py is installed (use async client)
from supabase import AsyncClient as SupabaseAsyncClient
from postgrest.base_request_builder import APIResponse

from nodetool.models.database_adapter import DatabaseAdapter
from nodetool.models.condition_builder import (
    Condition,
    ConditionBuilder,
    ConditionGroup,
    Operator,
    LogicalOperator,
)

log = logging.getLogger(__name__)


# --- Type Conversion Helpers (Similar to Postgres, adjust if needed for Supabase client) ---


def convert_to_supabase_format(value: Any, py_type: Type | None) -> Any:
    """Converts Python types to Supabase-compatible formats (mostly handles JSON)."""
    if value is None:
        return None
    # Supabase client generally handles basic types and JSON directly
    # Add specific conversions if needed (e.g., for datetime formatting, enums)
    if isinstance(value, datetime):
        # Ensure timezone-aware or format as ISO string if required by Supabase/PostgREST
        return value.isoformat()
    if hasattr(value, "value") and isinstance(type(value), EnumType):
        return value.value  # Store enum value
    # For lists/dicts, supabase-py usually handles JSON serialization
    return value


def convert_from_supabase_format(value: Any, py_type: Type | None) -> Any:
    """Converts Supabase return values back to Python types."""
    if value is None or py_type is None:
        return value

    origin = get_origin(py_type)
    if origin is Union:
        # Handle Optional[T]
        args = [t for t in get_args(py_type) if t is not type(None)]
        if len(args) == 1:
            return convert_from_supabase_format(value, args[0])
        else:  # Handle Union[T1, T2] - more complex, might need type hints in data
            return value  # Or attempt conversion based on value type

    if py_type is datetime and isinstance(value, str):
        try:
            # Parse ISO format string (linter might warn, but this handles 'Z' correctly)
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            log.warning(f"Could not parse datetime string from Supabase: {value}")
            return value  # Return original if parsing fails
    elif isinstance(py_type, EnumType) and not isinstance(value, py_type):
        try:
            return py_type(value)  # type: ignore
        except ValueError:
            log.warning(f"Could not cast value '{value}' to Enum {py_type}")
            return value
    # Basic types and JSON usually handled correctly by supabase-py
    return value


def convert_from_supabase_attributes(
    attributes: Dict[str, Any], fields: Dict[str, FieldInfo]
) -> Dict[str, Any]:
    """Converts a dictionary of attributes from Supabase types."""
    return {
        key: (
            convert_from_supabase_format(attributes[key], fields[key].annotation)
            if key in fields
            else attributes[key]
        )
        for key in attributes
    }


# --- Supabase Adapter Class ---


class SupabaseAdapter(DatabaseAdapter):
    """Adapts DBModel operations to a Supabase backend."""

    table_name: str
    table_schema: Dict[str, Any]
    fields: Dict[str, FieldInfo]
    # Note: Supabase/PostgREST doesn't directly expose index management via client lib
    # Index operations might require raw SQL execution if needed.

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        fields: Dict[str, FieldInfo],
        table_schema: Dict[str, Any],
        # indexes: List[Dict[str, Any]], # Index management might differ
    ):
        """Initializes the Supabase adapter."""
        # Instantiate async client; direct constructor avoids needing to await factory
        self.supabase_client: SupabaseAsyncClient = SupabaseAsyncClient(
            supabase_url, supabase_key
        )
        self.table_name = table_schema["table_name"]
        self.table_schema = table_schema
        self.fields = fields
        # self.indexes = indexes # Store if needed for raw SQL index ops

        # Optional: Check if table exists on init? Supabase doesn't have a simple check.
        # Could try a select with limit 1, but might be slow.
        # Table creation is often handled by Supabase migrations UI/CLI.

    def _get_primary_key(self) -> str:
        """Gets the primary key column name from the schema."""
        # Assuming 'id' or defined in table_schema, like PostgresAdapter
        return self.table_schema.get("primary_key", "id")

    async def create_table(self) -> None:
        """Creates the database table.
        NOTE: Table creation in Supabase is typically handled via migrations (UI or CLI).
        Implementing this via the client is less common and might require executing raw SQL.
        This implementation will raise a NotImplementedError.
        """
        log.warning(
            f"Table creation for '{self.table_name}' should ideally be handled by Supabase migrations."
        )
        # If direct creation is absolutely needed, implement using execute_sql with CREATE TABLE DDL.
        # Need to translate Python types to PostgreSQL types (similar to PostgresAdapter).
        raise NotImplementedError(
            "Table creation via adapter is not standard practice for Supabase."
        )

    async def drop_table(self) -> None:
        """Drops the database table.
        NOTE: Like creation, dropping tables is usually done via Supabase UI/CLI or migrations.
        """
        log.warning(
            f"Table dropping for '{self.table_name}' should ideally be handled by Supabase migrations/UI."
        )
        # If needed, implement using execute_sql: f"DROP TABLE IF EXISTS {self.table_name}"
        raise NotImplementedError(
            "Table dropping via adapter is not standard practice for Supabase."
        )

    async def save(self, item: Dict[str, Any]) -> None:
        """Saves (inserts or updates) an item in the Supabase table using upsert."""
        self._get_primary_key()
        # Prepare item data, converting types if necessary
        supabase_item = {
            key: convert_to_supabase_format(value, self.fields[key].annotation)
            for key, value in item.items()
            if key in self.fields  # Ensure only model fields are sent
        }

        try:
            response: APIResponse = await (
                self.supabase_client.table(self.table_name)
                .upsert(
                    supabase_item  # , on_conflict=pk # 'on_conflict' is often implicit based on PK
                )
                .execute()
            )

            if not response.data:
                # Handle potential errors if needed, PostgREST errors might be in response directly
                # or raise exceptions depending on the client version/config.
                log.error(
                    f"Supabase upsert failed for table {self.table_name}. Response: {response}"
                )
                # Attempt to parse PostgrestError if available
                # raise Exception(f"Supabase upsert failed: {getattr(response, 'error', 'Unknown error')}")

        except Exception as e:
            log.exception(f"Error saving item to Supabase table {self.table_name}: {e}")
            raise

    async def get(self, key: Any) -> Dict[str, Any] | None:
        """Retrieves an item from Supabase by its primary key."""
        pk = self._get_primary_key()
        select_columns = ", ".join(self.fields.keys())

        try:
            response: APIResponse = await (
                self.supabase_client.table(self.table_name)
                .select(select_columns)
                .eq(pk, key)
                .limit(1)
                .execute()
            )

            if response.data:
                return convert_from_supabase_attributes(response.data[0], self.fields)
            else:
                # Check for errors in response if necessary
                return None
        except Exception as e:
            log.exception(
                f"Error getting item {key} from Supabase table {self.table_name}: {e}"
            )
            raise

    async def delete(self, primary_key: Any) -> None:
        """Deletes an item from Supabase by its primary key."""
        pk = self._get_primary_key()
        try:
            response: APIResponse = await (
                self.supabase_client.table(self.table_name)
                .delete()
                .eq(pk, primary_key)
                .execute()
            )
            # Check response for errors if needed
            if not response.data:  # Or check status code/error attribute
                log.warning(
                    f"Potential issue deleting item {primary_key} from {self.table_name}. Response: {response}"
                )

        except Exception as e:
            log.exception(
                f"Error deleting item {primary_key} from Supabase table {self.table_name}: {e}"
            )
            raise

    def _apply_conditions(
        self, query_builder, condition: Union[Condition, ConditionGroup]
    ):
        """Applies conditions recursively to the Supabase query builder."""
        if isinstance(condition, Condition):
            field = condition.field
            op = condition.operator
            value = condition.value  # Already converted? Assume yes for now.

            # Map Operator enum to Supabase client filter methods
            if op == Operator.EQ:
                query_builder = query_builder.eq(field, value)
            elif op == Operator.NE:
                query_builder = query_builder.neq(field, value)
            elif op == Operator.GT:
                query_builder = query_builder.gt(field, value)
            elif op == Operator.GTE:
                query_builder = query_builder.gte(field, value)
            elif op == Operator.LT:
                query_builder = query_builder.lt(field, value)
            elif op == Operator.LTE:
                query_builder = query_builder.lte(field, value)
            elif op == Operator.IN:
                query_builder = query_builder.in_(field, value)
            elif op == Operator.LIKE:
                query_builder = query_builder.like(field, value)
            elif op == Operator.CONTAINS:
                query_builder = query_builder.contains(field, value)
            else:
                raise NotImplementedError(
                    f"Supabase adapter does not support operator: {op}"
                )

        elif isinstance(condition, ConditionGroup):
            # Supabase client uses .or_() and .and_() for grouping, requiring specific syntax.
            # This needs careful implementation based on how filters are chained.
            # Simple chaining implies AND. OR requires explicit .or_ filter string.
            # Example (may need refinement based on supabase-py version):
            filters = []
            for sub_condition in condition.conditions:
                query_builder = self._apply_conditions(
                    query_builder, sub_condition
                )  # Apply sequentially

            if condition.operator == LogicalOperator.AND:
                # Chaining typically handles AND implicitly, but explicit might be needed
                # query_builder = query_builder.filter(field, "and", f"({','.join(filters)})") # Check syntax
                # Or simply chain the filters if supabase-py allows:
                for sub_condition in condition.conditions:
                    query_builder = self._apply_conditions(
                        query_builder, sub_condition
                    )  # Apply sequentially
            elif condition.operator == LogicalOperator.OR:
                or_filter = f"or({','.join(filters)})"
                query_builder = query_builder.or_(or_filter)  # Pass combined OR filter
            else:
                raise ValueError(
                    f"Unsupported ConditionGroup operator: {condition.operator}"
                )

        return query_builder

    async def query(
        self,
        condition: ConditionBuilder | None = None,
        order_by: str | None = None,
        limit: int = 100,
        reverse: bool = False,
        columns: List[str] | None = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """Queries Supabase based on conditions."""
        pk = self._get_primary_key()

        if columns:
            select_columns = ", ".join(columns)
        else:
            select_columns = ", ".join(self.fields.keys())

        # Base query
        query = self.supabase_client.table(self.table_name).select(select_columns)

        # Apply conditions (potentially complex with AND/OR groups)
        if condition is not None:
            built_condition = condition.build()
            try:
                query = self._apply_conditions(query, built_condition)
            except NotImplementedError as e:
                log.error(f"Query failed due to unsupported operator/condition: {e}")
                raise  # Or return empty result?

        if order_by:
            query = query.order(order_by, desc=reverse)
        else:
            query = query.order(pk, desc=reverse)

        query = query.limit(limit)

        # Execute
        try:
            response: APIResponse = await query.execute()
            if response.data:
                results = [
                    convert_from_supabase_attributes(dict(row), self.fields)
                    for row in response.data
                ]
                return results, ""
            else:
                return [], ""
        except Exception as e:
            log.exception(f"Error querying Supabase table {self.table_name}: {e}")
            raise  # Or return [], ""

    # --- Index Management (Likely requires raw SQL via execute_sql) ---

    async def create_index(
        self, index_name: str, columns: List[str], unique: bool = False
    ) -> None:
        """Creates an index using raw SQL."""
        raise NotImplementedError("Index creation is not supported for Supabase.")

    async def drop_index(self, index_name: str) -> None:
        """Drops an index using raw SQL."""
        raise NotImplementedError("Index creation is not supported for Supabase.")

    async def list_indexes(self) -> List[Dict[str, Any]]:
        """Lists indexes using raw SQL querying pg_catalog."""
        raise NotImplementedError("Index listing is not supported for Supabase.")

    async def close(self) -> None:
        """
        Close the database connection and clean up resources.
        """
        # Supabase client typically manages its own connection lifecycle
        # Check if the client has a close method or similar cleanup
        if hasattr(self.supabase_client, 'close'):
            await self.supabase_client.close()
        # For most Supabase client implementations, explicit cleanup isn't required
        # as connections are managed automatically
