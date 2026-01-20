"""
Supabase scope resources for ResourceScope.

Provides per-scope Supabase resources with adapter memoization.
Note: Supabase client handles its own connection pooling.
"""

from typing import Any, Dict, Type

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import DBResources

log = get_logger(__name__)


class SupabaseScopeResources(DBResources):
    """Per-scope Supabase resources (client + adapters)."""

    def __init__(self, client: Any):
        """Initialize scope resources.

        Args:
            client: The Supabase client
        """
        self.client = client
        self._adapters: dict[str, Any] = {}

    async def adapter_for_model(self, model_cls: type[Any]) -> Any:
        """Get or create an adapter for the given model class.

        Memoizes adapters per table within this scope.

        Args:
            model_cls: The model class to get an adapter for

        Returns:
            A SupabaseAdapter instance
        """
        from nodetool.models.supabase_adapter import SupabaseAdapter

        table_name = model_cls.get_table_schema().get("table_name", "unknown")

        # Return memoized adapter if available
        if table_name in self._adapters:
            return self._adapters[table_name]

        # Create new adapter
        log.debug(f"Creating new Supabase adapter for table '{table_name}'")
        adapter = SupabaseAdapter(
            client=self.client,
            fields=model_cls.db_fields(),
            table_schema=model_cls.get_table_schema(),
        )

        # Memoize
        self._adapters[table_name] = adapter
        return adapter

    async def cleanup(self) -> None:
        """Clean up resources.

        Note: Supabase client is stateless and handles pooling internally,
        so no explicit connection release needed.
        """
        # Clean up any adapters that have close methods
        for table_name, adapter in self._adapters.items():
            try:
                if hasattr(adapter, "close"):
                    await adapter.close()
            except Exception as e:
                log.warning(f"Error closing Supabase adapter for '{table_name}': {e}")
