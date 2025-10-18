import pytest
from types import SimpleNamespace
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from unittest.mock import patch

from nodetool.models.base_model import DBModel, DBField
from nodetool.models.condition_builder import Field
from nodetool.models.supabase_adapter import (
    SupabaseAdapter,
    convert_to_supabase_format,
    convert_from_supabase_format,
)


# Skip global DB setup/teardown from tests/conftest.py for this module
pytestmark = pytest.mark.no_setup


class TestEnum(Enum):
    VALUE1 = "value1"
    VALUE2 = "value2"


class TestModel(DBModel):
    id: str = DBField(hash_key=True)
    name: str = DBField()
    age: int = DBField()
    height: float = DBField()
    is_active: bool = DBField()
    tags: List[str] = DBField()
    metadata: Dict[str, str] = DBField()
    created_at: datetime = DBField()
    enum_field: TestEnum = DBField()
    optional_field: Optional[str] = DBField(default=None)

    @classmethod
    def get_table_schema(cls) -> dict:
        return {"table_name": "test_table", "primary_key": "id"}


class DummyAsyncBuilder:
    def __init__(self, response_data):
        self._response_data = response_data
        self.last_order = None
        self.last_limit = None
        self.called_methods = []

    # Chainable filter/order/limit methods
    def upsert(self, _):
        self.called_methods.append("upsert")
        return self

    def select(self, _):
        self.called_methods.append("select")
        return self

    def delete(self):
        self.called_methods.append("delete")
        return self

    def eq(self, *_):
        self.called_methods.append("eq")
        return self

    def gt(self, *_):
        self.called_methods.append("gt")
        return self

    def gte(self, *_):
        self.called_methods.append("gte")
        return self

    def lt(self, *_):
        self.called_methods.append("lt")
        return self

    def lte(self, *_):
        self.called_methods.append("lte")
        return self

    def in_(self, *_):
        self.called_methods.append("in_")
        return self

    def like(self, *_):
        self.called_methods.append("like")
        return self

    def contains(self, *_):
        self.called_methods.append("contains")
        return self

    def order(self, column, desc=False):
        self.last_order = (column, desc)
        self.called_methods.append("order")
        return self

    def limit(self, n):
        self.last_limit = n
        self.called_methods.append("limit")
        return self

    async def execute(self):
        return SimpleNamespace(data=self._response_data)


def build_adapter_with_mocked_client(response_data):
    builder = DummyAsyncBuilder(response_data)
    with patch("nodetool.models.supabase_adapter.SupabaseAsyncClient") as MockClient:
        mock_client = MockClient.return_value
        mock_client.table.return_value = builder
        adapter = SupabaseAdapter(
            supabase_url="https://example.supabase.co",
            supabase_key="service_key",
            fields=TestModel.db_fields(),
            table_schema=TestModel.get_table_schema(),
        )
    # Ensure the adapter uses our builder for any subsequent calls
    adapter.supabase_client.table.return_value = builder  # type: ignore[attr-defined]
    return adapter, builder


@pytest.mark.asyncio
async def test_save_calls_upsert_and_succeeds():
    adapter, builder = build_adapter_with_mocked_client(response_data=[{"id": "1"}])

    item = TestModel(
        id="1",
        name="John Doe",
        age=30,
        height=1.75,
        is_active=True,
        tags=["a", "b"],
        metadata={"x": "y"},
        created_at=datetime.now(),
        enum_field=TestEnum.VALUE1,
        optional_field=None,
    ).model_dump()

    await adapter.save(item)

    assert "upsert" in builder.called_methods


@pytest.mark.asyncio
async def test_get_returns_item_when_found():
    row = {
        "id": "1",
        "name": "Jane",
        "age": 31,
        "height": 1.8,
        "is_active": True,
        "tags": ["t1"],
        "metadata": {"k": "v"},
        "created_at": datetime.now().isoformat(),
        "enum_field": TestEnum.VALUE2.value,
        "optional_field": None,
    }
    adapter, _ = build_adapter_with_mocked_client(response_data=[row])

    result = await adapter.get("1")
    assert result is not None
    assert result["id"] == "1"
    assert result["name"] == "Jane"


@pytest.mark.asyncio
async def test_get_returns_none_when_not_found():
    adapter, _builder = build_adapter_with_mocked_client(response_data=[])
    result = await adapter.get("missing")
    assert result is None


@pytest.mark.asyncio
async def test_delete_executes_without_error():
    adapter, builder = build_adapter_with_mocked_client(response_data=[{"deleted": 1}])
    await adapter.delete("1")
    assert "delete" in builder.called_methods
    assert "eq" in builder.called_methods


@pytest.mark.asyncio
async def test_query_with_filters_order_and_limit():
    items = [
        {
            "id": str(i),
            "name": f"User {i}",
            "age": 25 + i,
            "height": 1.6 + i * 0.05,
            "is_active": i % 2 == 0,
            "tags": [f"t{i}"],
            "metadata": {"a": f"{i}"},
            "created_at": datetime.now().isoformat(),
            "enum_field": TestEnum.VALUE1.value,
            "optional_field": None,
        }
        for i in range(5)
    ]
    adapter, builder = build_adapter_with_mocked_client(response_data=items)

    results, last_key = await adapter.query(
        condition=Field("age").greater_than(26),
        order_by="age",
        limit=3,
        reverse=True,
    )

    assert isinstance(results, list)
    assert last_key == ""
    # Order and limit captured on builder
    assert builder.last_order == ("age", True)
    assert builder.last_limit == 3
    # Filter methods applied
    assert "gt" in builder.called_methods


def test_convert_to_supabase_format_and_back():
    now = datetime(2024, 1, 1, 12, 0, 0)
    assert convert_to_supabase_format(now, datetime) == now.isoformat()
    dt = convert_from_supabase_format(now.isoformat() + "Z", datetime)
    assert dt.replace(tzinfo=None).isoformat() == now.isoformat()

    assert convert_to_supabase_format(TestEnum.VALUE1, TestEnum) == "value1"
    assert convert_from_supabase_format("value1", TestEnum) == TestEnum.VALUE1
