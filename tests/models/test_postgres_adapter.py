import pytest
import pytest_asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock
from psycopg.types.json import Jsonb
from nodetool.models.condition_builder import Field
from nodetool.models.postgres_adapter import (
    PostgresAdapter,
    convert_to_postgres_format,
    convert_from_postgres_format,
    translate_condition_to_sql,
)
from nodetool.models.base_model import DBModel, DBField


# Mock Pydantic model for testing
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
        return {"table_name": "test_table"}

    @classmethod
    def adapter(cls):
        return PostgresAdapter(
            db_params=dict(
                database="test_db",
                user="test_user",
                password="test_password",
                host="localhost",
                port="5432",
            ),
            fields=TestModel.db_fields(),
            table_schema=TestModel.get_table_schema(),
            indexes=[
                dict(
                    name="age_index",
                    columns=["age"],
                    unique=False,
                )
            ],
        )


@pytest_asyncio.fixture
async def mock_db_adapter():
    with patch(
        "nodetool.models.postgres_adapter.AsyncConnectionPool"
    ) as mock_pool_class:
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()

        mock_pool_class.return_value = mock_pool
        mock_pool.connection.return_value.__aenter__.return_value = mock_conn
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor

        adapter = TestModel.adapter()
        adapter._pool = mock_pool
        yield adapter


@pytest.mark.asyncio
async def test_table_creation(mock_db_adapter):
    mock_db_adapter.table_exists = AsyncMock(return_value=True)
    assert await mock_db_adapter.table_exists()


@pytest.mark.asyncio
async def test_save_and_get(mock_db_adapter):
    item = TestModel(
        id="1",
        name="John Doe",
        age=30,
        height=1.75,
        is_active=True,
        tags=["tag1", "tag2"],
        metadata={"key": "value"},
        created_at=datetime.now(),
        enum_field=TestEnum.VALUE1,
        optional_field="test",
    )
    mock_db_adapter.save = AsyncMock()
    mock_db_adapter.get = AsyncMock(return_value=item.model_dump())

    await mock_db_adapter.save(item.model_dump())
    retrieved_item = TestModel(**await mock_db_adapter.get("1"))
    assert retrieved_item == item


@pytest.mark.asyncio
async def test_update(mock_db_adapter):
    item = TestModel(
        id="1",
        name="John Doe",
        age=30,
        height=1.75,
        is_active=True,
        tags=["tag1", "tag2"],
        metadata={"key": "value"},
        created_at=datetime.now(),
        enum_field=TestEnum.VALUE1,
        optional_field="test",
    )
    mock_db_adapter.save = AsyncMock()
    mock_db_adapter.get = AsyncMock()

    await mock_db_adapter.save(item.model_dump())

    updated_item = item.model_copy()
    updated_item.name = "Jane Doe"
    updated_item.age = 31
    await mock_db_adapter.save(updated_item.model_dump())
    mock_db_adapter.get.return_value = updated_item.model_dump()

    retrieved_item = TestModel(**await mock_db_adapter.get("1"))
    assert retrieved_item == updated_item


@pytest.mark.asyncio
async def test_delete(mock_db_adapter):
    mock_db_adapter.delete = AsyncMock()
    mock_db_adapter.get = AsyncMock(return_value=None)

    await mock_db_adapter.delete("1")
    assert await mock_db_adapter.get("1") is None


@pytest.mark.asyncio
async def test_query(mock_db_adapter):
    items = [
        TestModel(
            id=str(i),
            name=f"User {i}",
            age=25 + i,  # Start age at 25 to ensure all results are > 25
            height=1.7 + i * 0.1,
            is_active=i % 2 == 0,
            tags=[f"tag{i}"],
            metadata={"key": f"value{i}"},
            created_at=datetime.now(),
            enum_field=TestEnum.VALUE1 if i % 2 == 0 else TestEnum.VALUE2,
            optional_field=f"test{i}" if i % 2 == 0 else None,
        ).model_dump()
        for i in range(10)
    ]
    # Mock the query to return items 5-8 (indices 5, 6, 7, 8)
    mock_db_adapter.query = AsyncMock(return_value=(items[5:9], ""))

    results, last_key = await mock_db_adapter.query(
        Field("age").greater_than(25), limit=5
    )

    assert len(results) == 4
    assert all(result["age"] > 25 for result in results)
    assert (
        min(result["age"] for result in results) == 30
    )  # Youngest person should be 30
    assert max(result["age"] for result in results) == 33  # Oldest person should be 33
    assert last_key == ""


def test_convert_to_postgres_format():
    assert convert_to_postgres_format("test", str) == "test"
    assert convert_to_postgres_format(123, int) == 123
    assert convert_to_postgres_format(1.23, float) == 1.23
    assert convert_to_postgres_format(True, bool) is True

    # For lists and dicts, check if the result is a Jsonb object and compare its value
    list_result = convert_to_postgres_format(["a", "b"], List[str])
    assert isinstance(list_result, Jsonb)
    assert list_result.obj == ["a", "b"]

    dict_result = convert_to_postgres_format({"a": 1}, Dict[str, int])
    assert isinstance(dict_result, Jsonb)
    assert dict_result.obj == {"a": 1}

    # For datetime, check if it's returned as-is
    test_datetime = datetime(2023, 1, 1)
    assert convert_to_postgres_format(test_datetime, datetime) == test_datetime

    # For Enum, check if it's converted to its value
    assert convert_to_postgres_format(TestEnum.VALUE1, TestEnum) == "value1"


def test_convert_from_postgres_format():
    assert convert_from_postgres_format("test", str) == "test"
    assert convert_from_postgres_format(123, int) == 123
    assert convert_from_postgres_format(1.23, float) == 1.23
    assert convert_from_postgres_format(True, bool) is True

    assert convert_from_postgres_format(["a", "b"], List[str]) == ["a", "b"]
    assert convert_from_postgres_format({"a": 1}, Dict[str, int]) == {"a": 1}

    assert convert_from_postgres_format(datetime(2023, 1, 1), datetime) == datetime(
        2023, 1, 1
    )
    assert convert_from_postgres_format("value1", TestEnum) == TestEnum.VALUE1


def test_translate_condition_to_sql():
    condition = "user_id = %(user_id)s AND content_type LIKE %(content_type)s || '%%'"
    expected = "user_id = %(user_id)s AND content_type LIKE %(content_type)s || '%%'"
    assert translate_condition_to_sql(condition) == expected


@pytest.mark.asyncio
async def test_table_migration(mock_db_adapter):
    mock_db_adapter.get_current_schema = AsyncMock(return_value=set())
    mock_db_adapter.migrate_table = AsyncMock()

    # Add a new field
    mock_db_adapter.fields["new_field"] = DBField()
    mock_db_adapter.fields["new_field"].annotation = str
    await mock_db_adapter.migrate_table()

    # Check if the new field was added
    mock_db_adapter.get_current_schema.return_value = {"new_field"}
    current_schema = await mock_db_adapter.get_current_schema()
    assert "new_field" in current_schema

    # Remove a field
    del mock_db_adapter.fields["optional_field"]
    await mock_db_adapter.migrate_table()

    # Check if the field was removed
    mock_db_adapter.get_current_schema.return_value = set()
    current_schema = await mock_db_adapter.get_current_schema()
    assert "optional_field" not in current_schema
