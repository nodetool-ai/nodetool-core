import pytest
import pytest_asyncio

from nodetool.config.logging_config import get_logger
from nodetool.models.base_model import DBField, DBModel
from nodetool.models.condition_builder import Field

log = get_logger(__name__)


class TestModel(DBModel):
    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "test_table",
            "key_schema": {"id": "HASH"},
            "attribute_definitions": {"id": "S", "username": "S"},
            "global_secondary_indexes": {
                "test_table_username_index": {"username": "HASH"},
            },
        }

    id: str = DBField(hash_key=True)
    username: str = DBField()


@pytest_asyncio.fixture(scope="function")
async def model():
    """Mock for unit tests."""
    try:
        await TestModel.create_table()
    except Exception as e:
        log.info(f"create test table: {e}")

    model = TestModel(id="1", username="Test")
    yield model
    await TestModel.drop_table()


@pytest.mark.asyncio
async def test_model_get(model: TestModel):
    await model.save()

    retrieved_instance = await TestModel.get("1")

    assert retrieved_instance is not None
    assert retrieved_instance.id == "1"
    assert retrieved_instance.username == "Test"


@pytest.mark.asyncio
async def test_model_delete(model: TestModel):
    await model.delete()
    retrieved_instance = await TestModel.get("1")
    assert retrieved_instance is None


@pytest.mark.asyncio
async def test_model_query(model: TestModel):
    await model.save()
    retrieved_instances, _ = await TestModel.query(
        condition=Field("username").equals("Test")
    )
    assert len(retrieved_instances) > 0
    assert retrieved_instances[0].id == "1"
    assert retrieved_instances[0].username == "Test"
