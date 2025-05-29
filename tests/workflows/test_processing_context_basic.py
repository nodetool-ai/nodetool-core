import asyncio
import pytest

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.common.environment import Environment


class SumNode(BaseNode):
    a: int = 0
    b: int = 0

    async def process(self, context: ProcessingContext) -> int:
        return self.a + self.b


def test_get_set(context: ProcessingContext):
    context.set("foo", 123)
    assert context.get("foo") == 123
    assert context.get("missing", "default") == "default"


@pytest.mark.asyncio
async def test_message_queue(context: ProcessingContext):
    context.post_message("hello")
    assert context.has_messages()
    msg = await context.pop_message_async()
    assert msg == "hello"
    assert not context.has_messages()


def test_asset_storage_url():
    ctx = ProcessingContext()
    url = ctx.asset_storage_url("file.txt")
    assert url.endswith("/file.txt")
    assert Environment.get_storage_api_url() in url


def test_cache_result_and_retrieve(context: ProcessingContext):
    Environment.get_node_cache().clear()
    node = SumNode(id="1", a=1, b=2)
    context.cache_result(node, 3, ttl=60)
    cached = context.get_cached_result(node)
    assert cached == 3


def test_generate_node_cache_key_unique():
    ctx = ProcessingContext()
    node1 = SumNode(id="1", a=1, b=2)
    node2 = SumNode(id="1", a=1, b=3)
    key1 = ctx.generate_node_cache_key(node1)
    key2 = ctx.generate_node_cache_key(node2)
    assert key1 != key2

