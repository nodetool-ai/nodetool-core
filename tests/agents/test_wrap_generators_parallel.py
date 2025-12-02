import asyncio

import pytest

from nodetool.agents.wrap_generators_parallel import wrap_generators_parallel


@pytest.mark.asyncio
async def test_wrap_generators_parallel_order():
    async def slow():
        for i in range(3):
            await asyncio.sleep(0.02)
            yield f"slow{i}"

    async def fast():
        for i in range(2):
            await asyncio.sleep(0.01)
            yield f"fast{i}"

    results = []
    async for item in wrap_generators_parallel(slow(), fast()):
        results.append(item)

    assert set(results) == {"slow0", "slow1", "slow2", "fast0", "fast1"}
    assert results[0].startswith("fast")


@pytest.mark.asyncio
async def test_wrap_generators_parallel_exception():
    async def good():
        yield "ok"

    async def bad():
        yield "start"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        async for _ in wrap_generators_parallel(good(), bad()):
            pass
