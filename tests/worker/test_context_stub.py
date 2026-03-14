import asyncio
import pytest
from nodetool.runtime.resources import ResourceScope
from nodetool.worker.context_stub import WorkerContext


@pytest.mark.asyncio
async def test_get_secret():
    async with ResourceScope():
        ctx = WorkerContext(secrets={"HF_TOKEN": "abc123"})
        assert await ctx.get_secret("HF_TOKEN") == "abc123"
        assert await ctx.get_secret("MISSING") is None


@pytest.mark.asyncio
async def test_post_message_captures_progress():
    from nodetool.workflows.types import NodeProgress

    async with ResourceScope():
        ctx = WorkerContext()
        ctx.post_message(NodeProgress(node_id="n1", progress=5, total=20))
        messages = ctx.drain_progress()
        assert len(messages) == 1
        assert messages[0].progress == 5


@pytest.mark.asyncio
async def test_cancellation_flag():
    async with ResourceScope():
        ctx = WorkerContext()
        assert not ctx.is_cancelled
        ctx._cancel_event.set()
        assert ctx.is_cancelled


@pytest.mark.asyncio
async def test_image_roundtrip():
    """Test image_from_pil produces output blob."""
    import PIL.Image
    import io

    async with ResourceScope():
        img = PIL.Image.new("RGB", (4, 4), color="red")

        ctx = WorkerContext()
        out_ref = await ctx.image_from_pil(img, name="result")
        blobs = ctx.get_output_blobs()
        assert len(blobs) == 1
        out_img = PIL.Image.open(io.BytesIO(list(blobs.values())[0]))
        assert out_img.size == (4, 4)
