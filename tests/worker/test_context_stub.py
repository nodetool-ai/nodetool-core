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


@pytest.mark.asyncio
async def test_drain_messages_keeps_previews_and_logs():
    """drain_messages must forward previews/logs instead of silently dropping them."""
    from nodetool.workflows.types import LogUpdate, NodeProgress, NodeUpdate, PreviewUpdate

    async with ResourceScope():
        ctx = WorkerContext()
        ctx.post_message(NodeProgress(node_id="n1", progress=5, total=20))
        ctx.post_message(PreviewUpdate(node_id="n1", value="frame"))
        ctx.post_message(
            LogUpdate(node_id="n1", node_name="n", content="log line", severity="info")
        )
        # NodeUpdate has no worker wire representation — still dropped.
        ctx.post_message(
            NodeUpdate(node_id="n1", node_name="n", node_type="t", status="running")
        )
        messages = ctx.drain_messages()
        assert [type(m).__name__ for m in messages] == [
            "NodeProgress",
            "PreviewUpdate",
            "LogUpdate",
        ]


@pytest.mark.asyncio
async def test_drain_progress_still_returns_only_progress():
    from nodetool.workflows.types import NodeProgress, PreviewUpdate

    async with ResourceScope():
        ctx = WorkerContext()
        ctx.post_message(PreviewUpdate(node_id="n1", value="frame"))
        ctx.post_message(NodeProgress(node_id="n1", progress=1, total=2))
        messages = ctx.drain_progress()
        assert len(messages) == 1
        assert messages[0].progress == 1


@pytest.mark.asyncio
async def test_raise_if_cancelled():
    from nodetool.workflows.processing_context import NodeCancelledError

    async with ResourceScope():
        ctx = WorkerContext()
        ctx.raise_if_cancelled()  # not cancelled: no-op
        ctx._cancel_event.set()
        with pytest.raises(NodeCancelledError):
            ctx.raise_if_cancelled()
