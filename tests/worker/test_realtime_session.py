"""Unit tests for the realtime session substrate (PLAN-REALTIME.md item 6c).

Two layers are exercised:

1. :class:`nodetool.worker.realtime_session.RealtimeNodeInstance` directly,
   to lock down the lifecycle contract (start awaits hooks before returning,
   identity passthrough preserves frame ordering and conservation,
   parameter routing assigns to pydantic fields, stop runs ``on_session_stop``
   on every termination path).
2. :class:`nodetool.worker.stdio_server.StdioWorkerServer` via its dispatch
   coroutine, with a stub transport that captures the wire-format messages.
   This guarantees the bridge protocol verbs and the
   ``realtime_output_frame`` events match the shape consumed by the TS
   ``PythonStdioBridge``.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, ClassVar

import pytest
from pydantic import Field

from nodetool.worker.realtime_session import (
    RealtimeNodeInstance,
    RealtimeSessionError,
)
from nodetool.worker.stdio_server import StdioWorkerServer
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.realtime import RealtimeSessionInfo


# ---------------------------------------------------------------------------
# Test node fixtures
# ---------------------------------------------------------------------------


class _IdentityFrameNode(BaseNode):
    """Realtime node: read frames from inbox, echo each one to ``output``."""

    _is_realtime_capable: ClassVar[bool] = True
    _owns_warm_state: ClassVar[bool] = True
    multiplier: int = Field(default=1)

    started: ClassVar[list[str]] = []
    stopped: ClassVar[list[str]] = []

    @classmethod
    def get_node_type(cls) -> str:
        return "test_realtime.IdentityFrameNode"

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    async def on_session_start(self, context: Any, session: Any) -> None:
        type(self).started.append(session.session_id)

    async def on_session_stop(self, context: Any, session: Any) -> None:
        type(self).stopped.append(session.session_id)

    async def gen_process(
        self, context: Any
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for frame in self.iter_input("frame"):
            yield {"output": frame * self.multiplier}


class _NotRealtimeNode(BaseNode):
    """A non-realtime node, used to verify the opt-in guard."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test_realtime.NotRealtimeNode"

    async def process(self, context: Any) -> str:
        return "no"


class _StartFailsNode(BaseNode):
    """Realtime node whose ``on_session_start`` raises."""

    _is_realtime_capable: ClassVar[bool] = True
    cleanup_called: ClassVar[list[str]] = []

    @classmethod
    def get_node_type(cls) -> str:
        return "test_realtime.StartFailsNode"

    async def on_session_start(self, context: Any, session: Any) -> None:
        raise RuntimeError("boom from on_session_start")

    async def on_session_stop(self, context: Any, session: Any) -> None:
        type(self).cleanup_called.append(session.session_id)

    async def process(self, context: Any) -> str:
        return ""


@pytest.fixture(autouse=True)
def _register_test_nodes():
    NODE_BY_TYPE["test_realtime.IdentityFrameNode"] = _IdentityFrameNode
    NODE_BY_TYPE["test_realtime.NotRealtimeNode"] = _NotRealtimeNode
    NODE_BY_TYPE["test_realtime.StartFailsNode"] = _StartFailsNode
    _IdentityFrameNode.started.clear()
    _IdentityFrameNode.stopped.clear()
    _StartFailsNode.cleanup_called.clear()
    try:
        yield
    finally:
        NODE_BY_TYPE.pop("test_realtime.IdentityFrameNode", None)
        NODE_BY_TYPE.pop("test_realtime.NotRealtimeNode", None)
        NODE_BY_TYPE.pop("test_realtime.StartFailsNode", None)


def _make_session(session_id: str = "s-1", **overrides: Any) -> RealtimeSessionInfo:
    return RealtimeSessionInfo(
        session_id=session_id,
        workflow_id=overrides.get("workflow_id", "wf-1"),
        transport=overrides.get("transport", "websocket"),
        parameters=overrides.get("parameters", {}),
        media_tracks=overrides.get("media_tracks", []),
    )


# ---------------------------------------------------------------------------
# RealtimeNodeInstance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_rejects_unknown_node_type():
    async def _emit(_h: str, _v: Any) -> None:  # pragma: no cover - unreachable
        return None

    with pytest.raises(RealtimeSessionError):
        await RealtimeNodeInstance.start(
            session=_make_session(),
            node_type="test_realtime.DoesNotExist",
            fields={},
            secrets=None,
            emit_frame=_emit,
        )


@pytest.mark.asyncio
async def test_start_rejects_non_realtime_node():
    async def _emit(_h: str, _v: Any) -> None:  # pragma: no cover - unreachable
        return None

    with pytest.raises(RealtimeSessionError) as excinfo:
        await RealtimeNodeInstance.start(
            session=_make_session(),
            node_type="test_realtime.NotRealtimeNode",
            fields={},
            secrets=None,
            emit_frame=_emit,
        )
    assert "realtime-capable" in str(excinfo.value)


@pytest.mark.asyncio
async def test_start_failure_runs_cleanup_hook():
    """If on_session_start raises, on_session_stop must still run so warm
    state acquired in start is freed."""

    async def _emit(_h: str, _v: Any) -> None:  # pragma: no cover - unreachable
        return None

    with pytest.raises(RealtimeSessionError):
        await RealtimeNodeInstance.start(
            session=_make_session("fail-1"),
            node_type="test_realtime.StartFailsNode",
            fields={},
            secrets=None,
            emit_frame=_emit,
        )
    assert _StartFailsNode.cleanup_called == ["fail-1"]


@pytest.mark.asyncio
async def test_identity_passthrough_preserves_order_and_count():
    """100 frames in, 100 frames out, same order — the canonical 6c smoke test."""
    emitted: list[Any] = []

    async def _emit(handle: str, value: Any) -> None:
        assert handle == "output"
        emitted.append(value)

    instance = await RealtimeNodeInstance.start(
        session=_make_session(),
        node_type="test_realtime.IdentityFrameNode",
        fields={"multiplier": 1},
        secrets=None,
        emit_frame=_emit,
        # Use a generous capacity so the deterministic test doesn't drop
        # frames; backpressure is exercised in its own test below.
        input_buffer_size=256,
    )
    assert _IdentityFrameNode.started == ["s-1"]

    for i in range(100):
        dropped = instance.push_input_frame("frame", i)
        assert dropped == 0
        # Yield to let the runner drain so we don't accidentally exercise
        # backpressure on this test.
        await asyncio.sleep(0)

    # Drain the loop until everything is forwarded.
    for _ in range(50):
        if len(emitted) >= 100:
            break
        await asyncio.sleep(0.01)

    await instance.stop()

    assert emitted == list(range(100))
    assert _IdentityFrameNode.stopped == ["s-1"]


@pytest.mark.asyncio
async def test_push_input_frame_drops_oldest_on_overflow():
    """With a capacity of 2 and a paused consumer, the third push must drop
    one frame and the bridge must observe the drop count."""
    emitted: list[Any] = []
    consumer_block = asyncio.Event()

    class _BlockingNode(_IdentityFrameNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "test_realtime.BlockingNode"

        async def gen_process(
            self, context: Any
        ) -> AsyncGenerator[dict[str, Any], None]:
            await consumer_block.wait()
            async for frame in self.iter_input("frame"):
                yield {"output": frame}

    NODE_BY_TYPE["test_realtime.BlockingNode"] = _BlockingNode
    try:

        async def _emit(handle: str, value: Any) -> None:
            emitted.append((handle, value))

        instance = await RealtimeNodeInstance.start(
            session=_make_session("s-block"),
            node_type="test_realtime.BlockingNode",
            fields={},
            secrets=None,
            emit_frame=_emit,
            input_buffer_size=2,
        )
        # Three pushes while the consumer is blocked: third one must drop.
        d0 = instance.push_input_frame("frame", "a")
        d1 = instance.push_input_frame("frame", "b")
        d2 = instance.push_input_frame("frame", "c")
        assert d0 == 0
        assert d1 == 0
        assert d2 == 1

        # Release the consumer and let it drain.
        consumer_block.set()
        for _ in range(50):
            if len(emitted) >= 2:
                break
            await asyncio.sleep(0.01)

        await instance.stop()

        # Only "b" and "c" survived (oldest "a" was dropped).
        assert [v for _, v in emitted] == ["b", "c"]
    finally:
        NODE_BY_TYPE.pop("test_realtime.BlockingNode", None)


@pytest.mark.asyncio
async def test_update_parameter_routes_to_field_and_takes_effect():
    """A live parameter update must mutate the node attribute and affect
    subsequent processing."""
    emitted: list[Any] = []

    async def _emit(_handle: str, value: Any) -> None:
        emitted.append(value)

    instance = await RealtimeNodeInstance.start(
        session=_make_session("s-param"),
        node_type="test_realtime.IdentityFrameNode",
        fields={"multiplier": 1},
        secrets=None,
        emit_frame=_emit,
        input_buffer_size=64,
    )

    instance.push_input_frame("frame", 10)
    await asyncio.sleep(0.05)

    routed = instance.update_parameter("multiplier", 5)
    assert routed is True

    instance.push_input_frame("frame", 10)
    for _ in range(50):
        if len(emitted) >= 2:
            break
        await asyncio.sleep(0.01)
    await instance.stop()

    assert emitted == [10, 50]


@pytest.mark.asyncio
async def test_update_parameter_rejects_unknown_field():
    async def _emit(_h: str, _v: Any) -> None:
        return None

    instance = await RealtimeNodeInstance.start(
        session=_make_session("s-unk"),
        node_type="test_realtime.IdentityFrameNode",
        fields={},
        secrets=None,
        emit_frame=_emit,
    )
    try:
        assert instance.update_parameter("nope", 1) is False
    finally:
        await instance.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    async def _emit(_h: str, _v: Any) -> None:
        return None

    instance = await RealtimeNodeInstance.start(
        session=_make_session("s-idemp"),
        node_type="test_realtime.IdentityFrameNode",
        fields={},
        secrets=None,
        emit_frame=_emit,
    )
    await instance.stop()
    await instance.stop()  # second call must not raise
    assert _IdentityFrameNode.stopped.count("s-idemp") == 1


# ---------------------------------------------------------------------------
# StdioWorkerServer dispatch wiring
# ---------------------------------------------------------------------------


class _StubTransport:
    """In-memory stand-in for StdioTransport that records sent messages."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send(self, _data: bytes) -> None:  # pragma: no cover - unused
        return None

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.sent.append(msg)


def _request(verb: str, request_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": verb, "request_id": request_id, "data": data}


@pytest.mark.asyncio
async def test_dispatch_round_trip_full_session_lifecycle():
    """Drive a session start → push → output_frame → stop through the
    actual dispatcher to lock down the wire format."""
    server = StdioWorkerServer()
    transport = _StubTransport()
    server._transport = transport  # attach the stub so dispatch can use it

    await server._dispatch(
        _request(
            "start_session",
            "r-start",
            {
                "session_id": "s-100",
                "node_type": "test_realtime.IdentityFrameNode",
                "fields": {"multiplier": 2},
                "session_info": {
                    "session_id": "s-100",
                    "workflow_id": "wf-100",
                    "transport": "webrtc",
                },
            },
        )
    )

    # The start ack must arrive before any frames so the bridge can flip
    # the session to "running".
    start_ack = transport.sent[0]
    assert start_ack["type"] == "result"
    assert start_ack["request_id"] == "r-start"
    assert start_ack["data"]["session_id"] == "s-100"
    assert start_ack["data"]["status"] == "running"

    await server._dispatch(
        _request(
            "push_input_frame",
            "r-push-1",
            {"session_id": "s-100", "handle": "frame", "data": 7},
        )
    )

    # Drain so the runner can emit the output_frame event.
    for _ in range(50):
        if any(m["type"] == "realtime_output_frame" for m in transport.sent):
            break
        await asyncio.sleep(0.01)

    push_ack = next(
        m for m in transport.sent if m.get("request_id") == "r-push-1"
    )
    assert push_ack["data"] == {
        "session_id": "s-100",
        "ok": True,
        "dropped_count": 0,
    }

    out = next(m for m in transport.sent if m["type"] == "realtime_output_frame")
    assert out == {
        "type": "realtime_output_frame",
        "session_id": "s-100",
        "handle": "output",
        "data": 14,
    }

    await server._dispatch(
        _request(
            "update_parameter",
            "r-upd",
            {"session_id": "s-100", "name": "multiplier", "value": 3},
        )
    )
    upd_ack = next(m for m in transport.sent if m.get("request_id") == "r-upd")
    assert upd_ack["data"] == {
        "session_id": "s-100",
        "ok": True,
        "routed": True,
    }

    await server._dispatch(
        _request("stop_session", "r-stop", {"session_id": "s-100"})
    )
    stop_ack = next(m for m in transport.sent if m.get("request_id") == "r-stop")
    assert stop_ack["data"]["session_id"] == "s-100"
    assert stop_ack["data"]["ok"] is True
    assert stop_ack["data"]["error"] is None

    # The session is gone after stop — push to it should error.
    await server._dispatch(
        _request(
            "push_input_frame",
            "r-after",
            {"session_id": "s-100", "handle": "frame", "data": 0},
        )
    )
    after = next(m for m in transport.sent if m.get("request_id") == "r-after")
    assert after["type"] == "error"
    assert "Unknown session_id" in after["data"]["error"]


@pytest.mark.asyncio
async def test_dispatch_rejects_duplicate_start_session():
    server = StdioWorkerServer()
    transport = _StubTransport()
    server._transport = transport

    payload = {
        "session_id": "s-dup",
        "node_type": "test_realtime.IdentityFrameNode",
        "fields": {},
    }
    await server._dispatch(_request("start_session", "r1", payload))
    await server._dispatch(_request("start_session", "r2", payload))

    second = next(m for m in transport.sent if m.get("request_id") == "r2")
    assert second["type"] == "error"
    assert "already running" in second["data"]["error"]

    await server._dispatch(
        _request("stop_session", "r-cleanup", {"session_id": "s-dup"})
    )


@pytest.mark.asyncio
async def test_dispatch_unknown_node_type_returns_error():
    server = StdioWorkerServer()
    transport = _StubTransport()
    server._transport = transport

    await server._dispatch(
        _request(
            "start_session",
            "r-bad",
            {"session_id": "s-bad", "node_type": "test_realtime.NoSuchThing"},
        )
    )
    msg = next(m for m in transport.sent if m.get("request_id") == "r-bad")
    assert msg["type"] == "error"
    assert "Unknown node type" in msg["data"]["error"]
