"""Unit tests for the realtime capability flags and lifecycle hooks on BaseNode.

These cover the substrate that PLAN-REALTIME.md item 6b adds — the flags must
default to False, subclasses must be able to opt in by setting the underscore
ClassVar, the lifecycle methods must default to safe no-ops, and the discover
payload must surface the flags so the TS-side NodeRegistry sees Python and TS
realtime nodes uniformly.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from nodetool.worker.node_loader import node_to_metadata
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.realtime import (
    AudioFrame,
    RealtimeFrame,
    RealtimeMediaTrack,
    RealtimeSessionInfo,
    VideoFrame,
)


class _DefaultNode(BaseNode):
    """Vanilla node — no realtime opt-in."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test_realtime.DefaultNode"

    async def process(self, context: Any) -> str:
        return "ok"


class _RealtimeNode(BaseNode):
    """Realtime-capable node with warm state and a media-adapter role."""

    _is_realtime_capable: ClassVar[bool] = True
    _owns_warm_state: ClassVar[bool] = True
    _is_media_adapter: ClassVar[bool] = True
    _realtime_profile: ClassVar[dict[str, Any]] = {
        "browser_capable": True,
        "requires_browser_frame": True,
        "requires_webgpu": True,
        "emits_analysis_event": True,
        "emits_parameter_update": False,
        "emits_media_frame": False,
    }

    @classmethod
    def get_node_type(cls) -> str:
        return "test_realtime.RealtimeNode"

    async def process(self, context: Any) -> str:
        return "ok"


@pytest.fixture(autouse=True)
def _register_test_nodes():
    """Register the test node classes for the duration of each test."""
    NODE_BY_TYPE["test_realtime.DefaultNode"] = _DefaultNode
    NODE_BY_TYPE["test_realtime.RealtimeNode"] = _RealtimeNode
    try:
        yield
    finally:
        NODE_BY_TYPE.pop("test_realtime.DefaultNode", None)
        NODE_BY_TYPE.pop("test_realtime.RealtimeNode", None)


def test_capability_flags_default_to_false():
    assert _DefaultNode.is_realtime_capable() is False
    assert _DefaultNode.owns_warm_state() is False
    assert _DefaultNode.is_media_adapter() is False


def test_capability_flags_can_be_overridden():
    assert _RealtimeNode.is_realtime_capable() is True
    assert _RealtimeNode.owns_warm_state() is True
    assert _RealtimeNode.is_media_adapter() is True


def test_capability_flags_are_inherited():
    """Subclasses inherit the parent's flags unless they override."""

    class _Sub(_RealtimeNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "test_realtime.Sub"

    assert _Sub.is_realtime_capable() is True
    assert _Sub.owns_warm_state() is True
    assert _Sub.is_media_adapter() is True


@pytest.mark.asyncio
async def test_lifecycle_hooks_default_to_noops():
    """Default hook implementations must accept the canonical signature
    without raising — non-realtime nodes must remain blissfully unaware
    that the hooks exist."""
    node = _DefaultNode()
    session = RealtimeSessionInfo(
        session_id="s-1",
        workflow_id="wf-1",
        transport="websocket",
    )

    assert await node.on_session_start(context=None, session=session) is None
    assert await node.on_session_stop(context=None, session=session) is None
    assert node.reset_warm_state() is None


@pytest.mark.asyncio
async def test_lifecycle_hooks_invoked_on_realtime_node():
    """A realtime-capable node sees its hook overrides called with the
    session info passed through unchanged."""
    captured: dict[str, Any] = {}

    class _CapturingNode(_RealtimeNode):
        async def on_session_start(
            self, context: Any, session: RealtimeSessionInfo
        ) -> None:
            captured["start"] = session
            captured["start_ctx"] = context

        async def on_session_stop(
            self, context: Any, session: RealtimeSessionInfo
        ) -> None:
            captured["stop"] = session

        def reset_warm_state(self) -> None:
            captured["reset"] = True

    node = _CapturingNode()
    session = RealtimeSessionInfo(
        session_id="s-2",
        workflow_id="wf-2",
        transport="webrtc",
        parameters={"prompt": "hello"},
        media_tracks=[
            RealtimeMediaTrack(
                track_id="t-1",
                kind="video",
                node_id="src-1",
                input_name="frame",
            )
        ],
    )

    await node.on_session_start(context="ctx-token", session=session)
    node.reset_warm_state()
    await node.on_session_stop(context="ctx-token", session=session)

    assert captured["start"] is session
    assert captured["start_ctx"] == "ctx-token"
    assert captured["reset"] is True
    assert captured["stop"] is session


def test_node_to_metadata_emits_realtime_flags_for_default_node():
    """A node with no realtime opt-in still emits the keys with False."""
    meta = node_to_metadata(_DefaultNode)
    assert meta["is_realtime_capable"] is False
    assert meta["owns_warm_state"] is False
    assert meta["is_media_adapter"] is False


def test_node_to_metadata_emits_realtime_flags_for_realtime_node():
    """An opted-in node surfaces the flags so the TS registry treats it
    the same way it treats a TS-side realtime node."""
    meta = node_to_metadata(_RealtimeNode)
    assert meta["is_realtime_capable"] is True
    assert meta["owns_warm_state"] is True
    assert meta["is_media_adapter"] is True
    assert meta["realtime_profile"] == {
        "browser_capable": True,
        "requires_browser_frame": True,
        "requires_webgpu": True,
        "emits_analysis_event": True,
        "emits_parameter_update": False,
        "emits_media_frame": False,
    }


def test_realtime_session_info_from_dict_round_trip():
    """The wire-format helper tolerates a complete payload and produces a
    structurally-correct RealtimeSessionInfo."""
    info = RealtimeSessionInfo.from_dict(
        {
            "session_id": "s-3",
            "workflow_id": "wf-3",
            "transport": "webrtc",
            "parameters": {"strength": 0.7},
            "media_tracks": [
                {
                    "track_id": "t-9",
                    "kind": "video",
                    "node_id": "node-9",
                    "input_name": "frame",
                }
            ],
        }
    )
    assert info.session_id == "s-3"
    assert info.workflow_id == "wf-3"
    assert info.transport == "webrtc"
    assert info.parameters == {"strength": 0.7}
    assert len(info.media_tracks) == 1
    assert info.media_tracks[0].kind == "video"


def test_realtime_session_info_from_dict_tolerates_missing_optional_fields():
    """Older bridges may not send parameters or media_tracks; defaults must
    keep node startup from crashing."""
    info = RealtimeSessionInfo.from_dict(
        {"session_id": "s-4", "workflow_id": None, "transport": "websocket"}
    )
    assert info.session_id == "s-4"
    assert info.workflow_id is None
    assert info.parameters == {}
    assert info.media_tracks == []


def test_realtime_frame_dataclasses_match_protocol_shape():
    """Realtime frames stay substrate-only: raw bytes plus format metadata."""
    video = VideoFrame(
        data=b"\xff\x00\x00\xff",
        width=1,
        height=1,
        stride=4,
        pixel_format="rgba8",
        timestamp_ns=123_000_000,
        sequence=7,
    )
    audio = AudioFrame(
        data=b"\x00\x00\xff\x7f",
        sample_rate=48_000,
        channels=2,
        sample_format="s16le",
        samples=1,
        timestamp_ns=456_000_000,
        sequence=8,
    )
    frames: list[RealtimeFrame] = [video, audio]

    assert frames[0].type == "realtime_video_frame"
    assert frames[1].type == "realtime_audio_frame"
    assert video.data == b"\xff\x00\x00\xff"
    assert audio.data == b"\x00\x00\xff\x7f"
