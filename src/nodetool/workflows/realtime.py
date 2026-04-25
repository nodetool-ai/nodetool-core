"""Realtime session primitives shared by the workflow runtime and node authors.

This module is intentionally **substrate-only**: it owns the value types and
small helpers that node authors need to write realtime-capable nodes
(`RealtimeSessionInfo`, the lifecycle hook signature) and that the runner /
worker need to pass that information across the stdio bridge.

Heavy model code (weight loaders, sampler loops, frame converters) lives in
the sister ``nodetool-realtime`` package, never here. See
``PLAN-REALTIME.md`` for the full architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

VideoPixelFormat: TypeAlias = Literal["rgba8", "rgb8", "yuv420p", "nv12"]
AudioSampleFormat: TypeAlias = Literal["s16le", "f32le"]


@dataclass(slots=True)
class VideoFrame:
    """Raw realtime video frame routed through the substrate.

    Mirrors ``VideoFrame`` from ``@nodetool/protocol``. The pixel buffer stays
    CPU-resident and binary; model nodes are responsible for converting it to
    tensors or any model-specific layout.
    """

    type: Literal["realtime_video_frame"] = field(
        default="realtime_video_frame", init=False
    )
    data: bytes
    width: int
    height: int
    stride: int
    pixel_format: VideoPixelFormat
    timestamp_ns: int
    sequence: int


@dataclass(slots=True)
class AudioFrame:
    """Raw realtime audio frame routed through the substrate.

    Mirrors ``AudioFrame`` from ``@nodetool/protocol``. Samples are PCM bytes,
    interleaved when ``channels`` is greater than one.
    """

    type: Literal["realtime_audio_frame"] = field(
        default="realtime_audio_frame", init=False
    )
    data: bytes
    sample_rate: int
    channels: int
    sample_format: AudioSampleFormat
    samples: int
    timestamp_ns: int
    sequence: int


RealtimeFrame: TypeAlias = VideoFrame | AudioFrame


@dataclass(slots=True)
class RealtimeMediaTrack:
    """Mapping between a transport-level media track and a graph input handle.

    Mirrors ``RealtimeMediaTrackMapping`` from
    ``packages/protocol/src/api-schemas/realtime.ts``. Worker-side code only
    needs the fields a node would actually inspect (which input handle the
    track feeds, what kind of media it carries); orchestration concerns
    (label, enabled flag, transport-level identifiers) stay on the TS side.
    """

    track_id: str
    kind: str  # "audio" | "video"
    node_id: str
    input_name: str


@dataclass(slots=True)
class RealtimeSessionInfo:
    """Snapshot of a realtime session passed into BaseNode lifecycle hooks.

    Mirrors the shape of ``RealtimeSessionRecord`` on the TS side
    (``packages/protocol/src/api-schemas/realtime.ts``) but only carries the
    fields a node author can act on:

    - ``session_id`` — stable identifier for the live session; nodes use this
      for logging, telemetry, and as a key when caching warm state across
      hooks.
    - ``workflow_id`` — the workflow this session is running. Useful when the
      same node class is shared across multiple workflows.
    - ``transport`` — ``"websocket"`` or ``"webrtc"``; nodes that need to
      negotiate codecs or framing decide based on this.
    - ``parameters`` — initial control-plane parameter values applied at
      session start. Subsequent updates arrive via ``pushParameter`` (handled
      by the runner, not by the node directly).
    - ``media_tracks`` — declared input/output track mappings; media-adapter
      nodes (``is_media_adapter = True``) use this to know which track they
      should bind to.

    Non-actionable fields (``status``, ``signaling``, ``created_at``,
    ``updated_at``) are deliberately omitted — they belong to the orchestration
    layer, not to node code.
    """

    session_id: str
    workflow_id: str | None
    transport: str  # "websocket" | "webrtc"
    parameters: dict[str, Any] = field(default_factory=dict)
    media_tracks: list[RealtimeMediaTrack] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RealtimeSessionInfo:
        """Construct from the wire format used by the stdio bridge.

        Tolerant of missing fields: ``parameters`` and ``media_tracks``
        default to empty containers when absent so partial payloads from
        older bridges don't crash node startup.
        """
        raw_tracks = payload.get("media_tracks") or []
        tracks = [
            RealtimeMediaTrack(
                track_id=str(t["track_id"]),
                kind=str(t["kind"]),
                node_id=str(t["node_id"]),
                input_name=str(t["input_name"]),
            )
            for t in raw_tracks
            if isinstance(t, dict)
            and "track_id" in t
            and "kind" in t
            and "node_id" in t
            and "input_name" in t
        ]
        return cls(
            session_id=str(payload["session_id"]),
            workflow_id=(
                None
                if payload.get("workflow_id") is None
                else str(payload["workflow_id"])
            ),
            transport=str(payload.get("transport", "websocket")),
            parameters=dict(payload.get("parameters") or {}),
            media_tracks=tracks,
        )
