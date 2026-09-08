from __future__ import annotations

import hashlib

import pytest

from nodetool.worker.protocol import WorkerProtocolServer


class RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send_msg(self, msg: dict) -> None:
        self.frames.append(msg)


def _server(blob: bytes) -> WorkerProtocolServer:
    server = WorkerProtocolServer(transport_name="test")

    async def handler(data, cancel_event, emit_progress, emit_chunk, emit_update):
        return {
            "outputs": {"video": {"type": "video", "format": "mp4"}},
            "blobs": {"video": blob},
        }

    server.set_execute_handler(handler)
    return server


@pytest.mark.asyncio
async def test_chunked_result_blob_has_ordered_offsets_and_digest(monkeypatch):
    monkeypatch.setattr("nodetool.worker.protocol.BLOB_CHUNK_SIZE", 4)
    blob = b"abcdefghij"
    transport = RecordingTransport()

    await _server(blob).dispatch(
        {
            "type": "execute",
            "request_id": "request-1",
            "data": {
                "node_type": "test.Node",
                "blob_transfer": "chunked-v1",
            },
        },
        transport,
    )

    assert [frame["type"] for frame in transport.frames] == [
        "blob.start",
        "blob.chunk",
        "blob.chunk",
        "blob.chunk",
        "blob.end",
        "result",
    ]
    assert [
        frame["data"]["offset"]
        for frame in transport.frames
        if frame["type"] == "blob.chunk"
    ] == [0, 4, 8]
    assert b"".join(
        frame["data"]["bytes"]
        for frame in transport.frames
        if frame["type"] == "blob.chunk"
    ) == blob
    assert transport.frames[-2]["data"] == {
        "name": "video",
        "size": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
    assert transport.frames[-1]["data"]["blobs"] == {}


@pytest.mark.asyncio
async def test_legacy_result_keeps_inline_blob():
    blob = b"legacy"
    transport = RecordingTransport()

    await _server(blob).dispatch(
        {
            "type": "execute",
            "request_id": "request-1",
            "data": {"node_type": "test.Node"},
        },
        transport,
    )

    assert [frame["type"] for frame in transport.frames] == ["result"]
    assert transport.frames[0]["data"]["blobs"] == {"video": blob}


@pytest.mark.asyncio
async def test_empty_blob_still_has_start_and_end_frames():
    transport = RecordingTransport()

    await _server(b"").dispatch(
        {
            "type": "execute",
            "request_id": "request-1",
            "data": {
                "node_type": "test.Node",
                "blob_transfer": "chunked-v1",
            },
        },
        transport,
    )

    assert [frame["type"] for frame in transport.frames] == [
        "blob.start",
        "blob.end",
        "result",
    ]
