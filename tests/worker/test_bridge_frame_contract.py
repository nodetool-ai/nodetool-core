"""Validate the worker's emitted frames against the JS repo's frame schema.

`tests/worker/fixtures/bridge-frames.schema.json` is a verbatim copy of the
`dist/bridge-frames.schema.json` artifact the `nodetool` repo generates from
its Zod schemas (see that directory's README). Validating the frames this
worker actually puts on the wire against it is what stops the two sides of the
bridge from silently drifting apart — particularly for protocol v4, where the
JS side swallows `job.*` failures and would never surface a malformed response.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nodetool.ml.core.model_manager import ModelManager
from nodetool.worker import BRIDGE_PROTOCOL_VERSION
from nodetool.worker.executor import execute_node, read_run_identity
from nodetool.worker.job_registry import JobRegistry
from nodetool.worker.protocol import WorkerProtocolServer

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = Path(__file__).parent / "fixtures" / "bridge-frames.schema.json"

# The frame types the JS dispatcher switches on. Anything else the worker sends
# is not covered by the contract, so validating it would be validating nothing.
DISPATCHED_TYPES = {"discover", "result", "error", "chunk", "progress", "comfy.event"}


@pytest.fixture(scope="module")
def validator():
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


def assert_valid(validator, frame: dict[str, Any]) -> None:
    if frame.get("type") not in DISPATCHED_TYPES:
        return
    errors = sorted(validator.iter_errors(frame), key=lambda e: e.path)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


class RecordingTransport:
    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.frames.append(msg)


@pytest.fixture(autouse=True)
def clean_state():
    ModelManager.clear()
    JobRegistry.clear()
    yield
    ModelManager.clear()
    JobRegistry.clear()


def _server() -> WorkerProtocolServer:
    server = WorkerProtocolServer(transport_name="test")

    async def handler(data, cancel_event, emit_progress, emit_chunk, emit_update):
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
            emit_progress=emit_progress,
            emit_chunk=emit_chunk,
            emit_update=emit_update,
            **read_run_identity(data),
        )

    server.set_execute_handler(handler)
    return server


async def _frames(msg: dict[str, Any], *, nodes: list[dict] | None = None) -> list[dict]:
    server = _server()
    if nodes is not None:
        server.set_nodes_metadata(nodes)
    transport = RecordingTransport()
    await server.dispatch(msg, transport)
    return transport.frames


def test_schema_fixture_is_the_generated_artifact():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["$id"].endswith("bridge-frames.schema.json")
    assert {branch["properties"]["type"]["const"] for branch in schema["oneOf"]} == DISPATCHED_TYPES


@pytest.mark.asyncio
async def test_discover_frame_with_vram_hint_validates(validator):
    from nodetool.worker.node_loader import node_to_metadata
    from tests.worker.test_bridge_protocol_v4 import VramHintNode

    frames = await _frames(
        {"type": "discover", "request_id": "d1", "data": {}},
        nodes=[node_to_metadata(VramHintNode)],
    )

    assert len(frames) == 1
    assert_valid(validator, frames[0])
    node = frames[0]["data"]["nodes"][0]
    assert node["requires_vram_gb"] == 6.5
    assert frames[0]["data"]["protocol_version"] == BRIDGE_PROTOCOL_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        {"type": "job.start", "request_id": "j1", "data": {"job_id": "job-1"}},
        {
            "type": "job.end",
            "request_id": "j2",
            "data": {"job_id": "job-1", "reason": "cancelled"},
        },
        {"type": "job.end", "request_id": "j3", "data": {"job_id": "unknown"}},
        {"type": "job.end", "request_id": "j4", "data": {}},
        {"type": "worker.status", "request_id": "w1", "data": {}},
    ],
)
async def test_v4_response_frames_validate(validator, message):
    frames = await _frames(message)
    assert frames, "every v4 request must answer with a frame"
    for frame in frames:
        assert frame["type"] in ("result", "error")
        assert_valid(validator, frame)


@pytest.mark.asyncio
async def test_models_evict_result_frame_validates(validator):
    from nodetool.worker.model_handler import handle_models_message
    from tests.worker.test_bridge_protocol_v4 import GB, FakeModel

    ModelManager.set_model("n", "m", FakeModel(GB))
    transport = RecordingTransport()
    await handle_models_message(
        msg_type="models.evict",
        request_id="ev1",
        data={"target_vram_gb": 0.5},
        transport=transport,
        cancel_flags={},
    )

    assert len(transport.frames) == 1
    assert_valid(validator, transport.frames[0])
    assert transport.frames[0]["data"]["evicted"] == ["m"]


@pytest.mark.asyncio
async def test_execute_result_frame_with_identity_validates(validator):
    frames = await _frames(
        {
            "type": "execute",
            "request_id": "e1",
            "data": {
                "node_type": "test.QuietNode",
                "fields": {},
                "secrets": {},
                "blobs": {},
                "node_id": "n1",
                "job_id": "j1",
                "workflow_id": "wf1",
                "user_id": "u1",
                "requires_vram_gb": 2.5,
            },
        }
    )

    assert [f["type"] for f in frames] == ["result"]
    assert_valid(validator, frames[0])


def test_validator_rejects_a_malformed_frame(validator):
    """The validator is not a no-op — a bad frame must actually fail.

    Without this, every assertion above would be indistinguishable from one
    that examines nothing.
    """
    errors = list(validator.iter_errors({"type": "result", "request_id": "", "data": {}}))
    assert errors, "empty request_id must violate the schema"

    errors = list(validator.iter_errors({"type": "discover", "request_id": "d", "data": {}}))
    assert errors, "discover data without `nodes` must violate the schema"
