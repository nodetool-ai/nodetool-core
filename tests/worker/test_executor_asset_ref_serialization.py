"""Serialization of AssetRef outputs on the worker bridge.

`_serialize_value` used to reduce every AssetRef to {"uri", "type"}. That is
right only for a ref whose payload travels separately as a blob keyed by a
`blob://` uri. It is wrong for a ref that carries its payload inline:
DataframeRef holds the whole table in `columns` and `data` with no blob and an
empty uri, so a dataframe output arrived at the host empty and the node
reported success.
"""

import asyncio

import msgpack
import numpy as np
import pytest
from pydantic import Field

from nodetool.metadata.types import (
    AudioRef,
    ColumnDef,
    DataframeRef,
    ImageRef,
    Model3DRef,
    VideoRef,
)
from nodetool.worker.executor import (
    _serialize_value,
    execute_node,
    msgpack_default,
)
from nodetool.workflows.base_node import NODE_BY_TYPE, BaseNode
from nodetool.workflows.processing_context import ProcessingContext


def test_dataframe_ref_keeps_columns_and_rows():
    """The whole payload of a DataframeRef is inline; none of it may be dropped."""
    df = DataframeRef(
        columns=[
            ColumnDef(name="token", data_type="string"),
            ColumnDef(name="score", data_type="float"),
        ],
        data=[["paris", 0.9], ["lyon", 0.05]],
    )

    out = _serialize_value(df)

    assert out["type"] == "dataframe"
    assert out["data"] == [["paris", 0.9], ["lyon", 0.05]]
    assert [c["name"] for c in out["columns"]] == ["token", "score"]
    assert [c["data_type"] for c in out["columns"]] == ["string", "float"]


def test_model3d_ref_keeps_format_and_files():
    m = Model3DRef(
        uri="blob://model",
        format="obj",
        material_file=ImageRef(uri="file://model.mtl"),
        texture_files=[ImageRef(uri="file://wood.png")],
    )

    out = _serialize_value(m)

    assert out["type"] == "model_3d"
    assert out["format"] == "obj"
    assert out["material_file"]["uri"] == "file://model.mtl"
    assert [t["uri"] for t in out["texture_files"]] == ["file://wood.png"]


def test_video_ref_keeps_duration_and_format():
    out = _serialize_value(VideoRef(uri="blob://clip", duration=12.5, format="mp4"))

    assert out["type"] == "video"
    assert out["duration"] == 12.5
    assert out["format"] == "mp4"


@pytest.mark.parametrize(
    "ref",
    [
        ImageRef(uri="blob://img", data=b"\x89PNG-bytes"),
        AudioRef(uri="blob://aud", data=b"RIFF-bytes"),
        VideoRef(uri="blob://vid", data=[b"chunk-a", b"chunk-b"]),
    ],
)
def test_blob_backed_ref_keeps_its_pointer_and_sheds_its_bytes(ref):
    """The blob path is unchanged: the uri still points at the blobs map entry.

    Bytes must not ride along in `data` — _extract_outputs already puts them in
    the frame's separate blobs map, and inlining them would send the payload
    twice.
    """
    out = _serialize_value(ref)

    assert out["uri"] == ref.uri
    assert out["uri"].startswith("blob://")
    # `.get` deliberately: this invariant held before the fix too (the old
    # flattening emitted no `data` key at all), so it is a regression guard on
    # the blob path rather than a red-before-green assertion.
    assert out.get("data") is None


def test_nested_texture_bytes_are_not_inlined():
    """Model3DRef nests ImageRefs, so the bytes check cannot be top-level only."""
    m = Model3DRef(
        uri="blob://model",
        material_file=ImageRef(uri="blob://mtl", data=b"material-bytes"),
        texture_files=[ImageRef(uri="blob://tex", data=b"texture-bytes")],
    )

    out = _serialize_value(m)

    assert out["material_file"]["data"] is None
    assert out["texture_files"][0]["data"] is None
    # The pointers survive; only the bytes are shed.
    assert out["texture_files"][0]["uri"] == "blob://tex"


def test_serialized_dataframe_survives_msgpack():
    """The frame must still pack — including numpy scalars from a model output."""
    df = DataframeRef(
        columns=[ColumnDef(name="score", data_type="float")],
        data=[[np.float32(0.25)], [0.5]],
    )

    packed = msgpack.packb(
        {"outputs": {"output": _serialize_value(df)}},
        default=msgpack_default,
        datetime=True,
    )
    rows = msgpack.unpackb(packed, raw=False)["outputs"]["output"]["data"]

    assert rows == [[pytest.approx(0.25)], [0.5]]


class DataframeNode(BaseNode):
    """Return an inline dataframe, the way FillMask does."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.DataframeNode"

    async def process(self, context: ProcessingContext) -> DataframeRef:
        return DataframeRef(
            columns=[ColumnDef(name="token", data_type="string")],
            data=[["paris"], ["lyon"]],
        )


@pytest.fixture
def dataframe_node():
    NODE_BY_TYPE["test.DataframeNode"] = DataframeNode
    yield
    NODE_BY_TYPE.pop("test.DataframeNode", None)


@pytest.mark.asyncio
async def test_execute_node_returns_dataframe_rows(dataframe_node):
    """End to end through the executor: the rows reach the result frame."""
    result = await execute_node(
        node_type="test.DataframeNode",
        fields={},
        secrets={},
        input_blobs={},
    )

    output = result["outputs"]["output"]
    assert output["type"] == "dataframe"
    assert output["data"] == [["paris"], ["lyon"]]
    assert [c["name"] for c in output["columns"]] == ["token"]
    assert result["blobs"] == {}
