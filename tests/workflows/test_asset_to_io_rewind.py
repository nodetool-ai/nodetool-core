"""Regression test: reading a memory:// asset holding an IO object is idempotent.

``asset_to_io`` used to return the cached file-like object without rewinding, so
the first ``read()`` left it at EOF and every subsequent read returned b"".
"""

import base64
from io import BytesIO

import pytest

from nodetool.metadata.types import AssetRef
from nodetool.workflows.processing_context import ProcessingContext


def _make_ref(context: ProcessingContext, payload: bytes) -> AssetRef:
    uri = "memory://test-io-object"
    context._memory_set(uri, BytesIO(payload))
    return AssetRef(uri=uri)


@pytest.mark.asyncio
async def test_asset_to_bytes_is_idempotent_for_io_objects(user_id: str):
    context = ProcessingContext(user_id=user_id, auth_token="t")
    payload = b"hello world"
    ref = _make_ref(context, payload)

    assert await context.asset_to_bytes(ref) == payload
    # Second read used to return b"" because the cached buffer was left at EOF
    assert await context.asset_to_bytes(ref) == payload


@pytest.mark.asyncio
async def test_asset_to_base64_is_idempotent_for_io_objects(user_id: str):
    context = ProcessingContext(user_id=user_id, auth_token="t")
    payload = b"some binary \x00\x01 payload"
    ref = _make_ref(context, payload)

    expected = base64.b64encode(payload).decode("utf-8")
    assert await context.asset_to_base64(ref) == expected
    assert await context.asset_to_base64(ref) == expected


@pytest.mark.asyncio
async def test_asset_to_io_returns_rewound_buffer(user_id: str):
    context = ProcessingContext(user_id=user_id, auth_token="t")
    payload = b"abcdef"
    ref = _make_ref(context, payload)

    io_obj = await context.asset_to_io(ref)
    assert io_obj.tell() == 0
    assert io_obj.read() == payload

    io_again = await context.asset_to_io(ref)
    assert io_again.tell() == 0
    assert io_again.read() == payload
