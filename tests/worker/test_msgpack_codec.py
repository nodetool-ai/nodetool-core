"""Tests for the worker's shared msgpack decoding (ext_hook handling)."""

import msgpack

from nodetool.worker.msgpack_codec import decode_message


def test_decodes_plain_message():
    payload = msgpack.packb({"type": "execute", "node": {"model": "whisper"}})
    assert decode_message(payload) == {
        "type": "execute",
        "node": {"model": "whisper"},
    }


def test_unknown_extension_decodes_to_none():
    # The frontend encodes an unset/empty value with a custom extension type
    # (e.g. ExtType(code=0, data=b"\x00")). It must decode to None rather than
    # surface a raw ExtType that later breaks property validation.
    payload = msgpack.packb({"model": msgpack.ExtType(0, b"\x00")})
    assert decode_message(payload) == {"model": None}


def test_nested_unknown_extension_decodes_to_none():
    payload = msgpack.packb(
        {"properties": {"model": msgpack.ExtType(0, b"\x00"), "name": "x"}}
    )
    assert decode_message(payload) == {"properties": {"model": None, "name": "x"}}
