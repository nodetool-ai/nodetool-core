"""Shared msgpack decoding for the worker transports.

The TypeScript server encodes some values (for example an unset/empty model
selection) using msgpack *extension* types. The Python side has no encoder for
those custom extensions, so a bare ``msgpack.unpackb`` leaves them as raw
``msgpack.ExtType`` objects. Those objects then flow into node property
assignment, where pydantic rejects them with a confusing error such as::

    Error converting value for property `model`:
    Input should be a valid string [input_value=ExtType(code=0, data=b'\\x00')]

There is no meaningful Python value for an unknown extension, so we decode any
unrecognized extension to ``None`` (treated downstream as "no value", which
falls back to the property default). This keeps a single un-decodable sentinel
from crashing the whole execute request.
"""

from __future__ import annotations

import logging
from typing import Any

import msgpack

log = logging.getLogger(__name__)


def _ext_hook(code: int, data: bytes) -> Any:
    """Decode unknown msgpack extension types to ``None``.

    Code ``-1`` (the standard msgpack timestamp) is left to msgpack's own
    handling; everything else is an extension the Python worker has no decoder
    for, so it is dropped to ``None`` rather than surfacing a raw ``ExtType``.
    """
    if code == -1:
        return msgpack.ExtType(code, data)
    log.debug("Dropping unknown msgpack extension type code=%s to None", code)
    return None


def decode_message(payload: bytes) -> Any:
    """Decode a msgpack payload from a worker transport.

    Uses :func:`_ext_hook` so unknown extension types become ``None`` instead
    of raw ``ExtType`` objects.
    """
    return msgpack.unpackb(payload, raw=False, ext_hook=_ext_hook)
