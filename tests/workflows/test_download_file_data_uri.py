"""Regression tests for data: URI handling in ProcessingContext.download_file.

The handler unconditionally base64-decoded the payload, so a percent-encoded
data URI (``data:text/plain,hi`` — the non-base64 form the RFC allows) raised
``binascii.Error``. It also indexed the media type blindly, so a URI without
one (``data:,hi``) raised IndexError.
"""

import base64

import pytest

from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def ctx():
    return ProcessingContext()


@pytest.mark.asyncio
async def test_percent_encoded_data_uri(ctx):
    f = await ctx.download_file("data:text/plain,hello world!")

    assert f.read() == b"hello world!"
    assert f.name.endswith(".plain")


@pytest.mark.asyncio
async def test_percent_escapes_are_decoded(ctx):
    f = await ctx.download_file("data:text/plain;charset=utf-8,a%20b")

    assert f.read() == b"a b"


@pytest.mark.asyncio
async def test_base64_data_uri_still_works(ctx):
    payload = b"\x89PNG\r\n\x1a\n"
    uri = "data:image/png;base64," + base64.b64encode(payload).decode()

    f = await ctx.download_file(uri)

    assert f.read() == payload
    assert f.name.endswith(".png")


@pytest.mark.asyncio
async def test_data_uri_without_media_type(ctx):
    f = await ctx.download_file("data:,bare")

    assert f.read() == b"bare"
    assert f.name.endswith(".bin")


@pytest.mark.asyncio
async def test_malformed_data_uri_raises_value_error(ctx):
    with pytest.raises(ValueError, match="Malformed data URI"):
        await ctx.download_file("data:text/plain;base64")
