"""Regression tests for resumable HuggingFace downloads.

The resume path decided whether to append based on the *request* (was a Range
header sent) rather than the *response* (did the server actually serve one).
When the server ignored the range — or when ``accept_ranges`` was False so no
Range header was sent at all — the full body was appended onto the partial
``.incomplete`` file, producing a corrupt result. With ``expected_size`` unknown
the size check could not catch it, so the corruption was silent.
"""

import httpx
import pytest

from nodetool.integrations.huggingface.async_downloader import _download_with_resume

FULL_BODY = b"0123456789ABCDEF"
PARTIAL = FULL_BODY[:6]


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_full_body_response_replaces_partial_file(tmp_path):
    """A 200 (range ignored) must restart, not append onto the partial file."""
    dest = tmp_path / "weights.bin"
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    tmp.write_bytes(PARTIAL)

    def handler(request: httpx.Request) -> httpx.Response:
        # Server ignores the Range header and returns the whole file.
        return httpx.Response(200, content=FULL_BODY)

    async with _client(handler) as client:
        await _download_with_resume(
            client,
            "https://example.invalid/weights.bin",
            dest,
            token=None,
            expected_size=None,  # no size check to mask the corruption
            accept_ranges=True,
        )

    assert dest.read_bytes() == FULL_BODY
    assert not tmp.exists()


@pytest.mark.asyncio
async def test_no_range_support_replaces_partial_file(tmp_path):
    """With accept_ranges False no Range is sent, so the body is the whole file."""
    dest = tmp_path / "weights.bin"
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    tmp.write_bytes(PARTIAL)

    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("Range"))
        return httpx.Response(200, content=FULL_BODY)

    async with _client(handler) as client:
        await _download_with_resume(
            client,
            "https://example.invalid/weights.bin",
            dest,
            token=None,
            expected_size=None,
            accept_ranges=False,
        )

    assert seen_headers == [None]
    assert dest.read_bytes() == FULL_BODY


@pytest.mark.asyncio
async def test_partial_content_response_is_appended(tmp_path):
    """A genuine 206 still resumes — the fix must not disable resuming."""
    dest = tmp_path / "weights.bin"
    tmp = dest.with_suffix(dest.suffix + ".incomplete")
    tmp.write_bytes(PARTIAL)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Range") == f"bytes={len(PARTIAL)}-"
        return httpx.Response(
            206,
            content=FULL_BODY[len(PARTIAL) :],
            headers={
                "Content-Range": f"bytes {len(PARTIAL)}-{len(FULL_BODY) - 1}/{len(FULL_BODY)}"
            },
        )

    async with _client(handler) as client:
        await _download_with_resume(
            client,
            "https://example.invalid/weights.bin",
            dest,
            token=None,
            expected_size=len(FULL_BODY),
            accept_ranges=True,
        )

    assert dest.read_bytes() == FULL_BODY


@pytest.mark.asyncio
async def test_persistent_416_gives_up_instead_of_looping(tmp_path):
    """A server stuck on 416 must exhaust retries rather than spin forever."""
    dest = tmp_path / "weights.bin"
    dest.with_suffix(dest.suffix + ".incomplete").write_bytes(PARTIAL)

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(416)

    async with _client(handler) as client:
        with pytest.raises(RuntimeError, match="416"):
            await _download_with_resume(
                client,
                "https://example.invalid/weights.bin",
                dest,
                token=None,
                expected_size=len(FULL_BODY),
                accept_ranges=True,
                max_retries=3,
            )

    assert calls == 3
