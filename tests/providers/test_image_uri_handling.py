import asyncio
import base64
from typing import Any

import PIL.Image
from io import BytesIO

import pytest

from nodetool.config.environment import Environment
from nodetool.chat.providers.openai_provider import OpenAIProvider
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from nodetool.chat.providers.huggingface_provider import HuggingFaceProvider
from nodetool.metadata.types import Message, MessageImageContent, ImageRef, Provider


@pytest.fixture(autouse=True)
def clear_memory_cache() -> None:
    try:
        Environment.get_memory_uri_cache().clear()
    except Exception:
        pass


def _make_png_bytes() -> bytes:
    img = PIL.Image.new("RGB", (4, 4), (128, 64, 32))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_factory",
    [OpenAIProvider, GeminiProvider],
)
async def test_providers_handle_data_and_file_and_memory_image_uris(
    provider_factory: Any, tmp_path
) -> None:
    # Prepare data URI
    png_bytes = _make_png_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    # Prepare file URI
    file_path = tmp_path / "test.png"
    file_path.write_bytes(png_bytes)
    file_uri = file_path.resolve().as_uri()

    # Prepare memory URI
    mem_uri = "memory://unit-test-image"
    Environment.get_memory_uri_cache().set(mem_uri, png_bytes)

    provider = provider_factory()

    # Build messages with image URIs
    for uri in (data_uri, file_uri, mem_uri):
        msg = Message(
            role="user", content=[MessageImageContent(image=ImageRef(uri=uri))]
        )
        # Providers convert messages; ensure no exceptions
        if isinstance(provider, OpenAIProvider):
            await provider.convert_message(msg)
        elif isinstance(provider, GeminiProvider):
            await provider._prepare_message_content(msg)


@pytest.mark.asyncio
async def test_anthropic_and_hf_handle_non_http_and_bytes(tmp_path) -> None:
    png_bytes = _make_png_bytes()
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

    # Anthropic: non-http should become base64 source
    anthropic = AnthropicProvider()
    msg = Message(
        role="user", content=[MessageImageContent(image=ImageRef(uri=data_uri))]
    )
    converted = anthropic.convert_message(msg)
    assert isinstance(converted, dict)
    assert converted["role"] == "user"

    # HuggingFace: raw bytes converted to data URI
    hf = HuggingFaceProvider()
    msg2 = Message(
        role="user", content=[MessageImageContent(image=ImageRef(data=png_bytes))]
    )
    converted2 = hf.convert_message(msg2)
    assert isinstance(converted2, dict)
    assert converted2["role"] == "user"
