"""The provider handler's side of the TS bridge contract.

- B4: an omitted TS option arrives as msgpack nil (None) and must not replace
  the provider's default.
- B5: tools arrive as ``ProviderTool`` maps and must reach the provider as
  ``Tool`` objects.
- HF1: media requests arrive as ``{provider, params, secrets}`` with camelCase
  keys and a bare model id. Providers need typed params and a context.
- B9: in-process provider calls must stop on a ``cancel`` frame.
- W3/W5: adapter runs are serialized per adapter and long lines are readable.
"""

import asyncio
import sys
from typing import Any, AsyncGenerator

import pytest

from nodetool.metadata.tool_types import Tool
from nodetool.metadata.types import Message, VideoRef
from nodetool.providers.types import ImageToImageParams, TextToImageParams, TextToVideoParams
from nodetool.worker import provider_handler
from nodetool.worker.provider_handler import _tts_kwargs, handle_provider_message


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.sent.append(msg)


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.started = asyncio.Event()
        self.stream_closed = False

    async def generate_message(self, messages, model, max_tokens=8192, temperature=None, **kwargs):
        self.calls.append(("generate", (), {"max_tokens": max_tokens, **kwargs}))
        return Message(role="assistant", content="ok")

    async def generate_messages(self, messages, model, max_tokens=8192, **kwargs) -> AsyncGenerator[Any, None]:
        self.calls.append(("stream", (), {"max_tokens": max_tokens, **kwargs}))
        try:
            self.started.set()
            await asyncio.sleep(30)
            yield Message(role="assistant", content="late")
        finally:
            self.stream_closed = True

    async def text_to_image(self, params, timeout_s=None, context=None, node_id=None):
        self.calls.append(("text_to_image", (params,), {"context": context}))
        return b"png"

    async def image_to_image(self, image, params, timeout_s=None, context=None, node_id=None):
        self.calls.append(("image_to_image", (image, params), {"context": context}))
        self.started.set()
        await asyncio.sleep(30)
        return b"never"

    async def automatic_speech_recognition(self, audio, model, language=None, temperature=0.0, context=None):
        self.calls.append(("asr", (), {"language": language, "temperature": temperature}))
        return "text"


class FlatVideoProvider:
    """The HuggingFace local provider's flat text_to_video signature."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def text_to_video(
        self,
        prompt: str,
        model: str,
        negative_prompt: str | None = None,
        num_frames: int = 49,
        seed: int | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> VideoRef:
        self.kwargs = {"prompt": prompt, "model": model, "num_frames": num_frames, "seed": seed, **kwargs}
        return await context.video_from_io(__import__("io").BytesIO(b"mp4"))


class TypedVideoProvider:
    def __init__(self) -> None:
        self.params: Any = None

    async def text_to_video(self, params, timeout_s=None, context=None, node_id=None):
        self.params = params
        return b"mp4"


@pytest.fixture
def provider(monkeypatch):
    instance = FakeProvider()
    monkeypatch.setattr(provider_handler, "_get_provider", lambda *_args: instance)
    return instance


def test_tts_kwargs_drop_none_options():
    kwargs = _tts_kwargs({"text": "hi", "model": "kokoro", "voice": None, "speed": None, "language": "en"})
    assert kwargs == {"text": "hi", "model": "kokoro", "language": "en"}


@pytest.mark.asyncio
async def test_generate_drops_none_and_converts_tools(provider):
    transport = FakeTransport()
    await handle_provider_message(
        "provider.generate",
        "r1",
        {
            "provider": "fake",
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": None,
            "temperature": None,
            "tools": [{"name": "search", "description": "Search", "inputSchema": {"type": "object"}}],
        },
        transport,
        {},
    )
    assert transport.sent[-1]["type"] == "result", transport.sent
    _, _, kwargs = provider.calls[0]
    assert kwargs["max_tokens"] == 8192
    assert "temperature" not in kwargs
    (tool,) = kwargs["tools"]
    assert isinstance(tool, Tool)
    assert tool.tool_param()["function"] == {
        "name": "search",
        "description": "Search",
        "parameters": {"type": "object"},
    }


@pytest.mark.asyncio
async def test_asr_drops_none_options(provider):
    transport = FakeTransport()
    await handle_provider_message(
        "provider.asr",
        "r1",
        {"provider": "fake", "model": "whisper", "audio": b"wav", "language": None, "temperature": None},
        transport,
        {},
    )
    assert transport.sent[-1]["data"] == {"text": "text"}
    assert provider.calls[0][2] == {"language": None, "temperature": 0.0}


@pytest.mark.asyncio
async def test_text_to_image_builds_typed_params_with_context(provider):
    transport = FakeTransport()
    cancel_flags: dict[str, asyncio.Event] = {}
    await handle_provider_message(
        "provider.text_to_image",
        "r1",
        {
            "provider": "huggingface",
            "params": {
                "model": "black-forest-labs/FLUX.1-dev",
                "prompt": "a cat",
                "negativePrompt": "blurry",
                "numInferenceSteps": 20,
                "guidanceScale": None,
                "width": 768,
            },
            "secrets": {},
        },
        transport,
        cancel_flags,
    )
    assert transport.sent[-1] == {"type": "result", "request_id": "r1", "data": {"blobs": {"image": b"png"}}}
    _, (params,), kwargs = provider.calls[0]
    assert isinstance(params, TextToImageParams)
    assert params.model.id == "black-forest-labs/FLUX.1-dev"
    assert params.model.provider.value == "huggingface"
    assert params.negative_prompt == "blurry"
    assert params.num_inference_steps == 20
    assert params.guidance_scale is None
    assert params.width == 768
    assert kwargs["context"] is not None
    assert cancel_flags == {}


@pytest.mark.asyncio
async def test_image_to_image_cancel_stops_the_call(provider):
    transport = FakeTransport()
    cancel_flags: dict[str, asyncio.Event] = {}
    task = asyncio.create_task(
        handle_provider_message(
            "provider.image_to_image",
            "r1",
            {
                "provider": "mlx",
                "image": b"src",
                "params": {"model": "flux", "prompt": "edit", "targetWidth": 512},
            },
            transport,
            cancel_flags,
        )
    )
    await asyncio.wait_for(provider.started.wait(), 5)
    _, (image, params), kwargs = provider.calls[0]
    assert image == b"src"
    assert isinstance(params, ImageToImageParams)
    assert params.target_width == 512
    context = kwargs["context"]

    cancel_flags["r1"].set()
    await asyncio.wait_for(task, 5)
    assert transport.sent[-1]["type"] == "error"
    assert transport.sent[-1]["data"]["error"] == "Provider operation cancelled"
    assert context.is_cancelled
    assert cancel_flags == {}


@pytest.mark.asyncio
async def test_stream_cancel_closes_the_provider_generator(provider):
    transport = FakeTransport()
    cancel_flags: dict[str, asyncio.Event] = {}
    task = asyncio.create_task(
        handle_provider_message(
            "provider.stream",
            "r1",
            {"provider": "fake", "model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": None},
            transport,
            cancel_flags,
        )
    )
    await asyncio.wait_for(provider.started.wait(), 5)
    assert provider.calls[0][2]["max_tokens"] == 8192
    cancel_flags["r1"].set()
    await asyncio.wait_for(task, 5)
    assert provider.stream_closed
    assert transport.sent[-1]["data"] == {"done": True}


@pytest.mark.asyncio
async def test_text_to_video_reads_nested_params_for_flat_signature(monkeypatch):
    instance = FlatVideoProvider()
    monkeypatch.setattr(provider_handler, "_get_provider", lambda *_args: instance)
    transport = FakeTransport()
    await handle_provider_message(
        "provider.text_to_video",
        "r1",
        {
            "provider": "huggingface",
            "params": {"model": "Wan-AI/Wan2.2", "prompt": "waves", "numFrames": 33, "seed": None},
        },
        transport,
        {},
    )
    assert transport.sent[-1]["data"] == {"blobs": {"video": b"mp4"}}, transport.sent
    assert instance.kwargs == {"prompt": "waves", "model": "Wan-AI/Wan2.2", "num_frames": 33, "seed": None}


@pytest.mark.asyncio
async def test_text_to_video_builds_typed_params_for_base_signature(monkeypatch):
    instance = TypedVideoProvider()
    monkeypatch.setattr(provider_handler, "_get_provider", lambda *_args: instance)
    transport = FakeTransport()
    await handle_provider_message(
        "provider.text_to_video",
        "r1",
        {"provider": "mlx", "params": {"model": "ltx", "prompt": "waves", "aspectRatio": "16:9"}},
        transport,
        {},
    )
    assert transport.sent[-1]["data"] == {"blobs": {"video": b"mp4"}}
    assert isinstance(instance.params, TextToVideoParams)
    assert instance.params.aspect_ratio == "16:9"


ADAPTER = """
import json, pathlib, sys, time
request = json.loads(sys.stdin.readline())
marker = pathlib.Path(sys.argv[1])
if request["operation"] == "models":
    print(json.dumps({"type": "result", "data": {"models": [{"id": "x" * 200_000}]}}), flush=True)
    sys.exit(0)
if marker.exists():
    print(json.dumps({"type": "error", "data": {"error": "concurrent run"}}), flush=True)
    sys.exit(0)
marker.write_text("busy")
time.sleep(0.5)
marker.unlink()
out = pathlib.Path(sys.argv[2]) / (request["params"]["prompt"] + ".png")
out.write_bytes(b"img")
print(json.dumps({"type": "result", "data": {"path": str(out)}}), flush=True)
"""


@pytest.mark.asyncio
async def test_adapter_runs_are_serialized_and_long_lines_are_read(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    marker = tmp_path / "busy"
    monkeypatch.setenv(
        "NODETOOL_PROVIDER_ADAPTER_COMMAND_SERIAL", f"{sys.executable} {adapter} {marker} {tmp_path}"
    )
    monkeypatch.setenv("NODETOOL_PROVIDER_ADAPTER_CAPABILITIES_SERIAL", "text_to_image")
    transport = FakeTransport()

    async def run(rid: str) -> None:
        await handle_provider_message(
            "provider.text_to_image",
            rid,
            {"provider": "serial", "params": {"model": "m", "prompt": rid}},
            transport,
            {},
        )

    await asyncio.gather(run("a"), run("b"))
    results = [m for m in transport.sent if m["type"] == "result"]
    assert len(results) == 2, transport.sent
    assert not list(tmp_path.glob("*.png"))

    # A model list over the 64 KiB asyncio default stream limit.
    await handle_provider_message(
        "provider.models", "m", {"provider": "serial", "model_type": "image"}, transport, {}
    )
    assert transport.sent[-1]["type"] == "result", transport.sent[-1]
    assert len(transport.sent[-1]["data"]["models"][0]["id"]) == 200_000
