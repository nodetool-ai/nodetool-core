"""Tests that bridge provider requests give providers a processing context."""

from typing import Any

import numpy as np
import pytest

from nodetool.worker import provider_handler


class _ContextRequiringProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def automatic_speech_recognition(self, **kwargs: Any) -> str:
        if kwargs.get("context") is None:
            raise ValueError("ProcessingContext is required for HuggingFace ASR")
        self.calls.append(kwargs)
        return "hello"


@pytest.mark.asyncio
async def test_asr_request_passes_processing_context(monkeypatch):
    provider = _ContextRequiringProvider()
    monkeypatch.setattr(provider_handler, "_get_provider", lambda *_: provider)

    result = await provider_handler._handle_asr(
        {
            "provider": "huggingface",
            "model": "openai/whisper-small",
            "audio": b"RIFF",
            "language": "en",
        }
    )

    assert result == {"text": "hello"}
    assert provider.calls[0]["language"] == "en"


class _ContextRequiringTTSProvider:
    async def text_to_speech(self, **kwargs: Any):
        if kwargs.get("context") is None:
            raise ValueError("ProcessingContext is required for HuggingFace TTS generation")
        yield np.zeros(4, dtype=np.int16)


class _RecordingTransport:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_tts_request_passes_processing_context(monkeypatch):
    monkeypatch.setattr(provider_handler, "_get_provider", lambda *_: _ContextRequiringTTSProvider())
    transport = _RecordingTransport()

    await provider_handler.handle_provider_message(
        "provider.tts",
        "req-1",
        {"provider": "huggingface", "text": "Hello", "model": "hexgrad/Kokoro-82M"},
        transport,
        {},
    )

    types = [msg["type"] for msg in transport.messages]
    assert "error" not in types
    assert types[-1] == "result"
