"""Tests for TTS requests crossing the TypeScript/Python provider bridge."""

from nodetool.worker.provider_handler import _tts_kwargs


def test_tts_kwargs_forwards_common_optional_inputs():
    reference_audio = b"RIFF-test-wave"
    data = {
        "provider": "huggingface",
        "text": "Hello",
        "model": "example/voice-cloner",
        "voice": "speaker-1",
        "speed": 1.2,
        "reference_audio": reference_audio,
        "reference_text": "Reference transcript",
        "language": "en",
        "instructions": "Sound cheerful",
        "secrets": {"HF_TOKEN": "do-not-forward"},
    }

    assert _tts_kwargs(data) == {
        "text": "Hello",
        "model": "example/voice-cloner",
        "voice": "speaker-1",
        "speed": 1.2,
        "reference_audio": reference_audio,
        "reference_text": "Reference transcript",
        "language": "en",
        "instructions": "Sound cheerful",
    }


def test_tts_kwargs_keeps_legacy_request_shape():
    assert _tts_kwargs({"text": "Hello", "model": "legacy"}) == {
        "text": "Hello",
        "model": "legacy",
    }
