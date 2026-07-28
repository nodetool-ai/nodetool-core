"""Regression tests for wire <-> Message content-part conversion.

The wire protocol names the image part ``"image"`` while
:class:`MessageImageContent` declares ``Literal["image_url"]``. Forwarding the
wire name into the constructor raised a pydantic ValidationError, so every
provider request carrying image content failed before reaching the provider.
"""

import pytest

from nodetool.metadata.types import (
    ImageRef,
    MessageAudioContent,
    MessageImageContent,
    MessageTextContent,
    MessageVideoContent,
    VideoRef,
)
from nodetool.worker.provider_handler import (
    _deserialize_messages,
    _serialize_content_part,
)


def _parts(content):
    return _deserialize_messages([{"role": "user", "content": content}])[0].content


def test_image_part_deserializes_without_validation_error():
    parts = _parts([{"type": "image", "image": {"type": "image", "uri": "http://x/y.png"}}])

    assert len(parts) == 1
    assert isinstance(parts[0], MessageImageContent)
    # The model keeps its own discriminator, not the wire name.
    assert parts[0].type == "image_url"
    assert parts[0].image.uri == "http://x/y.png"


def test_image_url_wire_type_is_also_accepted():
    parts = _parts([{"type": "image_url", "image": {"type": "image", "uri": "http://x/y.png"}}])

    assert len(parts) == 1
    assert isinstance(parts[0], MessageImageContent)


def test_video_part_is_preserved():
    parts = _parts([{"type": "video", "video": {"type": "video", "uri": "http://x/y.mp4"}}])

    assert len(parts) == 1
    assert isinstance(parts[0], MessageVideoContent)
    assert parts[0].video.uri == "http://x/y.mp4"


def test_text_and_audio_parts_still_work():
    parts = _parts([
        {"type": "text", "text": "hello"},
        {"type": "audio", "audio": {"type": "audio", "uri": "http://x/y.wav"}},
    ])

    assert isinstance(parts[0], MessageTextContent)
    assert parts[0].text == "hello"
    assert isinstance(parts[1], MessageAudioContent)


@pytest.mark.parametrize(
    "part",
    [
        {"type": "bogus", "value": 1},
        {"type": "image"},  # missing payload field
        {},
    ],
)
def test_unknown_or_incomplete_parts_are_dropped_not_raised(part):
    assert _parts([part]) == []


@pytest.mark.parametrize(
    ("part", "expected_type", "field"),
    [
        (MessageTextContent(text="hi"), "text", "text"),
        (MessageImageContent(image=ImageRef(uri="http://x/y.png")), "image", "image"),
        (MessageVideoContent(video=VideoRef(uri="http://x/y.mp4")), "video", "video"),
    ],
)
def test_serialize_uses_wire_names(part, expected_type, field):
    wire = _serialize_content_part(part)

    assert wire["type"] == expected_type
    assert field in wire


def test_round_trip_preserves_part_types():
    original = [
        {"type": "text", "text": "hello"},
        {"type": "image", "image": {"type": "image", "uri": "http://x/y.png"}},
        {"type": "video", "video": {"type": "video", "uri": "http://x/y.mp4"}},
    ]

    round_tripped = [_serialize_content_part(p) for p in _parts(original)]

    assert [p["type"] for p in round_tripped] == ["text", "image", "video"]
