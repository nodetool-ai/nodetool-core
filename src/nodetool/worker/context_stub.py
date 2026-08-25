"""
Thin ProcessingContext subclass for the worker server.

Overrides only what's needed:
- Injects per-request secrets (from the bridge protocol)
- Captures output blobs produced by media conversion methods
"""

import asyncio
import os
import uuid
from io import BytesIO
from typing import IO, TYPE_CHECKING, Any

from nodetool.metadata.types import AudioRef, ImageRef, Model3DRef, VideoRef
from nodetool.workflows.processing_context import ProcessingContext, _read_buffer

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image


class WorkerContext(ProcessingContext):
    """ProcessingContext configured for isolated worker execution."""

    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        cancel_event: asyncio.Event | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._request_secrets = secrets or {}
        self._cancel_event = cancel_event or asyncio.Event()
        self._output_blobs: dict[str, bytes] = {}

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    async def get_secret(self, key: str) -> str | None:
        val = self._request_secrets.get(key)
        if val:
            return val
        return os.environ.get(key)

    async def get_secret_required(self, key: str) -> str:
        val = await self.get_secret(key)
        if val is None:
            raise ValueError(f"Required secret not found: {key}")
        return val

    # -- Override media output methods to capture blobs --

    async def image_from_pil(
        self,
        image: "PIL.Image.Image",
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        buf = BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        blob_key = f"image_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = data
        return ImageRef(uri=f"blob://{blob_key}", metadata=metadata)

    async def image_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        blob_key = f"image_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = b
        return ImageRef(uri=f"blob://{blob_key}", metadata=metadata)

    async def audio_from_numpy(
        self,
        data: "np.ndarray",
        sample_rate: int,
        num_channels: int = 1,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> AudioRef:
        import struct

        import numpy as np

        if data.dtype == np.int16:
            raw = data.tobytes()
        elif data.dtype in (np.float16, np.float32, np.float64):
            raw = (np.asarray(data, dtype=np.float32).clip(-1.0, 1.0) * np.float32(32767.0)).astype(np.int16).tobytes()
        else:
            raise ValueError(f"Unsupported dtype {data.dtype}")
        # Always honour the caller's channel count, like the base
        # ProcessingContext: samples are taken as interleaved frames.
        channels = num_channels

        buf = BytesIO()
        data_size = len(raw)
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, sample_rate * channels * 2, channels * 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(raw)

        blob_key = f"audio_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = buf.getvalue()
        return AudioRef(uri=f"blob://{blob_key}")

    async def video_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoRef:
        # Every other video factory funnels here — video_from_bytes,
        # video_from_numpy and video_from_frames all call it — so this one
        # override captures them all. Without it a video node fell through to
        # the base implementation, which returns VideoRef(data=...) with no
        # uri: the executor only extracts blobs from a `blob://` uri, and
        # _serialize_asset_ref strips raw `data` at any depth, so the mp4 was
        # dropped while its metadata reached the host.
        blob_key = f"video_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = await _read_buffer(buffer)
        return VideoRef(uri=f"blob://{blob_key}", metadata=metadata)

    async def audio_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        content_type: str = "audio/wav",
    ) -> AudioRef:
        # audio_from_numpy encodes its own WAV, but a node handed already
        # encoded audio (a provider's mp3, a file read from disk) calls
        # audio_from_bytes / audio_from_base64, which both funnel here. Without
        # this override they fell through to the base implementation and got
        # AudioRef(data=..., uri=""); the executor extracts blobs only from a
        # `blob://` uri and the serializer strips raw `data` at any depth, so
        # the bytes were dropped.
        #
        # `content_type` selects an asset's MIME type in the base
        # implementation. A worker creates no asset, and AudioRef carries no
        # field for it, so it stays dropped like `parent_id`.
        blob_key = f"audio_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = await _read_buffer(buffer)
        return AudioRef(uri=f"blob://{blob_key}")

    async def image_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        # image_from_url and every other caller that holds an open buffer
        # funnels here. Without the override the base implementation returned
        # ImageRef(data=..., uri="") and the bytes never reached the host.
        blob_key = f"image_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = await _read_buffer(buffer)
        return ImageRef(uri=f"blob://{blob_key}", metadata=metadata)

    async def model3d_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Model3DRef:
        blob_key = f"model3d_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = await _read_buffer(buffer)
        return Model3DRef(uri=f"blob://{blob_key}", format=format, metadata=metadata)

    async def audio_from_segment(
        self,
        audio_segment: Any,
        name: str | None = None,
        parent_id: str | None = None,
        **kwargs: Any,
    ) -> AudioRef:
        # With a name the base implementation funnels through audio_from_io
        # and comes back as a blob. With no name it takes a shortcut the host
        # cannot follow: AudioRef(uri="memory://<uuid>", data=...), pointing at
        # an in-process store that lives and dies inside the worker. The
        # executor recognizes only blob://, so that ref crossed the bridge
        # empty. Re-home the bytes; the encoding and the metadata stay the
        # base implementation's.
        ref = await super().audio_from_segment(audio_segment, name=name, parent_id=parent_id, **kwargs)
        if ref.uri.startswith("blob://"):
            return ref
        blob_key = f"audio_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = ref.data or b""
        return AudioRef(uri=f"blob://{blob_key}", metadata=ref.metadata)

    async def model3d_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Model3DRef:
        blob_key = f"model3d_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = b
        # `format` and `metadata` are declared fields of the ref. Accepting
        # them and building the ref without them left a node that used the
        # canonical factory with no format to carry, which is the other half of
        # why a Model3DRef reached the host bare.
        #
        # `parent_id` has no field on the ref — it addresses a folder in the
        # asset store, which a worker does not have — so it stays dropped.
        return Model3DRef(uri=f"blob://{blob_key}", format=format, metadata=metadata)

    def get_output_blobs(self) -> dict[str, bytes]:
        return dict(self._output_blobs)

    def take_output_blobs(self) -> dict[str, bytes]:
        """Return the captured blobs and clear them.

        Used on the streaming path so per-chunk blobs are released as soon as
        they have been emitted, instead of accumulating for the whole request.
        """
        blobs = self._output_blobs
        self._output_blobs = {}
        return blobs

    def drain_progress(self) -> list[Any]:
        """Drain NodeProgress messages from the message queue.

        Anything else in the queue is discarded — callers that also want
        previews/logs should use :meth:`drain_messages` instead.
        """
        from nodetool.workflows.types import NodeProgress

        return [msg for msg in self.drain_messages() if isinstance(msg, NodeProgress)]

    def drain_messages(self) -> list[Any]:
        """Drain all forwardable messages from the message queue.

        Returns NodeProgress plus the update types a client can render live
        (PreviewUpdate, LogUpdate, BinaryUpdate) — previously everything but
        NodeProgress was silently discarded, so in-flight previews and logs
        could never reach the UI. Message types with no wire representation
        are still dropped.
        """
        from nodetool.workflows.types import (
            BinaryUpdate,
            LogUpdate,
            NodeProgress,
            PreviewUpdate,
        )

        keep = (NodeProgress, PreviewUpdate, LogUpdate, BinaryUpdate)
        messages = []
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                if isinstance(msg, keep):
                    messages.append(msg)
            except Exception:
                break
        return messages
