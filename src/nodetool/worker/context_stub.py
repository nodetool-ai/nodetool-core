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
from typing import TYPE_CHECKING, Any

from nodetool.metadata.types import AudioRef, ImageRef, Model3DRef
from nodetool.workflows.processing_context import ProcessingContext

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
        return ImageRef(uri=f"blob://{blob_key}")

    async def image_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        blob_key = f"image_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = b
        return ImageRef(uri=f"blob://{blob_key}")

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

        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        channels = num_channels if data.ndim == 1 else data.shape[1]
        raw = data.tobytes()

        buf = BytesIO()
        data_size = len(raw)
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + data_size))
        buf.write(b"WAVE")
        buf.write(b"fmt ")
        buf.write(struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                              sample_rate * channels * 2, channels * 2, 16))
        buf.write(b"data")
        buf.write(struct.pack("<I", data_size))
        buf.write(raw)

        blob_key = f"audio_{name or 'output'}_{uuid.uuid4().hex[:8]}"
        self._output_blobs[blob_key] = buf.getvalue()
        return AudioRef(uri=f"blob://{blob_key}")

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
        return Model3DRef(uri=f"blob://{blob_key}")

    def get_output_blobs(self) -> dict[str, bytes]:
        return dict(self._output_blobs)

    def drain_progress(self) -> list[Any]:
        """Drain progress messages from the message queue."""
        from nodetool.workflows.types import NodeProgress
        messages = []
        while not self.message_queue.empty():
            try:
                msg = self.message_queue.get_nowait()
                if isinstance(msg, NodeProgress):
                    messages.append(msg)
            except Exception:
                break
        return messages
