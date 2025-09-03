from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageOps


def numpy_to_pil_image(arr: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy array of various common shapes/dtypes into a PIL Image.
    Handles float arrays (0..1 or 0..255), integer types, boolean, and
    common layout conversions (CHW -> HWC, single-channel squeeze, batch dim).
    """
    a = np.asarray(arr)

    # Drop simple batch dimension (1, H, W, C) or (1, C, H, W)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]

    # Convert CHW -> HWC if likely channel-first
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))

    # If single-channel in last axis, squeeze to 2D for PIL L mode
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]

    # Normalize dtype to uint8
    if a.dtype in (np.float32, np.float64, np.float16):
        if a.size == 0:
            a = a.astype(np.uint8)
        else:
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
            if amin >= 0.0 and amax <= 1.0:
                a = a * 255.0
            elif amax > 255.0 or amin < 0.0:
                if amax != amin:
                    a = (a - amin) * (255.0 / (amax - amin))
                else:
                    a = np.zeros_like(a)
            a = np.clip(a, 0, 255).astype(np.uint8)
    elif a.dtype == np.uint16:
        a = (a / 257.0).astype(np.uint8)
    elif a.dtype in (np.int16, np.int32, np.int64):
        a = np.clip(a, 0, 255).astype(np.uint8)
    elif a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        try:
            a = np.clip(a, 0, 255).astype(np.uint8)
        except Exception:
            a = a.astype(np.uint8)

    a = np.ascontiguousarray(a)
    return PIL.Image.fromarray(a)


def pil_to_png_bytes(image: PIL.Image.Image) -> bytes:
    buf = BytesIO()
    PIL.ImageOps.exif_transpose(image).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = numpy_to_pil_image(arr)
    return pil_to_png_bytes(img)
