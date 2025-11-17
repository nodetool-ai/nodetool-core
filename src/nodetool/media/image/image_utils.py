from __future__ import annotations

from io import BytesIO

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
                a = (
                    (a - amin) * (255.0 / (amax - amin))
                    if amax != amin
                    else np.zeros_like(a)
                )
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


def pil_image_to_base64_jpeg(
    image: PIL.Image.Image, max_size: tuple[int, int] = (512, 512), quality: int = 85
) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.

    Args:
        image: PIL Image to convert
        max_size: Maximum size (width, height) to resize to while maintaining aspect ratio
        quality: JPEG quality (0-100)

    Returns:
        str: Base64-encoded JPEG data
    """
    import base64
    from io import BytesIO

    # Convert to RGB if needed (removes alpha channel)
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        background = PIL.Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3] if image.mode == "RGBA" else None)
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Resize if needed
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, PIL.Image.Resampling.LANCZOS)

    # Save as JPEG
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality)

    # Base64 encode without data URI prefix
    base64_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_data


def image_data_to_base64_jpeg(
    image_data: bytes, max_size: tuple[int, int] = (512, 512), quality: int = 85
) -> str:
    """
    Convert image data (bytes) to a base64-encoded JPEG string.

    Args:
        image_data: Raw image data as bytes
        max_size: Maximum size (width, height) to resize to while maintaining aspect ratio
        quality: JPEG quality (0-100)

    Returns:
        str: Base64-encoded JPEG data
    """
    from io import BytesIO

    # Open image with PIL
    with PIL.Image.open(BytesIO(image_data)) as img:
        return pil_image_to_base64_jpeg(img, max_size, quality)


def image_ref_to_base64_jpeg(
    image_ref, max_size: tuple[int, int] = (512, 512), quality: int = 85
) -> str:
    """
    Convert an ImageRef to a base64-encoded JPEG string.

    Handles various ImageRef types:
    - Direct data bytes
    - data: URIs
    - http(s) URLs (downloads the image)
    - file:// URIs (loads from local file)
    - Other URI types

    Args:
        image_ref: The ImageRef object to convert
        max_size: Maximum size (width, height) to resize to while maintaining aspect ratio
        quality: JPEG quality (0-100)

    Returns:
        str: Base64-encoded JPEG data

    Raises:
        ValueError: If the ImageRef cannot be processed
    """
    from nodetool.io.media_fetch import fetch_uri_bytes_and_mime_sync

    # Handle direct data
    if hasattr(image_ref, "data") and image_ref.data is not None:
        return image_data_to_base64_jpeg(image_ref.data, max_size, quality)

    # Handle URI-based images
    uri = getattr(image_ref, "uri", "") if hasattr(image_ref, "uri") else ""

    if not uri:
        raise ValueError("ImageRef has no data or URI")

    # Delegate URI handling to shared sync helper. For http(s), wrap errors with
    # a friendlier message expected by tests.
    try:
        _mime, data = fetch_uri_bytes_and_mime_sync(uri)
    except Exception as e:
        if uri.startswith("http://") or uri.startswith("https://"):
            raise ValueError(f"Failed to download image: {e}") from e
        raise
    # Accept only image-like content; attempt conversion regardless of mime
    return image_data_to_base64_jpeg(data, max_size, quality)
