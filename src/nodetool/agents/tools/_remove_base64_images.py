from typing import Any, TypeVar

T = TypeVar("T", bound="dict[str, Any] | list[Any]")


def _remove_base64_images(data: T) -> T:
    """Remove image elements entirely from the API response to reduce size."""
    if isinstance(data, dict):
        keys_to_remove = ["image", "image_alt", "image_base64", "image_url"]
        for key in list(data.keys()):
            if key in keys_to_remove:
                data.pop(key, None)
            elif isinstance(data[key], str):
                if data[key].startswith("data:"):
                    data.pop(key, None)
            elif isinstance(data[key], dict | list):
                data[key] = _remove_base64_images(data[key])  # type: ignore[index]
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = _remove_base64_images(data[i])  # type: ignore[index]
    return data
