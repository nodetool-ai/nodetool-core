"""Image provider interfaces and registry."""

from nodetool.image.providers.base import ImageProvider
from nodetool.image.providers.registry import (
    register_image_provider,
    get_image_provider,
    list_image_providers,
)
from nodetool.image.providers.huggingface_inference_provider import (
    HuggingFaceInferenceImageProvider,
)
# Use the unified GeminiProvider from chat.providers (supports both chat and images)
from nodetool.chat.providers.gemini_provider import GeminiProvider

register_image_provider("hf_inference", lambda: HuggingFaceInferenceImageProvider())
register_image_provider("gemini", lambda: GeminiProvider())

__all__ = [
    "ImageProvider",
    "register_image_provider",
    "get_image_provider",
    "list_image_providers",
    "GeminiProvider",
    "HuggingFaceInferenceImageProvider",
]
