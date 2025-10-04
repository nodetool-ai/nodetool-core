"""
Base ImageProvider class for image generation services.

This module provides the foundation for all image provider implementations, defining
a common interface that all providers must implement for text-to-image and image-to-image generation.

Note: ImageProvider now inherits from BaseProvider for unified provider interface.
"""

from nodetool.chat.providers.base import BaseProvider, ProviderCapability
from nodetool.image.types import ImageBytes, TextToImageParams, ImageToImageParams
from nodetool.metadata.types import ImageModel
from typing import List, Set
from nodetool.workflows.processing_context import ProcessingContext


class ImageProvider(BaseProvider):
    """
    Abstract base class for image generation providers (HuggingFace, FAL, MLX, etc.).

    Inherits from BaseProvider to provide unified interface across chat and image providers.
    Subclasses must implement:
    - get_capabilities() - Return set containing TEXT_TO_IMAGE and/or IMAGE_TO_IMAGE
    - text_to_image() - If TEXT_TO_IMAGE capability is declared
    - image_to_image() - If IMAGE_TO_IMAGE capability is declared
    - get_available_image_models() - Return list of available image models

    Note: This class provides default implementations that raise NotImplementedError.
    Subclasses should override the methods corresponding to their declared capabilities.

    All logging, error detection, and container environment methods are inherited
    from BaseProvider. The text_to_image() and image_to_image() methods are already
    defined in BaseProvider and will raise NotImplementedError if not overridden.
    """

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get a list of available image models for this provider.

        Subclasses should override this to return their available models.

        Returns:
            List of ImageModel instances available for this provider
        """
        return []
