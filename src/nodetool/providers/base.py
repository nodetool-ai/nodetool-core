"""
Base provider class for AI service providers.

This module provides the foundation for all provider implementations, defining
a common interface that providers must implement for chat completions, image generation,
and other AI capabilities. Providers declare their capabilities at runtime.
"""

import datetime
import json
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    ClassVar,
    Sequence,
)

import numpy as np

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    ASRModel,
    EmbeddingModel,
    ImageModel,
    LanguageModel,
    Message,
    MessageFile,
    Provider,
    ToolCall,
    TTSModel,
    VideoModel,
)
from nodetool.metadata.types import (
    Provider as ProviderEnum,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

log = get_logger(__name__)


class ProviderCapability(str, Enum):
    """Capabilities that a provider can support.

    Providers expose capabilities automatically via BaseProvider.get_capabilities(),
    which inspects whether subclasses override the relevant interface methods.
    This allows runtime discovery of provider features and enables multi-modal
    providers (like Gemini) to expose all their features through a single interface.
    """

    GENERATE_MESSAGE = "generate_message"  # Single message generation
    GENERATE_MESSAGES = "generate_messages"  # Streaming message generation
    GENERATE_EMBEDDING = "generate_embedding"  # Text → Embedding vectors
    TEXT_TO_IMAGE = "text_to_image"  # Text → Image generation
    IMAGE_TO_IMAGE = "image_to_image"  # Image transformation
    TEXT_TO_SPEECH = "text_to_speech"  # Text → Speech/Audio generation
    AUTOMATIC_SPEECH_RECOGNITION = "automatic_speech_recognition"  # Speech → Text transcription
    TEXT_TO_VIDEO = "text_to_video"  # Text → Video generation
    IMAGE_TO_VIDEO = "image_to_video"  # Image → Video generation


_PROVIDER_REGISTRY: dict[ProviderEnum, tuple[type["BaseProvider"], dict[str, Any]]] = {}


def register_provider(
    provider: ProviderEnum,
    **kwargs: Any,
) -> Callable[[type["BaseProvider"]], type["BaseProvider"]]:
    """Decorator to register a Provider implementation.

    Args:
        provider: The provider enum value to register
        **kwargs: Additional provider-specific configuration

    Returns:
        Decorator function for registering the provider class
    """

    def decorator(cls: type["BaseProvider"]) -> type["BaseProvider"]:
        _PROVIDER_REGISTRY[provider] = (cls, kwargs)
        return cls

    return decorator


def get_registered_provider(
    provider: ProviderEnum,
) -> tuple[type["BaseProvider"], dict[str, Any]]:
    """Get a registered provider class and its configuration.

    Args:
        provider: The provider enum value to look up

    Returns:
        Tuple of (provider_class, configuration_dict)

    Raises:
        ValueError: If the provider is not registered
    """
    if provider == Provider.Empty:
        raise ValueError("Please specify a provider")
    provider_cls, kwargs = _PROVIDER_REGISTRY.get(provider, (None, {}))
    if provider_cls is None:
        raise ValueError(f"Provider {provider} is not installed")
    return provider_cls, kwargs


class BaseProvider:
    """
    Abstract base class for AI service providers.

    Defines a common interface for different providers (OpenAI, Anthropic, Gemini, FAL, etc.),
    allowing the system to work with any supported provider interchangeably. Capabilities are
    determined automatically based on which capability methods a subclass overrides.

    Capabilities that can be detected include:
    - GENERATE_MESSAGE: Single message generation
    - GENERATE_MESSAGES: Streaming message generation
    - TEXT_TO_IMAGE: Text-to-image generation
    - IMAGE_TO_IMAGE: Image transformation
    - TEXT_TO_SPEECH: Text-to-speech/audio generation
    - AUTOMATIC_SPEECH_RECOGNITION: Audio-to-text transcription
    - TEXT_TO_VIDEO: Text-to-video generation
    - IMAGE_TO_VIDEO: Image-to-video generation
    - STRUCTURED_OUTPUT: Structured JSON output support

    Subclasses should implement:
    - The capability methods (generate_message, text_to_image, text_to_video, image_to_video, etc.) they support
    - get_available_language_models(), get_available_image_models(), get_available_video_models(), etc.
    """

    _CAPABILITY_METHODS: ClassVar[dict[ProviderCapability, str]] = {
        ProviderCapability.GENERATE_MESSAGE: "generate_message",
        ProviderCapability.GENERATE_MESSAGES: "generate_messages",
        ProviderCapability.GENERATE_EMBEDDING: "generate_embedding",
        ProviderCapability.TEXT_TO_IMAGE: "text_to_image",
        ProviderCapability.IMAGE_TO_IMAGE: "image_to_image",
        ProviderCapability.TEXT_TO_SPEECH: "text_to_speech",
        ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION: "automatic_speech_recognition",
        ProviderCapability.TEXT_TO_VIDEO: "text_to_video",
        ProviderCapability.IMAGE_TO_VIDEO: "image_to_video",
    }

    log_file: str | None = None
    provider_name: str = ""
    usage: ClassVar[dict[str, Any]] = {}

    @classmethod
    def required_secrets(cls) -> list[str]:
        """Secrets required for this provider. Override to declare keys."""
        return []

    def __init__(self, secrets: dict[str, str] | None = None):
        self.secrets = secrets or {}
        self.cost = 0.0  # Instance cost tracking
        self._usage_info: Any = None

    def track_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        input_characters: int = 0,
        duration_seconds: float = 0.0,
        image_count: int = 0,
    ) -> float:
        """
        Track usage and calculate cost for the current operation.

        This method should be called by provider implementations after
        each API call to record usage and accumulate cost.

        Args:
            model: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens
            input_characters: Number of input characters (for TTS)
            duration_seconds: Duration in seconds (for ASR)
            image_count: Number of images generated

        Returns:
            The cost of this operation in credits
        """
        from nodetool.providers.cost_calculator import CostCalculator, UsageInfo

        usage = UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_characters=input_characters,
            duration_seconds=duration_seconds,
            image_count=image_count,
        )
        self._usage_info = usage

        cost = CostCalculator.calculate(model, usage, provider=self.provider_name)
        self.cost += cost
        return cost

    def reset_cost(self) -> None:
        """Reset accumulated cost to zero."""
        self.cost = 0.0
        self._usage_info = None

    async def log_provider_call(
        self,
        user_id: str,
        provider: str,
        model: str,
        cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        input_size: int | None = None,
        output_size: int | None = None,
        parameters: dict[str, Any] | None = None,
        node_id: str = "",
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an API call to the database for cost tracking using Prediction model.

        Args:
            user_id: ID of the user making the call
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model identifier (e.g., "gpt-4o-mini")
            cost: Cost of the call in credits
            input_tokens: Number of input/prompt tokens (for text models)
            output_tokens: Number of output/completion tokens (for text models)
            total_tokens: Total number of tokens used (for text models)
            cached_tokens: Number of cached tokens (if applicable)
            reasoning_tokens: Number of reasoning tokens (if applicable)
            input_size: Input data size in bytes (for image/audio/video models)
            output_size: Output data size in bytes (for image/audio/video models)
            parameters: Model-specific parameters (resolution, quality, voice, etc.)
            node_id: Optional node ID for tracking
            workflow_id: Optional workflow ID for tracking
            metadata: Additional metadata about the call
        """
        try:
            from nodetool.models.prediction import Prediction

            await Prediction.create(
                user_id=user_id,
                node_id=node_id,
                provider=provider,
                model=model,
                workflow_id=workflow_id,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                input_size=input_size,
                output_size=output_size,
                parameters=parameters,
                metadata=metadata,
                status="completed",
            )
        except ImportError as e:
            # Handle missing module gracefully
            log.warning(f"Prediction model not available: {e}")
        except (ValueError, TypeError) as e:
            # Handle invalid parameter values
            log.warning(f"Invalid parameters for provider call logging: {e}")
        except Exception as e:
            # Don't fail the API call if logging fails
            log.error(f"Unexpected error logging provider call: {e}", exc_info=True)

    def get_capabilities(self) -> set[ProviderCapability]:
        """Determine supported capabilities based on implemented methods."""
        return {
            capability
            for capability, method_name in self._CAPABILITY_METHODS.items()
            if self._supports_capability(method_name)
        }

    @classmethod
    def _supports_capability(cls, method_name: str) -> bool:
        """Check if a subclass overrides the capability method from BaseProvider."""
        base_method = getattr(BaseProvider, method_name, None)
        subclass_method = getattr(cls, method_name, None)
        if base_method is None or subclass_method is None:
            return False
        return base_method is not subclass_method

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables needed when running inside Docker."""
        return {}

    async def get_available_language_models(self) -> list[LanguageModel]:
        """Get a list of available language models for this provider.

        This method should return all language models that are available for use with this provider.
        The implementation may check for API keys, fetch from external APIs, check local
        cache, or use static model lists.

        Returns:
            List of LanguageModel instances available for this provider.
            Returns empty list if provider doesn't support language models.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_image_models(self) -> list[ImageModel]:
        """Get a list of available image models for this provider.

        This method should return all image models that are available for use with this provider.
        The implementation may check for API keys, local cache, or other requirements.

        Returns:
            List of ImageModel instances available for this provider.
            Returns empty list if provider doesn't support image models.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_tts_models(self) -> list[TTSModel]:
        """Get a list of available text-to-speech models for this provider.

        This method should return all TTS models that are available for use with this provider.
        The implementation may check for API keys, supported voices, or other requirements.

        Returns:
            List of TTSModel instances available for this provider.
            Returns empty list if provider doesn't support TTS.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_asr_models(self) -> list[ASRModel]:
        """Get a list of available automatic speech recognition models for this provider.

        This method should return all ASR models that are available for use with this provider.
        The implementation may check for API keys, supported languages, or other requirements.

        Returns:
            List of ASRModel instances available for this provider.
            Returns empty list if provider doesn't support ASR.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_video_models(self) -> list[VideoModel]:
        """Get a list of available video generation models for this provider.

        This method should return all video models that are available for use with this provider.
        The implementation may check for API keys, local cache, or other requirements.

        Returns:
            List of VideoModel instances available for this provider.
            Returns empty list if provider doesn't support video generation.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_embedding_models(self) -> list[EmbeddingModel]:
        """Get a list of available embedding models for this provider.
        The implementation may check for API keys, local cache, or other requirements.

        Returns:
            List of EmbeddingModel instances available for this provider.
            Returns empty list if provider doesn't support embeddings.

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        return []

    async def get_available_models(
        self,
    ) -> list[LanguageModel | ImageModel | TTSModel | ASRModel | VideoModel | EmbeddingModel]:
        """Get a list of all available models for this provider.

        Returns language, image, TTS, ASR, video, and embedding models combined. Use get_available_language_models(),
        get_available_image_models(), get_available_tts_models(), get_available_asr_models(),
        get_available_video_models(), or get_available_embedding_models() to filter to specific model types.

        Returns:
            List containing LanguageModel, ImageModel, TTSModel, ASRModel, VideoModel, and EmbeddingModel instances

        Raises:
            Exception: If model discovery fails (should be caught and return empty list)
        """
        language_models = await self.get_available_language_models()
        image_models = await self.get_available_image_models()
        tts_models = await self.get_available_tts_models()
        asr_models = await self.get_available_asr_models()
        video_models = await self.get_available_video_models()
        embedding_models = await self.get_available_embedding_models()
        return language_models + image_models + tts_models + asr_models + video_models + embedding_models  # type: ignore

    def is_context_length_error(self, error: Exception) -> bool:
        """Return True if the given error indicates a context window overflow.

        Only relevant for providers with GENERATE_MESSAGE or GENERATE_MESSAGES capability.
        Subclasses can override this method for provider specific logic.
        The default implementation checks for common substrings in the error
        message that typically appear when the prompt exceeds the model's
        context length.
        """
        msg = str(error).lower()
        keywords = [
            "context length",
            "maximum context",
            "context window",
            "token limit",
            "too many tokens",
        ]
        return any(k in msg for k in keywords)

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Return True if the given error indicates a rate limit.

        Subclasses can override this method for provider specific logic.
        """
        msg = str(error).lower()
        keywords = ["rate limit", "too many requests", "quota exceeded"]
        return any(k in msg for k in keywords)

    def is_auth_error(self, error: Exception) -> bool:
        """Return True if the given error indicates an authentication failure.

        Subclasses can override this method for provider specific logic.
        """
        msg = str(error).lower()
        keywords = ["unauthorized", "authentication", "api key", "invalid token"]
        return any(k in msg for k in keywords)

    def _log_api_request(
        self,
        method: str,
        messages: Sequence[Message] | None = None,
        params: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log an API request to the specified log file.

        Args:
            method: The API method being called
            messages: The conversation history (for chat methods)
            params: The generation parameters (for image methods)
            **kwargs: Additional parameters to pass to the API
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                log_entry: dict[str, Any] = {
                    "timestamp": timestamp,
                    "type": "request",
                    "method": method,
                }

                if messages is not None:
                    log_entry["messages"] = [msg.model_dump() for msg in messages]

                if params is not None:
                    # Handle both dict and Pydantic model params
                    if hasattr(params, "model_dump"):
                        log_entry["params"] = params.model_dump()
                    else:
                        log_entry["params"] = params

                log_entry.update(kwargs)
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging API request: {e}")

    def _log_tool_call(self, tool_call: ToolCall) -> None:
        """Log a tool call to the specified log file.

        Args:
            tool_call: The tool call to log
        """
        if not self.log_file:
            return

        try:
            with open(self.log_file, "a") as f:
                timestamp = datetime.datetime.now().isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "type": "tool_call",
                    "tool_name": tool_call.name,
                    "arguments": tool_call.args,
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging tool call: {e}")

    def _log_api_response(
        self,
        method: str,
        response: Message | None = None,
        image_count: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log an API response to the specified log file.

        Args:
            method: The API method that was called
            response: The Message response (for chat methods)
            image_count: Number of images generated (for image methods)
            **kwargs: Additional response metadata
        """
        if not self.log_file:
            return

        try:
            log_entry: dict[str, Any] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "response",
                "method": method,
            }

            # Handle chat response
            if response is not None:
                response_dict = {
                    "role": response.role,
                    "content": response.content,
                }

                # Add tool calls if present
                if response.tool_calls:
                    response_dict["tool_calls"] = [
                        {
                            "function": {
                                "name": tc.name,
                                "arguments": tc.args,
                            }
                        }
                        for tc in response.tool_calls
                    ]

                    # Log each tool call as a separate entry
                    for tool_call in response.tool_calls:
                        self._log_tool_call(tool_call)

                log_entry["response"] = response_dict

            # Handle image response
            if image_count is not None:
                log_entry["image_count"] = image_count

            log_entry.update(kwargs)

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging API response: {e}")

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """Generate a single message completion from the provider.

        Only implemented by providers with GENERATE_MESSAGE capability.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: str containing model information
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Returns:
            A message returned by the provider

        Raises:
            NotImplementedError: If provider doesn't support GENERATE_MESSAGE capability
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support GENERATE_MESSAGE capability")

    def generate_messages(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk | ToolCall | MessageFile]:
        """Generate message completions from the provider, yielding chunks or tool calls.

        Only implemented by providers with GENERATE_MESSAGES capability.
        Subclass implementations should declare this method as async.

        Args:
            messages: Sequence of Message objects representing the conversation
            model: str containing model information
            tools: Available tools for the model to use
            max_tokens: Maximum number of tokens to generate
            response_format: Format of the response
            **kwargs: Additional provider-specific parameters

        Yields:
            Async iterator of Chunk objects with content and completion status or ToolCall objects

        Raises:
            NotImplementedError: If provider doesn't support GENERATE_MESSAGES capability
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support GENERATE_MESSAGES capability")

    async def text_to_image(
        self,
        params: Any,  # TextToImageParams, but imported later to avoid circular deps
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        node_id: str | None = None,
    ) -> bytes:
        """Generate an image from a text prompt.

        Only implemented by providers with TEXT_TO_IMAGE capability.

        Args:
            params: Text-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes (PNG, JPEG, etc.)

        Raises:
            NotImplementedError: If provider doesn't support TEXT_TO_IMAGE capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support TEXT_TO_IMAGE capability")

    async def image_to_image(  # type: ignore[override]
        self,
        image: bytes,
        params: Any,  # ImageToImageParams, but imported later to avoid circular deps
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        node_id: str | None = None,
    ) -> bytes:
        """Transform an image based on a text prompt.

        Only implemented by providers with IMAGE_TO_IMAGE capability.

        Args:
            image: Input image as bytes
            params: Image-to-image generation parameters
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            node_id: Optional node ID for progress tracking
        Returns:
            Raw image bytes (PNG, JPEG, etc.)

        Raises:
            NotImplementedError: If provider doesn't support IMAGE_TO_IMAGE capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support IMAGE_TO_IMAGE capability")

    def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        **kwargs: Any,
    ) -> AsyncGenerator[np.ndarray[Any, np.dtype[np.int16]], None]:
        """Generate speech audio from text as a streaming generator.

        Only implemented by providers with TEXT_TO_SPEECH capability.

        This method yields audio chunks as numpy int16 arrays at 24kHz mono.
        Providers that don't support true streaming should yield a single chunk.

        Args:
            text: Input text to convert to speech
            model: Model identifier for TTS
            voice: Voice identifier/name (provider-specific)
            speed: Speech speed multiplier (typically 0.25 to 4.0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional provider-specific parameters

        Yields:
            numpy.ndarray: Int16 audio chunks at 24kHz mono.
                          Each chunk is a 1D numpy array with dtype=int16.

        Raises:
            NotImplementedError: If provider doesn't support TEXT_TO_SPEECH capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support TEXT_TO_SPEECH capability")

    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        **kwargs: Any,
    ) -> str:
        """Transcribe audio to text using automatic speech recognition.

        Only implemented by providers with AUTOMATIC_SPEECH_RECOGNITION capability.

        Args:
            audio: Input audio as bytes (various formats supported: mp3, mp4, mpeg, mpga, m4a, wav, webm)
            model: Model identifier for ASR (e.g., "whisper-1")
            language: Optional ISO-639-1 language code to improve accuracy and latency
            prompt: Optional text to guide the model's style or continue a previous segment
            temperature: Sampling temperature between 0 and 1 (default 0)
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Transcribed text from the audio

        Raises:
            NotImplementedError: If provider doesn't support AUTOMATIC_SPEECH_RECOGNITION capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If transcription fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support AUTOMATIC_SPEECH_RECOGNITION capability")

    async def text_to_video(
        self,
        params: Any,  # TextToVideoParams, but imported later to avoid circular deps
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        node_id: str | None = None,
    ) -> bytes:
        """Generate a video from a text prompt.

        Only implemented by providers with TEXT_TO_VIDEO capability.

        Args:
            params: Text-to-video generation parameters including:
                - prompt: Text description of the video to generate
                - negative_prompt: Elements to exclude from generation
                - model: Video model to use
                - duration: Video duration in seconds (if supported)
                - fps: Frames per second (if supported)
                - aspect_ratio: Video aspect ratio (e.g., "16:9", "9:16")
                - resolution: Video resolution (e.g., "720p", "1080p")
            timeout_s: Optional timeout in seconds
            context: Optional processing context
            node_id: Optional node ID for progress tracking
        Returns:
            Raw video bytes (MP4, WebM, etc.)

        Raises:
            NotImplementedError: If provider doesn't support TEXT_TO_VIDEO capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support TEXT_TO_VIDEO capability")

    async def image_to_video(  # type: ignore[override]
        self,
        image: bytes,
        params: Any,  # ImageToVideoParams, but imported later to avoid circular deps
        timeout_s: int | None = None,
        context: Any = None,  # ProcessingContext, but imported later
        node_id: str | None = None,
        **kwargs: Any,
    ) -> bytes:
        """Generate a video from an input image.

        Only implemented by providers with IMAGE_TO_VIDEO capability.

        Args:
            image: Input image as bytes
            params: Image-to-video generation parameters including:
                - prompt: Optional text description to guide video generation
                - negative_prompt: Elements to exclude from generation
                - model: Video model to use
                - duration: Video duration in seconds (if supported)
                - fps: Frames per second (if supported)
                - aspect_ratio: Video aspect ratio (e.g., "16:9", "9:16")
                - resolution: Video resolution (e.g., "720p", "1080p")
            timeout_s: Optional timeout in seconds
            context: Optional processing context

        Returns:
            Raw video bytes (MP4, WebM, etc.)

        Raises:
            NotImplementedError: If provider doesn't support IMAGE_TO_VIDEO capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support IMAGE_TO_VIDEO capability")

    async def generate_embedding(
        self,
        text: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate embedding vectors for the given text input(s).

        Only implemented by providers with GENERATE_EMBEDDING capability.

        Args:
            text: Single text string or list of text strings to embed
            model: Model identifier for embedding generation (e.g., "text-embedding-3-small")
            **kwargs: Additional provider-specific parameters (e.g., dimensions)

        Returns:
            List of embedding vectors, one for each input text.
            Each embedding is a list of floats representing the vector.

        Raises:
            NotImplementedError: If provider doesn't support GENERATE_EMBEDDING capability
            ValueError: If required parameters are missing or invalid
            RuntimeError: If embedding generation fails
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support GENERATE_EMBEDDING capability")


class MockProvider(BaseProvider):
    """
    A mock chat provider for testing purposes.

    Allows defining a sequence of responses (text or tool calls) that the
    provider will return upon subsequent calls to generate_message or generate_messages.
    """

    def __init__(self, responses: Sequence[Message], log_file: str | None = None):
        """
        Initialize the MockProvider.

        Args:
            responses: A sequence of Message objects to be returned by generate calls.
                       Each call consumes one message from the sequence.
            log_file: Optional path to a log file.
        """
        super().__init__()
        self.responses = list(responses)  # Store responses
        self.call_log: list[dict[str, Any]] = []  # Log calls made to the provider
        self.response_index = 0
        self.log_file = log_file

    def _get_next_response(self) -> Message:
        """Returns the next predefined response or raises an error if exhausted."""
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        else:
            raise IndexError("MockProvider has run out of predefined responses.")

    async def get_available_models(self) -> list[LanguageModel]:  # type: ignore[override]
        """Mock provider has no models."""
        return []

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> Message:
        """
        Simulates generating a single message.

        Logs the call and returns the next predefined response.
        """
        self._log_api_request("generate_message", messages=messages, model=model, tools=tools, **kwargs)
        self.call_log.append(
            {
                "method": "generate_message",
                "messages": messages,
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        response = self._get_next_response()
        self._log_api_response("generate_message", response)

        return response

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] | None = None,
        max_tokens: int = 8192,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Chunk | ToolCall, Any]:
        """
        Simulates generating messages, yielding chunks or tool calls.

        Currently yields the entire next predefined response. Can be adapted
        to yield individual chunks/tool calls if needed for more granular testing.
        """
        self._log_api_request("generate_messages", messages=messages, model=model, tools=tools, **kwargs)
        self.call_log.append(
            {
                "method": "generate_messages",
                "messages": messages,
                "model": model,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        response = self._get_next_response()
        self._log_api_response("generate_messages", response=response)  # Log the full conceptual response

        # Simulate streaming behavior
        if response.tool_calls:
            for tool_call in response.tool_calls:
                self._log_tool_call(tool_call)  # Log individual tool calls
                yield tool_call
        elif response.content:
            # Yield content as a single chunk for simplicity in this mock
            yield Chunk(content=str(response.content))
