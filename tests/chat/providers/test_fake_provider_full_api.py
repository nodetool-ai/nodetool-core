"""
Tests for the FakeProvider full API implementation.

This test suite verifies all provider capabilities:
- Image generation (text_to_image, image_to_image)
- Audio generation (text_to_speech)
- Speech recognition (automatic_speech_recognition)
- Video generation (text_to_video, image_to_video)
- Embeddings (generate_embedding)
- Model discovery methods
"""

import io

import numpy as np
import pytest
from PIL import Image

from nodetool.metadata.types import (
    ASRModel,
    EmbeddingModel,
    ImageModel,
    LanguageModel,
    Provider,
    TTSModel,
    VideoModel,
)
from nodetool.providers.base import ProviderCapability
from nodetool.providers.fake_provider import FakeProvider
from nodetool.providers.types import (
    ImageToImageParams,
    ImageToVideoParams,
    TextToImageParams,
    TextToVideoParams,
)


class TestFakeProviderModelDiscovery:
    """Tests for model discovery methods."""

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test getting available language models."""
        provider = FakeProvider()
        models = await provider.get_available_language_models()

        assert len(models) == 3
        assert all(isinstance(m, LanguageModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-model-v1" in [m.id for m in models]

    @pytest.mark.asyncio
    async def test_get_available_image_models(self):
        """Test getting available image models."""
        provider = FakeProvider()
        models = await provider.get_available_image_models()

        assert len(models) == 2
        assert all(isinstance(m, ImageModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-image-model" in [m.id for m in models]
        # Verify supported tasks
        assert "text_to_image" in models[0].supported_tasks
        assert "image_to_image" in models[0].supported_tasks

    @pytest.mark.asyncio
    async def test_get_available_tts_models(self):
        """Test getting available TTS models."""
        provider = FakeProvider()
        models = await provider.get_available_tts_models()

        assert len(models) == 2
        assert all(isinstance(m, TTSModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-tts" in [m.id for m in models]
        # Verify voices are listed
        assert len(models[0].voices) > 0

    @pytest.mark.asyncio
    async def test_get_available_asr_models(self):
        """Test getting available ASR models."""
        provider = FakeProvider()
        models = await provider.get_available_asr_models()

        assert len(models) == 2
        assert all(isinstance(m, ASRModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-asr" in [m.id for m in models]

    @pytest.mark.asyncio
    async def test_get_available_video_models(self):
        """Test getting available video models."""
        provider = FakeProvider()
        models = await provider.get_available_video_models()

        assert len(models) == 2
        assert all(isinstance(m, VideoModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-video-model" in [m.id for m in models]
        # Verify supported tasks
        assert "text_to_video" in models[0].supported_tasks
        assert "image_to_video" in models[0].supported_tasks

    @pytest.mark.asyncio
    async def test_get_available_embedding_models(self):
        """Test getting available embedding models."""
        provider = FakeProvider()
        models = await provider.get_available_embedding_models()

        assert len(models) == 2
        assert all(isinstance(m, EmbeddingModel) for m in models)
        assert all(m.provider == Provider.Fake for m in models)
        assert "fake-embedding" in [m.id for m in models]
        # Verify dimensions
        assert models[0].dimensions > 0


class TestFakeProviderCapabilities:
    """Tests for capability detection."""

    def test_capabilities_detection(self):
        """Test that all capabilities are properly detected."""
        provider = FakeProvider()
        capabilities = provider.get_capabilities()

        expected_capabilities = {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_MESSAGES,
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.IMAGE_TO_IMAGE,
            ProviderCapability.TEXT_TO_SPEECH,
            ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION,
            ProviderCapability.TEXT_TO_VIDEO,
            ProviderCapability.IMAGE_TO_VIDEO,
            ProviderCapability.GENERATE_EMBEDDING,
        }

        assert capabilities == expected_capabilities


class TestFakeProviderImageGeneration:
    """Tests for image generation capabilities."""

    @pytest.mark.asyncio
    async def test_text_to_image_generates_valid_png(self):
        """Test that text_to_image generates a valid PNG."""
        provider = FakeProvider()
        image_model = ImageModel(
            id="fake-image-model",
            name="Fake Image Model",
            provider=Provider.Fake,
        )
        params = TextToImageParams(
            model=image_model,
            prompt="A test image",
            width=256,
            height=256,
        )

        image_bytes = await provider.text_to_image(params)

        # Verify it's valid PNG data
        assert image_bytes is not None
        assert len(image_bytes) > 0

        # Verify it can be opened as an image
        img = Image.open(io.BytesIO(image_bytes))
        assert img.format == "PNG"
        assert img.size == (256, 256)
        assert img.mode == "RGB"

    @pytest.mark.asyncio
    async def test_text_to_image_with_custom_color(self):
        """Test that text_to_image uses custom color."""
        red_color = (255, 0, 0)
        provider = FakeProvider(image_color=red_color)
        image_model = ImageModel(
            id="fake-image-model",
            name="Fake Image Model",
            provider=Provider.Fake,
        )
        params = TextToImageParams(
            model=image_model,
            prompt="A red image",
            width=64,
            height=64,
        )

        image_bytes = await provider.text_to_image(params)
        img = Image.open(io.BytesIO(image_bytes))

        # Check that the center pixel is red
        pixel = img.getpixel((32, 32))
        assert pixel == red_color

    @pytest.mark.asyncio
    async def test_image_to_image_generates_valid_png(self):
        """Test that image_to_image generates a valid PNG."""
        provider = FakeProvider()
        image_model = ImageModel(
            id="fake-image-model",
            name="Fake Image Model",
            provider=Provider.Fake,
        )
        params = ImageToImageParams(
            model=image_model,
            prompt="Transform image",
            target_width=128,
            target_height=128,
        )

        # Create a dummy input image
        input_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        input_buffer = io.BytesIO()
        input_img.save(input_buffer, format="PNG")
        input_bytes = input_buffer.getvalue()

        output_bytes = await provider.image_to_image(input_bytes, params)

        # Verify output is valid PNG
        output_img = Image.open(io.BytesIO(output_bytes))
        assert output_img.format == "PNG"
        assert output_img.size == (128, 128)

    @pytest.mark.asyncio
    async def test_image_generation_count_tracking(self):
        """Test that image generation calls are tracked."""
        provider = FakeProvider()
        assert provider.image_generation_count == 0

        image_model = ImageModel(
            id="fake-image-model",
            name="Fake Image Model",
            provider=Provider.Fake,
        )
        params = TextToImageParams(
            model=image_model,
            prompt="Test",
        )

        await provider.text_to_image(params)
        assert provider.image_generation_count == 1

        await provider.text_to_image(params)
        assert provider.image_generation_count == 2

        provider.reset_all_counts()
        assert provider.image_generation_count == 0


class TestFakeProviderAudioGeneration:
    """Tests for audio generation capabilities."""

    @pytest.mark.asyncio
    async def test_text_to_speech_generates_valid_audio(self):
        """Test that text_to_speech generates valid audio chunks."""
        provider = FakeProvider(audio_duration_ms=500)

        chunks = []
        async for chunk in provider.text_to_speech("Hello world", "fake-tts"):
            chunks.append(chunk)

        # Verify we got audio chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        assert all(chunk.dtype == np.int16 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_text_to_speech_count_tracking(self):
        """Test that TTS calls are tracked."""
        provider = FakeProvider()
        assert provider.audio_generation_count == 0

        async for _ in provider.text_to_speech("Test", "fake-tts"):
            pass

        assert provider.audio_generation_count == 1


class TestFakeProviderSpeechRecognition:
    """Tests for speech recognition capabilities."""

    @pytest.mark.asyncio
    async def test_asr_returns_configured_response(self):
        """Test that ASR returns the configured response."""
        custom_response = "This is my custom transcription"
        provider = FakeProvider(asr_response=custom_response)

        result = await provider.automatic_speech_recognition(
            audio=b"fake audio data",
            model="fake-asr",
        )

        assert result == custom_response

    @pytest.mark.asyncio
    async def test_asr_default_response(self):
        """Test that ASR returns default response."""
        provider = FakeProvider()

        result = await provider.automatic_speech_recognition(
            audio=b"fake audio data",
            model="fake-asr",
        )

        assert result == "This is a fake transcription."

    @pytest.mark.asyncio
    async def test_asr_count_tracking(self):
        """Test that ASR calls are tracked."""
        provider = FakeProvider()
        assert provider.asr_count == 0

        await provider.automatic_speech_recognition(
            audio=b"fake audio",
            model="fake-asr",
        )
        assert provider.asr_count == 1


class TestFakeProviderVideoGeneration:
    """Tests for video generation capabilities."""

    @pytest.mark.asyncio
    async def test_text_to_video_generates_valid_mp4(self):
        """Test that text_to_video generates valid MP4 data."""
        provider = FakeProvider(video_duration_s=0.5, video_fps=10)
        video_model = VideoModel(
            id="fake-video-model",
            name="Fake Video Model",
            provider=Provider.Fake,
        )
        params = TextToVideoParams(
            model=video_model,
            prompt="A test video",
        )

        video_bytes = await provider.text_to_video(params)

        # Verify we got MP4 data (check for MP4 header signature)
        assert video_bytes is not None
        assert len(video_bytes) > 0
        # MP4 files typically start with 'ftyp' box
        assert b"ftyp" in video_bytes[:100]

    @pytest.mark.asyncio
    async def test_text_to_video_with_resolution(self):
        """Test that text_to_video respects resolution parameter."""
        provider = FakeProvider(video_duration_s=0.5, video_fps=10)
        video_model = VideoModel(
            id="fake-video-model",
            name="Fake Video Model",
            provider=Provider.Fake,
        )
        params = TextToVideoParams(
            model=video_model,
            prompt="A test video",
            resolution="720p",
        )

        video_bytes = await provider.text_to_video(params)

        # Just verify it generates without error
        assert video_bytes is not None
        assert len(video_bytes) > 0

    @pytest.mark.asyncio
    async def test_image_to_video_generates_valid_mp4(self):
        """Test that image_to_video generates valid MP4 data."""
        provider = FakeProvider(video_duration_s=0.5, video_fps=10)
        video_model = VideoModel(
            id="fake-video-model",
            name="Fake Video Model",
            provider=Provider.Fake,
        )
        params = ImageToVideoParams(
            model=video_model,
            prompt="Animate this image",
        )

        # Create a dummy input image
        input_img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        input_buffer = io.BytesIO()
        input_img.save(input_buffer, format="PNG")
        input_bytes = input_buffer.getvalue()

        video_bytes = await provider.image_to_video(input_bytes, params)

        # Verify we got MP4 data
        assert video_bytes is not None
        assert len(video_bytes) > 0
        assert b"ftyp" in video_bytes[:100]

    @pytest.mark.asyncio
    async def test_video_generation_count_tracking(self):
        """Test that video generation calls are tracked."""
        provider = FakeProvider(video_duration_s=0.5, video_fps=10)
        assert provider.video_generation_count == 0

        video_model = VideoModel(
            id="fake-video-model",
            name="Fake Video Model",
            provider=Provider.Fake,
        )
        params = TextToVideoParams(
            model=video_model,
            prompt="Test",
        )

        await provider.text_to_video(params)
        assert provider.video_generation_count == 1


class TestFakeProviderEmbeddings:
    """Tests for embedding generation capabilities."""

    @pytest.mark.asyncio
    async def test_generate_embedding_single_text(self):
        """Test embedding generation for a single text."""
        provider = FakeProvider(embedding_dimensions=512)

        embeddings = await provider.generate_embedding(
            text="Hello world",
            model="fake-embedding",
        )

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 512
        assert all(isinstance(v, float) for v in embeddings[0])

    @pytest.mark.asyncio
    async def test_generate_embedding_multiple_texts(self):
        """Test embedding generation for multiple texts."""
        provider = FakeProvider(embedding_dimensions=256)

        texts = ["Hello", "World", "Test"]
        embeddings = await provider.generate_embedding(
            text=texts,
            model="fake-embedding",
        )

        assert len(embeddings) == 3
        assert all(len(emb) == 256 for emb in embeddings)

    @pytest.mark.asyncio
    async def test_generate_embedding_deterministic(self):
        """Test that embeddings are deterministic for same input."""
        provider = FakeProvider()

        emb1 = await provider.generate_embedding("test", "fake-embedding")
        emb2 = await provider.generate_embedding("test", "fake-embedding")

        # Same input should produce same embedding
        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_generate_embedding_different_for_different_input(self):
        """Test that different inputs produce different embeddings."""
        provider = FakeProvider()

        emb1 = await provider.generate_embedding("hello", "fake-embedding")
        emb2 = await provider.generate_embedding("goodbye", "fake-embedding")

        # Different inputs should produce different embeddings
        assert emb1 != emb2

    @pytest.mark.asyncio
    async def test_embedding_count_tracking(self):
        """Test that embedding calls are tracked."""
        provider = FakeProvider()
        assert provider.embedding_count == 0

        await provider.generate_embedding("test", "fake-embedding")
        assert provider.embedding_count == 1


class TestFakeProviderResetCounts:
    """Tests for count reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_all_counts(self):
        """Test that reset_all_counts resets all counters."""
        provider = FakeProvider(
            audio_duration_ms=100,
            video_duration_s=0.1,
            video_fps=10,
        )

        # Generate some data to increment counts
        from nodetool.metadata.types import Message, MessageTextContent

        messages = [Message(role="user", content=[MessageTextContent(text="Hi")])]
        await provider.generate_message(messages, "test-model")

        image_model = ImageModel(
            id="fake-image-model",
            name="Fake Image Model",
            provider=Provider.Fake,
        )
        await provider.text_to_image(
            TextToImageParams(model=image_model, prompt="Test")
        )

        async for _ in provider.text_to_speech("Test", "fake-tts"):
            pass

        await provider.automatic_speech_recognition(b"audio", "fake-asr")

        video_model = VideoModel(
            id="fake-video-model",
            name="Fake Video Model",
            provider=Provider.Fake,
        )
        await provider.text_to_video(
            TextToVideoParams(model=video_model, prompt="Test")
        )

        await provider.generate_embedding("Test", "fake-embedding")

        # Verify all counts are incremented
        assert provider.call_count > 0
        assert provider.image_generation_count > 0
        assert provider.audio_generation_count > 0
        assert provider.asr_count > 0
        assert provider.video_generation_count > 0
        assert provider.embedding_count > 0

        # Reset and verify
        provider.reset_all_counts()

        assert provider.call_count == 0
        assert provider.image_generation_count == 0
        assert provider.audio_generation_count == 0
        assert provider.asr_count == 0
        assert provider.video_generation_count == 0
        assert provider.embedding_count == 0
