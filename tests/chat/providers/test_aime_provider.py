"""
Tests for AIME provider.

AIME provides access to AI models through the AIME Model API.
This test suite verifies that the AIME provider correctly:
- Initializes with the correct credentials
- Supports streaming and non-streaming completions via progress polling
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.metadata.types import Message, MessageTextContent
from nodetool.providers.aime_provider import AIMEProvider


class TestAIMEProvider:
    """Test suite for AIME provider."""

    def test_initialization(self):
        """Test that AIME provider initializes with correct configuration."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.provider.value == "aime"

    def test_required_secrets(self):
        """Test that AIME provider requires the correct credentials."""
        required = AIMEProvider.required_secrets()
        assert "AIME_API_KEY" in required
        assert "AIME_USER" not in required

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "AIME_API_KEY" in env
        assert env["AIME_API_KEY"] == "test-key"
        assert "AIME_USER" not in env

    def test_tool_support(self):
        """Test that AIME does not support tools."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # AIME doesn't support tools
        assert not provider.has_tool_support("Mistral-Small-3.1-24B-Instruct")

    def test_messages_to_prompt(self):
        """AIMEProvider now uses _messages_to_chat_context, but let's check internal message conversion if any."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
        ]

        # The provider now uses _messages_to_chat_context which returns JSON
        chat_context = provider._messages_to_chat_context(messages)
        import json
        data = json.loads(chat_context)
        
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "Hello, how are you?"
        assert data[1]["role"] == "assistant"
        assert data[1]["content"] == "I'm doing well, thank you!"

    def test_messages_to_prompt_with_content_list(self):
        """Test message conversion with content as list."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        messages = [
            Message(role="user", content=[MessageTextContent(text="Hello world")]),
        ]

        chat_context = provider._messages_to_chat_context(messages)
        import json
        data = json.loads(chat_context)

        assert data[0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        # Mock response
        api_response = {
            "success": True,
            "job_id": "JID123",
            "job_result": {
                "success": True,
                "text": "This is a test response from AIME.",
                "num_generated_tokens": 10,
                "model_name": "llama4_chat",
            },
            "job_state": "done",
        }

        # Create mock session
        class MockResponse:
            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def __init__(self, *args, **kwargs):
                self.call_count = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                return MockResponse({"success": True, "job_state": "done", "job_result": api_response["job_result"]})

            def post(self, url, **kwargs):
                return MockResponse(api_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test message")]

            result = await provider.generate_message(
                messages=messages,
                model="llama4_chat",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "This is a test response from AIME."

    @pytest.mark.asyncio
    async def test_generate_message_with_polling(self):
        """Test non-streaming message generation with progress polling."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        initial_response = {
            "success": True,
            "job_id": "JID456",
            "job_state": "processing",
        }

        progress_response = {
            "success": True,
            "job_id": "JID456",
            "job_state": "done",
            "job_result": {
                "success": True,
                "text": "Response after polling.",
                "num_generated_tokens": 5,
            },
        }

        class MockResponse:
            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                if "progress" in url:
                    return MockResponse(progress_response)
                return MockResponse({"success": False})

            def post(self, url, **kwargs):
                return MockResponse(initial_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test")]

            result = await provider.generate_message(
                messages=messages,
                model="llama4_chat",
            )

            assert result.content == "Response after polling."

    @pytest.mark.asyncio
    async def test_generate_messages_streaming(self):
        """Test streaming message generation with progress updates."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        initial_response = {
            "success": True,
            "job_id": "JID789",
            "job_state": "processing",
        }

        # Simulate progress updates
        progress_responses = [
            {
                "success": True,
                "job_id": "JID789",
                "job_state": "processing",
                "progress": {
                    "progress": 1,
                    "progress_data": {"text": "Hello", "num_generated_tokens": 1},
                },
            },
            {
                "success": True,
                "job_id": "JID789",
                "job_state": "processing",
                "progress": {
                    "progress": 2,
                    "progress_data": {"text": "Hello world", "num_generated_tokens": 2},
                },
            },
            {
                "success": True,
                "job_id": "JID789",
                "job_state": "done",
                "progress": {
                    "progress": 3,
                    "progress_data": {"text": "Hello world!", "num_generated_tokens": 3},
                },
                "job_result": {
                    "success": True,
                    "text": "Hello world!",
                    "num_generated_tokens": 3,
                },
            },
        ]

        class MockResponse:
            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        progress_index = 0

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                nonlocal progress_index
                if "progress" in url:
                    resp = progress_responses[min(progress_index, len(progress_responses) - 1)]
                    progress_index += 1
                    return MockResponse(resp)
                return MockResponse({"success": False})

            def post(self, url, **kwargs):
                return MockResponse(initial_response)

        with patch("aiohttp.ClientSession", MockSession), patch("asyncio.sleep", new_callable=AsyncMock):  # Skip sleep delays
            messages = [Message(role="user", content="Test")]

            chunks = []
            async for chunk in provider.generate_messages(
                messages=messages,
                model="llama4_chat",
            ):
                chunks.append(chunk)

            # Should have received chunks
            assert len(chunks) > 0

            # Last chunk should be done
            assert chunks[-1].done is True

            # Reconstruct full text
            full_text = "".join(c.content for c in chunks)
            assert "Hello" in full_text
            assert "world!" in full_text

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that AIME returns available language models."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        models = await provider.get_available_language_models()

        assert len(models) > 0
        model_ids = [m.id for m in models]
        assert "llama4_chat" in model_ids

        # Verify provider is set correctly
        for model in models:
            assert model.provider.value == "aime"

    @pytest.mark.asyncio
    async def test_api_error(self):
        """Test handling of API errors."""
        provider = AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

        api_error_response = {
            "success": False,
            "error": "Model unavailable",
        }

        class MockResponse:
            def __init__(self, data):
                self._data = data

            async def json(self):
                return self._data

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                return MockResponse(api_error_response)

            def post(self, url, **kwargs):
                return MockResponse(api_error_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test")]

            with pytest.raises(RuntimeError, match="AIME API error: Model unavailable"):
                await provider.generate_message(
                    messages=messages,
                    model="llama4_chat",
                )

    def test_initialization_missing_key(self):
        """Test that initialization fails without AIME_API_KEY."""
        with pytest.raises(AssertionError, match="AIME_API_KEY is required"):
            AIMEProvider(secrets={})
