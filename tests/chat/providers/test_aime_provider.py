"""
Tests for AIME provider.

AIME provides access to AI models through the AIME Model API.
This test suite verifies that the AIME provider correctly:
- Initializes with the correct credentials
- Authenticates and gets a session auth key
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
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})
        assert provider.user == "test-user"
        assert provider.api_key == "test-key"
        assert provider.provider.value == "aime"

    def test_required_secrets(self):
        """Test that AIME provider requires the correct credentials."""
        required = AIMEProvider.required_secrets()
        assert "AIME_USER" in required
        assert "AIME_API_KEY" in required

    def test_container_env(self):
        """Test that container environment variables are correctly set."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        mock_context = MagicMock()

        env = provider.get_container_env(mock_context)
        assert "AIME_USER" in env
        assert env["AIME_USER"] == "test-user"
        assert "AIME_API_KEY" in env
        assert env["AIME_API_KEY"] == "test-key"

    def test_tool_support(self):
        """Test that AIME does not support tools."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        # AIME doesn't support tools
        assert not provider.has_tool_support("Mistral-Small-3.1-24B-Instruct")

    def test_messages_to_prompt(self):
        """Test message conversion to prompt format."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        messages = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm doing well, thank you!"),
            Message(role="user", content="What's the weather like?"),
        ]

        prompt = provider._messages_to_prompt(messages)

        assert "User: Hello, how are you?" in prompt
        assert "Assistant: I'm doing well, thank you!" in prompt
        assert "User: What's the weather like?" in prompt
        assert prompt.endswith("Assistant:")

    def test_messages_to_prompt_with_content_list(self):
        """Test message conversion with content as list."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        messages = [
            Message(role="user", content=[MessageTextContent(text="Hello world")]),
        ]

        prompt = provider._messages_to_prompt(messages)

        assert "User: Hello world" in prompt
        assert prompt.endswith("Assistant:")

    @pytest.mark.asyncio
    async def test_generate_message(self):
        """Test non-streaming message generation."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        # Mock responses
        login_response = {
            "success": True,
            "client_session_auth_key": "session-auth-key-123",
        }

        api_response = {
            "success": True,
            "job_id": "JID123",
            "job_result": {
                "success": True,
                "text": "This is a test response from AIME.",
                "num_generated_tokens": 10,
                "model_name": "Mistral-Small-3.1-24B-Instruct",
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
                # Return login response for login endpoint
                return MockResponse(login_response)

            def post(self, url, **kwargs):
                # Return API response for main endpoint
                return MockResponse(api_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test message")]

            result = await provider.generate_message(
                messages=messages,
                model="Mistral-Small-3.1-24B-Instruct",
                max_tokens=100,
            )

            assert result.role == "assistant"
            assert result.content == "This is a test response from AIME."

    @pytest.mark.asyncio
    async def test_generate_message_with_polling(self):
        """Test non-streaming message generation with progress polling."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        login_response = {
            "success": True,
            "client_session_auth_key": "session-auth-key-123",
        }

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

        get_call_count = 0

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, **kwargs):
                nonlocal get_call_count
                get_call_count += 1
                if "login" in url:
                    return MockResponse(login_response)
                elif "progress" in url:
                    return MockResponse(progress_response)
                return MockResponse(login_response)

            def post(self, url, **kwargs):
                return MockResponse(initial_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test")]

            result = await provider.generate_message(
                messages=messages,
                model="Mistral-Small-3.1-24B-Instruct",
            )

            assert result.content == "Response after polling."

    @pytest.mark.asyncio
    async def test_generate_messages_streaming(self):
        """Test streaming message generation with progress updates."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        login_response = {
            "success": True,
            "client_session_auth_key": "session-auth-key-123",
        }

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
                if "login" in url:
                    return MockResponse(login_response)
                elif "progress" in url:
                    resp = progress_responses[min(progress_index, len(progress_responses) - 1)]
                    progress_index += 1
                    return MockResponse(resp)
                return MockResponse(login_response)

            def post(self, url, **kwargs):
                return MockResponse(initial_response)

        with patch("aiohttp.ClientSession", MockSession), patch("asyncio.sleep", new_callable=AsyncMock):  # Skip sleep delays
            messages = [Message(role="user", content="Test")]

            chunks = []
            async for chunk in provider.generate_messages(
                messages=messages,
                model="Mistral-Small-3.1-24B-Instruct",
            ):
                chunks.append(chunk)

            # Should have received chunks
            assert len(chunks) > 0

            # Last chunk should be done
            assert chunks[-1].done is True

            # Reconstruct full text
            full_text = "".join(c.content for c in chunks)
            assert "Hello" in full_text

    @pytest.mark.asyncio
    async def test_get_available_language_models(self):
        """Test that AIME returns available language models."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        models = await provider.get_available_language_models()

        assert len(models) > 0
        model_ids = [m.id for m in models]
        assert "Mistral-Small-3.1-24B-Instruct" in model_ids

        # Verify provider is set correctly
        for model in models:
            assert model.provider.value == "aime"

    @pytest.mark.asyncio
    async def test_login_failure(self):
        """Test handling of login failures."""
        provider = AIMEProvider(secrets={"AIME_USER": "bad-user", "AIME_API_KEY": "bad-key"})

        login_error_response = {
            "success": False,
            "error": "Invalid credentials",
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
                return MockResponse(login_error_response)

            def post(self, url, **kwargs):
                return MockResponse(login_error_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test")]

            with pytest.raises(RuntimeError, match="AIME login failed"):
                await provider.generate_message(
                    messages=messages,
                    model="Mistral-Small-3.1-24B-Instruct",
                )

    @pytest.mark.asyncio
    async def test_api_error(self):
        """Test handling of API errors."""
        provider = AIMEProvider(secrets={"AIME_USER": "test-user", "AIME_API_KEY": "test-key"})

        login_response = {
            "success": True,
            "client_session_auth_key": "session-auth-key-123",
        }

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
                return MockResponse(login_response)

            def post(self, url, **kwargs):
                return MockResponse(api_error_response)

        with patch("aiohttp.ClientSession", MockSession):
            messages = [Message(role="user", content="Test")]

            with pytest.raises(RuntimeError, match="AIME API error"):
                await provider.generate_message(
                    messages=messages,
                    model="Mistral-Small-3.1-24B-Instruct",
                )

    def test_initialization_missing_user(self):
        """Test that initialization fails without AIME_USER."""
        with pytest.raises(AssertionError, match="AIME_USER is required"):
            AIMEProvider(secrets={"AIME_API_KEY": "test-key"})

    def test_initialization_missing_key(self):
        """Test that initialization fails without AIME_API_KEY."""
        with pytest.raises(AssertionError, match="AIME_API_KEY is required"):
            AIMEProvider(secrets={"AIME_USER": "test-user"})
