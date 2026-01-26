"""
AIME provider implementation for chat completions.

This module implements the ChatProvider interface for AIME,
which provides access to AI models through the AIME Model API.

AIME API Documentation: https://www.aime.info/api
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Sequence

import aiohttp

if TYPE_CHECKING:
    from nodetool.workflows.processing_context import ProcessingContext

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    LanguageModel,
    Message,
    Provider,
    ToolCall,
)
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# AIME API version string
AIME_API_VERSION = "Python AIME API Client Interface 0.1.0"


@register_provider(Provider.AIME)
class AIMEProvider(BaseProvider):
    """AIME implementation of the ChatProvider interface.

    AIME provides access to AI models through the AIME Model API.
    This provider implements the AIME-specific authentication and request flow:
    1. Login with user/apiKey to get a session auth key
    2. Make API requests with the session auth key
    3. Poll for progress until the job is done

    Key features:
    1. Base URL: https://api.aime.info/api/
    2. Uses AIME_USER and AIME_API_KEY for authentication
    3. Supports progress polling for long-running requests
    """

    provider: Provider = Provider.AIME

    DEFAULT_API_URL = "https://api.aime.info/api/"
    DEFAULT_ENDPOINT = "llm_chat"
    DEFAULT_PROGRESS_INTERVAL = 0.3  # 300ms like the JS client

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["AIME_USER", "AIME_API_KEY"]

    def __init__(self, secrets: dict[str, str]):
        """Initialize the AIME provider with client credentials.

        Reads ``AIME_USER`` and ``AIME_API_KEY`` from secrets.
        """
        super().__init__(secrets=secrets)
        assert "AIME_USER" in secrets, "AIME_USER is required"
        assert "AIME_API_KEY" in secrets, "AIME_API_KEY is required"
        self.user = secrets["AIME_USER"]
        self.api_key = secrets["AIME_API_KEY"]
        self.api_url = self.DEFAULT_API_URL
        self.endpoint = self.DEFAULT_ENDPOINT
        self.client_session_auth_key: str | None = None
        self.cost = 0.0
        log.debug("AIMEProvider initialized. User present: True, API key present: True")

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        """Return environment variables required for containerized execution.

        Returns:
            A mapping containing ``AIME_USER`` and ``AIME_API_KEY`` if available.
        """
        env = {}
        if self.user:
            env["AIME_USER"] = self.user
        if self.api_key:
            env["AIME_API_KEY"] = self.api_key
        return env

    async def _fetch_async(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict[str, Any] | None = None,
        do_post: bool = True,
    ) -> dict[str, Any]:
        """Make an async HTTP request to the AIME API.

        Args:
            session: The aiohttp session to use
            url: The URL to request
            params: Optional parameters to send
            do_post: Whether to use POST (True) or GET (False)

        Returns:
            The JSON response as a dictionary
        """
        if do_post:
            headers = {"Content-type": "application/json; charset=UTF-8"}
            async with session.post(url, json=params, headers=headers) as response:
                return await response.json()
        else:
            async with session.get(url) as response:
                return await response.json()

    async def _login(self, session: aiohttp.ClientSession) -> str:
        """Login to the AIME API and get a session auth key.

        Args:
            session: The aiohttp session to use

        Returns:
            The client session auth key

        Raises:
            RuntimeError: If login fails
        """
        url = f"{self.api_url}{self.endpoint}/login?user={self.user}&key={self.api_key}&version={AIME_API_VERSION}"
        log.debug(f"AIME login URL: {url}")

        response = await self._fetch_async(session, url, do_post=False)

        if response.get("success"):
            auth_key = response.get("client_session_auth_key")
            log.debug("AIME login successful, got session auth key")
            return auth_key
        else:
            error_msg = response.get("error", "Unknown login error")
            ep_version = response.get("ep_version", "")
            if ep_version:
                error_msg += f" Endpoint version: {ep_version}"
            raise RuntimeError(f"AIME login failed: {error_msg}")

    async def _ensure_authenticated(self, session: aiohttp.ClientSession) -> None:
        """Ensure we have a valid session auth key.

        Args:
            session: The aiohttp session to use
        """
        if self.client_session_auth_key is None:
            self.client_session_auth_key = await self._login(session)

    async def _poll_progress(
        self,
        session: aiohttp.ClientSession,
        job_id: str,
        progress_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Poll for job progress until completion.

        Args:
            session: The aiohttp session to use
            job_id: The job ID to poll for
            progress_callback: Optional callback for progress updates

        Returns:
            The final job result
        """
        progress_url = f"{self.api_url}{self.endpoint}/progress?key={self.api_key}&job_id={job_id}"

        while True:
            response = await self._fetch_async(session, progress_url, do_post=False)

            if not response.get("success"):
                raise RuntimeError(f"AIME progress check failed: {response}")

            job_state = response.get("job_state")
            progress = response.get("progress", {})

            if progress_callback and progress:
                progress_info = {
                    "progress": progress.get("progress", 0),
                    "queue_position": progress.get("queue_position", -1),
                    "estimate": progress.get("estimate", -1),
                    "num_workers_online": progress.get("num_workers_online", -1),
                }
                progress_data = progress.get("progress_data")
                progress_callback(progress_info, progress_data)

            if job_state == "done":
                return response.get("job_result", {})
            elif job_state == "canceled":
                raise RuntimeError("AIME job was canceled")
            elif job_state == "failed":
                raise RuntimeError(f"AIME job failed: {response}")

            await asyncio.sleep(self.DEFAULT_PROGRESS_INTERVAL)

    def _messages_to_prompt(self, messages: Sequence[Message]) -> str:
        """Convert messages to a prompt string for AIME.

        Args:
            messages: The messages to convert

        Returns:
            A formatted prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.role.capitalize()
            content = ""

            if isinstance(msg.content, str):
                content = msg.content
            elif isinstance(msg.content, list):
                # Extract text from content list
                for item in msg.content:
                    if hasattr(item, "text"):
                        content += item.text
                    elif isinstance(item, dict) and "text" in item:
                        content += item["text"]
                    elif isinstance(item, str):
                        content += item

            if content:
                prompt_parts.append(f"{role}: {content}")

        # Add the assistant prefix for the response
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    async def generate_message(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> Message:
        """Generate a non-streaming completion from AIME.

        Args:
            messages: The message history
            model: The model to use
            tools: Optional tools to provide to the model (not currently supported by AIME)
            max_tokens: The maximum number of tokens to generate
            json_schema: Optional JSON schema for structured output
            temperature: Optional sampling temperature
            top_p: Optional nucleus sampling parameter
            top_k: Optional top-k sampling parameter
            **kwargs: Additional arguments to pass to the API

        Returns:
            A Message object containing the model's response
        """
        log.debug(f"AIME generating non-streaming message for model: {model}")
        log.debug(f"AIME non-streaming with {len(messages)} messages")

        if not messages:
            raise ValueError("messages must not be empty")

        prompt = self._messages_to_prompt(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "prompt_input": prompt,
            "wait_for_result": True,
            "key": self.api_key,
        }

        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = top_k
        if max_tokens:
            params["max_new_tokens"] = max_tokens

        self._log_api_request("chat", messages, **params)

        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for long requests

        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._ensure_authenticated(session)
            params["client_session_auth_key"] = self.client_session_auth_key

            url = f"{self.api_url}{self.endpoint}"
            log.debug(f"AIME API URL: {url}")

            response = await self._fetch_async(session, url, params, do_post=True)

            if response.get("success"):
                # If wait_for_result=True, the response should contain the result
                job_result = response.get("job_result", {})
                if not job_result and response.get("job_id"):
                    # Need to poll for result
                    job_result = await self._poll_progress(session, response["job_id"])

                text = job_result.get("text", "")
                log.debug(f"AIME response text length: {len(text)}")

                message = Message(
                    role="assistant",
                    content=text,
                )

                self._log_api_response("chat", message)
                return message
            else:
                error_msg = response.get("error", "Unknown error")
                raise RuntimeError(f"AIME API error: {error_msg}")

    async def generate_messages(  # type: ignore[override]
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 8192,
        json_schema: dict | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> AsyncIterator[Chunk | ToolCall]:
        """Stream assistant deltas from AIME using progress polling.

        AIME doesn't support true streaming, but provides progress updates
        with partial text during generation. This method polls for progress
        and yields chunks as they become available.

        Args:
            messages: Conversation history to send.
            model: Target model.
            tools: Optional tool definitions (not currently supported by AIME).
            max_tokens: Maximum tokens to generate.
            json_schema: Optional response schema.
            temperature: Optional sampling temperature.
            top_p: Optional nucleus sampling parameter.
            top_k: Optional top-k sampling parameter.
            **kwargs: Additional parameters.

        Yields:
            Text ``Chunk`` items as progress updates arrive.
        """
        log.debug(f"AIME starting streaming generation for model: {model}")
        log.debug(f"AIME streaming with {len(messages)} messages")

        if not messages:
            raise ValueError("messages must not be empty")

        prompt = self._messages_to_prompt(messages)

        # Build request parameters - wait_for_result=False for streaming
        params: dict[str, Any] = {
            "prompt_input": prompt,
            "wait_for_result": False,
            "key": self.api_key,
        }

        # Add optional parameters
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if top_k is not None:
            params["top_k"] = top_k
        if max_tokens:
            params["max_new_tokens"] = max_tokens

        self._log_api_request("chat_stream", messages, **params)

        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._ensure_authenticated(session)
            params["client_session_auth_key"] = self.client_session_auth_key

            url = f"{self.api_url}{self.endpoint}"
            log.debug(f"AIME API URL: {url}")

            response = await self._fetch_async(session, url, params, do_post=True)

            if not response.get("success"):
                error_msg = response.get("error", "Unknown error")
                raise RuntimeError(f"AIME API error: {error_msg}")

            job_id = response.get("job_id")
            if not job_id:
                raise RuntimeError("AIME API did not return a job_id")

            # Track the previous text to yield only new content
            previous_text = ""
            progress_url = f"{self.api_url}{self.endpoint}/progress?key={self.api_key}&job_id={job_id}"

            while True:
                progress_response = await self._fetch_async(session, progress_url, do_post=False)

                if not progress_response.get("success"):
                    raise RuntimeError(f"AIME progress check failed: {progress_response}")

                job_state = progress_response.get("job_state")
                progress = progress_response.get("progress", {})
                progress_data = progress.get("progress_data", {})

                # Get the current text from progress data
                current_text = progress_data.get("text", "")

                # Yield only the new content
                if len(current_text) > len(previous_text):
                    new_content = current_text[len(previous_text) :]
                    previous_text = current_text
                    yield Chunk(content=new_content, done=False)

                if job_state == "done":
                    # Get the final result
                    job_result = progress_response.get("job_result", {})
                    final_text = job_result.get("text", "")

                    # Yield any remaining content
                    if len(final_text) > len(previous_text):
                        remaining = final_text[len(previous_text) :]
                        yield Chunk(content=remaining, done=True)
                    else:
                        yield Chunk(content="", done=True)

                    self._log_api_response(
                        "chat_stream",
                        Message(role="assistant", content=final_text),
                    )
                    break
                elif job_state == "canceled":
                    raise RuntimeError("AIME job was canceled")
                elif job_state == "failed":
                    raise RuntimeError(f"AIME job failed: {progress_response}")

                await asyncio.sleep(self.DEFAULT_PROGRESS_INTERVAL)

    async def get_available_language_models(self) -> list[LanguageModel]:
        """Get available AIME models.

        Returns a list of known AIME models. AIME doesn't have a public
        models endpoint, so we return a static list of known models.

        Returns:
            List of LanguageModel instances for AIME
        """
        # AIME models - based on their documentation
        # These may need to be updated as AIME adds/removes models
        models = [
            LanguageModel(
                id="Mistral-Small-3.1-24B-Instruct",
                name="Mistral Small 3.1 24B Instruct",
                provider=Provider.AIME,
            ),
        ]

        log.debug(f"Returning {len(models)} AIME models")
        return models

    def has_tool_support(self, model: str) -> bool:
        """Return True if the given model supports tools/function calling.

        AIME currently does not support function calling.

        Args:
            model: Model identifier string.

        Returns:
            False, as AIME doesn't support tools.
        """
        return False
