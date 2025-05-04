from typing import Dict, Any
import openai
from nodetool.agents.tools.base import Tool
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
import base64


class OpenAIWebSearchTool(Tool):
    """
    🔍 OpenAI Web Search Tool - Searches the web using OpenAI's web search API

    This tool uses OpenAI's web search API to perform web searches and return structured results.
    Requires an OpenAI API key with web search access enabled.
    """

    name = "openai_web_search"
    description = "Search the web using OpenAI's web search API"

    def __init__(self):
        self.client = openai.AsyncClient(api_key=Environment.get("OPENAI_API_KEY"))
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                }
            },
            "required": ["query"],
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a web search using OpenAI's API.

        Args:
            context: The processing context
            params: The search parameters including the query

        Returns:
            Dict containing the search results
        """
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")

        completion = await self.client.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={},
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
        )

        # Format the results
        formatted_results = {
            "query": query,
            "results": completion.choices[0].message.content,
            "status": "success",
        }

        return formatted_results


class OpenAIImageGenerationTool(Tool):
    """
    🎨 OpenAI Image Generation Tool - Creates images from text prompts using DALL-E

    This tool uses OpenAI's DALL-E models to generate images based on textual descriptions.
    Requires an OpenAI API key.
    """

    name = "openai_image_generation"
    description = "Generate an image from a text prompt using OpenAI DALL-E"

    def __init__(self):
        self.client = openai.AsyncClient(api_key=Environment.get("OPENAI_API_KEY"))
        # Define schema based on OpenAI API parameters for dall-e-3
        # Reference: https://platform.openai.com/docs/api-reference/images/create
        self.input_schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A text description of the desired image(s).",
                },
            },
            "required": ["prompt"],
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an image using OpenAI's DALL-E API.

        Args:
            context: The processing context
            params: The image generation parameters including the prompt

        Returns:
            Dict containing the image generation result (e.g., image URL)
        """
        prompt = params.get("prompt")
        if not prompt:
            raise ValueError("Image generation prompt is required")

        response = await self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
        )

        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            # Safely access url and revised_prompt
            b64_image = getattr(image_data, "b64_json", None)

            if b64_image:
                formatted_results = {
                    "prompt": prompt,
                    "image": b64_image,
                    "status": "success",
                }
                return formatted_results
            else:
                raise ValueError("No image data received from OpenAI.")
        else:
            raise ValueError("No image data received from OpenAI.")


class OpenAITextToSpeechTool(Tool):
    """
    🗣️ OpenAI Text-to-Speech Tool - Converts text into spoken audio using OpenAI TTS

    This tool uses OpenAI's TTS models to synthesize speech from text.
    Requires an OpenAI API key.
    """

    name = "openai_text_to_speech"
    description = "Convert text into spoken audio using OpenAI TTS"

    def __init__(self):
        self.client = openai.AsyncClient(api_key=Environment.get("OPENAI_API_KEY"))
        # Define schema based on OpenAI API parameters for TTS
        # Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
        self.input_schema = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text to synthesize speech from (max 4096 characters).",
                },
                "model": {
                    "type": "string",
                    "description": "The TTS model to use (e.g., 'tts-1', 'tts-1-hd').",
                    "default": "tts-1",
                },
                "voice": {
                    "type": "string",
                    "description": "The voice to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer').",
                },
                "response_format": {
                    "type": "string",
                    "description": "The format of the audio output (e.g., 'mp3', 'opus', 'aac', 'flac').",
                    "default": "mp3",
                },
                "speed": {
                    "type": "number",
                    "description": "The speed of the speech (0.25 to 4.0).",
                    "default": 1.0,
                },
            },
            "required": ["input", "voice"],
        }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate audio from text using OpenAI's TTS API.

        Args:
            context: The processing context
            params: The TTS parameters including the input text, voice, and model.

        Returns:
            Dict containing the TTS result (e.g., base64 encoded audio data)
        """
        text_input = params.get("input")
        voice = params.get("voice")
        model = params.get("model", "tts-1")
        response_format = params.get("response_format", "mp3")
        speed = params.get("speed", 1.0)

        if not text_input:
            raise ValueError("Input text is required for TTS.")
        if not voice:
            raise ValueError("Voice selection is required for TTS.")
        if len(text_input) > 4096:
            raise ValueError("Input text exceeds maximum length of 4096 characters.")

        response = await self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text_input,
            response_format=response_format,
            speed=speed,
        )

        # The API response streams the audio content directly.
        # We read the content and encode it in base64 for easier handling in JSON.
        audio_content = response.content
        b64_audio = base64.b64encode(audio_content).decode("utf-8")

        formatted_results = {
            "input_text": text_input,
            "voice": voice,
            "model": model,
            "format": response_format,
            "speed": speed,
            "audio": b64_audio,
            "status": "success",
        }

        return formatted_results
