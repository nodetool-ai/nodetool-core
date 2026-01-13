import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncClient

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext


def get_openai_client() -> "AsyncClient":
    from openai import AsyncClient

    env = Environment.get_environment()
    api_key = env.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"
    return AsyncClient(api_key=api_key)


class OpenAIWebSearchTool(Tool):
    """
    🔍 OpenAI Web Search Tool - Searches the web using OpenAI's web search API

    This tool uses OpenAI's web search API to perform web searches and return structured results.
    Requires an OpenAI API key with web search access enabled.
    """

    name = "openai_web_search"
    description = "Search the web using OpenAI's web search API"

    def __init__(self):
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute",
                }
            },
            "required": ["query"],
        }

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        key = Environment.get("OPENAI_API_KEY")
        return {"OPENAI_API_KEY": key} if key else {}

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
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

        client = get_openai_client()
        completion = await client.chat.completions.create(
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

    def user_message(self, params: dict) -> str:
        query = params.get("query", "something")
        msg = f"Searching the web for '{query}' using OpenAI..."
        if len(msg) > 80:
            msg = "Searching the web using OpenAI..."
        return msg


class OpenAIImageGenerationTool(Tool):
    """
    🎨 OpenAI Image Generation Tool - Creates images from text prompts using DALL-E

    This tool uses OpenAI's DALL-E models to generate images based on textual descriptions.
    Requires an OpenAI API key.
    """

    name = "openai_image_generation"
    description = "Generate an image from a text prompt using OpenAI DALL-E"

    def __init__(self):
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A text description of the desired image(s).",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to save the generated image as png file.",
                },
            },
            "required": ["prompt", "output_file"],
        }

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
        """
        Generate an image using OpenAI's Image Generation API.

        Args:
            context: The processing context
            params: The image generation parameters including the prompt

        Returns:
            Dict containing the image generation result (e.g., image URL)
        """
        prompt = params.get("prompt")
        output_file = params.get("output_file")

        if not prompt:
            raise ValueError("Image generation prompt is required")

        if not output_file:
            raise ValueError("Output file is required")

        client = get_openai_client()
        response = await client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
        )

        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            # Safely access url and revised_prompt
            b64_image = getattr(image_data, "b64_json", None)
            if b64_image:
                file_path = context.resolve_workspace_path(output_file)
                with open(file_path, "wb") as f:
                    f.write(base64.b64decode(b64_image))
                formatted_results = {
                    "type": "image",
                    "prompt": prompt,
                    "output_file": output_file,
                    "status": "success",
                }
                return formatted_results
            else:
                raise ValueError("No image data received from OpenAI.")
        else:
            raise ValueError("No image data received from OpenAI.")

    def user_message(self, params: dict) -> str:
        prompt = params.get("prompt", "an image")
        msg = f"Generating {prompt} using OpenAI..."
        if len(msg) > 80:
            msg = "Generating an image using OpenAI..."
        return msg


class OpenAITextToSpeechTool(Tool):
    """
    🗣️ OpenAI Text-to-Speech Tool - Converts text into spoken audio using OpenAI TTS

    This tool uses OpenAI's TTS models to synthesize speech from text.
    Requires an OpenAI API key.
    """

    name = "openai_text_to_speech"
    description = "Convert text into spoken audio using OpenAI TTS"

    def __init__(self):
        # Define schema based on OpenAI API parameters for TTS
        # Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The text to synthesize speech from (max 4096 characters).",
                },
                "output_file": {
                    "type": "string",
                    "description": "The path to save the generated audio as mp3 file.",
                },
                "voice": {
                    "type": "string",
                    "description": "The voice to use (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer').",
                },
                "speed": {
                    "type": "number",
                    "description": "The speed of the speech (0.25 to 4.0).",
                    "default": 1.0,
                },
            },
            "required": ["input", "voice", "output_file"],
        }

    def get_container_env(self, context: ProcessingContext) -> dict[str, str]:
        key = Environment.get("OPENAI_API_KEY")
        return {"OPENAI_API_KEY": key} if key else {}

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
        """
        Generate audio from text using OpenAI's TTS API.

        Args:
            context: The processing context
            params: The TTS parameters including the input text, voice, and model.

        Returns:
            Dict containing the TTS result (e.g., base64 encoded audio data)
        """
        text_input = params.get("input")
        voice = params.get("voice", "alloy")
        model = params.get("model", "tts-1")
        response_format = "mp3"
        speed = params.get("speed", 1.0)
        output_file = params.get("output_file")

        if not text_input:
            raise ValueError("Input text is required for TTS.")
        if not output_file:
            raise ValueError("Output file is required for TTS.")
        if len(text_input) > 4096:
            raise ValueError("Input text exceeds maximum length of 4096 characters.")

        client = get_openai_client()
        response = await client.audio.speech.create(
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

        if output_file:
            file_path = context.resolve_workspace_path(output_file)
            with open(file_path, "wb") as f:
                f.write(base64.b64decode(b64_audio))

        formatted_results = {
            "type": "audio",
            "input_text": text_input,
            "voice": voice,
            "model": model,
            "format": response_format,
            "speed": speed,
            "output_file": output_file,
            "status": "success",
        }

        return formatted_results

    def user_message(self, params: dict) -> str:
        text = params.get("input", "some text")
        voice = params.get("voice", "a voice")
        msg = f"Converting text to speech with voice {voice}..."
        if len(text) < 30 and len(msg) + len(text) + 4 < 80:  # Add preview if short
            msg = f"Converting '{text}' to speech with voice {voice}..."
        elif len(msg) > 80:
            msg = "Converting text to speech..."
        return msg
