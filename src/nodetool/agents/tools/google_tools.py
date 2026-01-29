from typing import Any

import aiofiles

from nodetool.agents.tools.base import Tool
from nodetool.config.environment import Environment
from nodetool.metadata.types import Provider
from nodetool.providers.gemini_provider import GeminiProvider
from nodetool.workflows.processing_context import ProcessingContext


class GoogleGroundedSearchTool(Tool):
    """
    🔍 Google Grounded Search Tool - Searches the web using Gemini API's grounding capabilities

    This tool uses Google's Gemini API to perform web searches and return structured results
    with source information. Requires a Gemini API key.
    """

    name = "google_grounded_search"
    description = "Search the web using Google's Gemini API with grounding capabilities"

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
        key = Environment.get_environment().get("GEMINI_API_KEY")
        return {"GEMINI_API_KEY": key} if key else {}

    async def process(self, context: ProcessingContext, params: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a web search using Gemini API with grounding.

        Args:
            context: The processing context
            params: The search parameters including the query

        Returns:
            Dict containing the search results and sources
        """
        query = params.get("query")
        if not query:
            raise ValueError("Search query is required")

        from google.genai.types import GenerateContentConfig, GoogleSearch
        from google.genai.types import Tool as GenAITool

        # Configure Google Search as a tool
        google_search_tool = GenAITool(google_search=GoogleSearch())

        # Generate content with search grounding
        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]
        response = await client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )

        # Extract search results and source information
        results = []
        sources = []

        # Check if we have a valid response with candidates
        if not response or not response.candidates:
            raise ValueError("No response received from Gemini API")

        candidate = response.candidates[0]
        if not candidate or not candidate.content:
            raise ValueError("Invalid response format from Gemini API")

        # Get the main response text
        if candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    results.append(part.text)

        # Extract source information if available
        if candidate.grounding_metadata and candidate.grounding_metadata.grounding_chunks:
            # Extract sources from grounding chunks
            chunks = candidate.grounding_metadata.grounding_chunks
            for chunk in chunks:
                if hasattr(chunk, "web") and chunk.web:
                    source = {
                        "title": (chunk.web.title if hasattr(chunk.web, "title") else "Unknown Source"),
                        "url": chunk.web.uri if hasattr(chunk.web, "uri") else None,
                    }
                    if source not in sources and source["url"]:
                        sources.append(source)

        # Extract grounding supports if available
        grounding_supports = []
        if candidate.grounding_metadata and candidate.grounding_metadata.grounding_supports:
            supports = candidate.grounding_metadata.grounding_supports
            for support in supports:
                if support.segment:
                    support_info = {
                        "text": support.segment.text,
                        "start_index": support.segment.start_index,
                        "end_index": support.segment.end_index,
                        "chunk_indices": support.grounding_chunk_indices,
                        "confidence_scores": support.confidence_scores,
                    }
                    grounding_supports.append(support_info)

        # Format the results
        formatted_sources = []

        # Resolve redirect URLs
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                for source in sources:
                    url = source.get("url")
                    if url and "grounding-api-redirect" in url:
                        try:
                            async with session.head(url, allow_redirects=True, timeout=5) as resp:
                                source["url"] = str(resp.url)
                        except Exception:
                            # If resolution fails, keep original URL
                            pass
                    formatted_sources.append(source)
        except Exception:
            # If session creation fails, keep original sources
            formatted_sources = sources

        formatted_results = {
            "query": query,
            "results": results,
            "sources": formatted_sources,
            "grounding_supports": grounding_supports,
            "status": "success",
        }

        return formatted_results

    def user_message(self, params: dict) -> str:
        query = params.get("query", "something")
        msg = f"Searching Google (grounded) for '{query}'..."
        if len(msg) > 80:
            msg = "Searching Google (grounded)..."
        return msg


class GoogleImageGenerationTool(Tool):
    """
    🎨 Google Image Generation Tool - Generates images using Gemini API

    This tool uses Google's Gemini API (gemini-2.0-flash-exp-image-generation)
    to generate images based on a text prompt.

    Returns a base64 encoded image data.
    """

    name = "google_image_generation"
    description = "Generate images based on a text prompt using Google's Gemini API"

    def __init__(self):
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt describing the image to generate",
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
        Generate an image using the Gemini API based on the provided prompt.

        Args:
            context: The processing context
            params: The parameters including the text prompt

        Returns:
            Dict containing generated text and base64 encoded image data
        """
        prompt = params.get("prompt")
        output_file = params.get("output_file")
        if not prompt:
            raise ValueError("Image generation prompt is required")
        if not output_file:
            raise ValueError("Output file is required")

        from google.genai.types import GenerateImagesConfig

        provider = await context.get_provider(Provider.Gemini)
        assert isinstance(provider, GeminiProvider)
        client = provider.get_client()  # pyright: ignore[reportAttributeAccessIssue]
        response = await client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=GenerateImagesConfig(
                number_of_images=1,
            ),
        )
        assert response.generated_images, "No images generated"

        image_bytes = None
        for generated_image in response.generated_images:
            assert generated_image.image, "No image"
            assert generated_image.image.image_bytes, "No image bytes"
            image_bytes = generated_image.image.image_bytes
            break

        if image_bytes is None:
            raise ValueError("No image bytes found in response")

        file_path = context.resolve_workspace_path(output_file)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(image_bytes)

        formatted_results = {
            "type": "image",
            "prompt": prompt,
            "output_file": output_file,
            "status": "success",
        }
        return formatted_results

    def user_message(self, params: dict) -> str:
        prompt = params.get("prompt", "an image")
        msg = f"Generating image '{prompt}' using Google..."
        if len(msg) > 80:
            msg = "Generating an image using Google..."
        return msg


if __name__ == "__main__":
    import asyncio

    from nodetool.workflows.processing_context import ProcessingContext

    async def main():
        # Workspace dir must be explicitly provided (user-defined workspaces only)
        context = ProcessingContext()
        print(f"Using workspace directory: {context.workspace_dir}")

        # --- Test Google Grounded Search Tool ---
        print("\n--- Testing Google Grounded Search Tool ---")
        search_tool = GoogleGroundedSearchTool()
        search_params = {"query": "What are the latest advancements in AI?"}
        search_result = await search_tool.process(context, search_params)
        print("Search Result:")
        # Pretty print the dictionary
        import json

        print(json.dumps(search_result, indent=2))

        # --- Test Google Image Generation Tool ---
        print("\n--- Testing Google Image Generation Tool ---")
        image_tool = GoogleImageGenerationTool()
        # Output path relative to the workspace directory
        output_image_path = "workspace/generated_test_image.png"
        image_params = {
            "prompt": "Abase64istic cityscape at sunset, digital art",
            "output_file": output_image_path,
        }
        image_result = await image_tool.process(context, image_params)
        print("Image Generation Result:")
        import json

        print(json.dumps(image_result, indent=2))
        if image_result.get("status") == "success":
            # Resolve the path using context to get the absolute path
            abs_path = context.resolve_workspace_path(output_image_path)
            print(f"Image saved to: {abs_path}")

    # Run the async main function
    asyncio.run(main())
