from typing import Dict, Any
import openai
from nodetool.chat.tools import Tool
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext


class OpenAIWebSearchTool(Tool):
    """
    🔍 OpenAI Web Search Tool - Searches the web using OpenAI's web search API

    This tool uses OpenAI's web search API to perform web searches and return structured results.
    Requires an OpenAI API key with web search access enabled.
    """

    name = "openai_web_search"
    description = "Search the web using OpenAI's web search API"

    def __init__(self, workspace_dir: str):
        super().__init__(workspace_dir)
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
