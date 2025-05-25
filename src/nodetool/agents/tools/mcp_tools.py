from typing import Any, Dict, Optional
import httpx

from nodetool.agents.tools.base import Tool
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.common.environment import Environment


class MCPTool(Tool):
    """Generic tool for calling a Model Context Protocol service."""

    def __init__(
        self,
        tool: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.tool = tool
        self.name = name or tool
        self.description = description or f"Execute the MCP tool '{tool}'"
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
        self.base_url = base_url or Environment.get(
            "MCP_API_URL", "http://localhost:8000"
        )
        self.token = token or Environment.get("MCP_TOKEN")

    def get_container_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if self.base_url:
            env["MCP_API_URL"] = self.base_url
        if self.token:
            env["MCP_TOKEN"] = self.token
        return env

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url.rstrip('/')}/tools/{self.tool}"
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = await context.http_post(url, json=params, headers=headers)
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}
