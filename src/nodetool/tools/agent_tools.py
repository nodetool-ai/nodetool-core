"""Agent execution tools.

These tools provide functionality for running autonomous AI agents with various tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Context

from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.agents.tools.email_tools import SearchEmailTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.runtime.resources import ResourceScope, maybe_scope, require_scope
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.types import (
    Chunk,
    LogUpdate,
    PlanningUpdate,
    TaskUpdate,
)


class AgentTools:
    """Agent execution tools."""

    @staticmethod
    async def run_agent(
        objective: str,
        provider: str,
        model: str = "gpt-4o",
        tools: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Execute a NodeTool agent to perform autonomous task execution.

        Args:
            objective: The task description for agent to accomplish
            provider: AI provider ("openai", "anthropic", "ollama", "gemini", etc.)
            model: Model to use (default: "gpt-4o")
            tools: List of tool names to enable ("google_search", "browser", "email")
            output_schema: Optional JSON schema to structure agent's output
            ctx: FastMCP context for progress reporting

        Returns:
            Dictionary with status, results, events, and workspace directory
        """
        tools = tools or []
        context = ProcessingContext(asset_output_mode=AssetOutputMode.TEMP_URL)

        async with ResourceScope():
            try:
                tool_instances = []
                tool_map = {
                    "google_search": GoogleSearchTool,
                    "browser": BrowserTool,
                    "email": SearchEmailTool,
                }

                for tool_name in tools:
                    if tool_name in tool_map:
                        tool_instances.append(tool_map[tool_name]())
                    else:
                        pass

                provider_enum = Provider(provider)
                provider_instance = await get_provider(provider_enum)

                agent = Agent(
                    name=f"Agent: {objective[:50]}...",
                    objective=objective,
                    provider=provider_instance,
                    model=model,
                    tools=tool_instances,
                    output_schema=output_schema,
                )

                output_chunks = []
                events = []
                async for event in agent.execute(context):
                    if isinstance(event, Chunk):
                        output_chunks.append(event.content)

                    elif isinstance(event, PlanningUpdate):
                        if ctx:
                            from fastmcp import Context

                            await ctx.info(f"Plan: {event.phase} - {event.content}")
                        events.append(event.model_dump())

                    elif isinstance(event, TaskUpdate):
                        if ctx:
                            task_title = event.task.title if event.task else "Task"
                            await ctx.info(f"Task: {event.event} - {task_title}")
                        events.append(event.model_dump())

                    elif isinstance(event, LogUpdate):
                        if ctx:
                            await ctx.info(f"Log: {event.content}")
                        events.append(event.model_dump())

                    else:
                        if hasattr(event, "model_dump"):
                            events.append(event.model_dump())

                results = agent.get_results()

                return {
                    "status": "success",
                    "results": results,
                    "events": events,
                    "workspace_dir": str(context.workspace_dir),
                }

            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e),
                }

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all agent tool functions."""
        return {
            "run_agent": AgentTools.run_agent,
        }
