import asyncio
import json
import sys
from pathlib import Path

from nodetool.agents.agent import Agent
from nodetool.agents.tools import get_tool_by_name
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.workflows.processing_context import ProcessingContext


async def _run(cfg: dict) -> None:
    provider = await get_provider(Provider[cfg["provider"]])
    tools = []
    for name in cfg.get("tools", []):
        tool_cls = get_tool_by_name(name)
        if not tool_cls:
            raise ValueError(f"Unknown tool: {name}")
        tools.append(tool_cls())

    agent = Agent(
        name=cfg["name"],
        objective=cfg["objective"],
        provider=provider,
        model=cfg["model"],
        planning_model=cfg.get("planning_model"),
        reasoning_model=cfg.get("reasoning_model"),
        tools=tools,
        description=cfg.get("description", ""),
        inputs=cfg.get("inputs", {}),
        system_prompt=cfg.get("system_prompt"),
        max_steps=cfg.get("max_steps", 10),
        max_step_iterations=cfg.get("max_step_iterations", 5),
        max_token_limit=cfg.get("max_token_limit"),
        output_schema=cfg.get("output_schema"),
        verbose=cfg.get("verbose", True),
    )

    context = ProcessingContext(workspace_dir=cfg["workspace_dir"])

    async for _ in agent.execute(context):
        pass

    results = agent.get_results()
    with Path(cfg["result_path"]).open("w") as f:
        json.dump(results, f)


def main() -> None:
    config_path = sys.argv[1]
    with Path(config_path).open() as f:
        cfg = json.load(f)
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
