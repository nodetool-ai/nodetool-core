import asyncio
import json
import sys
import os

from nodetool.agents.agent import Agent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import get_tool_by_name
# Import tools modules to ensure registration happens
import nodetool.agents.tools
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext


async def _run(cfg: dict) -> None:
    provider = get_provider(Provider[cfg["provider"]])
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
        input_files=cfg.get("input_files", []),
        system_prompt=cfg.get("system_prompt"),
        max_subtasks=cfg.get("max_subtasks", 10),
        max_steps=cfg.get("max_steps", 50),
        max_subtask_iterations=cfg.get("max_subtask_iterations", 5),
        max_token_limit=cfg.get("max_token_limit"),
        output_schema=cfg.get("output_schema"),
        output_type=cfg.get("output_type"),
        enable_analysis_phase=cfg.get("enable_analysis_phase", True),
        enable_data_contracts_phase=cfg.get("enable_data_contracts_phase", True),
        verbose=cfg.get("verbose", True),
    )

    context = ProcessingContext(workspace_dir=cfg["workspace_dir"])

    async for _ in agent.execute(context):
        pass

    results = agent.get_results()
    with open(cfg["result_path"], "w") as f:
        json.dump(results, f)


def main() -> None:
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
