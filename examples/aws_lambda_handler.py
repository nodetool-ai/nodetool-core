#!/usr/bin/env python3
"""AWS Lambda handler example for running a NodeTool workflow.

This script defines a minimal `lambda_handler` function that executes a simple
ChatCompletion workflow using NodeTool Core. You can package this file with its
dependencies and deploy it to AWS Lambda.
"""

import asyncio
import json
from typing import Any, Dict

from nodetool.dsl.graph import graph, run_graph
from nodetool.dsl.providers.openai import ChatCompletion
from nodetool.metadata.types import OpenAIModel


async def run_workflow(prompt: str) -> str:
    """Execute the workflow using the provided prompt."""
    workflow = ChatCompletion(
        model=OpenAIModel(model="gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
    )
    return await run_graph(graph(workflow))


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Entry point for AWS Lambda."""
    prompt = event.get("prompt", "Hello from NodeTool!")
    result = asyncio.run(run_workflow(prompt))
    return {"statusCode": 200, "body": json.dumps({"result": result})}


if __name__ == "__main__":
    # Simple local test
    print(lambda_handler({"prompt": "Test"}, None))
