import asyncio
import json
import sys
from uuid import uuid4

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.workflow_runner import WorkflowRunner


async def _run(cfg: dict) -> None:
    context = ProcessingContext(workspace_dir=cfg["workspace_dir"])
    req = RunJobRequest(**cfg["request"])
    runner = WorkflowRunner(job_id=uuid4().hex)

    async for _ in run_workflow(req, runner=runner, context=context):
        pass

    with open(cfg["result_path"], "w") as f:
        json.dump(runner.outputs, f)


def main() -> None:
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
