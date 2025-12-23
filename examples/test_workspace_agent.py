#!/usr/bin/env python3
"""
Simple Workspace Test Agent

This script demonstrates basic workspace functionality by having an agent:
1. Create test files in the workspace
2. Write content to files
3. List workspace contents
4. Read files back

This is a minimal test to verify that the workspace feature works correctly.

Usage:
    python test_workspace_agent.py [--docker-image DOCKER_IMAGE]

By default the agent runs directly on your machine. Provide a Docker image via
``--docker-image`` to execute the agent inside a container.
"""

import asyncio
from pathlib import Path

import dotenv

from nodetool.agents.agent import Agent
from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.metadata.types import Provider
from nodetool.providers import get_provider
from nodetool.providers.base import BaseProvider
from nodetool.runtime.resources import ResourceScope
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

# Load environment variables
dotenv.load_dotenv()


async def run_workspace_test_agent(
    provider: BaseProvider,
    model: str,
    docker_image: str | None = None,
):
    context = ProcessingContext()

    code_tools = [
        ExecutePythonTool(),
    ]

    # Simple objective to test workspace functionality
    workspace_objective = """
        Your objective is to test the workspace feature:

        - Create a file called 'test.txt' with the content "Hello, Workspace!"
        - Create a file called 'data.json' with simple JSON: {"test": true, "count": 42}
        - Create a file called 'readme.md' with a brief markdown note

        Keep it simple - this is just a workspace functionality test.
        """

    agent = Agent(
        name="Workspace Test Agent",
        objective=workspace_objective,
        provider=provider,
        model=model,
        tools=code_tools,
        docker_image=docker_image,
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n\n" + "=" * 60)
    print("=== WORKSPACE TEST COMPLETE ===")
    print("=" * 60)

    # Verify workspace contents
    workspace_path = Path(context.workspace_dir)
    print(f"\n📂 Workspace directory: {workspace_path}")

    if workspace_path.exists():
        all_files = list(workspace_path.glob("*"))
        files = [f for f in all_files if f.is_file()]
        dirs = [d for d in all_files if d.is_dir()]

        print(f"\n📁 Files created: {len(files)}")
        for file in sorted(files):
            size = file.stat().st_size
            print(f"   ✓ {file.name} ({size:,} bytes)")

        if dirs:
            print(f"\n📁 Directories: {len(dirs)}")
            for dir in sorted(dirs):
                print(f"   📂 {dir.name}")

        # Check for expected files
        expected_files = ["test.txt", "data.json", "readme.md"]
        print("\n🔍 Verification:")
        for expected in expected_files:
            file_path = workspace_path / expected
            if file_path.exists():
                print(f"   ✓ {expected} exists")
                if expected == "test.txt":
                    content = file_path.read_text()
                    print(f"     Content: {content.strip()}")
            else:
                print(f"   ✗ {expected} NOT FOUND")

    else:
        print(f"\n⚠️  Workspace directory does not exist: {workspace_path}")


async def main(args):
    print("🚀 Starting Workspace Test Agent")
    print("-" * 60)

    async with ResourceScope():
        try:
            await run_workspace_test_agent(
                provider=await get_provider(Provider.HuggingFaceCerebras),
                model="openai/gpt-oss-120b",
                docker_image=args.docker_image,
            )
        except Exception as e:
            print(f"❌ Error during workspace test: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test workspace functionality with a simple agent")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Run the agent inside this Docker image (optional)",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
