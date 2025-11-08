#!/usr/bin/env python3
"""
Advanced Code Analysis and Development Agent

This script demonstrates the use of an intelligent coding agent that can:
1. Perform comprehensive data analysis with visualization
2. Generate production-quality code with best practices
3. Create detailed documentation and reports
4. Execute and validate code in a controlled environment

The agent is capable of:
- Exploratory Data Analysis (EDA)
- Statistical analysis and visualization
- Machine learning model development
- Code optimization and refactoring
- Automated testing and validation

**NEW: DYNAMIC SUBTASK SUPPORT**
The agent can now dynamically add subtasks during execution! If the agent discovers
that additional analysis steps are needed (e.g., deeper statistical tests, additional
visualizations, model comparisons), it can use the add_subtask tool to create new
tasks on-the-fly. This enables more adaptive and thorough analysis.

Usage:
    python test_coding_agent.py [--docker-image DOCKER_IMAGE]

By default the agent runs directly on your machine.  Provide a Docker image via
``--docker-image`` to execute the agent inside a container.

The agent will download data, perform analysis, create visualizations, and
generate a comprehensive report with all findings.
"""

import asyncio
from nodetool.agents.agent import Agent
from nodetool.agents.tools.code_tools import ExecuteDatascienceTool
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.providers.base import BaseProvider
from nodetool.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.runtime.resources import ResourceScope
from pathlib import Path

import dotenv

# Load environment variables
dotenv.load_dotenv()


async def run_coding_agent(
    provider: BaseProvider,
    model: str,
):
    context = ProcessingContext()

    code_tools = [
        ExecuteDatascienceTool(),
        DownloadFileTool(),
    ]

    # Define analysis objectives based on type
    analysis_objective = """
        Your objective is to build and evaluate classification models:

        1. **Data Preparation**:
           - Download Iris data from: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
           - Save as 'iris.csv' (columns: sepal_length, sepal_width, petal_length, petal_width, species)
           - Split data into training and testing sets

        2. **Feature Engineering**:
           - Analyze feature correlations
           - Create polynomial features if beneficial
           - Normalize/standardize features

        3. **Model Development**:
           - Implement a logistic regression classifier
           - Perform cross-validation

        4. **Model Evaluation**:
           - Generate a confusion matrix
           - Calculate precision, recall, F1-score
           - Create an ROC curve
           - Visualize decision boundaries

        5. **Report**:
           - Generate a markdown report with key findings
           - Document each step
           - Reference the plots as images with correct file names
        """

    agent = Agent(
        name="Coding Agent",
        objective=analysis_objective,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        provider=provider,
        model=model,
        tools=code_tools,
        # display_manager=AgentConsole(),
    )

    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            print(item.content, end="", flush=True)

    print("\n\n" + "=" * 60)
    print("=== DATA ANALYSIS COMPLETE ===")
    print("=" * 60)

    # Save the markdown report
    if agent.results:
        report_path = Path(context.workspace_dir) / "analysis_report.md"
        with open(report_path, "w") as f:
            # Handle both string and dict results
            if isinstance(agent.results, str):
                f.write(agent.results)
            else:
                # If results is a dict or other type, convert to JSON string
                import json

                f.write(json.dumps(agent.results, indent=2))
        print(f"\n✓ Analysis report saved to: {report_path}")

        # List all generated files
        workspace_path = Path(context.workspace_dir)
        generated_files = list(workspace_path.glob("*"))

        if generated_files:
            print("\n📁 Generated Files:")
            for file in sorted(generated_files):
                if file.is_file():
                    size = file.stat().st_size
                    print(f"   - {file.name} ({size:,} bytes)")

        # Display summary statistics if available
        print("\n📊 Analysis Summary:")
        print(f"   - Model Used: {model}")
        print(
            f"   - Total Files Generated: {len([f for f in generated_files if f.is_file()])}"
        )

        # Check for specific output files
        visualization_files = list(workspace_path.glob("*.png")) + list(
            workspace_path.glob("*.jpg")
        )

        if visualization_files:
            print(f"   - Visualizations Created: {len(visualization_files)}")
            for viz in visualization_files[:5]:  # Show first 5
                print(f"     • {viz.name}")
            if len(visualization_files) > 5:
                print(f"     • ... and {len(visualization_files) - 5} more")

    print(f"\n📂 Full workspace path: {context.workspace_dir}")


async def main():
    print("🚀 Starting Data Analysis Agent")
    print("-" * 60)

    async with ResourceScope():
        try:
            await run_coding_agent(
                provider=await get_provider(Provider.HuggingFaceCerebras),
                model="openai/gpt-oss-120b",
            )
        except Exception as e:
            print(f"❌ Error during analysis: {e}")



if __name__ == "__main__":
    asyncio.run(main())
