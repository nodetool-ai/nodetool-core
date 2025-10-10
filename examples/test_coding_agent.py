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
from nodetool.agents.tools.code_tools import ExecutePythonTool
from nodetool.agents.tools.http_tools import DownloadFileTool
from nodetool.providers.base import BaseProvider
from nodetool.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from pathlib import Path

import dotenv

# Load environment variables
dotenv.load_dotenv()


async def run_data_analysis_agent(
    provider: BaseProvider,
    model: str,
    analysis_type: str = "comprehensive",
    docker_image: str | None = None,
):
    context = ProcessingContext()

    code_tools = [
        ExecutePythonTool(),
        DownloadFileTool(),
    ]

    # Define analysis objectives based on type
    analysis_objectives = {
        "comprehensive": """
        Perform a comprehensive data science analysis of the Iris dataset.

        Your mission is to conduct a professional-grade analysis following these steps:

        1. **Data Acquisition**:
           - Download the Iris dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
           - Save it as 'iris.csv' in the workspace
           - Note: The file has NO headers. Columns are: sepal_length, sepal_width, petal_length, petal_width, species

        2. **Data Exploration & Cleaning**:
           - Load the data with proper column names
           - Check for missing values, duplicates, and data types
           - Generate basic statistics (mean, std, min, max, quartiles)
           - Create a data quality report

        3. **Exploratory Data Analysis (EDA)**:
           - Create distribution plots for each feature
           - Generate pair plots to show relationships between features
           - Create correlation heatmaps
           - Analyze class distributions and balance
           - Use box plots to identify outliers

        4. **Statistical Analysis**:
           - Perform ANOVA tests to check feature significance
           - Calculate feature importance using correlation analysis
           - Test for normality of distributions
           - Identify key distinguishing features between species

        5. **Advanced Visualizations**:
           - Create violin plots for feature distributions by species
           - Generate 3D scatter plots for multi-dimensional relationships
           - Design a dashboard-style figure combining multiple insights
           - Use appropriate color schemes for accessibility

        6. **Machine Learning Insights**:
           - Apply PCA to understand variance in the data
           - Create a simple classification model and evaluate performance
           - Visualize decision boundaries if possible
           - Generate a confusion matrix

        7. **Report Generation**:
           - Create a professional markdown report with:
             * Executive summary of findings
             * Methodology section
             * Key insights with supporting visualizations
             * Statistical findings with p-values
             * Recommendations for further analysis
             * All plots saved as high-quality PNG files
             * Code snippets for reproducibility

        Remember to:
        - Use professional plotting styles (seaborn style 'paper' or 'whitegrid')
        - Add proper titles, labels, and legends to all plots
        - Save all figures with descriptive names (e.g., 'feature_distributions.png')
        - Include interpretation for each visualization
        - Write clear, concise explanations suitable for both technical and non-technical audiences

        **Dynamic Subtask Addition**:
        You have access to the add_subtask tool. If you discover that certain analyses need
        deeper investigation (e.g., outlier analysis, specific feature relationships, or
        additional model types), you can create new subtasks dynamically for more focused work.
        """,
        "quick": """
        Perform a quick exploratory analysis of the Iris dataset.

        Steps:
        1. Download the Iris dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
        2. Save as 'iris.csv' (no headers - columns: sepal_length, sepal_width, petal_length, petal_width, species)
        3. Create basic visualizations (distributions, pair plot)
        4. Generate a brief markdown report with key findings
        """,
        "ml_focused": """
        Perform a machine learning focused analysis of the Iris dataset.

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
           - Implement multiple classifiers (LogisticRegression, RandomForest, SVM)
           - Perform cross-validation
           - Tune hyperparameters
           - Compare model performances

        4. **Model Evaluation**:
           - Generate confusion matrices
           - Calculate precision, recall, F1-scores
           - Create ROC curves
           - Visualize decision boundaries

        5. **Report**:
           - Document model performances
           - Provide recommendations
           - Include reproducible code
        """,
    }

    agent = Agent(
        name="Data Science Agent",
        objective=analysis_objectives.get(
            analysis_type, analysis_objectives["comprehensive"]
        ),
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,
        provider=provider,
        model=model,
        tools=code_tools,
        docker_image=docker_image,
        output_schema={
            "type": "string",
            "description": "A comprehensive markdown report with embedded visualizations and code snippets",
        },
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
            f.write(agent.results)
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
        print(f"   - Analysis Type: {analysis_type}")
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the data analysis agent example")
    parser.add_argument(
        "--analysis-type",
        choices=["comprehensive", "quick", "ml_focused"],
        default="comprehensive",
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Run the agent inside this Docker image (optional)",
    )

    args = parser.parse_args()

    print("🚀 Starting Data Analysis Agent")
    print(f"📊 Analysis Type: {args.analysis_type}")
    print("-" * 60)

    try:
        asyncio.run(
            run_data_analysis_agent(
                provider=get_provider(Provider.HuggingFaceCerebras),
                model="openai/gpt-oss-120b",
                analysis_type=args.analysis_type,
                docker_image=args.docker_image,
            )
        )
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

    # Alternative examples:

    # Quick analysis with OpenAI
    # asyncio.run(
    #     run_data_analysis_agent(
    #         provider=get_provider(Provider.OpenAI),
    #         model="gpt-4o-mini",
    #         planning_model="gpt-4o-mini",
    #         reasoning_model="gpt-4o-mini",
    #         analysis_type="quick"
    #     )
    # )

    # ML-focused analysis with Anthropic Claude
    # asyncio.run(
    #     run_data_analysis_agent(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #         planning_model="claude-3-5-sonnet-20241022",
    #         reasoning_model="claude-3-5-sonnet-20241022",
    #         analysis_type="ml_focused"
    #     )
    # )

    # Comprehensive analysis with Google Gemini
    # asyncio.run(
    #     run_data_analysis_agent(
    #         provider=get_provider(Provider.Gemini),
    #         model="gemini-2.0-flash",
    #         planning_model="gemini-2.0-flash",
    #         reasoning_model="gemini-2.0-flash",
    #         analysis_type="comprehensive"
    #     )
    # )
