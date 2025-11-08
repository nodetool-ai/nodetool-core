#!/usr/bin/env python3
"""
Simple GraphPlanner Evaluation Test

Tests the evaluation system with a simpler objective that's more likely to succeed.
"""

import asyncio
from nodetool.agents.graph_planner import GraphPlanner
from nodetool.agents.graph_planner_evaluator import GraphPlannerEvaluator
from nodetool.providers.anthropic_provider import AnthropicProvider
from nodetool.api.types.workflow import GraphInput, GraphOutput
from nodetool.metadata.types import TypeMetadata
from nodetool.runtime.resources import ResourceScope


async def test_simple_evaluation():
    """Test with a simpler, more likely to succeed objective"""

    print("🧪 Testing GraphPlanner Evaluation with Simple Objective")
    print("=" * 60)

    # Create evaluator
    evaluator = GraphPlannerEvaluator(
        results_dir="simple_test_results", enable_logging=True
    )

    # Override the test case with a simpler one
    from nodetool.agents.graph_planner_evaluator import TestCase

    simple_test = TestCase(
        name="simple_math",
        objective="Take two numbers as input and add them together to produce a sum",
        input_schema=[
            GraphInput(
                name="number_a",
                type=TypeMetadata(type="float"),
                description="First number to add",
            ),
            GraphInput(
                name="number_b",
                type=TypeMetadata(type="float"),
                description="Second number to add",
            ),
        ],
        output_schema=[
            GraphOutput(
                name="sum_result",
                type=TypeMetadata(type="float"),
                description="The sum of the two input numbers",
            )
        ],
        expected_nodes=[
            "nodetool.input.FloatInput",
            "nodetool.math.Add",
            "nodetool.output.FloatOutput",
        ],
        expected_connections=[
            ("number_a", "add_node"),
            ("number_b", "add_node"),
            ("add_node", "sum_result"),
        ],
        expected_properties={"operation": "add"},
        complexity_score=2,
        description="Simple addition workflow for testing",
    )

    # Add the simple test case
    evaluator.test_cases["simple_math"] = simple_test

    # Create planner with simple objective
    planner = GraphPlanner(
        provider=AnthropicProvider(),
        model="claude-3-5-sonnet-20241022",
        objective=simple_test.objective,
        input_schema=simple_test.input_schema,
        output_schema=simple_test.output_schema,
        verbose=True,
    )

    try:
        # Run evaluation
        result = await evaluator.evaluate_graph_planner(
            planner=planner, test_case_name="simple_math", prompt_version="simple_test"
        )

        print("\n✅ Evaluation completed successfully!")
        print(f"📊 Overall Score: {result.overall_score:.2%}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"🔧 Total Metrics: {len(result.metrics)}")

        # Show metric breakdown
        print("\n📈 Metric Results:")
        for metric in result.metrics:
            status = "✅" if metric.passed else "❌"
            score_pct = metric.score * 100
            print(f"  {status} {metric.name}: {score_pct:.1f}%")

        if result.recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"  {i}. {rec}")

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


async def main() -> bool:
    async with ResourceScope():
        return await test_simple_evaluation()


if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n😞 Test failed")
