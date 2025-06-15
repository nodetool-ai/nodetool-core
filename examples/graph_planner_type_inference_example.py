#!/usr/bin/env python3
"""
Example demonstrating automatic type inference in GraphPlanner.

This example shows how the GraphPlanner can automatically infer input schema
from provided input values, eliminating the need to manually define GraphInput
objects for common data types.
"""

import asyncio
from nodetool.agents.graph_planner import GraphPlanner, GraphOutput
from nodetool.chat.providers import OpenAIProvider
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.workflows.processing_context import ProcessingContext


async def main():
    """Demonstrate automatic type inference in GraphPlanner."""

    print("GraphPlanner Type Inference Example")
    print("=" * 40)

    # Initialize provider (you'll need to set OPENAI_API_KEY)
    provider = OpenAIProvider()

    # Example 1: Basic type inference
    print("\n1. Basic Type Inference Example")
    print("-" * 30)

    inputs = {
        "name": "Mike",
        "age": 22,
        "score": 95.5,
        "is_active": True,
        "tags": ["python", "ai", "development"],
        "preferences": {"theme": "dark", "language": "en"},
    }

    print("Input values:")
    for key, value in inputs.items():
        print(f"  {key}: {value} ({type(value).__name__})")

    # Create GraphPlanner with automatic type inference
    planner = GraphPlanner(
        provider=provider,
        model="gpt-4o-mini",
        objective="Generate a personalized report based on user data",
        inputs=inputs,  # Types will be inferred automatically
        output_schema=[
            GraphOutput(
                name="report",
                type=TypeMetadata(type="str"),
                description="Generated personalized report",
            )
        ],
        verbose=False,
    )

    print("\nInferred input schema:")
    for inp in planner.input_schema:
        print(f"  {inp.name}: {inp.type} - {inp.description}")

    # Example 2: Special types and asset references
    print("\n2. Special Types Example")
    print("-" * 25)

    special_inputs = {
        "optional_field": None,
        "image_asset": {"type": "image", "uri": "path/to/image.jpg"},
        "coordinates": (42.3601, -71.0589),  # Boston coordinates
        "empty_list": [],
        "nested_data": {
            "users": [{"name": "Alice", "id": 1}, {"name": "Bob", "id": 2}]
        },
    }

    print("Special input values:")
    for key, value in special_inputs.items():
        print(f"  {key}: {value}")

    special_planner = GraphPlanner(
        provider=provider,
        model="gpt-4o-mini",
        objective="Process special data types",
        inputs=special_inputs,
        verbose=False,
    )

    print("\nInferred types for special inputs:")
    for inp in special_planner.input_schema:
        print(f"  {inp.name}: {inp.type}")

    # Example 3: Manual schema takes precedence
    print("\n3. Manual Schema Precedence Example")
    print("-" * 35)

    from nodetool.agents.graph_planner import GraphInput

    manual_inputs = {"number": 42}
    manual_schema = [
        GraphInput(
            name="number",
            type=TypeMetadata(type="str"),  # Override inferred int type
            description="Number as string",
        )
    ]

    manual_planner = GraphPlanner(
        provider=provider,
        model="gpt-4o-mini",
        objective="Demonstrate manual override",
        inputs=manual_inputs,
        input_schema=manual_schema,  # Manual schema takes precedence
        verbose=False,
    )

    print("Input value: number = 42 (int)")
    print("Manual schema overrides to: str")
    print(f"Final schema: {manual_planner.input_schema[0].type}")

    print("\n" + "=" * 40)
    print("Type inference demonstration complete!")
    print("\nKey benefits:")
    print("- Automatic type detection from values")
    print("- Support for complex nested types")
    print("- Asset reference recognition")
    print("- Manual schema override capability")


if __name__ == "__main__":
    asyncio.run(main())
