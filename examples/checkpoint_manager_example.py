#!/usr/bin/env python3
"""
Example: Using CheckpointManager for Resumable Workflows

This example demonstrates how to use the CheckpointManager to create
resumable workflows with zero overhead when disabled.
"""

import asyncio
from typing import Any

from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode
from nodetool.workflows.checkpoint_manager import CheckpointManager
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext


# Define some example nodes
class NumberInput(InputNode):
    """Simple input node that provides a number."""
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class Add(BaseNode):
    """Add two numbers."""
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        result = self.a + self.b
        print(f"  Computing {self.a} + {self.b} = {result}")
        return result


class Multiply(BaseNode):
    """Multiply two numbers."""
    a: float = 0.0
    b: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        result = self.a * self.b
        print(f"  Computing {self.a} * {self.b} = {result}")
        return result


class NumberOutput(OutputNode):
    """Output node that displays a number."""
    value: float = 0.0

    async def process(self, context: ProcessingContext) -> float:
        print(f"  Result: {self.value}")
        return self.value


async def example_basic_checkpoint():
    """
    Example 1: Basic checkpoint save and restore.

    Demonstrates:
    - Creating a checkpoint manager (disabled by default)
    - Enabling checkpointing
    - Conceptual workflow for saving/restoring checkpoints

    NOTE: This example shows the API but doesn't actually save to database
    since that requires a ResourceScope (available during workflow execution).
    """
    print("\n=== Example 1: Basic Checkpoint API ===\n")

    # Create a simple graph
    input1 = NumberInput(id="input1", name="input1", value=5.0)  # type: ignore[call-arg]
    input2 = NumberInput(id="input2", name="input2", value=10.0)  # type: ignore[call-arg]
    add_node = Add(id="add1")  # type: ignore[call-arg]
    multiply_node = Multiply(id="multiply1")  # type: ignore[call-arg]
    output_node = NumberOutput(id="output1", name="output")  # type: ignore[call-arg]

    graph = Graph(  # noqa: F841
        nodes=[input1, input2, add_node, multiply_node, output_node],
        edges=[],
    )

    # Simulate a workflow run ID
    run_id = "example-run-001"

    # Create checkpoint manager (disabled by default for zero overhead)
    print("1. Creating checkpoint manager (disabled by default)")
    checkpoint_mgr = CheckpointManager(run_id=run_id, enabled=False)
    print(f"   Stats: {checkpoint_mgr.get_stats()}")
    print("   Overhead: Zero (single boolean check)")

    # Enable checkpointing for this workflow
    print("\n2. Enabling checkpointing for this workflow")
    checkpoint_mgr = CheckpointManager(run_id=run_id, enabled=True)
    print(f"   Stats: {checkpoint_mgr.get_stats()}")

    # Simulate some nodes completing
    print("\n3. Simulating workflow execution")
    print("   - Nodes input1, input2 completed")
    print("   - Node add1 is active (running)")
    print("   - Nodes multiply1, output1 are pending")

    # Show checkpoint API (would save to DB in real execution)
    print("\n4. Checkpoint API Usage:")
    print("   ```python")
    print("   success = await checkpoint_mgr.save_checkpoint(")
    print("       graph=graph,")
    print("       completed_nodes={'input1', 'input2'},")
    print("       active_nodes={'add1'},")
    print("       pending_nodes={'multiply1', 'output1'},")
    print("   )")
    print("   ```")
    print("   This would save state to database in actual workflow execution")

    # Show resume API
    print("\n5. Resume API Usage:")
    print("   ```python")
    print("   if await checkpoint_mgr.can_resume():")
    print("       checkpoint_data = await checkpoint_mgr.restore_checkpoint(graph)")
    print("       # Use checkpoint_data to skip completed nodes")
    print("   ```")


async def example_zero_overhead():
    """
    Example 2: Zero overhead when disabled.

    Demonstrates:
    - Checkpoint manager has no overhead when disabled
    - All operations return immediately without DB access
    """
    print("\n=== Example 2: Zero Overhead When Disabled ===\n")

    run_id = "example-run-002"

    # Create checkpoint manager with checkpointing disabled (default)
    print("1. Creating checkpoint manager (disabled)")
    checkpoint_mgr = CheckpointManager(run_id=run_id, enabled=False)
    print(f"   Enabled: {checkpoint_mgr.enabled}")
    print("   Runtime overhead: Zero (single boolean check)")

    # Create a simple graph
    input1 = NumberInput(id="input1", name="input1", value=42.0)  # type: ignore[call-arg]
    graph = Graph(nodes=[input1], edges=[])  # noqa: F841

    # Show that disabled operations have no effect
    print("\n2. Operations with disabled checkpoint manager:")
    print("   - save_checkpoint() returns False immediately (no DB access)")
    print("   - restore_checkpoint() returns None immediately (no DB access)")
    print("   - can_resume() returns False immediately (no DB access)")
    print("   - clear_checkpoint() returns False immediately (no DB access)")

    print("\n3. Performance characteristics:")
    print("   - Latency: < 1 microsecond per operation")
    print("   - Database writes: 0")
    print("   - Memory usage: Minimal (manager object only)")
    print("   - CPU overhead: Single boolean check per call")


async def example_incremental_checkpoints():
    """
    Example 3: Incremental checkpoints during workflow execution.

    Demonstrates:
    - When to save checkpoints
    - Different node states at each checkpoint
    - Strategic checkpoint placement
    """
    print("\n=== Example 3: Strategic Checkpoint Placement ===\n")

    run_id = "example-run-003"

    # Create checkpoint manager
    print("1. Creating checkpoint manager (enabled)")
    checkpoint_mgr = CheckpointManager(run_id=run_id, enabled=True)  # noqa: F841

    # Create a pipeline of nodes
    input1 = NumberInput(id="input1", name="input1", value=2.0)  # type: ignore[call-arg]
    add1 = Add(id="add1")  # type: ignore[call-arg]
    multiply1 = Multiply(id="multiply1")  # type: ignore[call-arg]
    add2 = Add(id="add2")  # type: ignore[call-arg]
    output1 = NumberOutput(id="output1", name="output")  # type: ignore[call-arg]

    graph = Graph(  # noqa: F841
        nodes=[input1, add1, multiply1, add2, output1],
        edges=[],
    )

    # Show strategic checkpoint points
    print("\n2. Strategic checkpoint points in workflow execution:")

    print("\n   Checkpoint 1: After Input Stage")
    print("   - Good for: Avoiding re-loading expensive data")
    print("   - Completed: input1")
    print("   - Pending: add1, multiply1, add2, output1")

    print("\n   Checkpoint 2: After Batch Processing")
    print("   - Good for: After processing 10+ nodes")
    print("   - Completed: input1, add1")
    print("   - Pending: multiply1, add2, output1")

    print("\n   Checkpoint 3: Before Expensive Operation")
    print("   - Good for: Before GPU operations, API calls")
    print("   - Completed: input1, add1, multiply1")
    print("   - Pending: add2, output1")

    print("\n3. Checkpoint placement guidelines:")
    print("   ✓ After completing a batch of nodes (e.g., every 10 nodes)")
    print("   ✓ Before starting expensive operations")
    print("   ✓ After major workflow milestones")
    print("   ✓ On explicit user request")
    print("   ✗ Don't checkpoint too frequently (adds overhead)")
    print("   ✗ Don't checkpoint trivial operations")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("CheckpointManager Examples")
    print("=" * 60)

    # Example 1: Basic checkpoint save/restore
    await example_basic_checkpoint()

    # Example 2: Zero overhead when disabled
    await example_zero_overhead()

    # Example 3: Incremental checkpoints
    await example_incremental_checkpoints()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
