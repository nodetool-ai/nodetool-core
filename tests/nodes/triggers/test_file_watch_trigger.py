"""
Tests for the FileWatchTrigger node.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from nodetool.nodes.triggers.file_watch import FileWatchTrigger
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_file_watch_trigger_detects_created_file():
    """Test that file watch trigger detects created files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trigger = FileWatchTrigger(
            id="test-trigger",
            path=tmpdir,
            patterns=["*.txt"],
            events=["created"],
            debounce_seconds=0.1,
            max_events=1,
        )
        context = ProcessingContext()

        await trigger.initialize(context)

        events = []

        async def collect_events():
            async for result in trigger.gen_process(context):
                events.append(result["event"])

        async def create_file():
            # Wait for observer to start
            await asyncio.sleep(0.3)
            # Create a file
            Path(tmpdir, "test.txt").write_text("hello")
            # Give time for debounce
            await asyncio.sleep(0.2)

        # Run with timeout to prevent hanging
        try:
            await asyncio.wait_for(
                asyncio.gather(create_file(), collect_events()),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            trigger.stop()
            pytest.fail("File watch trigger timed out")

        await trigger.finalize(context)

        assert len(events) == 1
        assert events[0]["event_type"] == "created"
        assert "test.txt" in events[0]["data"]["path"]


@pytest.mark.asyncio
async def test_file_watch_trigger_pattern_filter():
    """Test that file watch trigger respects pattern filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trigger = FileWatchTrigger(
            id="test-trigger",
            path=tmpdir,
            patterns=["*.txt"],  # Only watch .txt files
            events=["created"],
            debounce_seconds=0.05,
            max_events=1,
        )
        context = ProcessingContext()

        await trigger.initialize(context)

        events = []

        async def collect_events():
            async for result in trigger.gen_process(context):
                events.append(result["event"])

        async def create_files():
            await asyncio.sleep(0.2)
            # Create a .json file (should be ignored)
            Path(tmpdir, "ignore.json").write_text("{}")
            await asyncio.sleep(0.1)
            # Create a .txt file (should be captured)
            Path(tmpdir, "capture.txt").write_text("hello")

        await asyncio.gather(create_files(), collect_events())
        await trigger.finalize(context)

        # Should only have the .txt file event
        assert len(events) == 1
        assert "capture.txt" in events[0]["data"]["path"]


@pytest.mark.asyncio
async def test_file_watch_trigger_stop():
    """Test that file watch trigger can be stopped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trigger = FileWatchTrigger(
            id="test-trigger",
            path=tmpdir,
            patterns=["*"],
            max_events=100,
        )
        context = ProcessingContext()

        await trigger.initialize(context)

        events = []

        async def collect_events():
            async for result in trigger.gen_process(context):
                events.append(result["event"])

        task = asyncio.create_task(collect_events())

        # Let it start, then stop
        await asyncio.sleep(0.1)
        trigger.stop()

        await asyncio.wait_for(task, timeout=2.0)
        await trigger.finalize(context)

        # No events expected if no files were created
        # Just verify it stopped cleanly


@pytest.mark.asyncio
async def test_file_watch_trigger_invalid_path():
    """Test that file watch trigger raises error for invalid path."""
    trigger = FileWatchTrigger(
        id="test-trigger",
        path="/nonexistent/path/that/does/not/exist",
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    with pytest.raises(ValueError, match="does not exist"):
        async for _ in trigger.gen_process(context):
            pass
