"""
Tests for the WebhookTrigger node.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
import httpx

from nodetool.nodes.triggers.webhook import WebhookTrigger
from nodetool.workflows.processing_context import ProcessingContext


def get_free_port() -> int:
    """Get a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))  # Bind to localhost only for port discovery
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_webhook_trigger_receives_request():
    """Test that webhook trigger receives HTTP requests."""
    port = get_free_port()
    trigger = WebhookTrigger(
        id="test-trigger",
        port=port,
        path="/webhook",
        methods=["POST"],
        max_events=1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    async def send_request():
        # Wait for server to start
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                json={"message": "hello"},
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200

    await asyncio.gather(send_request(), collect_events())
    await trigger.finalize(context)

    assert len(events) == 1
    assert events[0]["event_type"] == "webhook"
    assert events[0]["data"]["body"]["message"] == "hello"
    assert events[0]["data"]["method"] == "POST"


@pytest.mark.asyncio
async def test_webhook_trigger_secret_validation():
    """Test that webhook trigger validates secrets."""
    port = get_free_port()
    trigger = WebhookTrigger(
        id="test-trigger",
        port=port,
        path="/webhook",
        methods=["POST"],
        secret="my-secret",
        max_events=1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    async def send_requests():
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            # Request without secret should fail
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                json={"message": "no secret"},
            )
            assert response.status_code == 401

            # Request with wrong secret should fail
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                json={"message": "wrong secret"},
                headers={"X-Webhook-Secret": "wrong"},
            )
            assert response.status_code == 401

            # Request with correct secret should succeed
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                json={"message": "correct secret"},
                headers={"X-Webhook-Secret": "my-secret"},
            )
            assert response.status_code == 200

    await asyncio.gather(send_requests(), collect_events())
    await trigger.finalize(context)

    # Only the request with correct secret should be captured
    assert len(events) == 1
    assert events[0]["data"]["body"]["message"] == "correct secret"


@pytest.mark.asyncio
async def test_webhook_trigger_method_filter():
    """Test that webhook trigger only accepts configured methods."""
    port = get_free_port()
    trigger = WebhookTrigger(
        id="test-trigger",
        port=port,
        path="/webhook",
        methods=["POST"],  # Only POST allowed
        max_events=1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    async def send_requests():
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            # GET should be rejected (method not allowed)
            response = await client.get(f"http://127.0.0.1:{port}/webhook")
            assert response.status_code == 405

            # POST should succeed
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                json={"method": "post"},
            )
            assert response.status_code == 200

    await asyncio.gather(send_requests(), collect_events())
    await trigger.finalize(context)

    assert len(events) == 1
    assert events[0]["data"]["method"] == "POST"


@pytest.mark.asyncio
async def test_webhook_trigger_stop():
    """Test that webhook trigger can be stopped."""
    port = get_free_port()
    trigger = WebhookTrigger(
        id="test-trigger",
        port=port,
        path="/webhook",
        max_events=100,  # High limit
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    task = asyncio.create_task(collect_events())

    # Let it start, then stop
    await asyncio.sleep(0.2)
    trigger.stop()

    await asyncio.wait_for(task, timeout=2.0)
    await trigger.finalize(context)

    # Should have stopped cleanly with no events
    assert len(events) == 0


@pytest.mark.asyncio
async def test_webhook_trigger_form_data():
    """Test that webhook trigger handles form data."""
    port = get_free_port()
    trigger = WebhookTrigger(
        id="test-trigger",
        port=port,
        path="/webhook",
        methods=["POST"],
        max_events=1,
    )
    context = ProcessingContext()

    await trigger.initialize(context)

    events = []

    async def collect_events():
        async for result in trigger.gen_process(context):
            events.append(result["event"])

    async def send_request():
        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/webhook",
                data={"field1": "value1", "field2": "value2"},
            )
            assert response.status_code == 200

    await asyncio.gather(send_request(), collect_events())
    await trigger.finalize(context)

    assert len(events) == 1
    # Form data should be parsed
    body = events[0]["data"]["body"]
    assert body.get("field1") == "value1" or "field1" in str(body)
