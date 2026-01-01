"""
Tests for Docker-based job execution.
"""

import asyncio
import subprocess

import docker.errors
import pytest

import docker
from nodetool.models.job import Job
from nodetool.types.graph import Edge, Graph, Node
from nodetool.workflows.docker_job_execution import DockerJobExecution
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import PreviewUpdate


def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_nodetool_image_available() -> bool:
    """Check if the nodetool Docker image is available."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "nodetool"],
            capture_output=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip all tests if Docker is not available or nodetool image is missing
pytestmark = [
    pytest.mark.skipif(
        not check_docker_available(),
        reason="Docker is not available",
    ),
    pytest.mark.skipif(
        not check_nodetool_image_available(),
        reason="nodetool Docker image is not available",
    ),
]


def _build_simple_workflow_graph() -> Graph:
    """Build a simple workflow graph for testing."""
    return Graph(
        nodes=[
            Node(
                id="input_text",
                type="nodetool.workflows.test_helper.StringInput",
                data={
                    "name": "text",
                    "label": "Input Text",
                    "value": "",
                },
            ),
            Node(
                id="format_text",
                type="nodetool.workflows.test_helper.FormatText",
                data={
                    "template": "Docker test: {{ text }}",
                },
            ),
            Node(
                id="output_result",
                type="nodetool.workflows.test_helper.StringOutput",
                data={
                    "name": "result",
                    "value": "",
                },
            ),
        ],
        edges=[
            Edge(
                id="edge_input_to_format",
                source="input_text",
                sourceHandle="output",
                target="format_text",
                targetHandle="text",
            ),
            Edge(
                id="edge_format_to_output",
                source="format_text",
                sourceHandle="output",
                target="output_result",
                targetHandle="value",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_docker_job_creation():
    """Test creating and starting a Docker job."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Creation test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        assert job is not None
        assert job.job_id is not None
        assert job.status == "running"

        # Wait for container to be created
        for _ in range(50):
            if job.container_id is not None:
                break
            await asyncio.sleep(0.1)

        assert job.container_id is not None
        assert isinstance(job, DockerJobExecution)

        # Container uses --rm so it will be removed automatically when finished
        # Just verify we can interact with the job
        assert not job.is_completed()
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_is_running():
    """Test that a started Docker job is running."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Running test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Initially running
        assert job.is_running()
        assert not job.is_completed()
        # assert not job.is_finished()
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_completion():
    """Test that a Docker job completes successfully."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Success!"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Wait for completion (with timeout)
        max_wait = 10  # Increased from 3 to allow for Docker container startup
        for _ in range(max_wait):
            print(job)
            if job.is_completed():
                break
            await asyncio.sleep(1)

        assert job.is_completed()
        assert job.status == "completed"

        # Verify database record
        db_job = await Job.get(job.job_id)
        assert db_job is not None
        assert db_job.status == "completed"
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_cancellation():
    """Test cancelling a running Docker job."""
    # Use a workflow that would take a while
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Cancel me"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Wait for container to be created
        for _ in range(50):
            if job.container_id is not None:
                break
            await asyncio.sleep(0.1)

        # Save container_id before cancellation (it may be cleared after)
        container_id = job.container_id
        assert container_id is not None

        # Cancel it
        await job.cancel()

        # Wait a bit for cancellation to take effect
        await asyncio.sleep(2)

        assert job.status == "cancelled"
        assert not job.is_running()

        # Verify container is stopped/removed
        docker_client = docker.from_env()
        try:
            container = docker_client.containers.get(container_id)
            # If it exists, it should be exited
            assert container.status == "exited"
        except docker.errors.NotFound:
            # Container was removed, which is also acceptable
            pass
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_cleanup():
    """Test that Docker job cleanup removes the container."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Cleanup test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    # Wait for container to be created
    for _ in range(50):
        if job.container_id is not None:
            break
        await asyncio.sleep(0.1)

    # Save container_id before cleanup (it may be cleared after)
    container_id = job.container_id
    assert container_id is not None

    # Cleanup
    await job.cleanup_resources()

    # Verify container is removed
    docker_client = docker.from_env()
    try:
        container = docker_client.containers.get(container_id)
        # If it still exists, it should be marked as exited or removed
        assert container.status in ("exited", "removed")
    except docker.errors.NotFound:
        # Container is completely gone, which is expected with auto_remove=True
        pass


@pytest.mark.asyncio
async def test_docker_job_database_record():
    """Test that Docker job creates and updates database record."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Database test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Check database record exists
        db_job = await Job.get(job.job_id)
        assert db_job is not None
        assert db_job.user_id == "test_user"
        assert db_job.status == "running"

        # Wait for completion
        max_wait = 30
        for _ in range(max_wait):
            if job.is_completed():
                break
            await asyncio.sleep(1)

        # Check database record is updated
        db_job = await Job.get(job.job_id)
        assert db_job is not None
        assert db_job.status == "completed"
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_messages_forwarded():
    """Test that messages from container are forwarded."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Messages test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    messages = []

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Wait for completion
        max_wait = 30
        for _ in range(max_wait):
            while context.has_messages():
                msg = await context.pop_message_async()
                messages.append(msg)
            if job.is_completed():
                break
            await asyncio.sleep(1)

        while context.has_messages():
            msg = await context.pop_message_async()
            messages.append(msg)

        print(messages)

        # Should have received some messages
        assert len(messages) > 0
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_container_id():
    """Test that Docker job exposes container ID."""
    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=_build_simple_workflow_graph(),
        params={"text": "Container ID test"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Wait for container to be created (runner does this asynchronously)
        max_wait = 5
        for _ in range(max_wait * 10):
            if job.container_id is not None:
                break
            await asyncio.sleep(0.1)

        assert job.container_id is not None, "Container ID should be set after creation"
        assert len(job.container_id) > 0

        # Verify it's a valid container ID (hex string)
        assert all(c in "0123456789abcdef" for c in job.container_id.lower())
    finally:
        await job.cleanup_resources()


@pytest.mark.asyncio
async def test_docker_job_preview_update_messages():
    """Test that PreviewUpdate messages are properly forwarded from Docker containers."""
    # Create a workflow with a Preview node
    graph = Graph(
        nodes=[
            Node(
                id="input_text",
                type="nodetool.workflows.test_helper.StringInput",
                data={
                    "name": "text",
                    "label": "Input Text",
                    "value": "",
                },
            ),
            Node(
                id="format_text",
                type="nodetool.workflows.test_helper.FormatText",
                data={
                    "template": "Preview test: {{ text }}",
                },
            ),
            Node(
                id="preview_node",
                type="nodetool.workflows.base_node.Preview",
                data={
                    "name": "test_preview",
                    "value": "",
                },
            ),
            Node(
                id="output_result",
                type="nodetool.workflows.test_helper.StringOutput",
                data={
                    "name": "result",
                    "value": "",
                },
            ),
        ],
        edges=[
            Edge(
                id="edge_input_to_format",
                source="input_text",
                sourceHandle="output",
                target="format_text",
                targetHandle="text",
            ),
            Edge(
                id="edge_format_to_preview",
                source="format_text",
                sourceHandle="output",
                target="preview_node",
                targetHandle="value",
            ),
            Edge(
                id="edge_preview_to_output",
                source="preview_node",
                sourceHandle="output",
                target="output_result",
                targetHandle="value",
            ),
        ],
    )

    request = RunJobRequest(
        user_id="test_user",
        auth_token="test_token",
        graph=graph,
        params={"text": "Hello Preview!"},
    )

    context = ProcessingContext(
        user_id=request.user_id,
        auth_token=request.auth_token,
    )

    messages = []
    preview_messages = []

    job = await DockerJobExecution.create_and_start(request, context)

    try:
        # Wait for completion and collect messages
        max_wait = 30
        for _ in range(max_wait):
            while context.has_messages():
                msg = await context.pop_message_async()
                messages.append(msg)
                if isinstance(msg, PreviewUpdate):
                    preview_messages.append(msg)
            if job.is_completed():
                break
            await asyncio.sleep(1)

        # Collect any remaining messages
        while context.has_messages():
            msg = await context.pop_message_async()
            messages.append(msg)
            if isinstance(msg, PreviewUpdate):
                preview_messages.append(msg)

        # Should have received messages
        assert len(messages) > 0, "Should have received some messages"

        # Should have received at least one PreviewUpdate message
        assert len(preview_messages) > 0, (
            f"Should have received PreviewUpdate messages. Got message types: {[type(m).__name__ for m in messages]}"
        )

        # Verify the PreviewUpdate has the correct node_id
        assert any(pm.node_id == "preview_node" for pm in preview_messages), (
            f"PreviewUpdate should have correct node_id. Got: {[pm.node_id for pm in preview_messages]}"
        )

        # Verify the preview value contains our text
        assert any("Preview test: Hello Preview!" in str(pm.value) for pm in preview_messages), (
            f"PreviewUpdate should contain our text. Got values: {[pm.value for pm in preview_messages]}"
        )

    finally:
        await job.cleanup_resources()
