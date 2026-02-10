"""Tests for VibeCoding API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nodetool.agents.vibecoding import extract_html_from_response
from nodetool.models.workflow import Workflow


def test_extract_html_from_markdown_code_block():
    """Test extracting HTML from markdown code blocks."""
    response = """Here's your HTML:

```html
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>Hello</body>
</html>
```

That should work!
"""
    html = extract_html_from_response(response)
    assert html is not None
    assert html.startswith("<!DOCTYPE html>")
    assert "<title>Test</title>" in html


def test_extract_html_from_raw_response():
    """Test extracting HTML from raw response without code block."""
    response = """<!DOCTYPE html>
<html>
<body>Test</body>
</html>"""
    html = extract_html_from_response(response)
    assert html is not None
    assert html.startswith("<!DOCTYPE html>")


def test_extract_html_no_html_found():
    """Test that None is returned when no HTML is found."""
    response = "This is just some text without any HTML."
    html = extract_html_from_response(response)
    assert html is None


@pytest.mark.asyncio
async def test_templates_endpoint(client: TestClient, headers: dict[str, str]):
    """Test that templates endpoint returns valid templates."""
    response = client.get("/api/vibecoding/templates", headers=headers)
    assert response.status_code == 200
    templates = response.json()
    assert isinstance(templates, list)
    assert len(templates) > 0
    # Check all required fields are present
    for template in templates:
        assert "id" in template
        assert "name" in template
        assert "description" in template
        assert "prompt" in template


@pytest.mark.asyncio
async def test_generate_requires_valid_workflow(client: TestClient, headers: dict[str, str]):
    """Test that generate fails for non-existent workflow."""
    response = client.post(
        "/api/vibecoding/generate",
        json={"workflow_id": "nonexistent", "prompt": "Create a simple form"},
        headers=headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_generate_requires_access(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that users can only generate for workflows they have access to."""
    # Create workflow owned by different user
    workflow = await Workflow.create(
        user_id="different_user",
        name="Private Workflow",
        graph={"nodes": [], "edges": []},
    )
    workflow.access = "private"
    await workflow.save()

    # Attempt to generate - should fail with 403
    response = client.post(
        "/api/vibecoding/generate",
        json={"workflow_id": workflow.id, "prompt": "Create a simple form"},
        headers=headers,
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_generate_returns_streaming_response(
    client: TestClient, headers: dict[str, str], workflow: Workflow
):
    """Test that generate returns a streaming response."""
    await workflow.save()

    # Mock the agent's generate method to return a simple response
    async def mock_generate(*args, **kwargs):
        yield "```html\n<!DOCTYPE html><html></html>\n```"

    with patch("nodetool.api.vibecoding.VibeCodingAgent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.generate = mock_generate
        MockAgent.return_value = mock_agent

        response = client.post(
            "/api/vibecoding/generate",
            json={"workflow_id": workflow.id, "prompt": "Create a minimal form"},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/plain; charset=utf-8"


@pytest.mark.asyncio
async def test_workflow_app_injects_config(
    client: TestClient, headers: dict[str, str], workflow: Workflow
):
    """Test that the /app endpoint injects runtime configuration."""
    # Set html_app on workflow
    workflow.html_app = "<!DOCTYPE html><html><head></head><body>Test</body></html>"
    workflow.access = "private"
    await workflow.save()

    response = client.get(f"/api/workflows/{workflow.id}/app", headers=headers)
    assert response.status_code == 200
    html = response.text

    assert "window.NODETOOL_API_URL" in html
    assert "window.NODETOOL_WS_URL" in html
    assert f'window.NODETOOL_WORKFLOW_ID = "{workflow.id}"' in html


@pytest.mark.asyncio
async def test_workflow_app_no_html_returns_404(
    client: TestClient, headers: dict[str, str], workflow: Workflow
):
    """Test that /app endpoint returns 404 when no HTML app is configured."""
    workflow.html_app = None
    await workflow.save()

    response = client.get(f"/api/workflows/{workflow.id}/app", headers=headers)
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "html app configured" in detail.lower()
    assert workflow.id in detail


@pytest.mark.asyncio
async def test_workflow_app_public_access(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that public workflows can be accessed."""
    # Create a public workflow with HTML app
    workflow = await Workflow.create(
        user_id="different_user",
        name="Public Workflow",
        graph={"nodes": [], "edges": []},
    )
    workflow.access = "public"
    workflow.html_app = "<!DOCTYPE html><html><head></head><body>Public</body></html>"
    await workflow.save()

    response = client.get(f"/api/workflows/{workflow.id}/app", headers=headers)
    assert response.status_code == 200
    assert "window.NODETOOL_WORKFLOW_ID" in response.text
