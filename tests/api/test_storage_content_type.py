
import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_html_download_attachment(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that HTML files are served with Content-Disposition: attachment."""
    filename = "test_exploit.html"
    content = b"<html><script>alert(1)</script></html>"

    response = client.put(
        f"/api/storage/{filename}",
        content=content,
        headers=headers
    )
    assert response.status_code == 200

    response = client.get(f"/api/storage/{filename}", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html"
    assert "attachment" in response.headers["Content-Disposition"]
    assert filename in response.headers["Content-Disposition"]

@pytest.mark.asyncio
async def test_image_inline(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that image files are served without attachment disposition (inline)."""
    filename = "test_image.jpg"
    content = b"fake image content"

    response = client.put(
        f"/api/storage/{filename}",
        content=content,
        headers=headers
    )
    assert response.status_code == 200

    response = client.get(f"/api/storage/{filename}", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/jpeg"
    # Content-Disposition should specificially NOT contain attachment
    content_disposition = response.headers.get("Content-Disposition", "")
    assert "attachment" not in content_disposition

@pytest.mark.asyncio
async def test_text_inline(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that text files are served without attachment disposition (inline)."""
    filename = "test_doc.txt"
    content = b"Just some text"

    response = client.put(
        f"/api/storage/{filename}",
        content=content,
        headers=headers
    )
    assert response.status_code == 200

    response = client.get(f"/api/storage/{filename}", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    content_disposition = response.headers.get("Content-Disposition", "")
    assert "attachment" not in content_disposition

@pytest.mark.asyncio
async def test_svg_attachment(client: TestClient, headers: dict[str, str], user_id: str):
    """Test that SVG files are served with attachment disposition."""
    filename = "test_image.svg"
    content = b"<svg><script>alert(1)</script></svg>"

    response = client.put(
        f"/api/storage/{filename}",
        content=content,
        headers=headers
    )
    assert response.status_code == 200

    response = client.get(f"/api/storage/{filename}", headers=headers)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "image/svg+xml"
    assert "attachment" in response.headers["Content-Disposition"]
    assert filename in response.headers["Content-Disposition"]
