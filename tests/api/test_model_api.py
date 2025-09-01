# Tests for the model API endpoints

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Assuming your FastAPI app instance is accessible for testing
# For example, if it's defined in nodetool.api.main:
# from nodetool.api.main import app
# For now, let's assume 'app' is the FastAPI instance from the router we are testing
# We need to get the 'app' that includes the 'model.router'
# This might require some setup. For now, let's define a placeholder app
# and focus on testing the router logic.
# Actual app instantiation might need to be adjusted based on project structure.
from fastapi import FastAPI
from nodetool.api.model import router as model_router

# Create a minimal app for testing this router
app = FastAPI()
app.include_router(model_router)

@pytest.fixture()
def client():
    # Context-manage TestClient to avoid leaking event loop/resources
    with TestClient(app) as c:
        yield c

# Define mock paths for consistent use in tests
MOCK_OLLAMA_ROOT = Path("/safe/ollama/models").resolve()
MOCK_HF_CACHE_ROOT = Path("/safe/hf/cache").resolve()

# --- Tests for get_ollama_base_path_endpoint ---


@patch("nodetool.api.model.common_get_ollama_models_dir")
def test_get_ollama_base_path_success(mock_get_ollama_dir, client):
    """Test successful retrieval of Ollama base path."""
    mock_get_ollama_dir.return_value = MOCK_OLLAMA_ROOT
    response = client.get("/api/models/ollama_base_path")
    assert response.status_code == 200
    assert response.json() == {"path": str(MOCK_OLLAMA_ROOT)}


@patch("nodetool.api.model.common_get_ollama_models_dir")
def test_get_ollama_base_path_not_found(mock_get_ollama_dir, client):
    """Test Ollama base path not found."""
    mock_get_ollama_dir.return_value = None
    response = client.get("/api/models/ollama_base_path")
    assert response.status_code == 200  # Endpoint returns 200 but with error status
    assert response.json() == {
        "status": "error",
        "message": "Could not determine Ollama models path. Please check server logs for details.",
    }


# --- Tests for open_in_explorer ---


@patch("nodetool.common.file_explorer.get_valid_explorable_roots")
@patch("nodetool.common.file_explorer.asyncio.create_subprocess_exec")
def test_open_in_explorer_success_ollama_path(mock_create_proc, mock_get_roots, client):
    """Test opening a valid path within the Ollama models directory."""
    mock_get_roots.return_value = [MOCK_OLLAMA_ROOT, MOCK_HF_CACHE_ROOT]
    valid_path_to_open = MOCK_OLLAMA_ROOT / "some_model"

    # Mock async subprocess returning success
    proc = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    mock_create_proc.return_value = proc

    response = client.post(
        f"/api/models/open_in_explorer?path={str(valid_path_to_open)}"
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "path": str(valid_path_to_open)}
    mock_create_proc.assert_called_once()


@patch("nodetool.common.file_explorer.get_valid_explorable_roots")
@patch("nodetool.common.file_explorer.asyncio.create_subprocess_exec")
def test_open_in_explorer_success_hf_cache_path(mock_create_proc, mock_get_roots, client):
    """Test opening a valid path within the Hugging Face cache directory."""
    mock_get_roots.return_value = [MOCK_OLLAMA_ROOT, MOCK_HF_CACHE_ROOT]
    valid_path_to_open = MOCK_HF_CACHE_ROOT / "models--some-model"

    # Mock async subprocess returning success
    proc = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    mock_create_proc.return_value = proc

    response = client.post(
        f"/api/models/open_in_explorer?path={str(valid_path_to_open)}"
    )

    assert response.status_code == 200
    assert response.json() == {"status": "success", "path": str(valid_path_to_open)}
    mock_create_proc.assert_called_once()


@patch("nodetool.common.file_explorer.get_valid_explorable_roots")
@patch("nodetool.common.file_explorer.asyncio.create_subprocess_exec")
def test_open_in_explorer_path_traversal_attempt(mock_create_proc, mock_get_roots, client):
    """Test path traversal attempt is blocked when path is outside all safe roots."""
    mock_get_roots.return_value = [MOCK_OLLAMA_ROOT, MOCK_HF_CACHE_ROOT]

    malicious_path_str = "/etc/passwd"  # A common example for *nix
    # For Windows, an equivalent might be C:\\Windows\\System32\\config\\SAM
    # However, Path.resolve() behavior for ".." can be tricky across OS for pure string paths.
    # Using an absolute path known to be outside the mocked safe roots is more robust.
    if MOCK_OLLAMA_ROOT.drive:  # Assuming windows if drive is present
        malicious_path_str = (
            f"{MOCK_OLLAMA_ROOT.drive}\\Windows\\System32\\drivers\\etc\\hosts"
        )

    response = client.post(f"/api/models/open_in_explorer?path={malicious_path_str}")

    assert response.status_code == 200
    assert response.json() == {
        "status": "error",
        "message": "Access denied: Path is outside the allowed directories.",
    }
    mock_create_proc.assert_not_called()


@patch("nodetool.common.file_explorer.get_valid_explorable_roots")
def test_open_in_explorer_no_safe_roots_found(mock_get_roots, client):
    """Test case where no safe explorable roots can be determined."""
    mock_get_roots.return_value = []  # No safe roots configured or found

    response = client.post("/api/models/open_in_explorer?path=/some/path")

    assert response.status_code == 200
    assert response.json() == {
        "status": "error",
        "message": "Cannot open path: No safe directories (like Ollama or Hugging Face cache) could be determined.",
    }


@patch("nodetool.common.file_explorer.get_valid_explorable_roots")
@patch("nodetool.common.file_explorer.asyncio.create_subprocess_exec")
def test_open_in_explorer_subprocess_error(mock_create_proc, mock_get_roots, client):
    """Test error handling when subprocess.run fails."""
    mock_get_roots.return_value = [MOCK_OLLAMA_ROOT]
    valid_path_to_open = MOCK_OLLAMA_ROOT / "some_model"

    # Simulate non-zero exit code from async subprocess
    proc = MagicMock()
    proc.wait = AsyncMock(return_value=1)
    mock_create_proc.return_value = proc

    response = client.post(
        f"/api/models/open_in_explorer?path={str(valid_path_to_open)}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "error",
        "message": "An internal error occurred while attempting to open the path. Please check server logs for details.",
    }
    mock_create_proc.assert_called_once()
