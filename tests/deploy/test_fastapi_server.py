"""Tests for the FastAPI server module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from nodetool.deploy.fastapi_server import (
    load_workflow,
    load_workflows_from_directory, 
    create_nodetool_server,
)
from nodetool.types.workflow import Workflow


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        "id": "test-workflow",
        "name": "Test Workflow",
        "access": "public",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "description": "Test workflow description",
        "graph": {
            "nodes": [
                {
                    "id": "node1",
                    "type": "nodetool.nodes.nodetool.input.TextInput",
                    "data": {"value": "Hello"}
                }
            ],
            "edges": []
        }
    }


@pytest.fixture
def sample_workflow_file(sample_workflow_data):
    """Create a temporary workflow file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_workflow_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_workflows_dir(sample_workflow_data):
    """Create a temporary directory with workflow files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create first workflow
        workflow1 = sample_workflow_data.copy()
        workflow1["id"] = "workflow1"
        with open(os.path.join(temp_dir, "workflow1.json"), "w") as f:
            json.dump(workflow1, f)
        
        # Create second workflow
        workflow2 = sample_workflow_data.copy()
        workflow2["id"] = "workflow2"
        workflow2["name"] = "Second Workflow"
        with open(os.path.join(temp_dir, "workflow2.json"), "w") as f:
            json.dump(workflow2, f)
        
        # Create non-JSON file (should be ignored)
        with open(os.path.join(temp_dir, "readme.txt"), "w") as f:
            f.write("This should be ignored")
        
        # Create invalid JSON file
        with open(os.path.join(temp_dir, "invalid.json"), "w") as f:
            f.write("{ invalid json")
        
        yield temp_dir


@pytest.fixture
def fastapi_app():
    """Create FastAPI app for testing."""
    app = create_nodetool_server(
        remote_auth=False,
        provider="test_provider",
        default_model="test-model",
        tools=["test_tool"],
        workflows=[]
    )
    return app


@pytest.fixture
def client(fastapi_app):
    """Create test client."""
    return TestClient(fastapi_app)


class TestWorkflowLoading:
    """Test workflow loading functionality (same as runpod_handler tests)."""
    
    def test_load_workflow_success(self, sample_workflow_file):
        """Test successful workflow loading."""
        workflow = load_workflow(sample_workflow_file)
        assert isinstance(workflow, Workflow)
        assert workflow.id == "test-workflow"
        assert workflow.name == "Test Workflow"
    
    def test_load_workflow_file_not_found(self):
        """Test loading non-existent workflow file."""
        with pytest.raises(FileNotFoundError):
            load_workflow("/nonexistent/file.json")
    
    def test_load_workflows_from_directory_success(self, temp_workflows_dir):
        """Test loading workflows from directory."""
        workflows = load_workflows_from_directory(temp_workflows_dir)
        
        assert len(workflows) == 2  # Only valid JSON files
        assert "workflow1" in workflows
        assert "workflow2" in workflows
        
        assert workflows["workflow1"].id == "workflow1"
        assert workflows["workflow2"] is not None
    
    def test_load_workflows_from_directory_not_exists(self):
        """Test loading from non-existent directory."""
        workflows = load_workflows_from_directory("/nonexistent/directory")
        assert workflows == {}


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestModelsEndpoint:
    """Test OpenAI-compatible models endpoint."""
    
    def test_models_endpoint_success(self, client):
        """Test successful models endpoint."""
        mock_models = [
            MagicMock(id="model1", name="model1", provider=MagicMock(value="test_provider")),
            MagicMock(id="model2", name="model2", provider=MagicMock(value="other_provider"))
        ]
        
        with patch("nodetool.deploy.fastapi_server.get_language_models") as mock_get_models:
            mock_get_models.return_value = mock_models
            
            response = client.get("/v1/models")
            assert response.status_code == 200
            
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1  # Only test_provider models
            assert data["data"][0]["id"] == "model1"
            assert data["data"][0]["object"] == "model"
            assert data["data"][0]["owned_by"] == "test_provider"
    
    def test_models_endpoint_error(self, client):
        """Test models endpoint with error."""
        with patch("nodetool.deploy.fastapi_server.get_language_models") as mock_get_models:
            mock_get_models.side_effect = Exception("Test error")
            
            response = client.get("/v1/models")
            assert response.status_code == 500


class TestChatCompletionsEndpoint:
    """Test OpenAI-compatible chat completions endpoint."""
    
    def test_chat_completions_non_streaming(self, client):
        """Test non-streaming chat completions."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        with patch("nodetool.deploy.fastapi_server.ChatSSERunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            
            # Mock the async generator to return SSE-formatted data
            async def mock_process(request_data):
                yield "data: " + json.dumps({
                    "id": "test-123",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hello "}}]
                })
                yield "data: " + json.dumps({
                    "id": "test-123", 
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "World!"}}]
                })
                yield "data: [DONE]"
            
            mock_runner.process_single_request = mock_process
            
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200
            
            # For non-streaming, should return the last chunk
            data = response.json()
            assert "id" in data
    
    def test_chat_completions_streaming(self, client):
        """Test streaming chat completions."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        
        with patch("nodetool.deploy.fastapi_server.ChatSSERunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            
            # Mock the async generator to return SSE-formatted data
            async def mock_process(request_data):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Hello"}}]
                })
                yield "data: [DONE]"
            
            mock_runner.process_single_request = mock_process
            
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_chat_completions_with_auth_header(self, client):
        """Test chat completions with authorization header."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        with patch("nodetool.deploy.fastapi_server.ChatSSERunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            
            async def mock_process(request_data):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Hello"}}]
                })
                yield "data: [DONE]"
            
            mock_runner.process_single_request = mock_process
            
            response = client.post("/v1/chat/completions", json=request_data, headers=headers)
            assert response.status_code == 200
            
            # Verify auth token was passed to runner
            call_args = mock_runner_class.call_args
            assert call_args[0][0] == "test-token"  # auth_token parameter
    
    def test_chat_completions_error(self, client):
        """Test chat completions with error."""
        request_data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        with patch("nodetool.deploy.fastapi_server.ChatSSERunner") as mock_runner_class:
            mock_runner_class.side_effect = Exception("Test error")
            
            response = client.post("/v1/chat/completions", json=request_data)
            assert response.status_code == 500


class TestWorkflowEndpoints:
    """Test workflow execution endpoints."""
    
    def test_list_workflows(self, client):
        """Test listing workflows."""
        mock_workflows = {
            "workflow1": MagicMock(name="Test Workflow 1"),
            "workflow2": MagicMock(name="Test Workflow 2")
        }
        
        with patch("nodetool.deploy.fastapi_server._workflow_registry", mock_workflows):
            response = client.get("/workflows")
            assert response.status_code == 200
            
            data = response.json()
            assert "workflows" in data
            assert len(data["workflows"]) == 2
            assert any(w["id"] == "workflow1" for w in data["workflows"])
            assert any(w["id"] == "workflow2" for w in data["workflows"])
    
    def test_execute_workflow_success(self, client, sample_workflow_data):
        """Test successful workflow execution."""
        request_data = {
            "workflow_id": "test-workflow",
            "params": {"input": "test"}
        }
        
        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()
        
        with patch("nodetool.deploy.fastapi_server.get_workflow_by_id") as mock_get_workflow, \
             patch("nodetool.deploy.fastapi_server.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.fastapi_server.ProcessingContext"):
            
            mock_get_workflow.return_value = mock_workflow
            
            # Mock workflow execution results
            async def mock_run_gen(req, context=None, use_thread=True):
                from nodetool.workflows.types import OutputUpdate
                yield OutputUpdate(
                    node_id="output_node",
                    node_name="output", 
                    output_name="result", 
                    value="result",
                    output_type="string"
                )
            
            mock_run_workflow.return_value = mock_run_gen(None, None, True)
            
            response = client.post("/workflows/execute", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "results" in data
            assert "output" in data["results"]
    
    def test_execute_workflow_missing_id(self, client):
        """Test workflow execution without workflow ID."""
        request_data = {
            "params": {"input": "test"}
        }
        
        response = client.post("/workflows/execute", json=request_data)
        assert response.status_code == 400
        assert "workflow_id is required" in response.json()["detail"]
    
    def test_execute_workflow_not_found(self, client):
        """Test workflow execution with non-existent workflow ID."""
        request_data = {
            "workflow_id": "nonexistent-workflow",
            "params": {"input": "test"}
        }
        
        with patch("nodetool.deploy.fastapi_server.get_workflow_by_id") as mock_get_workflow:
            mock_get_workflow.side_effect = ValueError("Workflow 'nonexistent-workflow' not found")
            
            response = client.post("/workflows/execute", json=request_data)
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    
    def test_execute_workflow_stream(self, client, sample_workflow_data):
        """Test streaming workflow execution."""
        request_data = {
            "workflow_id": "test-workflow",
            "params": {"input": "test"}
        }
        
        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()
        
        with patch("nodetool.deploy.fastapi_server.get_workflow_by_id") as mock_get_workflow, \
             patch("nodetool.deploy.fastapi_server.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.fastapi_server.ProcessingContext"):
            
            mock_get_workflow.return_value = mock_workflow
            
            # Mock workflow execution results
            async def mock_run_gen(req, context=None, use_thread=True):
                from nodetool.types.job import JobUpdate
                from nodetool.workflows.types import OutputUpdate
                yield JobUpdate(status="running")
                yield OutputUpdate(
                    node_id="output_node",
                    node_name="output", 
                    output_name="result", 
                    value="result",
                    output_type="string"
                )
            
            mock_run_workflow.return_value = mock_run_gen(None, None, True)
            
            response = client.post("/workflows/execute/stream", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    def test_execute_workflow_stream_error(self, client):
        """Test streaming workflow execution with error."""
        request_data = {
            "workflow_id": "test-workflow",
            "params": {"input": "test"}
        }
        
        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()
        
        with patch("nodetool.deploy.fastapi_server.get_workflow_by_id") as mock_get_workflow, \
             patch("nodetool.deploy.fastapi_server.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.fastapi_server.ProcessingContext"):
            
            mock_get_workflow.return_value = mock_workflow
            
            # Mock workflow execution with error
            async def mock_run_gen(req, context=None, use_thread=True):
                from nodetool.types.job import JobUpdate
                yield JobUpdate(status="error", error="Test workflow error")
            
            mock_run_workflow.return_value = mock_run_gen(None, None, True)
            
            response = client.post("/workflows/execute/stream", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestAdminEndpoints:
    """Test admin operation endpoints."""
    
    # Individual Admin Endpoints Tests
    def test_ping_health_check(self, client):
        """Test ping health check endpoint."""
        response = client.get("/ping")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_download_huggingface_model(self, client):
        """Test HuggingFace model download endpoint."""
        request_data = {
            "repo_id": "microsoft/DialoGPT-medium",
            "cache_dir": "/test/cache",
            "stream": True
        }
        
        with patch("nodetool.deploy.fastapi_server.download_hf_model") as mock_download:
            async def mock_download_gen(*args, **kwargs):
                yield {"status": "starting", "repo_id": "microsoft/DialoGPT-medium"}
                yield {"status": "progress", "progress": 50}
                yield {"status": "completed", "repo_id": "microsoft/DialoGPT-medium"}
            
            mock_download.return_value = mock_download_gen()
            
            response = client.post("/admin/models/huggingface/download", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_download_huggingface_model_missing_repo_id(self, client):
        """Test HuggingFace model download without repo_id."""
        request_data = {"cache_dir": "/test/cache"}
        
        response = client.post("/admin/models/huggingface/download", json=request_data)
        assert response.status_code == 400
        assert "repo_id is required" in response.json()["detail"]

    def test_download_ollama_model(self, client):
        """Test Ollama model download endpoint."""
        request_data = {
            "model_name": "gemma3n:latest",
            "stream": True
        }
        
        with patch("nodetool.deploy.fastapi_server.download_ollama_model") as mock_download:
            async def mock_download_gen(*args, **kwargs):
                yield {"status": "starting", "model": "gemma3n:latest"}
                yield {"status": "progress", "progress": 75}
                yield {"status": "completed", "model": "gemma3n:latest"}
            
            mock_download.return_value = mock_download_gen()
            
            response = client.post("/admin/models/ollama/download", json=request_data)
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_download_ollama_model_missing_model_name(self, client):
        """Test Ollama model download without model_name."""
        request_data = {"stream": True}
        
        response = client.post("/admin/models/ollama/download", json=request_data)
        assert response.status_code == 400
        assert "model_name is required" in response.json()["detail"]

    def test_scan_cache(self, client):
        """Test cache scan endpoint."""
        with patch("nodetool.deploy.fastapi_server.scan_hf_cache") as mock_scan:
            async def mock_scan_gen():
                yield {
                    "status": "completed", 
                    "cache_info": {
                        "size_on_disk": 1024000,
                        "repos": [],
                        "warnings": []
                    }
                }
            
            mock_scan.return_value = mock_scan_gen()
            
            response = client.get("/admin/cache/scan")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "completed"
            assert "cache_info" in data

    def test_get_cache_size(self, client):
        """Test cache size endpoint."""
        with patch("nodetool.deploy.fastapi_server.calculate_cache_size") as mock_calc:
            async def mock_calc_gen(*args, **kwargs):
                yield {
                    "success": True,
                    "cache_dir": "/test/cache",
                    "total_size_bytes": 5120000,
                    "size_gb": 5.12
                }
            
            mock_calc.return_value = mock_calc_gen()
            
            response = client.get("/admin/cache/size?cache_dir=/test/cache")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["size_gb"] == 5.12

    def test_delete_huggingface_model(self, client):
        """Test HuggingFace model deletion endpoint."""
        repo_id = "microsoft/DialoGPT-medium"
        
        with patch("nodetool.deploy.fastapi_server.delete_hf_model") as mock_delete:
            async def mock_delete_gen(*args, **kwargs):
                yield {
                    "status": "completed",
                    "repo_id": "microsoft/DialoGPT-medium",
                    "message": "Successfully deleted microsoft/DialoGPT-medium"
                }
            
            mock_delete.return_value = mock_delete_gen()
            
            response = client.delete(f"/admin/models/huggingface/{repo_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "completed"
            assert data["repo_id"] == repo_id

    def test_admin_workflows_status(self, client):
        """Test admin workflow status endpoint."""
        mock_workflows = {"workflow1": MagicMock(), "workflow2": MagicMock()}
        
        with patch("nodetool.deploy.fastapi_server._workflow_registry", mock_workflows):
            response = client.get("/admin/workflows/status")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["workflow_count"] == 2
            assert "workflow1" in data["available_workflows"]
            assert "workflow2" in data["available_workflows"]
            assert "timestamp" in data



class TestServerCreation:
    """Test server creation and configuration."""
    
    def test_create_nodetool_server(self):
        """Test creating server with custom configuration."""
        app = create_nodetool_server(
            remote_auth=True,
            provider="custom_provider",
            default_model="custom-model",
            tools=["tool1", "tool2"],
            workflows=[]
        )
        
        assert isinstance(app, FastAPI)
        assert app.title == "NodeTool API Server"
        assert app.version == "1.0.0"
    
    def test_create_nodetool_server_defaults(self):
        """Test creating server with default configuration."""
        app = create_nodetool_server()
        
        assert isinstance(app, FastAPI)
        assert app.title == "NodeTool API Server"


class TestSSEStreamingIntegration:
    """Test Server-Sent Events streaming functionality."""
    
    def test_workflow_stream_sse_format(self, client):
        """Test that workflow streaming returns proper SSE format."""
        request_data = {
            "workflow_id": "test-workflow", 
            "params": {"input": "test"}
        }
        
        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()
        
        with patch("nodetool.deploy.fastapi_server.get_workflow_by_id") as mock_get_workflow, \
             patch("nodetool.deploy.fastapi_server.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.fastapi_server.ProcessingContext") as mock_context_class:
            
            # Mock the context instance
            mock_context = MagicMock()
            mock_context.encode_assets_as_uri.return_value = "test_result"
            mock_context_class.return_value = mock_context
            
            mock_get_workflow.return_value = mock_workflow
            
            async def mock_run_gen(req, context=None, use_thread=True):
                from nodetool.workflows.types import OutputUpdate
                yield OutputUpdate(
                    node_id="output_node",
                    node_name="output", 
                    output_name="result", 
                    value="test_result",
                    output_type="string"
                )
            
            mock_run_workflow.return_value = mock_run_gen(None, None, True)
            
            with client.stream("POST", "/workflows/execute/stream", json=request_data) as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                
                # Read the stream content
                content = response.read().decode()
                lines = content.strip().split('\n')
                
                # Should contain data lines and [DONE] marker
                data_lines = [line for line in lines if line.startswith("data: ")]
                assert len(data_lines) >= 2  # At least output_update and complete events
                assert "data: [DONE]" in lines
    
    def test_admin_stream_sse_format(self, client):
        """Test that admin streaming returns proper SSE format."""
        request_data = {
            "repo_id": "test-model",
            "stream": True
        }
        
        with patch("nodetool.deploy.fastapi_server.download_hf_model") as mock_download:
            async def mock_download_gen(*args, **kwargs):
                yield {"progress": 100, "status": "complete"}
            
            mock_download.return_value = mock_download_gen()
            
            with client.stream("POST", "/admin/models/huggingface/download", json=request_data) as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                
                content = response.read().decode()
                lines = content.strip().split('\n')
                
                # Should contain data lines and [DONE] marker
                data_lines = [line for line in lines if line.startswith("data: ")]
                assert len(data_lines) >= 1
                assert "data: [DONE]" in lines


if __name__ == "__main__":
    pytest.main([__file__])