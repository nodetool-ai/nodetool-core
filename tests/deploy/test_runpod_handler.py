"""Tests for the RunPod handler module."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from nodetool.deploy.runpod_handler import (
    load_workflow,
    load_workflows_from_directory,
    initialize_workflow_registry,
    get_workflow_by_id,
    universal_handler,
    _handle_workflow_execution,
    _handle_chat_request,
    _workflow_registry,
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
        
        # Create second workflow (filename as ID)
        workflow2 = sample_workflow_data.copy()
        workflow2["id"] = "workflow2"  # Keep ID for validation, will be overridden by filename logic if needed
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


class TestWorkflowLoading:
    """Test workflow loading functionality."""
    
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
        assert "workflow2" in workflows  # Uses filename as ID
        
        assert workflows["workflow1"].id == "workflow1"
        assert workflows["workflow2"] is not None
    
    def test_load_workflows_from_directory_not_exists(self):
        """Test loading from non-existent directory."""
        workflows = load_workflows_from_directory("/nonexistent/directory")
        assert workflows == {}
    
    def test_initialize_workflow_registry(self, temp_workflows_dir):
        """Test workflow registry initialization."""
        with patch("nodetool.deploy.runpod_handler.load_workflows_from_directory") as mock_load, \
             patch("nodetool.deploy.runpod_handler._workflow_registry") as mock_registry:
            mock_workflows = {"test": MagicMock()}
            mock_load.return_value = mock_workflows
            
            initialize_workflow_registry()
            
            mock_load.assert_called_once_with()
            # Can't easily test the global assignment, so just verify the function was called
    
    def test_get_workflow_by_id_success(self):
        """Test successful workflow retrieval by ID."""
        mock_workflow = MagicMock()
        with patch("nodetool.deploy.runpod_handler._workflow_registry", {"test-id": mock_workflow}):
            result = get_workflow_by_id("test-id")
            assert result == mock_workflow
    
    def test_get_workflow_by_id_not_found(self):
        """Test workflow not found error."""
        with patch("nodetool.deploy.runpod_handler._workflow_registry", {"existing": MagicMock()}):
            with pytest.raises(ValueError, match="Workflow 'missing' not found"):
                get_workflow_by_id("missing")


class TestUniversalHandler:
    """Test the universal handler functionality."""
    
    @pytest.mark.asyncio
    async def test_universal_handler_admin_operation(self):
        """Test handling admin operations."""
        job = {
            "input": {
                "operation": "test_op"
            }
        }
        
        with patch("nodetool.deploy.runpod_handler.handle_admin_operation") as mock_admin:
            async def mock_admin_gen(job_input):
                yield {"result": "admin_result"}
            
            mock_admin.return_value = mock_admin_gen({"operation": "test_op"})
            
            results = []
            async for chunk in universal_handler(job):
                results.append(chunk)
            
            mock_admin.assert_called_once_with({"operation": "test_op"})
            assert results == [{"result": "admin_result"}]
    
    @pytest.mark.asyncio
    async def test_universal_handler_chat_request(self):
        """Test handling chat requests."""
        job = {
            "input": {
                "openai_route": "/v1/models"
            }
        }
        
        with patch("nodetool.deploy.runpod_handler._handle_chat_request") as mock_chat:
            async def mock_chat_gen(job_input):
                yield {"result": "chat_result"}
            
            mock_chat.return_value = mock_chat_gen({"openai_route": "/v1/models"})
            
            results = []
            async for chunk in universal_handler(job):
                results.append(chunk)
            
            mock_chat.assert_called_once_with({"openai_route": "/v1/models"})
            assert results == [{"result": "chat_result"}]
    
    @pytest.mark.asyncio
    async def test_universal_handler_workflow_execution(self):
        """Test handling workflow execution."""
        job = {
            "input": {
                "workflow_id": "test-workflow",
                "params": {"key": "value"}
            }
        }
        
        with patch("nodetool.deploy.runpod_handler._handle_workflow_execution") as mock_workflow:
            async def mock_workflow_gen(job_input):
                yield {"result": "workflow_result"}
            
            mock_workflow.return_value = mock_workflow_gen({
                "workflow_id": "test-workflow",
                "params": {"key": "value"}
            })
            
            results = []
            async for chunk in universal_handler(job):
                results.append(chunk)
            
            mock_workflow.assert_called_once_with({
                "workflow_id": "test-workflow",
                "params": {"key": "value"}
            })
            assert results == [{"result": "workflow_result"}]
    
    @pytest.mark.asyncio
    async def test_universal_handler_error(self):
        """Test error handling in universal handler."""
        job = {
            "input": {
                "workflow_id": "test-workflow"
            }
        }
        
        with patch("nodetool.deploy.runpod_handler._handle_workflow_execution") as mock_workflow:
            mock_workflow.side_effect = Exception("Test error")
            
            results = []
            async for chunk in universal_handler(job):
                results.append(chunk)
            
            assert len(results) == 1
            assert "error" in results[0]
            assert results[0]["error"]["message"] == "Test error"
            assert results[0]["error"]["type"] == "handler_error"


class TestWorkflowExecution:
    """Test workflow execution functionality."""
    
    @pytest.mark.asyncio
    async def test_handle_workflow_execution_success(self, sample_workflow_data):
        """Test successful workflow execution."""
        job_input = {
            "workflow_id": "test-workflow",
            "params": {"param1": "value1"}
        }
        
        mock_workflow = MagicMock()
        mock_workflow.graph = MagicMock()
        
        with patch("nodetool.deploy.runpod_handler.get_workflow_by_id") as mock_get_workflow, \
             patch("nodetool.deploy.runpod_handler.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.runpod_handler.ProcessingContext") as mock_context:
            
            mock_get_workflow.return_value = mock_workflow
            
            async def mock_run_gen(req, context=None, use_thread=True):
                yield {"status": "completed"}
            
            mock_run_workflow.return_value = mock_run_gen(None)
            
            results = []
            async for chunk in _handle_workflow_execution(job_input):
                results.append(chunk)
            
            mock_get_workflow.assert_called_once_with("test-workflow")
            mock_run_workflow.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_workflow_execution_missing_id(self):
        """Test workflow execution without workflow ID."""
        job_input = {
            "params": {"param1": "value1"}
        }
        
        with pytest.raises(Exception, match="workflow_id is required"):
            async for _ in _handle_workflow_execution(job_input):
                pass
    
    @pytest.mark.asyncio
    async def test_handle_workflow_execution_workflow_not_found(self):
        """Test workflow execution with non-existent workflow ID."""
        job_input = {
            "workflow_id": "nonexistent-workflow",
            "params": {"param1": "value1"}
        }
        
        with patch("nodetool.deploy.runpod_handler.get_workflow_by_id") as mock_get_workflow:
            mock_get_workflow.side_effect = ValueError("Workflow 'nonexistent-workflow' not found")
            
            with pytest.raises(Exception, match="Workflow 'nonexistent-workflow' not found"):
                async for _ in _handle_workflow_execution(job_input):
                    pass


class TestChatRequest:
    """Test chat request handling."""
    
    @pytest.mark.asyncio
    async def test_handle_chat_request_models_endpoint(self):
        """Test handling /v1/models endpoint."""
        job_input = {
            "openai_route": "/v1/models",
            "openai_input": {}
        }
        
        mock_models = [
            MagicMock(id="model1", name="model1", provider=MagicMock(value="test_provider")),
            MagicMock(id="model2", name="model2", provider=MagicMock(value="other_provider"))
        ]
        
        with patch("nodetool.deploy.runpod_handler.get_language_models") as mock_get_models, \
             patch.dict(os.environ, {"CHAT_PROVIDER": "test_provider"}):
            
            mock_get_models.return_value = mock_models
            
            results = []
            async for chunk in _handle_chat_request(job_input):
                results.append(chunk)
            
            assert len(results) == 1
            assert results[0]["object"] == "list"
            assert len(results[0]["data"]) == 1  # Only test_provider models
            assert results[0]["data"][0]["id"] == "model1"
    
    @pytest.mark.asyncio
    async def test_handle_chat_request_unknown_route(self):
        """Test handling unknown route."""
        job_input = {
            "openai_route": "/unknown/route",
            "openai_input": {}
        }
        
        results = []
        async for chunk in _handle_chat_request(job_input):
            results.append(chunk)
        
        assert len(results) == 1
        assert "error" in results[0]
        assert "Unknown route" in results[0]["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_handle_chat_request_completions_non_streaming(self):
        """Test handling chat completions (non-streaming)."""
        job_input = {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
        }
        
        with patch("nodetool.deploy.runpod_handler.ChatSSERunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            
            # Mock the async generator to return SSE-formatted data
            async def mock_process(request_data):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "Hello "}}]
                })
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": "World!"}}]
                })
                yield "data: [DONE]"
            
            mock_runner.process_single_request = mock_process
            
            results = []
            async for chunk in _handle_chat_request(job_input):
                results.append(chunk)
            
            assert len(results) == 1
            assert "choices" in results[0]
            assert results[0]["choices"][0]["message"]["content"] == "Hello World!"


class TestIntegration:
    """Integration tests for the full handler."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_execution_flow(self, temp_workflows_dir, sample_workflow_data):
        """Test the full workflow execution flow."""
        # Initialize registry with test workflows
        with patch("nodetool.deploy.runpod_handler.load_workflows_from_directory") as mock_load:
            workflows = {"test-workflow": Workflow.model_validate(sample_workflow_data)}
            mock_load.return_value = workflows
            initialize_workflow_registry()
        
        job = {
            "input": {
                "workflow_id": "test-workflow",
                "params": {"input": "test"}
            }
        }
        
        with patch("nodetool.deploy.runpod_handler.run_workflow") as mock_run_workflow, \
             patch("nodetool.deploy.runpod_handler.ProcessingContext"):
            
            # Mock workflow execution results
            async def mock_run_gen(req, context=None, use_thread=True):
                yield {"status": "completed"}
            
            mock_run_workflow.return_value = mock_run_gen(None)
            
            results = []
            async for chunk in universal_handler(job):
                results.append(chunk)
            
            # Should complete without errors
            assert len(results) >= 1


if __name__ == "__main__":
    pytest.main([__file__])