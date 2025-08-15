"""
Comprehensive tests for ProcessingContext database model integration.
Tests all the methods that were migrated from API client to direct database operations.
"""

import pytest
import asyncio
from datetime import datetime
from io import BytesIO
from unittest.mock import Mock, patch, AsyncMock
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.graph import Graph
from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.workflow import Workflow
from nodetool.models.message import Message
from nodetool.types.chat import MessageCreateRequest
from nodetool.types.job import JobUpdate
from nodetool.common.environment import Environment
from nodetool.metadata.types import AssetRef


@pytest.fixture
def context():
    """Create a test ProcessingContext instance."""
    return ProcessingContext(
        user_id="test_user",
        auth_token="test_token",
        workflow_id="test_workflow"
    )


@pytest.fixture
def sample_asset():
    """Create a sample Asset for testing."""
    return Asset.create(
        user_id="test_user",
        name="test_image.png",
        content_type="image/png",
        size=1024
    )


@pytest.fixture
def sample_workflow():
    """Create a sample Workflow for testing."""
    return Workflow.create(
        user_id="test_user",
        name="Test Workflow",
        graph={"nodes": [], "edges": []}
    )


@pytest.fixture
def sample_job():
    """Create a sample Job for testing."""
    return Job.create(
        workflow_id="test_workflow",
        user_id="test_user",
        job_type="workflow",
        status="created"
    )


@pytest.fixture
def sample_message():
    """Create a sample Message for testing."""
    return Message.create(
        thread_id="test_thread",
        user_id="test_user",
        role="user",
        content="Test message"
    )


class TestProcessingContextInit:
    """Test ProcessingContext initialization and basic properties."""
    
    def test_basic_initialization(self):
        """Test basic ProcessingContext initialization."""
        ctx = ProcessingContext(user_id="user1", auth_token="token1")
        assert ctx.user_id == "user1"
        assert ctx.auth_token == "token1"
        assert ctx.workflow_id == ""
        assert isinstance(ctx.graph, Graph)
        assert ctx.variables == {}
    
    def test_initialization_with_defaults(self):
        """Test ProcessingContext initialization with default values."""
        ctx = ProcessingContext()
        assert ctx.user_id == "1"
        assert ctx.auth_token == "local_token"
        assert ctx.workflow_id == ""
    
    def test_api_client_property_removed(self, context):
        """Test that api_client property no longer exists."""
        with pytest.raises(AttributeError):
            _ = context.api_client


class TestAssetMethods:
    """Test asset-related methods in ProcessingContext."""
    
    @pytest.mark.asyncio
    async def test_find_asset_success(self, context, sample_asset):
        """Test successful asset finding."""
        result = await context.find_asset(sample_asset.id)
        assert result is not None
        assert result.id == sample_asset.id
        assert result.name == sample_asset.name
    
    @pytest.mark.asyncio
    async def test_find_asset_not_found(self, context):
        """Test asset finding when asset doesn't exist."""
        result = await context.find_asset("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_asset_wrong_user(self, sample_asset):
        """Test asset finding with wrong user_id."""
        other_context = ProcessingContext(user_id="other_user", auth_token="token")
        result = await other_context.find_asset(sample_asset.id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_find_asset_by_filename_success(self, context, sample_asset):
        """Test successful asset finding by filename."""
        result = await context.find_asset_by_filename(sample_asset.name)
        assert result is not None
        assert result.name == sample_asset.name
    
    @pytest.mark.asyncio
    async def test_find_asset_by_filename_not_found(self, context):
        """Test asset finding by filename when asset doesn't exist."""
        result = await context.find_asset_by_filename("nonexistent.txt")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_assets_non_recursive(self, context, sample_asset):
        """Test listing assets without recursion."""
        result = await context.list_assets(parent_id=context.user_id)
        assert isinstance(result, dict)
        assert "assets" in result
        assert "next" in result
        assert any(asset["id"] == sample_asset.id for asset in result["assets"])
    
    @pytest.mark.asyncio
    async def test_list_assets_recursive(self, context, sample_asset):
        """Test listing assets with recursion."""
        result = await context.list_assets(recursive=True)
        assert isinstance(result, dict)
        assert "assets" in result
        assert "next" in result
        assert result["next"] is None
    
    @pytest.mark.asyncio
    async def test_list_assets_with_content_type_filter(self, context, sample_asset):
        """Test listing assets with content type filter."""
        result = await context.list_assets(content_type="image/")
        assert isinstance(result, dict)
        assert "assets" in result
        # Should include our image asset
        asset_ids = [asset["id"] for asset in result["assets"]]
        assert sample_asset.id in asset_ids
    
    @pytest.mark.asyncio
    async def test_paginate_assets(self, context, sample_asset):
        """Test asset pagination."""
        result = await context.paginate_assets(parent_id=context.user_id, page_size=50)
        assert isinstance(result, dict)
        assert "assets" in result
        assert "next" in result
        assert len(result["assets"]) <= 50
    
    @pytest.mark.asyncio 
    async def test_get_asset_url_success(self, context, sample_asset):
        """Test getting asset URL for existing asset."""
        with patch.object(context, 'asset_storage_url', return_value="http://test.com/file.png"):
            url = await context.get_asset_url(sample_asset.id)
            assert url == "http://test.com/file.png"
    
    @pytest.mark.asyncio
    async def test_get_asset_url_not_found(self, context):
        """Test getting asset URL for non-existent asset."""
        with pytest.raises(ValueError, match="Asset with ID .* not found"):
            await context.get_asset_url("nonexistent_id")
    
    @pytest.mark.asyncio
    async def test_create_asset_success(self, context):
        """Test successful asset creation."""
        content = BytesIO(b"test content")
        
        with patch.object(Environment, 'get_asset_storage') as mock_storage:
            mock_storage_instance = Mock()
            mock_storage_instance.upload = AsyncMock()
            mock_storage.return_value = mock_storage_instance
            
            result = await context.create_asset(
                name="test.txt",
                content_type="text/plain",
                content=content
            )
            
            assert result is not None
            assert result.name == "test.txt"
            assert result.content_type == "text/plain"
            assert result.user_id == context.user_id
            assert result.size == len(b"test content")
            mock_storage_instance.upload.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_download_asset_success(self, context, sample_asset):
        """Test successful asset download."""
        test_content = b"test file content"
        
        with patch.object(Environment, 'get_asset_storage') as mock_storage:
            mock_storage_instance = Mock()
            mock_storage_instance.download = AsyncMock()
            
            async def mock_download(filename, io_obj):
                io_obj.write(test_content)
            
            mock_storage_instance.download.side_effect = mock_download
            mock_storage.return_value = mock_storage_instance
            
            result = await context.download_asset(sample_asset.id)
            
            assert result is not None
            assert result.read() == test_content
    
    @pytest.mark.asyncio
    async def test_download_asset_not_found(self, context):
        """Test asset download when asset doesn't exist."""
        with pytest.raises(ValueError, match="Asset .* not found"):
            await context.download_asset("nonexistent_id")
    
    @pytest.mark.asyncio
    async def test_refresh_uri(self, context):
        """Test refreshing asset URI."""
        asset_ref = AssetRef(asset_id="test_id")
        
        with patch.object(context, 'get_asset_url', return_value="http://test.com/file.png") as mock_get_url:
            await context.refresh_uri(asset_ref)
            mock_get_url.assert_called_once_with("test_id")
            assert asset_ref.uri == "http://test.com/file.png"


class TestWorkflowMethods:
    """Test workflow-related methods in ProcessingContext."""
    
    @pytest.mark.asyncio
    async def test_get_workflow_success(self, context, sample_workflow):
        """Test successful workflow retrieval."""
        result = await context.get_workflow(sample_workflow.id)
        assert result is not None
        assert result.id == sample_workflow.id
        assert result.name == sample_workflow.name
    
    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self, context):
        """Test workflow retrieval when workflow doesn't exist."""
        result = await context.get_workflow("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_workflow_wrong_user(self, sample_workflow):
        """Test workflow retrieval with wrong user_id."""
        other_context = ProcessingContext(user_id="other_user", auth_token="token")
        result = await other_context.get_workflow(sample_workflow.id)
        assert result is None


class TestJobMethods:
    """Test job-related methods in ProcessingContext."""
    
    @pytest.mark.asyncio
    async def test_create_job_success(self, context):
        """Test successful job creation."""
        request = RunJobRequest(
            workflow_id="test_workflow",
            graph={"nodes": [], "edges": []}  # This will be serialized to dict
        )
        
        result = await context.create_job(request)
        assert result is not None
        assert result.workflow_id == "test_workflow"
        assert result.user_id == context.user_id
        assert result.job_type == "workflow"
        assert result.status == "created"
    
    @pytest.mark.asyncio
    async def test_get_job_success(self, context, sample_job):
        """Test successful job retrieval."""
        result = await context.get_job(sample_job.id)
        assert result is not None
        assert result.id == sample_job.id
        assert result.user_id == sample_job.user_id
    
    @pytest.mark.asyncio
    async def test_get_job_not_found(self, context):
        """Test job retrieval when job doesn't exist."""
        result = await context.get_job("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_job_wrong_user(self, sample_job):
        """Test job retrieval with wrong user_id."""
        other_context = ProcessingContext(user_id="other_user", auth_token="token")
        result = await other_context.get_job(sample_job.id)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_job_success(self, context, sample_job):
        """Test successful job update."""
        update_req = JobUpdate(status="completed", message="Job finished")
        
        result = await context.update_job(sample_job.id, update_req)
        assert result is not None
        assert result.status == "completed"
        assert result.finished_at is not None
    
    @pytest.mark.asyncio
    async def test_update_job_with_error(self, context, sample_job):
        """Test job update with error status."""
        update_req = JobUpdate(status="failed", error="Test error", message="Job failed")
        
        result = await context.update_job(sample_job.id, update_req)
        assert result is not None
        assert result.status == "failed"
        assert result.error == "Test error"
        assert result.finished_at is not None
    
    @pytest.mark.asyncio
    async def test_update_job_with_cost(self, context, sample_job):
        """Test job update with cost information."""
        class MockUpdate:
            status = "completed"
            cost = 0.50
            error = None
        
        update_req = MockUpdate()
        
        result = await context.update_job(sample_job.id, update_req)
        assert result is not None
        assert result.cost == 0.50
    
    @pytest.mark.asyncio
    async def test_update_job_not_found(self, context):
        """Test job update when job doesn't exist."""
        update_req = JobUpdate(status="completed")
        
        with pytest.raises(ValueError, match="Job .* not found"):
            await context.update_job("nonexistent_id", update_req)


class TestMessageMethods:
    """Test message-related methods in ProcessingContext."""
    
    @pytest.mark.asyncio
    async def test_create_message_success(self, context):
        """Test successful message creation."""
        request = MessageCreateRequest(
            thread_id="test_thread",
            role="user",
            content="Hello world",
            tool_calls=[]
        )
        
        result = await context.create_message(request)
        assert result is not None
        assert result.thread_id == "test_thread"
        assert result.user_id == context.user_id
        assert result.role == "user"
        assert result.content == "Hello world"
    
    @pytest.mark.asyncio
    async def test_create_message_no_thread_id(self, context):
        """Test message creation without thread_id raises error."""
        request = MessageCreateRequest(
            thread_id=None,  # This should cause an error
            role="user",
            content="Hello world"
        )
        
        with pytest.raises(ValueError, match="Thread ID is required"):
            await context.create_message(request)
    
    @pytest.mark.asyncio
    async def test_get_messages_success(self, context, sample_message):
        """Test successful message retrieval."""
        result = await context.get_messages(sample_message.thread_id)
        assert isinstance(result, dict)
        assert "messages" in result
        assert "next" in result
        assert len(result["messages"]) >= 1
        
        # Find our message in the results
        message_ids = [msg["id"] for msg in result["messages"]]
        assert sample_message.id in message_ids
    
    @pytest.mark.asyncio
    async def test_get_messages_with_pagination(self, context, sample_message):
        """Test message retrieval with pagination parameters."""
        result = await context.get_messages(
            sample_message.thread_id,
            limit=5,
            reverse=True
        )
        assert isinstance(result, dict)
        assert "messages" in result
        assert "next" in result
        assert len(result["messages"]) <= 5
    
    @pytest.mark.asyncio
    async def test_get_messages_empty_thread(self, context):
        """Test message retrieval for empty thread."""
        result = await context.get_messages("empty_thread")
        assert isinstance(result, dict)
        assert "messages" in result
        assert result["messages"] == []


class TestErrorCases:
    """Test error cases and edge cases."""
    
    def test_context_variables(self, context):
        """Test context variable get/set functionality."""
        # Test default value
        assert context.get("nonexistent_key", "default") == "default"
        
        # Test setting and getting
        context.set("test_key", "test_value")
        assert context.get("test_key") == "test_value"
    
    def test_context_copy(self, context):
        """Test context copying functionality."""
        context.set("test_var", "test_value")
        
        copied = context.copy()
        assert copied.user_id == context.user_id
        assert copied.auth_token == context.auth_token
        assert copied.workflow_id == context.workflow_id
        assert copied.variables == context.variables
    
    def test_has_messages(self, context):
        """Test message queue functionality."""
        assert not context.has_messages()
        
        from nodetool.workflows.types import NodeProgress
        message = NodeProgress(node_id="test", progress=50, total=100, type="node_progress")
        context.post_message(message)
        
        assert context.has_messages()
    
    @pytest.mark.asyncio
    async def test_pop_message_async(self, context):
        """Test async message popping."""
        from nodetool.workflows.types import NodeProgress
        message = NodeProgress(node_id="test", progress=50, total=100, type="node_progress")
        context.post_message(message)
        
        # Use asyncio.wait_for to prevent hanging
        retrieved = await asyncio.wait_for(context.pop_message_async(), timeout=1.0)
        assert retrieved.node_id == "test"


class TestNodeCaching:
    """Test node caching functionality."""
    
    def test_generate_cache_key(self, context):
        """Test cache key generation."""
        from nodetool.workflows.base_node import BaseNode
        
        class TestNode(BaseNode):
            value: int = 42
        
        node = TestNode(id="test_node")
        key = context.generate_node_cache_key(node)
        
        expected = f"{context.user_id}:{TestNode.get_node_type()}:{hash(repr(node.model_dump()))}"
        assert key == expected
    
    def test_cache_and_retrieve(self, context):
        """Test caching and retrieving results."""
        from nodetool.workflows.base_node import BaseNode
        
        class TestNode(BaseNode):
            value: int = 42
            
            def outputs(self):
                from nodetool.workflows.property import Property
                from nodetool.metadata.type_metadata import TypeMetadata
                return [Property(name="output", type=TypeMetadata(type="int"))]
        
        node = TestNode(id="test_node")
        test_result = {"output": 123}
        
        # Cache should be empty initially
        assert context.get_cached_result(node) is None
        
        # Cache the result
        context.cache_result(node, test_result)
        
        # Should be able to retrieve it
        cached = context.get_cached_result(node)
        assert cached == test_result


class TestUtilityMethods:
    """Test utility methods in ProcessingContext."""
    
    def test_asset_storage_url(self, context):
        """Test asset storage URL generation."""
        with patch.object(Environment, 'get_asset_storage') as mock_storage:
            mock_storage_instance = Mock()
            mock_storage_instance.get_url.return_value = "http://test.com/file.png"
            mock_storage.return_value = mock_storage_instance
            
            url = context.asset_storage_url("test_key")
            assert url == "http://test.com/file.png"
            mock_storage_instance.get_url.assert_called_once_with("test_key")


@pytest.mark.asyncio
class TestAsyncMethods:
    """Test that methods maintain their async nature correctly."""
    
    async def test_all_database_methods_are_async(self, context):
        """Test that all database-related methods are async."""
        import inspect
        
        async_methods = [
            'find_asset', 'find_asset_by_filename', 'list_assets',
            'paginate_assets', 'get_workflow', 'create_job',
            'get_job', 'update_job', 'create_message', 'get_messages',
            'create_asset', 'download_asset'
        ]
        
        for method_name in async_methods:
            method = getattr(context, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"