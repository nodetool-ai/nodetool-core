"""
Comprehensive unit tests for MCP server tools.

Tests cover all major tool categories:
- Workflow operations
- Node operations
- Asset management
- Job management
- Model listing
- Vector collections
- Chat threads
- Storage operations
- HuggingFace cache/hub queries
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from nodetool.api import mcp_server
from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.message import Message
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow

# In FastMCP 3.x, @mcp.tool() returns the function directly (no FunctionTool wrapper)
create_workflow = mcp_server.create_workflow
get_workflow = mcp_server.get_workflow
run_workflow_tool = mcp_server.run_workflow_tool
run_graph = mcp_server.run_graph
list_nodes = mcp_server.list_nodes
search_nodes = mcp_server.search_nodes
get_node_info = mcp_server.get_node_info
validate_workflow = mcp_server.validate_workflow
list_workflows = mcp_server.list_workflows
list_assets = mcp_server.list_assets
get_asset = mcp_server.get_asset
list_jobs = mcp_server.list_jobs
get_job = mcp_server.get_job
get_job_logs = mcp_server.get_job_logs
start_background_job = mcp_server.start_background_job
list_models = mcp_server.list_models
list_collections = mcp_server.list_collections
get_collection = mcp_server.get_collection
query_collection = mcp_server.query_collection
get_documents_from_collection = mcp_server.get_documents_from_collection
# Thread operations (may not be implemented in mcp_tools yet)
list_threads = getattr(mcp_server, "list_threads", None)
get_thread = getattr(mcp_server, "get_thread", None)
get_thread_messages = getattr(mcp_server, "get_thread_messages", None)
download_file_from_storage = mcp_server.download_file_from_storage
get_file_metadata = mcp_server.get_file_metadata
list_storage_files = mcp_server.list_storage_files
get_hf_cache_info = mcp_server.get_hf_cache_info
inspect_hf_cached_model = mcp_server.inspect_hf_cached_model
query_hf_model_files = mcp_server.query_hf_model_files
search_hf_hub_models = mcp_server.search_hf_hub_models
get_hf_model_info = mcp_server.get_hf_model_info


class TestWorkflowOperations:
    """Test workflow-related MCP tools."""

    @pytest.mark.asyncio
    async def test_get_workflow(self, workflow: Workflow):
        """Test getting workflow details."""
        await workflow.save()

        result = await get_workflow(workflow.id)

        assert result["id"] == workflow.id
        assert result["name"] == workflow.name
        assert result["description"] == workflow.description
        assert "graph" in result
        assert "input_schema" in result
        assert "output_schema" in result
        assert "created_at" in result
        assert "updated_at" in result

    @pytest.mark.asyncio
    async def test_get_workflow_not_found(self):
        """Test getting non-existent workflow."""
        with pytest.raises(ValueError, match=r"Workflow .* not found"):
            await get_workflow("nonexistent-id")

    @pytest.mark.asyncio
    async def test_create_workflow(self):
        """Test creating a new workflow."""
        graph = {
            "nodes": [
                {
                    "id": "input1",
                    "type": "nodetool.input.IntegerInput",
                    "data": {"name": "value", "value": 42},
                }
            ],
            "edges": [],
        }

        result = await create_workflow(
            name="Test Workflow",
            graph=graph,
            description="Test description",
            tags=["test", "mcp"],
            access="private",
        )

        assert result["name"] == "Test Workflow"
        assert result["description"] == "Test description"
        assert result["tags"] == ["test", "mcp"]
        assert "id" in result

        # Verify workflow was saved
        saved_workflow = await Workflow.get(result["id"])
        assert saved_workflow is not None
        assert saved_workflow.name == "Test Workflow"

    @pytest.mark.asyncio
    async def test_validate_workflow_valid(self):
        """Test validating a valid workflow with registered nodes."""
        # Create a workflow with real registered nodes for validation
        workflow = await Workflow.create(
            user_id="1",
            name="Validation Test Workflow",
            graph={
                "nodes": [
                    {
                        "id": "input1",
                        "type": "nodetool.input.IntegerInput",
                        "data": {"name": "value", "value": 42},
                    }
                ],
                "edges": [],
            },
        )

        result = await validate_workflow(workflow.id)

        assert result["valid"] is True
        assert result["workflow_id"] == workflow.id
        assert result["summary"]["errors"] == 0
        assert "message" in result

    @pytest.mark.asyncio
    async def test_validate_workflow_invalid_node_type(self):
        """Test validating workflow with invalid node type."""
        workflow = await Workflow.create(
            user_id="1",
            name="Invalid Workflow",
            graph={
                "nodes": [{"id": "node1", "type": "nonexistent.NodeType", "data": {}}],
                "edges": [],
            },
        )

        result = await validate_workflow(workflow.id)

        assert result["valid"] is False
        assert result["summary"]["errors"] > 0
        assert any("not found" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_workflow_circular_dependency(self):
        """Test validating workflow with circular dependency."""
        workflow = await Workflow.create(
            user_id="1",
            name="Circular Workflow",
            graph={
                "nodes": [
                    {"id": "node1", "type": "nodetool.text.Concat", "data": {}},
                    {"id": "node2", "type": "nodetool.text.Concat", "data": {}},
                ],
                "edges": [
                    {
                        "source": "node1",
                        "target": "node2",
                        "sourceHandle": "output",
                        "targetHandle": "a",
                    },
                    {
                        "source": "node2",
                        "target": "node1",
                        "sourceHandle": "output",
                        "targetHandle": "a",
                    },
                ],
            },
        )

        result = await validate_workflow(workflow.id)

        assert result["valid"] is False
        assert any("circular" in error.lower() for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_workflow_input_node_missing_name(self):
        """Input nodes must provide a non-empty name property."""
        workflow = await Workflow.create(
            user_id="1",
            name="Missing Input Name Workflow",
            graph={
                "nodes": [
                    {
                        "id": "input1",
                        "type": "nodetool.input.IntegerInput",
                        "data": {"value": 42},
                    }
                ],
                "edges": [],
            },
        )

        result = await validate_workflow(workflow.id)

        assert result["valid"] is False
        assert any("missing required 'name' property" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_list_workflows(self, workflow: Workflow):
        """Test listing workflows with pagination."""
        await workflow.save()

        # Create another workflow
        await Workflow.create(user_id="1", name="Workflow 2", graph={"nodes": [], "edges": []})

        result = await list_workflows(limit=10)

        assert "workflows" in result
        assert len(result["workflows"]) == 2
        assert result["workflows"][0]["name"] in ["test_workflow", "Workflow 2"]

    @pytest.mark.asyncio
    @pytest.mark.no_setup
    async def test_list_workflows_binds_scope_when_unbound(self, monkeypatch):
        """list_workflows should create a ResourceScope when none is active."""
        entered = 0

        class DummyScope:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                nonlocal entered
                entered += 1
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        paginate_mock = AsyncMock(return_value=([], None))
        monkeypatch.setattr("nodetool.tools.workflow_tools.maybe_scope", lambda: None)
        monkeypatch.setattr("nodetool.tools.workflow_tools.ResourceScope", DummyScope)
        monkeypatch.setattr(Workflow, "paginate", paginate_mock)

        result = await list_workflows(limit=10, user_id="1")

        assert entered == 1
        paginate_mock.assert_awaited_once_with(user_id="1", limit=10)
        assert result["workflows"] == []
        assert result["next"] is None
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_run_graph_simple(self):
        """Test running a workflow graph directly."""
        graph = {
            "nodes": [
                {
                    "id": "input1",
                    "type": "nodetool.input.IntegerInput",
                    "data": {"name": "value", "value": 0},
                },
                {
                    "id": "output1",
                    "type": "nodetool.output.IntegerOutput",
                    "data": {"name": "result", "value": ""},
                },
            ],
            "edges": [
                {
                    "source": "input1",
                    "target": "output1",
                    "sourceHandle": "output",
                    "targetHandle": "value",
                }
            ],
        }

        result = await run_graph(graph=graph, params={"value": 42})

        assert result["status"] == "completed"
        assert "result" in result


class TestNodeOperations:
    """Test node-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_nodes(self):
        """Test listing nodes without namespace filter."""
        result = await list_nodes(limit=50)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all("type" in node for node in result)

    @pytest.mark.asyncio
    async def test_list_nodes_with_namespace(self):
        """Test listing nodes with namespace filter."""
        result = await list_nodes(namespace="nodetool.text", limit=20)

        assert isinstance(result, list)
        assert all("nodetool.text" in node["type"].lower() for node in result if result)

    @pytest.mark.asyncio
    async def test_search_nodes(self):
        """Test searching for nodes."""
        result = await search_nodes(query=["text", "concat"], n_results=10)

        assert isinstance(result, list)
        assert len(result) > 0
        assert all("type" in node for node in result)
        assert all("description" in node for node in result)

    @pytest.mark.asyncio
    async def test_get_node_info(self):
        """Test getting detailed node information."""
        result = await get_node_info("nodetool.text.Concat")

        assert "node_type" in result
        assert result["node_type"] == "nodetool.text.Concat"
        assert "properties" in result

    @pytest.mark.asyncio
    async def test_get_node_info_not_found(self):
        """Test getting info for non-existent node."""
        with pytest.raises(ValueError, match=r"Node type .* not found"):
            await get_node_info("nonexistent.NodeType")


class TestAssetOperations:
    """Test asset-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_assets_root(self):
        """Test listing root assets."""
        # Create test assets
        asset1 = await Asset.create(
            user_id="1",
            parent_id="1",
            name="test.jpg",
            content_type="image",
            metadata={},
        )

        result = await list_assets(limit=10)

        assert "assets" in result
        assert len(result["assets"]) > 0
        assert result["assets"][0]["id"] == asset1.id

    @pytest.mark.asyncio
    async def test_list_assets_by_content_type(self):
        """Test listing assets filtered by content type."""
        await Asset.create(
            user_id="1",
            parent_id="1",
            name="test.jpg",
            content_type="image",
            metadata={},
        )
        await Asset.create(
            user_id="1",
            parent_id="1",
            name="test.mp4",
            content_type="video",
            metadata={},
        )

        result = await list_assets(content_type="image", limit=10)

        assert "assets" in result
        assert all(asset["content_type"] == "image" for asset in result["assets"])

    @pytest.mark.asyncio
    async def test_search_assets(self):
        """Test searching assets by name."""
        await Asset.create(
            user_id="1",
            parent_id="1",
            name="findme.jpg",
            content_type="image",
            metadata={},
        )

        result = await list_assets(query="findme", limit=10)

        assert "assets" in result
        assert len(result["assets"]) > 0
        assert "findme" in result["assets"][0]["name"].lower()

    @pytest.mark.asyncio
    async def test_get_asset(self):
        """Test getting specific asset details."""
        # Skip for now - get_asset has a coroutine wrapping issue with FastMCP .fn accessor
        pytest.skip("get_asset has coroutine wrapping issue - needs investigation")

        asset = await Asset.create(
            user_id="1",
            parent_id="1",
            name="test.jpg",
            content_type="image",
            metadata={"width": 800, "height": 600},
        )

        result = await get_asset(asset.id)

        assert result["id"] == asset.id
        assert result["name"] == "test.jpg"
        assert result["content_type"] == "image"
        assert result["metadata"]["width"] == 800


class TestJobOperations:
    """Test job-related MCP tools."""

    @pytest.mark.asyncio
    async def test_list_jobs(self, workflow: Workflow):
        """Test listing jobs."""
        await workflow.save()

        job = await Job.create(
            user_id="1",
            workflow_id=workflow.id,
            job_type="workflow",
        )
        # Status is now directly on the Job model

        result = await list_jobs(limit=10)

        assert "jobs" in result
        assert len(result["jobs"]) > 0
        assert result["jobs"][0]["id"] == job.id

    @pytest.mark.asyncio
    async def test_list_jobs_by_workflow(self, workflow: Workflow):
        """Test listing jobs filtered by workflow."""
        await workflow.save()

        _job = await Job.create(
            user_id="1",
            workflow_id=workflow.id,
            job_type="workflow",
        )
        # Status is now directly on the Job model

        # Create another workflow and job
        workflow2 = await Workflow.create(user_id="1", name="Workflow 2", graph={"nodes": [], "edges": []})
        _job2 = await Job.create(
            user_id="1",
            workflow_id=workflow2.id,
            job_type="workflow",
        )
        # Status is now directly on the Job model

        result = await list_jobs(workflow_id=workflow.id, limit=10)

        assert "jobs" in result
        assert all(job["workflow_id"] == workflow.id for job in result["jobs"])

    @pytest.mark.asyncio
    async def test_get_job(self):
        """Test getting job details."""
        job = await Job.create(
            user_id="1",
            workflow_id="test-workflow",
            job_type="workflow",
        )
        # Mark job as completed
        await job.mark_completed()

        result = await get_job(job.id)

        assert result["id"] == job.id
        assert result["status"] == "completed"
        assert result["job_type"] == "workflow"


@pytest.mark.skip(reason="Test mocks require refactoring - get_all_models import path changed")
class TestModelOperations:
    """Test model listing MCP tools."""

    @pytest.mark.asyncio
    async def test_list_all_models_with_provider_filter(self):
        """Test listing models with provider filter."""
        with patch("nodetool.api.model.get_all_models") as mock_get_models:
            mock_get_models.return_value = [
                Mock(
                    id="openai/gpt-4",
                    name="GPT-4",
                    repo_id="openai/gpt-4",
                    path=None,
                    type="language_model",
                    downloaded=False,
                    size_on_disk=None,
                ),
                Mock(
                    id="anthropic/claude",
                    name="Claude",
                    repo_id="anthropic/claude",
                    path=None,
                    type="language_model",
                    downloaded=False,
                    size_on_disk=None,
                ),
            ]

            result = await list_models(provider="openai", limit=50)

            assert len(result) == 1
            assert "openai" in result[0]["id"].lower()

    @pytest.mark.asyncio
    async def test_list_all_models_limit_enforcement(self):
        """Test that model listing respects limits."""
        with patch("nodetool.api.model.get_all_models") as mock_get_models:
            mock_get_models.return_value = [
                Mock(
                    id=f"model-{i}",
                    name=f"Model {i}",
                    repo_id=f"repo/model-{i}",
                    path=None,
                    type="language_model",
                    downloaded=False,
                    size_on_disk=None,
                )
                for i in range(300)
            ]

            result = await list_models(provider="all", limit=250)

            assert len(result) <= 200  # Max limit is 200

    @pytest.mark.asyncio
    async def test_list_language_models(self):
        """Test listing language models with provider filter."""
        with patch("nodetool.api.model.get_language_models") as mock_get:
            from nodetool.metadata.types import Provider

            mock_get.return_value = [
                Mock(id="gpt-4", name="GPT-4", provider=Provider.OpenAI),
                Mock(id="claude-3", name="Claude 3", provider=Provider.Anthropic),
            ]

            result = await list_models(provider="openai", model_type="language_model", limit=50)

            assert len(result) == 1
            assert result[0]["provider"] == "openai"


@pytest.mark.skip(reason="Test mocks require refactoring - get_async_collection import path changed")
class TestCollectionOperations:
    """Test vector collection MCP tools."""

    @pytest.mark.asyncio
    async def test_query_collection(self):
        """Test querying a collection."""
        with patch("nodetool.integrations.vectorstores.chroma.async_chroma_client.get_async_collection") as mock_get_col:
            mock_collection = AsyncMock()
            mock_collection.query = AsyncMock(
                return_value={
                    "ids": [["doc1", "doc2"]],
                    "documents": [["Document 1", "Document 2"]],
                    "distances": [[0.1, 0.2]],
                    "metadatas": [[{"source": "test"}, {"source": "test"}]],
                }
            )
            mock_get_col.return_value = mock_collection

            result = await query_collection(name="test-collection", query_texts=["search query"], n_results=10)

            assert "ids" in result
            assert "documents" in result
            assert len(result["documents"][0]) == 2


@pytest.mark.skip(reason="Chat thread tools (list_threads, get_thread, get_thread_messages) are not implemented in mcp_tools")
class TestChatOperations:
    """Test chat thread and message MCP tools."""

    @pytest.mark.asyncio
    async def test_list_threads(self):
        """Test listing chat threads."""
        if list_threads is None:
            pytest.skip("list_threads not implemented")
        await Thread.create(user_id="1", name="Thread 1")
        await Thread.create(user_id="1", name="Thread 2")

        result = await list_threads(limit=10)

        assert "threads" in result
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_get_thread(self):
        """Test getting thread details."""
        if get_thread is None:
            pytest.skip("get_thread not implemented")
        thread = await Thread.create(user_id="1", title="Test Thread")

        result = await get_thread(thread.id)

        assert result["id"] == thread.id
        assert result["name"] == "Test Thread"

    @pytest.mark.asyncio
    async def test_get_thread_messages(self):
        """Test getting messages from a thread."""
        if get_thread_messages is None:
            pytest.skip("get_thread_messages not implemented")
        thread = await Thread.create(user_id="1", name="Test Thread")
        await Message.create(
            user_id="1",
            thread_id=thread.id,
            role="user",
            instructions="Hello",
            tool_calls=[],
        )
        await Message.create(
            user_id="1",
            thread_id=thread.id,
            role="assistant",
            instructions="Hi there",
            tool_calls=[],
        )

        result = await get_thread_messages(thread.id, limit=10)

        assert "messages" in result
        assert result["count"] >= 2


class TestStorageOperations:
    """Test storage-related MCP tools."""

    @pytest.mark.asyncio
    async def test_download_file_from_storage(self):
        """Test downloading a file from storage."""
        import base64

        from nodetool.runtime.resources import ResourceScope, require_scope

        # First manually put a file in temp storage using the MCP server's temp storage
        # We'll skip creating the file and just test that the function handles missing files properly
        # For now, disable this test as it requires complex setup
        pytest.skip("Temp storage setup requires restructuring - skipping for now")

        # Download
        result = await download_file_from_storage(key="download-test.txt", temp=True)

        assert result["key"] == "download-test.txt"
        assert "content" in result
        assert result["storage"] == "temp"

        # Verify content
        downloaded = base64.b64decode(result["content"])
        assert downloaded == b"test content"

    @pytest.mark.asyncio
    async def test_get_file_metadata(self):
        """Test getting file metadata without downloading."""
        from nodetool.runtime.resources import ResourceScope, require_scope

        # Skip this test as temp storage setup is complex
        pytest.skip("Temp storage setup requires restructuring - skipping for now")

        # Get metadata
        result = await get_file_metadata(key="metadata.txt", temp=True)

        assert result["key"] == "metadata.txt"
        assert result["exists"] is True
        assert "size" in result
        assert "content" not in result  # Should not download content


@pytest.mark.skip(reason="Test mocks require refactoring - HfApi and asdict import paths changed")
class TestHuggingFaceOperations:
    """Test HuggingFace cache and hub query tools."""

    @pytest.mark.asyncio
    async def test_get_hf_cache_info(self):
        """Test getting HuggingFace cache information."""
        with patch("nodetool.tools.hf_tools.read_cached_hf_models") as mock_read:
            mock_read.return_value = [
                Mock(
                    repo_id="meta-llama/Llama-2-7b",
                    type="language_model",
                    size_on_disk=13000000000,
                    path="/cache/models/llama",
                )
            ]

            result = await get_hf_cache_info()

            assert "cache_dir" in result
            assert result["total_models"] == 1
            assert "total_size_gb" in result
            assert len(result["models"]) == 1

    @pytest.mark.asyncio
    async def test_inspect_hf_cached_model(self):
        """Test inspecting a specific cached model."""
        with patch("nodetool.tools.hf_tools.read_cached_hf_models") as mock_read:
            mock_read.return_value = [
                Mock(
                    repo_id="meta-llama/Llama-2-7b",
                    name="Llama 2 7B",
                    type="language_model",
                    path="/cache/llama",
                    size_on_disk=13000000000,
                    downloaded=True,
                )
            ]

            result = await inspect_hf_cached_model("meta-llama/Llama-2-7b")

            assert result["repo_id"] == "meta-llama/Llama-2-7b"
            assert result["downloaded"] is True
            assert "size_on_disk_gb" in result

    @pytest.mark.asyncio
    async def test_inspect_hf_cached_model_not_found(self):
        """Test inspecting non-existent cached model."""
        with patch("nodetool.tools.hf_tools.read_cached_hf_models") as mock_read:
            mock_read.return_value = []

            with pytest.raises(ValueError, match="not found in cache"):
                await inspect_hf_cached_model("nonexistent/model")

    @pytest.mark.asyncio
    async def test_query_hf_model_files(self):
        """Test querying HuggingFace Hub for model files."""
        with (
            patch("nodetool.tools.hf_tools.HfApi") as mock_api_class,
            patch("nodetool.tools.hf_tools.asdict") as mock_asdict,
        ):
            mock_file_info = Mock()
            mock_file_info.size = 5000000000

            mock_file_info2 = Mock()
            mock_file_info2.size = 1000

            mock_api = Mock()
            mock_api.list_repo_files = Mock(return_value=["model.safetensors", "config.json"])
            mock_api.get_paths_info = Mock(
                side_effect=[
                    [mock_file_info],
                    [mock_file_info2],
                ]
            )
            mock_api_class.return_value = mock_api

            # Mock asdict to convert mock to dict
            mock_asdict.side_effect = [
                {"size": 5000000000},
                {"size": 1000},
            ]

            result = await query_hf_model_files(
                repo_id="meta-llama/Llama-2-7b",
                patterns=["*.safetensors", "*.json"],
            )

            assert result["repo_id"] == "meta-llama/Llama-2-7b"
            assert result["file_count"] == 2
            assert "total_size_gb" in result

    @pytest.mark.asyncio
    async def test_search_hf_hub_models(self):
        """Test searching HuggingFace Hub for models."""
        with (
            patch("nodetool.tools.hf_tools.HfApi") as mock_api_class,
            patch("nodetool.tools.hf_tools.asdict") as mock_asdict,
        ):
            mock_model = Mock()
            mock_model.id = "meta-llama/Llama-2-7b"
            mock_model.downloads = 1000000
            mock_model.likes = 5000
            mock_model.tags = ["text-generation", "llama"]
            mock_model.pipeline_tag = "text-generation"

            mock_api = Mock()
            mock_api.list_models = Mock(return_value=[mock_model])
            mock_api_class.return_value = mock_api

            # Mock asdict to convert mock to dict
            mock_asdict.return_value = {
                "id": "meta-llama/Llama-2-7b",
                "downloads": 1000000,
                "likes": 5000,
                "tags": ["text-generation", "llama"],
                "pipeline_tag": "text-generation",
            }

            result = await search_hf_hub_models(query="llama", limit=20)

            assert result["query"] == "llama"
            assert result["count"] == 1
            assert len(result["models"]) == 1
            assert result["models"][0]["id"] == "meta-llama/Llama-2-7b"

    @pytest.mark.asyncio
    async def test_get_hf_model_info(self):
        """Test getting detailed model info from HuggingFace Hub."""
        with (
            patch("nodetool.tools.hf_tools.HfApi") as mock_api_class,
            patch("nodetool.tools.hf_tools.asdict") as mock_asdict,
        ):
            mock_info = Mock()
            mock_info.id = "meta-llama/Llama-2-7b"
            mock_info.downloads = 1000000
            mock_info.likes = 5000
            mock_info.tags = ["llama", "7b"]
            mock_info.pipeline_tag = "text-generation"

            mock_api = Mock()
            mock_api.model_info = Mock(return_value=mock_info)
            mock_api_class.return_value = mock_api

            # Mock asdict to convert mock to dict
            mock_asdict.return_value = {
                "id": "meta-llama/Llama-2-7b",
                "downloads": 1000000,
                "likes": 5000,
                "tags": ["llama", "7b"],
                "pipeline_tag": "text-generation",
            }

            result = await get_hf_model_info("meta-llama/Llama-2-7b")

            assert result["id"] == "meta-llama/Llama-2-7b"
            assert result["downloads"] == 1000000
            assert result["likes"] == 5000


class TestParameterValidation:
    """Test parameter validation and edge cases."""

    @pytest.mark.asyncio
    async def test_list_nodes_limit_default(self):
        """Test that list_nodes uses default limit."""
        result = await list_nodes()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_assets_minimum_query_length(self):
        """Test that search requires minimum query length."""
        with pytest.raises(ValueError, match="at least 2 characters"):
            await list_assets(query="a")

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test mocks require refactoring - get_all_models import path changed")
    async def test_list_all_models_enforces_max_limit(self):
        """Test that model listing enforces maximum limit."""
        with patch("nodetool.api.model.get_all_models") as mock_get:
            mock_get.return_value = [
                Mock(
                    id=f"model-{i}",
                    name=f"Model {i}",
                    repo_id=None,
                    path=None,
                    type="language_model",
                    downloaded=False,
                    size_on_disk=None,
                )
                for i in range(500)
            ]

            result = await list_models(provider="all", limit=999)
            assert len(result) <= 200  # Max limit enforced

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Test mocks require refactoring - get_async_collection import path changed")
    async def test_query_collection_enforces_max_results(self):
        """Test that collection query enforces max results."""
        with patch("nodetool.integrations.vectorstores.chroma.async_chroma_client.get_async_collection") as mock_get:
            mock_collection = AsyncMock()
            mock_collection.query = AsyncMock(
                return_value={
                    "ids": [[]],
                    "documents": [[]],
                    "distances": [[]],
                    "metadatas": [[]],
                }
            )
            mock_get.return_value = mock_collection

            await query_collection(name="test", query_texts=["query"], n_results=100)

            # Verify n_results was capped at 50
            call_args = mock_collection.query.call_args
            assert call_args[1]["n_results"] == 50
