"""
FastAPI Server for NodeTool Operations

This server provides REST API endpoints for:
1. OpenAI-compatible chat completions with SSE streaming (/v1/chat/completions)
2. Model listing (/v1/models)
3. Workflow execution (/workflows/*)
4. Admin operations (/admin/*)

Migrated from runpod_handler.py to provide a standard HTTP API interface.
"""

import os
import json
import datetime
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from rich.console import Console

from nodetool.common.environment import Environment
from nodetool.types.job import JobUpdate
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import OutputUpdate
from nodetool.deploy.admin_operations import (
    download_hf_model,
    download_ollama_model,
    scan_hf_cache,
    calculate_cache_size,
    delete_hf_model,
)
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.api.model import get_language_models
from nodetool.types.workflow import Workflow


console = Console()
log = Environment.get_logger()

# Global workflow registry
_workflow_registry: Dict[str, Workflow] = {}


def load_workflow(path: str) -> Workflow:
    """Load a workflow from a file."""
    with open(path, "r") as f:
        workflow = json.load(f)
    return Workflow.model_validate(workflow)


def load_workflows_from_directory(workflows_dir: str = "/app/workflows") -> Dict[str, Workflow]:
    """Load all workflow JSON files from the specified directory."""
    workflows = {}
    
    if not os.path.exists(workflows_dir):
        log.warning(f"Workflows directory not found: {workflows_dir}")
        return workflows
    
    for filename in os.listdir(workflows_dir):
        if not filename.endswith(".json"):
            continue
        
        filepath = os.path.join(workflows_dir, filename)
        try:
            workflow = load_workflow(filepath)
            workflow_id = (
                workflow.id
                if hasattr(workflow, "id") and workflow.id
                else filename[:-5]
            )
            workflows[workflow_id] = workflow
            log.info(f"Loaded workflow '{workflow_id}' from {filename}")
        except Exception as e:
            log.error(f"Failed to load workflow from {filename}: {str(e)}")
    
    return workflows


def initialize_workflow_registry():
    """Initialize the global workflow registry."""
    global _workflow_registry
    _workflow_registry = load_workflows_from_directory()
    log.info(f"Initialized workflow registry with {len(_workflow_registry)} workflows")


def get_workflow_by_id(workflow_id: str) -> Workflow:
    """Get a workflow by its ID from the registry."""
    if workflow_id not in _workflow_registry:
        raise ValueError(
            f"Workflow '{workflow_id}' not found. Available workflows: {list(_workflow_registry.keys())}"
        )
    return _workflow_registry[workflow_id]


def create_nodetool_server(
    remote_auth: bool = False,
    provider: str = "ollama", 
    default_model: str = "gemma3n:latest",
    tools: List[str] = [],
    workflows: List[Workflow] = [],
) -> FastAPI:
    """Create a FastAPI server instance for NodeTool operations.
    
    Args:
        remote_auth: Whether to use remote authentication
        provider: Default provider to use
        default_model: Default model to use when not specified by client
        tools: List of tool names to enable
        workflows: List of workflows to make available
        
    Returns:
        FastAPI application instance
    """
    # Set authentication mode
    Environment.set_remote_auth(remote_auth)
    
    app = FastAPI(
        title="NodeTool API Server", 
        version="1.0.0",
        description="FastAPI server for NodeTool operations including chat completions, workflows, and admin tasks"
    )

    @app.on_event("startup")
    async def startup_event():
        """Initialize workflow registry on startup."""
        initialize_workflow_registry()
        console.print("NodeTool server started successfully")

    @app.on_event("shutdown") 
    async def shutdown_event():
        console.print("NodeTool server shutting down...")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}

    # OpenAI-compatible models endpoint
    @app.get("/v1/models")
    async def openai_models():
        """Returns list of models filtered by provider in OpenAI format."""
        try:
            all_models = await get_language_models()
            filtered = [
                m for m in all_models
                if (m.provider.value if hasattr(m.provider, "value") else m.provider) == provider
            ]
            data = [
                {
                    "id": m.id or m.name,
                    "object": "model", 
                    "created": 0,
                    "owned_by": provider,
                }
                for m in filtered
            ]
            return {"object": "list", "data": data}
        except Exception as e:
            console.print(f"OpenAI Models error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # OpenAI-compatible chat completions endpoint  
    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint with SSE streaming support."""
        try:
            data = await request.json()
            auth_header = request.headers.get("authorization", "")
            auth_token = (
                auth_header.replace("Bearer ", "")
                if auth_header.startswith("Bearer ")
                else None
            )
            if auth_token:
                data["auth_token"] = auth_token

            # Load workflows if available
            chat_workflows = []
            workflows_dir = "/workflows"
            if os.path.exists(workflows_dir):
                chat_workflows = [
                    load_workflow(os.path.join(workflows_dir, f))
                    for f in os.listdir(workflows_dir)
                    if f.endswith(".json")
                ]

            runner = ChatSSERunner(
                auth_token,
                default_model=default_model,
                default_provider=provider,
                tools=tools,
                workflows=chat_workflows + workflows,
            )

            # Determine if streaming is requested (default true)
            stream = data.get("stream", True)
            if not stream:
                # Non-streaming: collect chunks into single response
                chunks = []
                async for event in runner.process_single_request(data):
                    if event.startswith("data: "):
                        payload = event[len("data: "):].strip()
                        if payload == "[DONE]":
                            break
                        chunks.append(payload)
                if chunks:
                    return json.loads(chunks[-1])
                return {}
            else:
                # Streaming response
                return StreamingResponse(
                    runner.process_single_request(data),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Authorization, Content-Type",
                        "Access-Control-Allow-Methods": "POST, OPTIONS",
                    },
                )
        except Exception as e:
            console.print(f"OpenAI Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Workflow management endpoints
    @app.get("/workflows")
    async def list_workflows():
        """List available workflows."""
        return {
            "workflows": [
                {"id": workflow_id, "name": workflow.name if hasattr(workflow, "name") else workflow_id}
                for workflow_id, workflow in _workflow_registry.items()
            ]
        }

    @app.post("/workflows/execute")
    async def execute_workflow(request: Request):
        """Execute a workflow by ID."""
        try:
            data = await request.json()
            workflow_id = data.get("workflow_id")
            params = data.get("params", {})
            
            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id is required")
                
            workflow = get_workflow_by_id(workflow_id)
            req = RunJobRequest(params=params)
            req.graph = workflow.graph
            
            # Create processing context
            context = ProcessingContext(upload_assets_to_s3=True)
            
            # Collect results
            results = {}
            async for msg in run_workflow(req, context=context, use_thread=True):
                if isinstance(msg, JobUpdate) and msg.status == "error":
                    raise HTTPException(status_code=500, detail=msg.error)
                if isinstance(msg, OutputUpdate):
                    value = context.encode_assets_as_uri(msg.value)
                    if hasattr(value, "model_dump"):
                        value = value.model_dump()
                    results[msg.node_name] = value
            
            return {"results": results}
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            console.print(f"Workflow execution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/workflows/execute/stream")
    async def execute_workflow_stream(request: Request):
        """Execute a workflow with streaming updates via SSE."""
        try:
            data = await request.json()
            workflow_id = data.get("workflow_id")
            params = data.get("params", {})
            
            if not workflow_id:
                raise HTTPException(status_code=400, detail="workflow_id is required")
                
            workflow = get_workflow_by_id(workflow_id)
            req = RunJobRequest(params=params)
            req.graph = workflow.graph
            
            # Create processing context
            context = ProcessingContext(upload_assets_to_s3=True)
            
            async def generate_sse():
                """Generate SSE events for workflow execution."""
                results = {}
                try:
                    async for msg in run_workflow(req, context=context, use_thread=True):
                        if isinstance(msg, JobUpdate):
                            event_data = {"type": "job_update", "data": msg.model_dump()}
                            yield f"data: {json.dumps(event_data)}\n\n"
                            if msg.status == "error":
                                error_data = {"type": "error", "error": msg.error}
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return
                        elif isinstance(msg, OutputUpdate):
                            value = context.encode_assets_as_uri(msg.value)
                            if hasattr(value, "model_dump"):
                                value = value.model_dump()
                            results[msg.node_name] = value
                            event_data = {
                                "type": "output_update", 
                                "node_name": msg.node_name, 
                                "value": value
                            }
                            yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Send final results
                    final_data = {"type": "complete", "results": results}
                    yield f"data: {json.dumps(final_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    error_data = {"type": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            console.print(f"Workflow streaming error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Individual Admin Endpoints
    @app.get("/ping")
    async def ping():
        """Health check with system information."""
        try:
            return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}
        except Exception as e:
            console.print(f"Health check error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/models/huggingface/download")
    async def download_huggingface_model(request: Request):
        """Download HuggingFace model with optional streaming progress."""
        try:
            data = await request.json()
            repo_id = data.get("repo_id")
            
            if not repo_id:
                raise HTTPException(status_code=400, detail="repo_id is required")
            
            # Always use streaming for HF downloads
            async def generate_sse():
                try:
                    async for chunk in download_hf_model(
                        repo_id=repo_id,
                        cache_dir=data.get("cache_dir", "/app/.cache/huggingface/hub"),
                        file_path=data.get("file_path"),
                        ignore_patterns=data.get("ignore_patterns"),
                        allow_patterns=data.get("allow_patterns"),
                        stream=data.get("stream", True)
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive", 
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            console.print(f"HuggingFace download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/models/ollama/download")
    async def download_ollama_model_endpoint(request: Request):
        """Download Ollama model with optional streaming progress."""
        try:
            data = await request.json()
            model_name = data.get("model_name")
            
            if not model_name:
                raise HTTPException(status_code=400, detail="model_name is required")
            
            # Always use streaming for Ollama downloads
            async def generate_sse():
                try:
                    async for chunk in download_ollama_model(
                        model_name=model_name,
                        stream=data.get("stream", True)
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_data = {"status": "error", "error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*", 
                    "Access-Control-Allow-Headers": "Authorization, Content-Type",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            console.print(f"Ollama download error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/cache/scan")
    async def scan_cache():
        """Scan HuggingFace cache directory."""
        try:
            results = []
            async for chunk in scan_hf_cache():
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "No cache data"}
        except Exception as e:
            console.print(f"Cache scan error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/cache/size")
    async def get_cache_size(cache_dir: str = "/app/.cache/huggingface/hub"):
        """Calculate total cache size."""
        try:
            results = []
            async for chunk in calculate_cache_size(cache_dir=cache_dir):
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "No size data"}
        except Exception as e:
            console.print(f"Cache size calculation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/admin/models/huggingface/{repo_id:path}")
    async def delete_huggingface_model(repo_id: str):
        """Delete HuggingFace model from cache."""
        try:
            results = []
            async for chunk in delete_hf_model(repo_id=repo_id):
                results.append(chunk)
            return results[0] if results else {"status": "error", "message": "Delete failed"}
        except Exception as e:
            console.print(f"HuggingFace model deletion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/admin/workflows/status")
    async def admin_workflows_status():
        """Admin workflow status with registry info."""
        return {
            "status": "healthy",
            "workflow_count": len(_workflow_registry),
            "available_workflows": list(_workflow_registry.keys()),
            "timestamp": datetime.datetime.now().isoformat()
        }

    return app


def run_nodetool_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    remote_auth: bool = False,
    provider: str = "ollama", 
    default_model: str = "gemma3n:latest",
    tools: List[str] = [],
    workflows: List[Workflow] = [],
):
    """Run the NodeTool server.
    
    Args:
        host: Host address to serve on
        port: Port to serve on  
        remote_auth: Whether to use remote authentication
        provider: Default provider to use
        default_model: Default model to use when not specified by client
        tools: List of tool names to enable
        workflows: List of workflows to make available
    """
    import dotenv
    dotenv.load_dotenv()
    
    app = create_nodetool_server(remote_auth, provider, default_model, tools, workflows)
    
    console.print(f"🚀 Starting NodeTool server on {host}:{port}")
    console.print(f"Chat completions endpoint: http://{host}:{port}/v1/chat/completions")
    console.print(f"Models endpoint: http://{host}:{port}/v1/models")
    console.print(f"Workflows endpoint: http://{host}:{port}/workflows")
    console.print("Admin endpoints:")
    console.print(f"  - Health check: http://{host}:{port}/admin/health")
    console.print(f"  - HuggingFace download: http://{host}:{port}/admin/models/huggingface/download")
    console.print(f"  - Ollama download: http://{host}:{port}/admin/models/ollama/download")
    console.print(f"  - Cache scan: http://{host}:{port}/admin/cache/scan")
    console.print(f"  - Cache size: http://{host}:{port}/admin/cache/size")
    console.print(f"  - Delete HF model: http://{host}:{port}/admin/models/huggingface/{{repo_id}}")
    console.print(f"  - Workflow status: http://{host}:{port}/admin/workflows/status")
    console.print(
        "Authentication mode:",
        "Remote (Supabase)" if remote_auth else "Local (user_id=1)",
    )
    console.print("Default model:", f"{default_model} (provider: {provider})")
    console.print("Tools:", tools)
    console.print("Workflows:", [w.name for w in workflows])
    console.print("\\nSend requests with Authorization: Bearer YOUR_TOKEN header")
    
    # Run the server
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        console.print("\\n👋 NodeTool server stopped by user")  
    except Exception as e:
        console.print(f"❌ Server error: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    # Get configuration from environment
    provider = os.getenv("CHAT_PROVIDER", "ollama")
    default_model = os.getenv("DEFAULT_MODEL", "gemma3n:latest")
    remote_auth = os.getenv("REMOTE_AUTH", "false").lower() == "true"
    tools_str = os.getenv("NODETOOL_TOOLS", "")
    tools = (
        [tool.strip() for tool in tools_str.split(",") if tool.strip()]
        if tools_str
        else []
    )
    
    run_nodetool_server(
        remote_auth=remote_auth,
        provider=provider,
        default_model=default_model,
        tools=tools
    )