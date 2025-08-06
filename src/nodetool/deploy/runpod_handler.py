# import dotenv
import json
import os
import datetime
from nodetool.types.job import JobUpdate
import runpod
from nodetool.types.graph import Graph
from nodetool.workflows.run_workflow import run_workflow
from nodetool.common.environment import Environment
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import OutputUpdate
from nodetool.deploy.admin_operations import handle_admin_operation
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.api.model import get_language_models
from nodetool.types.workflow import Workflow

"""
Universal RunPod Handler

This handler provides a unified interface for RunPod serverless operations including:
1. Workflow execution - Traditional NodeTool workflows 
2. Chat completions - OpenAI-compatible chat API
3. Admin operations - Model downloads, cache management, health checks

For local testing, run:

export NODETOOL_WORKFLOW_PATH=concat.json
python src/nodetool/deploy/runpod_handler.py --rp_serve_api
"""


# dotenv.load_dotenv()


log = Environment.get_logger()


def load_workflow(path: str) -> Workflow:
    """
    Load a workflow from a file.
    """
    with open(path, "r") as f:
        workflow = json.load(f)
    return Workflow.model_validate(workflow)




async def universal_handler(job):
    """
    Universal handler for RunPod serverless operations including workflows, chat completions, and admin tasks.
    
    This function processes:
    1. NodeTool workflow execution
    2. OpenAI-compatible chat completions 
    3. Admin operations (model downloads, cache management, health checks)

    Args:
        job (dict): RunPod job dictionary containing:
            - input (dict): Request data with keys:
                - operation (str): Admin operation type (optional)
                - openai_route (str): The endpoint route ("/v1/models" or "/v1/chat/completions") (optional)
                - openai_input (dict): The request parameters for chat (optional)
                - params (dict): Workflow parameters (optional)

    Yields:
        dict: Response data - workflow results, model list, chat completion chunks, or admin operation results
    """
    try:
        job_input = job.get("input", {})
        
        # Check if this is an admin operation
        if "operation" in job_input:
            log.info(f"Handling admin operation: {job_input.get('operation')}")
            async for chunk in handle_admin_operation(job_input):
                yield chunk
            return
        
        # Check if this is a chat completion request
        if "openai_route" in job_input:
            log.info(f"Handling chat request for route: {job_input.get('openai_route')}")
            async for chunk in _handle_chat_request(job_input):
                yield chunk
            return
        
        # Default to workflow execution
        log.info("Handling workflow execution")
        async for chunk in _handle_workflow_execution(job_input):
            yield chunk
        
    except Exception as e:
        log.error(f"Universal handler error: {str(e)}")
        yield {"error": {"message": str(e), "type": "handler_error"}}


async def _handle_chat_request(job_input):
    """Handle OpenAI-compatible chat completion requests."""
    route = job_input.get("openai_route")
    request_data = job_input.get("openai_input", {})
    
    # Get configuration from environment
    provider = os.getenv("CHAT_PROVIDER", "ollama")
    default_model = os.getenv("DEFAULT_MODEL", "gemma3n:latest")
    remote_auth = os.getenv("REMOTE_AUTH", "false").lower() == "true"
    tools_str = os.getenv("NODETOOL_TOOLS", "")
    tools = [tool.strip() for tool in tools_str.split(",") if tool.strip()] if tools_str else []
    
    # Load workflows if available
    workflows = []
    workflows_dir = "/workflows"
    if os.path.exists(workflows_dir):
        workflows = [load_workflow(os.path.join(workflows_dir, f)) 
                    for f in os.listdir(workflows_dir) if f.endswith(".json")]
    
    # Set authentication mode
    Environment.set_remote_auth(remote_auth)
    
    log.info(f"Processing chat route: {route}")
    
    if route == "/v1/models":
        # Handle models endpoint
        try:
            all_models = await get_language_models()
            filtered = [m for m in all_models if ((m.provider.value if hasattr(m.provider, 'value') else m.provider) == provider)]
            data = [
                {
                    "id": m.id or m.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider,
                }
                for m in filtered
            ]
            yield {"object": "list", "data": data}
        except Exception as e:
            log.error(f"Models endpoint error: {e}")
            yield {"error": {"message": str(e), "type": "models_error"}}
            
    elif route == "/v1/chat/completions":
        # Handle chat completions endpoint
        try:
            # Extract auth token from request data
            auth_token = None
            if "auth_token" in request_data:
                auth_token = request_data["auth_token"]
            elif "authorization" in request_data:
                auth_header = request_data["authorization"]
                if auth_header.startswith("Bearer "):
                    auth_token = auth_header[7:]
            
            # Create chat runner
            runner = ChatSSERunner(
                auth_token=auth_token,
                default_model=default_model,
                default_provider=provider,
                tools=tools,
                workflows=workflows,
            )
            
            stream = request_data.get("stream", False)
            
            if not stream:
                # Non-streaming: collect all chunks into single response
                content = ""
                async for event in runner.process_single_request(request_data):
                    if event.startswith("data: "):
                        payload = event[len("data: "):].strip()
                        if payload == "[DONE]":
                            break
                        json_payload = json.loads(payload)
                        content += json_payload["choices"][0]["delta"]["content"]
                
                if content:
                    yield {
                        "id": "chatcmpl-" + str(int(datetime.datetime.now().timestamp())),
                        "object": "chat.completion",
                        "created": int(datetime.datetime.now().timestamp() * 1000),
                        "model": request_data["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                        ]
                    }
                else:
                    yield {}
            else:
                # Streaming: yield each chunk as it comes
                async for event in runner.process_single_request(request_data):
                    yield event
        except Exception as e:
            log.error(f"Chat completions error: {e}")
            yield {"error": {"message": str(e), "type": "chat_completion_error"}}
    else:
        # Unknown route
        yield {"error": {"message": f"Unknown route: {route}", "type": "route_error"}}


async def _handle_workflow_execution(job_input):
    """Handle traditional NodeTool workflow execution."""
    req = RunJobRequest(params=job_input)
    workflow_path = "/app/workflow.json"

    if not os.path.exists(workflow_path):
        raise Exception(f"Workflow file not found at {workflow_path}")
    
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    req.graph = Graph.model_validate(workflow["graph"])

    print(req)

    # Create processing context for workflow execution
    # This context manages the workflow runtime environment
    context = ProcessingContext(
        upload_assets_to_s3=True,  # Store generated assets in S3 for persistence
    )

    # Collect all messages from workflow execution
    results = {}
    async for msg in run_workflow(req, context=context, use_thread=True):
        print(msg)
        if isinstance(msg, JobUpdate) and msg.status == "error":
            raise Exception(msg.error)
        if isinstance(msg, OutputUpdate):
            value = context.encode_assets_as_uri(msg.value)
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            results[msg.node_name] = value

    yield results


if __name__ == "__main__":
    log.info("Starting RunPod universal handler...")
    
    runpod.serverless.start(
        {
            "handler": universal_handler,
            "return_aggregate_stream": True,
        }
    )