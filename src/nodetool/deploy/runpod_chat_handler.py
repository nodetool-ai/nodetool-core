"""
RunPod Chat Handler

This module provides a RunPod serverless handler for OpenAI-compatible chat completions.
It receives requests through RunPod's serverless infrastructure and processes them using
the same chat functionality as the FastAPI server.

The handler supports:
- /v1/models endpoint: Returns available models for the configured provider
- /v1/chat/completions endpoint: Processes chat completion requests with streaming support

Usage:
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
    })

Environment Variables:
    - CHAT_PROVIDER: The AI provider to use (default: "ollama")
    - DEFAULT_MODEL: Default model to use (default: "gemma3n:latest")
    - REMOTE_AUTH: Enable remote authentication (default: False)
    - USE_DATABASE: Enable database storage (default: False)
"""

import json
import os
import runpod
import datetime
from typing import Dict, Any
from pathlib import Path
from nodetool.common.environment import Environment
from nodetool.chat.chat_sse_runner import ChatSSERunner
from nodetool.api.model import get_language_models
from nodetool.deploy.download_models import download_models_from_spec


log = Environment.get_logger()


def download_models_on_startup() -> None:
    """
    Download missing models on container startup.
    
    Reads model specifications from /app/models.json and downloads
    any models that are not available in the network volume.
    """
    models_file = "/app/models.json"
    
    if not os.path.exists(models_file):
        log.info("No models.json file found, skipping model downloads")
        return
    
    try:
        with open(models_file, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.error(f"Failed to read models file: {e}")
        return
    
    # Use consolidated download function
    hf_cache_dir = "/runpod-volume/.cache/huggingface/hub"
    download_models_from_spec(models, hf_cache_dir, log)


async def chat_handler(job):
    """
    Chat handler for RunPod serverless chat completions.
    
    This function processes OpenAI-compatible chat requests on RunPod infrastructure.
    
    Args:
        job (dict): RunPod job dictionary containing:
            - input (dict): Request data with keys:
                - openai_route (str): The endpoint route ("/v1/models" or "/v1/chat/completions")
                - openai_input (dict): The request parameters
    
    Yields:
        dict: Response data - either model list or chat completion chunks
    """
    try:
        # Extract request data from job input
        input_data = job.get("input", {})
        route = input_data.get("openai_route")
        request_data = input_data.get("openai_input", {})
        
        # Get configuration from environment
        provider = os.getenv("CHAT_PROVIDER", "ollama")
        default_model = os.getenv("DEFAULT_MODEL", "gemma3n:latest")
        remote_auth = os.getenv("REMOTE_AUTH", "false").lower() == "true"
        use_database = os.getenv("USE_DATABASE", "false").lower() == "true"
        
        # Set authentication mode
        Environment.set_remote_auth(remote_auth)
        
        log.info(f"Processing route: {route}")
        
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
                    use_database=use_database,
                    default_model=default_model,
                    default_provider=provider
                )
                
                # Determine if streaming is requested (default True)
                stream = request_data.get("stream", True)
                
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
                        if event.startswith("data: "):
                            payload = event[len("data: "):].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                yield json.loads(payload)
                            except json.JSONDecodeError:
                                # Skip malformed JSON
                                continue
                                
            except Exception as e:
                log.error(f"Chat completions error: {e}")
                yield {"error": {"message": str(e), "type": "chat_completion_error"}}
        else:
            # Unknown route
            yield {"error": {"message": f"Unknown route: {route}", "type": "route_error"}}
            
    except Exception as e:
        log.error(f"Handler error: {e}")
        yield {"error": {"message": str(e), "type": "handler_error"}}


if __name__ == "__main__":
    # Download models on startup
    log.info("Starting RunPod chat handler...")
    download_models_on_startup()
    
    runpod.serverless.start(
        {
            "handler": chat_handler,
        }
    )