"""
Hugging Face WebSocket Endpoint Module

This module provides the WebSocket endpoint for HuggingFace model downloads
with authentication and real-time progress tracking.
"""

from fastapi import WebSocket

from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_download import DownloadManager, get_download_manager

log = get_logger(__name__)


async def huggingface_download_endpoint(websocket: WebSocket):
    """WebSocket endpoint for HuggingFace model downloads with authentication."""
    from nodetool.config.environment import Environment
    from nodetool.runtime.resources import ResourceScope, get_static_auth_provider, get_user_auth_provider

    # Wrap entire websocket handler in ResourceScope for database access
    async with ResourceScope():
        enforce_auth = Environment.enforce_auth()

        # In dev mode, skip all authentication checks
        if not enforce_auth:
            user_id = "1"
            log.info(f"huggingface_download_endpoint: Dev mode - skipping authentication, using user_id={user_id}")
        else:
            # Get auth providers only when enforcing auth
            static_provider = get_static_auth_provider()
            user_provider = get_user_auth_provider()

            # Authenticate websocket
            token = static_provider.extract_token_from_ws(
                websocket.headers, websocket.query_params
            )
            if not token:
                await websocket.close(code=1008, reason="Missing authentication")
                log.warning("HF download WebSocket connection rejected: Missing authentication header")
                return

            static_result = await static_provider.verify_token(token)
            if static_result.ok and static_result.user_id:
                user_id = static_result.user_id
            elif Environment.get_auth_provider_kind() == "supabase" and user_provider:
                user_result = await user_provider.verify_token(token)
                if user_result.ok and user_result.user_id:
                    user_id = user_result.user_id
                else:
                    await websocket.close(code=1008, reason="Invalid authentication")
                    log.warning("HF download WebSocket connection rejected: Invalid token")
                    return
            else:
                await websocket.close(code=1008, reason="Invalid authentication")
                log.warning("HF download WebSocket connection rejected: Invalid token")
                return

        # Ensure user_id is set (fallback to "1" for local mode)
        if not user_id:
            user_id = "1"

        log.info(f"huggingface_download_endpoint: Authenticated connection verified for user_id={user_id}")
        await websocket.accept()

        download_manager: DownloadManager | None = None
        try:
            download_manager = await get_download_manager(user_id=user_id)
        except Exception as e:
            log.error(f"Failed to initialize DownloadManager: {e}", exc_info=True)
            try:
                await websocket.send_json(
                    {
                        "status": "error",
                        "repo_id": None,
                        "path": None,
                        "error": str(e),
                    }
                )
            except Exception:
                pass
            await websocket.close()
            return

        # Register websocket and sync state
        download_manager.add_websocket(websocket)
        await download_manager.sync_state(websocket)
        last_repo_id: str | None = None
        last_path: str | None = None

        try:
            while True:
                data = await websocket.receive_json()
                command = data.get("command")
                repo_id = data.get("repo_id")
                path = data.get("path")
                last_repo_id = repo_id
                last_path = path
                allow_patterns = data.get("allow_patterns")
                ignore_patterns = data.get("ignore_patterns")

                if command == "start_download":
                    log.info(f"huggingface_download_endpoint: Received start_download command for {repo_id}/{path} (user_id={user_id})")
                    print(f"Starting download for {repo_id}/{path} (user_id={user_id})")
                    
                    # Determine cache_dir based on model_type
                    model_type = data.get("model_type")
                    cache_dir = None
                    if model_type == "llama_cpp_model":
                        from nodetool.providers.llama_server_manager import get_llama_cpp_cache_dir
                        cache_dir = get_llama_cpp_cache_dir()
                        log.info(f"Using llama.cpp cache for model_type={model_type}: {cache_dir}")
                    
                    try:
                        # This is now non-blocking
                        await download_manager.start_download(
                            repo_id=repo_id,
                            path=path,
                            allow_patterns=allow_patterns,
                            ignore_patterns=ignore_patterns,
                            user_id=user_id,
                            cache_dir=cache_dir,
                        )
                        log.info(f"huggingface_download_endpoint: Download started successfully for {repo_id}/{path}")
                    except Exception as e:
                        # Error should already be sent by start_download, but send a final error message
                        # in case the WebSocket update failed
                        await websocket.send_json(
                            {
                                "status": "error",
                                "error": str(e),
                                "repo_id": repo_id,
                                "path": path,
                            }
                        )
                        # Don't re-raise here as we want to keep the socket open for other commands
                        log.error(f"Error starting download: {e}")

                elif command == "cancel_download":
                    download_id = data.get("id")
                    if not download_id:
                        repo_id = data.get("repo_id")
                        path = data.get("path")
                        if repo_id:
                            download_id = repo_id if path is None else f"{repo_id}/{path}"

                    log.info(f"Processing cancel_download for id={download_id}")
                    if download_id:
                        await download_manager.cancel_download(download_id)
                    else:
                        log.warning("Received cancel_download without id or repo_id")
                else:
                    await websocket.send_json(
                        {"status": "error", "message": "Unknown command"}
                    )
        except Exception as e:
            log.error(f"WebSocket error: {e}", exc_info=True)
            try:
                await websocket.send_json(
                    {
                        "status": "error",
                        "repo_id": last_repo_id,
                        "path": last_path,
                        "error": str(e),
                    }
                )
            except Exception:
                pass
        finally:
            if download_manager:
                download_manager.remove_websocket(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
