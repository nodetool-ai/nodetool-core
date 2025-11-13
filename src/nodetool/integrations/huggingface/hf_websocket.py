"""
Hugging Face WebSocket Endpoint Module

This module provides the WebSocket endpoint for HuggingFace model downloads
with authentication and real-time progress tracking.
"""

from fastapi import WebSocket
from nodetool.config.logging_config import get_logger
from nodetool.integrations.huggingface.hf_download import DownloadManager

log = get_logger(__name__)


async def huggingface_download_endpoint(websocket: WebSocket):
    """WebSocket endpoint for HuggingFace model downloads with authentication."""
    from nodetool.runtime.resources import get_static_auth_provider, get_user_auth_provider, ResourceScope
    from nodetool.config.environment import Environment
    
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
        
        log.info(f"huggingface_download_endpoint: Websocket connection with user_id={user_id}")
        
        # Create download manager with user_id for database secret lookup
        download_manager = await DownloadManager.create(user_id=user_id)
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                command = data.get("command")
                repo_id = data.get("repo_id")
                path = data.get("path")
                allow_patterns = data.get("allow_patterns")
                ignore_patterns = data.get("ignore_patterns")

                if command == "start_download":
                    log.info(f"huggingface_download_endpoint: Received start_download command for {repo_id}/{path} (user_id={user_id})")
                    print(f"Starting download for {repo_id}/{path} (user_id={user_id})")
                    try:
                        await download_manager.start_download(
                            repo_id=repo_id,
                            path=path,
                            websocket=websocket,
                            allow_patterns=allow_patterns,
                            ignore_patterns=ignore_patterns,
                            user_id=user_id,
                        )
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
                        raise  # Re-raise to be caught by outer handler
                elif command == "cancel_download":
                    await download_manager.cancel_download(data.get("id"))
                else:
                    await websocket.send_json(
                        {"status": "error", "message": "Unknown command"}
                    )
        except Exception as e:
            log.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            await websocket.close()

