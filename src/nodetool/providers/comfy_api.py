from __future__ import annotations

import base64
import json
import os
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from io import BytesIO
from typing import Any

import requests
import websocket

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS: int = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES: int = 500

# Websocket reconnection behaviour (can be overridden through environment variables)
WEBSOCKET_RECONNECT_ATTEMPTS: int = int(os.environ.get("WEBSOCKET_RECONNECT_ATTEMPTS", 5))
WEBSOCKET_RECONNECT_DELAY_S: int = int(os.environ.get("WEBSOCKET_RECONNECT_DELAY_S", 3))

# Extra verbose websocket trace logs (set WEBSOCKET_TRACE=true to enable)
if os.environ.get("WEBSOCKET_TRACE", "false").lower() == "true":
    websocket.enableTrace(True)

# Host where ComfyUI is running
COMFY_HOST: str = os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")
# Enforce a clean state after each job is done
REFRESH_WORKER: bool = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


def _comfy_server_status() -> dict[str, Any]:
    """Return a dictionary with basic reachability info for the ComfyUI HTTP server."""
    try:
        resp = requests.get(f"http://{COMFY_HOST}/", timeout=5)
        return {"reachable": resp.status_code == 200, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def _attempt_websocket_reconnect(
    ws_url: str,
    max_attempts: int,
    delay_s: int,
    initial_error: Exception,
) -> websocket.WebSocket:
    """
    Attempts to reconnect to the WebSocket server after a disconnect.

    Args:
        ws_url: The WebSocket URL (including client_id).
        max_attempts: Maximum number of reconnection attempts.
        delay_s: Delay in seconds between attempts.
        initial_error: The error that triggered the reconnect attempt.

    Returns:
        The newly connected WebSocket object.

    Raises:
        websocket.WebSocketConnectionClosedException: If reconnection fails after all attempts.
    """
    print(f"worker-comfyui - Websocket connection closed unexpectedly: {initial_error}. Attempting to reconnect...")
    last_reconnect_error: Exception = initial_error
    for attempt in range(max_attempts):
        srv_status = _comfy_server_status()
        if not srv_status.get("reachable", False):
            print(
                "worker-comfyui - ComfyUI HTTP unreachable - aborting websocket reconnect: "
                f"{srv_status.get('error', 'status ' + str(srv_status.get('status_code')))}"
            )
            raise websocket.WebSocketConnectionClosedException("ComfyUI HTTP unreachable during websocket reconnect")

        print(
            f"worker-comfyui - Reconnect attempt {attempt + 1}/{max_attempts}... (ComfyUI HTTP reachable, status {srv_status.get('status_code')})"
        )
        try:
            new_ws = websocket.WebSocket()
            new_ws.connect(ws_url, timeout=10)
            print("worker-comfyui - Websocket reconnected successfully.")
            return new_ws
        except (TimeoutError, websocket.WebSocketException, ConnectionRefusedError, OSError) as reconn_err:
            last_reconnect_error = reconn_err
            print(f"worker-comfyui - Reconnect attempt {attempt + 1} failed: {reconn_err}")
            if attempt < max_attempts - 1:
                print(f"worker-comfyui - Waiting {delay_s} seconds before next attempt...")
                time.sleep(delay_s)
            else:
                print("worker-comfyui - Max reconnection attempts reached.")

    print("worker-comfyui - Failed to reconnect websocket after connection closed.")
    raise websocket.WebSocketConnectionClosedException(
        f"Connection closed and failed to reconnect. Last error: {last_reconnect_error}"
    )


def validate_input(job_input: dict[str, Any] | str | None) -> tuple[dict[str, Any] | None, str | None]:
    """
    Validates the input for the handler function.

    Args:
        job_input: The input data to validate.

    Returns:
        Tuple containing the validated data and an error message, if any.
    """
    if job_input is None:
        return None, "Please provide input"

    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    assert isinstance(job_input, dict)
    workflow = job_input.get("workflow")
    if workflow is None:
        return None, "Missing 'workflow' parameter"

    images = job_input.get("images")
    if images is not None:
        if not isinstance(images, list) or not all(
            isinstance(image, dict) and ("name" in image and "image" in image) for image in images
        ):
            return None, "'images' must be a list of objects with 'name' and 'image' keys"

    comfy_org_api_key: str | None = job_input.get("comfy_org_api_key")

    return {"workflow": workflow, "images": images, "comfy_org_api_key": comfy_org_api_key}, None


def check_server(url: str, retries: int = 500, delay: int = 50) -> bool:
    """
    Check if a server is reachable via HTTP GET request.
    """
    print(f"worker-comfyui - Checking API server at {url}...")
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("worker-comfyui - API is reachable")
                return True
        except requests.Timeout:
            pass
        except requests.RequestException:
            pass
        time.sleep(delay / 1000)

    print(f"worker-comfyui - Failed to connect to server at {url} after {retries} attempts.")
    return False


def upload_images(images: list[dict[str, str]]) -> dict[str, Any]:
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses: list[str] = []
    upload_errors: list[str] = []

    print(f"worker-comfyui - Uploading {len(images)} image(s)...")

    for image in images:
        try:
            name = image["name"]
            image_data_uri = image["image"]

            base64_data = image_data_uri.split(",", 1)[1] if "," in image_data_uri else image_data_uri

            blob = base64.b64decode(base64_data)

            files = {
                "image": (name, BytesIO(blob), "image/png"),
                "overwrite": (None, "true"),
            }

            response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=30)
            response.raise_for_status()

            responses.append(f"Successfully uploaded {name}")
            print(f"worker-comfyui - Successfully uploaded {name}")

        except base64.binascii.Error as e:  # type: ignore[attr-defined]
            error_msg = f"Error decoding base64 for {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.Timeout:
            error_msg = f"Timeout uploading {image.get('name', 'unknown')}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except requests.RequestException as e:
            error_msg = f"Error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error uploading {image.get('name', 'unknown')}: {e}"
            print(f"worker-comfyui - {error_msg}")
            upload_errors.append(error_msg)

    if upload_errors:
        print("worker-comfyui - image(s) upload finished with errors")
        return {"status": "error", "message": "Some images failed to upload", "details": upload_errors}

    print("worker-comfyui - image(s) upload complete")
    return {"status": "success", "message": "All images uploaded successfully", "details": responses}


def get_available_models() -> dict[str, list[str]]:
    """Get list of available models from ComfyUI."""
    try:
        response = requests.get(f"http://{COMFY_HOST}/object_info", timeout=10)
        response.raise_for_status()
        object_info = response.json()

        available_models: dict[str, list[str]] = {}
        if "CheckpointLoaderSimple" in object_info:
            checkpoint_info = object_info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_options = checkpoint_info["input"]["required"].get("ckpt_name")
                if ckpt_options and len(ckpt_options) > 0:
                    available_models["checkpoints"] = ckpt_options[0] if isinstance(ckpt_options[0], list) else []

        return available_models
    except Exception as e:
        print(f"worker-comfyui - Warning: Could not fetch available models: {e}")
        return {}


def queue_workflow(workflow: dict[str, Any], client_id: str, comfy_org_api_key: str | None = None) -> dict[str, Any]:
    """
    Queue a workflow to be processed by ComfyUI.
    """
    payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}

    key_from_env = os.environ.get("COMFY_ORG_API_KEY")
    effective_key = comfy_org_api_key if comfy_org_api_key else key_from_env
    if effective_key:
        payload["extra_data"] = {"api_key_comfy_org": effective_key}
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    response = requests.post(f"http://{COMFY_HOST}/prompt", data=data, headers=headers, timeout=30)

    if response.status_code == 400:
        print(f"worker-comfyui - ComfyUI returned 400. Response body: {response.text}")
        try:
            error_data = response.json()
            print(f"worker-comfyui - Parsed error data: {error_data}")

            error_message = "Workflow validation failed"
            error_details: list[str] = []

            if "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", error_message)
                    if error_info.get("type") == "prompt_outputs_failed_validation":
                        error_message = "Workflow validation failed"
                else:
                    error_message = str(error_info)

            if "node_errors" in error_data:
                for node_id, node_error in error_data["node_errors"].items():
                    if isinstance(node_error, dict):
                        for error_type, error_msg in node_error.items():
                            error_details.append(f"Node {node_id} ({error_type}): {error_msg}")
                    else:
                        error_details.append(f"Node {node_id}: {node_error}")

            if error_data.get("type") == "prompt_outputs_failed_validation":
                error_message = error_data.get("message", "Workflow validation failed")
                available_models = get_available_models()
                if available_models.get("checkpoints"):
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += "\nAvailable checkpoint models: " + ", ".join(available_models["checkpoints"])
                else:
                    error_message += "\n\nThis usually means a required model or parameter is not available."
                    error_message += (
                        "\nNo checkpoint models appear to be available. Please check your model installation."
                    )

                raise ValueError(error_message)

            if error_details:
                detailed_message = f"{error_message}:\n" + "\n".join(f"• {detail}" for detail in error_details)

                if any("not in list" in detail and "ckpt_name" in detail for detail in error_details):
                    available_models = get_available_models()
                    if available_models.get("checkpoints"):
                        detailed_message += "\n\nAvailable checkpoint models: " + ", ".join(
                            available_models["checkpoints"]
                        )
                    else:
                        detailed_message += (
                            "\n\nNo checkpoint models appear to be available. Please check your model installation."
                        )

                raise ValueError(detailed_message)
            else:
                raise ValueError(f"{error_message}. Raw response: {response.text}")

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError("ComfyUI validation failed (could not parse error response): " + response.text) from e

    response.raise_for_status()
    return response.json()


def get_history(prompt_id: str) -> dict[str, Any]:
    """Retrieve the history for a given prompt ID."""
    response = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def get_image_data(filename: str, subfolder: str, image_type: str) -> bytes | None:
    """Fetch image bytes from the ComfyUI /view endpoint."""
    print(f"worker-comfyui - Fetching image data: type={image_type}, subfolder={subfolder}, filename={filename}")
    data = {"filename": filename, "subfolder": subfolder, "type": image_type}
    url_values = urllib.parse.urlencode(data)
    try:
        response = requests.get(f"http://{COMFY_HOST}/view?{url_values}", timeout=60)
        response.raise_for_status()
        print(f"worker-comfyui - Successfully fetched image data for {filename}")
        return response.content
    except requests.Timeout:
        print(f"worker-comfyui - Timeout fetching image data for {filename}")
        return None
    except requests.RequestException as e:
        print(f"worker-comfyui - Error fetching image data for {filename}: {e}")
        return None
    except Exception as e:
        print(f"worker-comfyui - Unexpected error fetching image data for {filename}: {e}")
        return None


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle a job using ComfyUI via websockets for status and image retrieval."""
    job_input = job["input"]

    validated_data, error_message = validate_input(job_input)
    if error_message or validated_data is None:
        return {"error": error_message or "Invalid input"}

    workflow = validated_data["workflow"]
    input_images = validated_data.get("images")

    if not check_server(
        f"http://{COMFY_HOST}/",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    ):
        return {"error": f"ComfyUI server ({COMFY_HOST}) not reachable after multiple retries."}

    if input_images:
        upload_result = upload_images(input_images)
        if upload_result["status"] == "error":
            return {"error": "Failed to upload one or more input images", "details": upload_result["details"]}

    ws: websocket.WebSocket | None = None
    client_id = str(uuid.uuid4())
    prompt_id: str | None = None
    output_data: list[dict[str, str]] = []
    errors: list[str] = []

    try:
        ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
        print(f"worker-comfyui - Connecting to websocket: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url, timeout=10)
        print("worker-comfyui - Websocket connected")

        try:
            queued_workflow = queue_workflow(
                workflow, client_id, comfy_org_api_key=validated_data.get("comfy_org_api_key")
            )
            prompt_id = queued_workflow.get("prompt_id")
            if not prompt_id:
                raise ValueError(f"Missing 'prompt_id' in queue response: {queued_workflow}")
            print(f"worker-comfyui - Queued workflow with ID: {prompt_id}")
        except requests.RequestException as e:
            print(f"worker-comfyui - Error queuing workflow: {e}")
            raise ValueError(f"Error queuing workflow: {e}") from e
        except Exception as e:
            print(f"worker-comfyui - Unexpected error queuing workflow: {e}")
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Unexpected error queuing workflow: {e}") from e

        print(f"worker-comfyui - Waiting for workflow execution ({prompt_id})...")
        execution_done = False
        while True:
            try:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "status":
                        status_data = message.get("data", {}).get("status", {})
                        print(
                            "worker-comfyui - Status update: "
                            f"{status_data.get('exec_info', {}).get('queue_remaining', 'N/A')} items remaining in queue"
                        )
                    elif message.get("type") == "executing":
                        data = message.get("data", {})
                        if data.get("node") is None and data.get("prompt_id") == prompt_id:
                            print(f"worker-comfyui - Execution finished for prompt {prompt_id}")
                            execution_done = True
                            break
                    elif message.get("type") == "execution_error":
                        data = message.get("data", {})
                        if data.get("prompt_id") == prompt_id:
                            error_details = (
                                f"Node Type: {data.get('node_type')}, Node ID: {data.get('node_id')}, "
                                f"Message: {data.get('exception_message')}"
                            )
                            print(f"worker-comfyui - Execution error received: {error_details}")
                            errors.append(f"Workflow execution error: {error_details}")
                            break
                else:
                    continue
            except websocket.WebSocketTimeoutException:
                print("worker-comfyui - Websocket receive timed out. Still waiting...")
                continue
            except websocket.WebSocketConnectionClosedException as closed_err:
                try:
                    ws = _attempt_websocket_reconnect(
                        ws_url,
                        WEBSOCKET_RECONNECT_ATTEMPTS,
                        WEBSOCKET_RECONNECT_DELAY_S,
                        closed_err,
                    )
                    print("worker-comfyui - Resuming message listening after successful reconnect.")
                    continue
                except websocket.WebSocketConnectionClosedException as reconn_failed_err:
                    raise reconn_failed_err
            except json.JSONDecodeError:
                print("worker-comfyui - Received invalid JSON message via websocket.")

        if not execution_done and not errors:
            raise ValueError("Workflow monitoring loop exited without confirmation of completion or error.")

        print(f"worker-comfyui - Fetching history for prompt {prompt_id}...")
        history = get_history(prompt_id or "")

        if not prompt_id or prompt_id not in history:
            error_msg = f"Prompt ID {prompt_id} not found in history after execution."
            print(f"worker-comfyui - {error_msg}")
            if not errors:
                return {"error": error_msg}
            else:
                errors.append(error_msg)
                return {"error": "Job processing failed, prompt ID not found in history.", "details": errors}

        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        if not outputs:
            warning_msg = f"No outputs found in history for prompt {prompt_id}."
            print(f"worker-comfyui - {warning_msg}")
            if not errors:
                errors.append(warning_msg)

        print(f"worker-comfyui - Processing {len(outputs)} output nodes...")
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                print(f"worker-comfyui - Node {node_id} contains {len(node_output['images'])} image(s)")
                for image_info in node_output["images"]:
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    img_type = image_info.get("type")

                    if img_type == "temp":
                        print(f"worker-comfyui - Skipping image {filename} because type is 'temp'")
                        continue

                    if not filename:
                        warn_msg = f"Skipping image in node {node_id} due to missing filename: {image_info}"
                        print(f"worker-comfyui - {warn_msg}")
                        errors.append(warn_msg)
                        continue

                    image_bytes = get_image_data(filename, subfolder, img_type or "output")

                    if image_bytes:
                        file_extension = os.path.splitext(filename)[1] or ".png"

                        if os.environ.get("BUCKET_ENDPOINT_URL"):
                            try:
                                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                                    temp_file.write(image_bytes)
                                    temp_file_path = temp_file.name
                                print(f"worker-comfyui - Wrote image bytes to temporary file: {temp_file_path}")

                                print(f"worker-comfyui - Uploading {filename} to S3...")
                                # Placeholder for rp_upload usage; integrate as needed
                                # s3_url = rp_upload.upload_image(job_id, temp_file_path)
                                # os.remove(temp_file_path)
                                # output_data.append({"filename": filename, "type": "s3_url", "data": s3_url})
                                output_data.append({"filename": filename, "type": "file", "data": temp_file_path})
                            except Exception as e:
                                error_msg = f"Error handling upload for {filename}: {e}"
                                print(f"worker-comfyui - {error_msg}")
                                errors.append(error_msg)
                        else:
                            try:
                                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                                output_data.append({"filename": filename, "type": "base64", "data": base64_image})
                                print(f"worker-comfyui - Encoded {filename} as base64")
                            except Exception as e:
                                error_msg = f"Error encoding {filename} to base64: {e}"
                                print(f"worker-comfyui - {error_msg}")
                                errors.append(error_msg)
                    else:
                        error_msg = f"Failed to fetch image data for {filename} from /view endpoint."
                        errors.append(error_msg)

            other_keys = [k for k in node_output if k != "images"]
            if other_keys:
                warn_msg = f"Node {node_id} produced unhandled output keys: {other_keys}."
                print(f"worker-comfyui - WARNING: {warn_msg}")
                print(
                    "worker-comfyui - --> If this output is useful, please consider opening an issue on GitHub to discuss adding support."
                )

    except websocket.WebSocketException as e:
        print(f"worker-comfyui - WebSocket Error: {e}")
        print(traceback.format_exc())
        return {"error": f"WebSocket communication error: {e}"}
    except requests.RequestException as e:
        print(f"worker-comfyui - HTTP Request Error: {e}")
        print(traceback.format_exc())
        return {"error": f"HTTP communication error with ComfyUI: {e}"}
    except ValueError as e:
        print(f"worker-comfyui - Value Error: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    except Exception as e:
        print(f"worker-comfyui - Unexpected Handler Error: {e}")
        print(traceback.format_exc())
        return {"error": f"An unexpected error occurred: {e}"}
    finally:
        if ws and ws.connected:
            print("worker-comfyui - Closing websocket connection.")
            ws.close()

    final_result: dict[str, Any] = {}

    if output_data:
        final_result["images"] = output_data

    if errors:
        final_result["errors"] = errors
        print(f"worker-comfyui - Job completed with errors/warnings: {errors}")

    if not output_data and errors:
        print("worker-comfyui - Job failed with no output images.")
        return {"error": "Job processing failed", "details": errors}
    elif not output_data and not errors:
        print("worker-comfyui - Job completed successfully, but the workflow produced no images.")
        final_result["status"] = "success_no_images"
        final_result["images"] = []

    print(f"worker-comfyui - Job completed. Returning {len(output_data)} image(s).")
    return final_result
