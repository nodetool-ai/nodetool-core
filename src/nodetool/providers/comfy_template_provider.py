from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import uuid
from typing import TYPE_CHECKING, Any, List

import requests
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageModel, VideoModel
from nodetool.metadata.types import Provider as ProviderEnum
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.comfy.template_loader import TemplateLoader
from nodetool.providers.comfy_api import (
    check_server,
    get_history,
    get_image_data,
    queue_workflow,
    upload_images,
)

if TYPE_CHECKING:
    from websockets.client import WebSocketClientProtocol

    from nodetool.providers.types import (
        ImageBytes,
        ImageToImageParams,
        TextToImageParams,
    )

log = get_logger(__name__)


def _server_addr() -> str:
    return os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")


def _ws_client_id() -> str:
    return str(uuid.uuid4())


@register_provider(ProviderEnum.ComfyTemplate)
class ComfyTemplateProvider(BaseProvider):
    """ComfyUI provider that uses YAML templates for workflow execution.

    This provider loads YAML template mappings that define how to construct
    ComfyUI graphs for different workflows. It dynamically builds graphs
    based on user parameters and executes them via the ComfyUI API.

    Templates are stored in the `templates/` directory and can be configured
    via the COMFY_TEMPLATE_DIR environment variable.
    """

    provider_name = "comfy_template"

    def __init__(self, secrets: dict[str, str] | None = None):
        super().__init__(secrets)
        self.template_loader = TemplateLoader()

    async def get_available_image_models(self) -> List[ImageModel]:
        """Get available image generation templates.

        Returns a list of ImageModel objects representing all available
        text_to_image and image_to_image templates.

        Returns:
            List of ImageModel objects for available templates.
        """
        templates = self.template_loader.get_image_templates()
        models: list[ImageModel] = []

        for template in templates:
            supported_tasks = [template.template_type]
            if template.template_type == "text_to_image":
                supported_tasks = ["text_to_image"]
            elif template.template_type == "image_to_image":
                supported_tasks = ["image_to_image"]

            models.append(
                ImageModel(
                    id=template.template_id,
                    name=template.template_name,
                    provider=ProviderEnum.ComfyTemplate,
                    supported_tasks=supported_tasks,
                    description=template.description,
                )
            )

        if not models:
            log.warning("No image templates found")
        return models

    async def get_available_video_models(self) -> List[VideoModel]:
        """Get available video generation templates.

        Returns a list of VideoModel objects representing all available
        image_to_video and text_to_video templates.

        Returns:
            List of VideoModel objects for available templates.
        """
        templates = self.template_loader.get_video_templates()
        models: list[VideoModel] = []

        for template in templates:
            models.append(
                VideoModel(
                    id=template.template_id,
                    name=template.template_name,
                    provider=ProviderEnum.ComfyTemplate,
                    supported_tasks=[template.template_type],
                    description=template.description,
                )
            )

        if not models:
            log.warning("No video templates found")
        return models

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context=None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Generate image from text using a template.

        Args:
            params: TextToImageParams containing the prompt and generation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            ImageBytes containing the generated image data.

        Raises:
            ValueError: If template_id is not provided or template not found.
            RuntimeError: If ComfyUI server is not reachable or execution fails.
        """
        template_id = params.model.id if params.model else None
        if not template_id:
            raise ValueError("ComfyTemplate: model.id (template_id) is required")

        log.debug(
            "ComfyTemplate.text_to_image called: template_id=%s prompt_len=%d",
            template_id,
            len(params.prompt) if isinstance(params.prompt, str) else -1,
        )

        mapping = self.template_loader.load(template_id)
        if mapping is None:
            raise ValueError(f"Template not found: {template_id}")

        if mapping.template_type not in ("text_to_image",):
            raise ValueError(
                f"Template '{template_id}' is not a text_to_image template (type: {mapping.template_type})"
            )

        await self._ensure_server()

        user_params = self._extract_params(params)
        graph = self._build_graph(mapping, user_params)

        images = await self._execute_graph(graph, template_id=template_id)

        return images[0] if images else b""

    async def image_to_image(
        self,
        image_bytes: bytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context=None,
        node_id: str | None = None,
    ) -> ImageBytes:
        """Transform image using a template.

        Args:
            image_bytes: Input image data.
            params: ImageToImageParams containing the prompt and transformation parameters.
            timeout_s: Optional timeout in seconds.
            context: Optional processing context.
            node_id: Optional node ID for tracking.

        Returns:
            ImageBytes containing the transformed image data.

        Raises:
            ValueError: If template_id is not provided, template not found, or template
                       is not an image_to_image template.
            RuntimeError: If ComfyUI server is not reachable or execution fails.
        """
        template_id = params.model.id if params.model else None
        if not template_id:
            raise ValueError("ComfyTemplate: model.id (template_id) is required")

        log.debug(
            "ComfyTemplate.image_to_image called: template_id=%s prompt_len=%d bytes=%d",
            template_id,
            len(params.prompt) if isinstance(params.prompt, str) else -1,
            len(image_bytes) if isinstance(image_bytes, bytes | bytearray) else -1,
        )

        mapping = self.template_loader.load(template_id)
        if mapping is None:
            raise ValueError(f"Template not found: {template_id}")

        if mapping.template_type not in ("image_to_image",):
            raise ValueError(
                f"Template '{template_id}' is not an image_to_image template (type: {mapping.template_type})"
            )

        await self._ensure_server()

        image_name = await self._upload_image(image_bytes)

        user_params = self._extract_params(params)
        user_params["image"] = image_name
        graph = self._build_graph(mapping, user_params)

        images = await self._execute_graph(graph, template_id=template_id)

        return images[0] if images else b""

    def _extract_params(self, params: Any) -> dict[str, Any]:
        """Extract user parameters from params object into a dict.

        Args:
            params: Params object with various attributes.

        Returns:
            Dict of parameter names to values.
        """
        result: dict[str, Any] = {}

        if hasattr(params, "prompt") and params.prompt:
            result["prompt"] = params.prompt
        if hasattr(params, "negative_prompt") and params.negative_prompt:
            result["negative_prompt"] = params.negative_prompt
        if hasattr(params, "width") and params.width:
            result["width"] = params.width
        if hasattr(params, "height") and params.height:
            result["height"] = params.height
        if hasattr(params, "seed") and params.seed is not None:
            result["seed"] = params.seed
        if hasattr(params, "num_inference_steps") and params.num_inference_steps:
            result["steps"] = params.num_inference_steps
        if hasattr(params, "guidance_scale") and params.guidance_scale:
            result["guidance"] = params.guidance_scale
        if hasattr(params, "strength") and params.strength:
            result["strength"] = params.strength
        if hasattr(params, "scheduler") and params.scheduler:
            result["scheduler"] = params.scheduler
        if hasattr(params, "cfg") and params.cfg:
            result["cfg"] = params.cfg
        if hasattr(params, "sampler") and params.sampler:
            result["sampler"] = params.sampler

        return result

    def _build_graph(self, mapping: Any, user_params: dict[str, Any]) -> dict[str, Any]:
        """Build a ComfyUI graph from a template mapping and user parameters.

        Args:
            mapping: TemplateMapping defining the workflow structure.
            user_params: Dict of user-provided parameter values.

        Returns:
            ComfyUI graph as a dict.
        """
        graph: dict[str, Any] = {}

        for input_name, input_mapping in mapping.inputs.items():
            value = user_params.get(input_name)

            if value is None:
                if input_mapping.required:
                    raise ValueError(f"Required input '{input_name}' not provided")
                value = input_mapping.default

            node_id = str(input_mapping.node_id)
            if node_id not in graph:
                graph[node_id] = {
                    "inputs": {},
                    "class_type": input_mapping.node_type,
                }
            graph[node_id]["inputs"][input_mapping.input_field] = value

        for _output_name, output_mapping in mapping.outputs.items():
            node_id = str(output_mapping.node_id)
            if node_id not in graph:
                graph[node_id] = {
                    "inputs": {},
                    "class_type": output_mapping.node_type,
                }

        if mapping.nodes:
            for node_id_str, node_mapping in mapping.nodes.items():
                node_id = str(node_id_str)
                if node_id not in graph:
                    graph[node_id] = {
                        "inputs": {},
                        "class_type": node_mapping.class_type,
                    }
                if node_mapping.images_directory:
                    graph[node_id]["inputs"]["images_directory"] = node_mapping.images_directory
                if node_mapping.filename_prefix:
                    graph[node_id]["inputs"]["filename_prefix"] = node_mapping.filename_prefix

        return graph

    async def _upload_image(self, image_bytes: bytes) -> str:
        """Upload an image to ComfyUI and return the filename.

        Args:
            image_bytes: Image data to upload.

        Returns:
            Filename assigned by ComfyUI.
        """
        image_name = f"nodetool_comfy_input_{uuid.uuid4().hex}.png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        upload_payload = [{"name": image_name, "image": f"data:image/png;base64,{image_b64}"}]

        upload_result = await asyncio.to_thread(upload_images, upload_payload)
        if upload_result.get("status") == "error":
            details = ", ".join(upload_result.get("details", []))
            raise ValueError(f"Failed to upload input image to ComfyUI: {details}")

        return image_name

    async def _ensure_server(self) -> None:
        """Ensure the ComfyUI server is reachable.

        Raises:
            RuntimeError: If the server is not reachable.
        """
        server_url = f"http://{_server_addr()}/"
        available = await asyncio.to_thread(check_server, server_url, 10, 100)
        if not available:
            raise RuntimeError(f"ComfyUI server {_server_addr()} not reachable")

    async def execute_graph(
        self,
        graph: dict[str, Any],
        template_id: str | None = None,
    ) -> list[bytes]:
        """Execute a ComfyUI graph and return the generated images.

        Public interface for executing ComfyUI workflow graphs. Used by
        ComfyTemplateNode subclasses and other callers that build their
        own graphs.

        Args:
            graph: ComfyUI workflow graph.
            template_id: Optional template ID for logging.

        Returns:
            List of generated image bytes.

        Raises:
            RuntimeError: If the server is not reachable or execution fails.
        """
        await self._ensure_server()
        return await self._execute_graph(graph, template_id)

    async def _execute_graph(
        self,
        graph: dict[str, Any],
        template_id: str | None = None,
    ) -> list[bytes]:
        """Execute a ComfyUI graph and return the generated images.

        Args:
            graph: ComfyUI workflow graph.
            template_id: Optional template ID for logging.

        Returns:
            List of generated image bytes.

        Raises:
            RuntimeError: If execution fails.
        """
        client_id = _ws_client_id()
        ws_url = f"ws://{_server_addr()}/ws?clientId={client_id}"
        log.debug("ComfyTemplate WS connecting: %s (template=%s)", ws_url, template_id)

        images: list[bytes] = []
        prompt_id: str | None = None

        try:
            async with websockets.connect(ws_url) as ws:
                try:
                    response = await asyncio.to_thread(queue_workflow, graph, client_id)
                except Exception as exc:
                    log.error("ComfyTemplate queue_workflow failed: %s", exc)
                    raise

                prompt_id = response.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(
                        f"ComfyUI did not return prompt_id in response: {response}"
                    )

                log.debug("ComfyTemplate queued prompt_id=%s template=%s", prompt_id, template_id)
                images = await self._collect_ws(ws, prompt_id)

        except Exception as exc:
            log.error("ComfyTemplate websocket execution failed: %s", exc)
            raise

        if not images and prompt_id:
            log.debug(
                "ComfyTemplate: No images received via websocket, falling back to history fetch (template=%s)",
                template_id,
            )
            images = await self._fetch_history_images(prompt_id)

        log.debug(
            "ComfyTemplate execution complete. Images collected=%d (template=%s)",
            len(images),
            template_id,
        )
        return images

    async def _collect_ws(self, ws: WebSocketClientProtocol, prompt_id: str) -> list[bytes]:
        """Collect images from WebSocket messages.

        Args:
            ws: WebSocket connection.
            prompt_id: Prompt ID to track execution.

        Returns:
            List of image bytes received.
        """
        images: list[bytes] = []
        current_node = ""

        while True:
            try:
                out = await ws.recv()
            except ConnectionClosedOK:
                log.debug("ComfyTemplate WS connection closed normally")
                break
            except ConnectionClosedError as exc:
                log.warning("ComfyTemplate WS connection closed with error: %s", exc)
                break

            if isinstance(out, str):
                log.debug("ComfyTemplate WS text: %s", out)
                try:
                    message = json.loads(out)
                except json.JSONDecodeError:
                    log.warning("ComfyTemplate WS received non-JSON text frame")
                    continue

                msg_type = message.get("type")
                if msg_type == "executing":
                    data = message.get("data", {})
                    node = data.get("node")
                    pid = data.get("prompt_id")
                    log.debug("ComfyTemplate WS executing event: node=%s pid=%s", node, pid)
                    if node is None and pid == prompt_id:
                        break
                    current_node = node or ""
                elif msg_type == "status":
                    log.debug("ComfyTemplate WS status: %s", message.get("data"))
                elif msg_type == "execution_error":
                    data = message.get("data", {})
                    log.error("ComfyTemplate execution error: %s", data)
                    raise RuntimeError(f"ComfyUI execution error: {data}")
            else:
                log.debug("ComfyTemplate WS binary: len=%d node=%s", len(out), current_node)
                if current_node == "save_image_websocket_node" and isinstance(
                    out, bytes | bytearray
                ):
                    images.append(out[8:])
                    log.debug("ComfyTemplate WS image frame received (total=%d)", len(images))

        return images

    async def _fetch_history_images(self, prompt_id: str) -> list[bytes]:
        """Fetch images from ComfyUI history API.

        Args:
            prompt_id: Prompt ID to fetch results for.

        Returns:
            List of image bytes.
        """
        try:
            history = await asyncio.to_thread(get_history, prompt_id)
        except Exception as exc:
            log.error("Failed to fetch ComfyUI history for %s: %s", prompt_id, exc)
            return []

        outputs = history.get(prompt_id, {}).get("outputs", {})
        images: list[bytes] = []

        for node_id, node_output in outputs.items():
            for image_info in node_output.get("images", []):
                filename = image_info.get("filename")
                if not filename:
                    continue
                subfolder = image_info.get("subfolder", "")
                image_type = image_info.get("type", "output")
                try:
                    data = await asyncio.to_thread(
                        get_image_data, filename, subfolder, image_type
                    )
                except Exception as exc:
                    log.warning(
                        "Failed to retrieve image from history (node=%s filename=%s): %s",
                        node_id,
                        filename,
                        exc,
                    )
                    continue
                if data:
                    images.append(data)
        return images
