from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import uuid
from typing import Any, List

import requests
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageModel, Provider as ProviderEnum
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.comfy_api import (
    check_server,
    get_history,
    get_image_data,
    queue_workflow,
    upload_images,
)
from nodetool.providers.types import ImageBytes, ImageToImageParams, TextToImageParams

log = get_logger(__name__)


def _server_addr() -> str:
    return os.environ.get("COMFYUI_ADDR", "127.0.0.1:8188")


def _ws_client_id() -> str:
    return str(uuid.uuid4())


@register_provider(ProviderEnum.Comfy)
class ComfyProvider(BaseProvider):
    provider_name = "comfy"

    async def get_available_image_models(self) -> List[ImageModel]:
        """Query ComfyUI for available checkpoints via /models/checkpoints."""
        url = f"http://{_server_addr()}/models/checkpoints"

        def _fetch() -> list[Any]:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()

        try:
            data = await asyncio.to_thread(_fetch)
            models: list[ImageModel] = []
            for item in data:
                name = item.get("name") if isinstance(item, dict) else str(item)
                if name:
                    models.append(ImageModel(id=name, name=name, provider=ProviderEnum.Comfy))
            if not models:
                log.warning("No checkpoints reported by ComfyUI")
            return models
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to fetch ComfyUI checkpoints: %s", exc)
            return []

    async def text_to_image(self, params: TextToImageParams, context=None) -> ImageBytes:
        log.debug(
            "ComfyProvider.text_to_image called: model_id=%s prompt_len=%d",
            getattr(params.model, "id", ""),
            len(params.prompt) if isinstance(params.prompt, str) else -1,
        )

        ckpt = params.model.id
        if not ckpt:
            raise ValueError("ComfyProvider: model.id (checkpoint) is required")

        await self._ensure_server()

        width = params.width or 512
        height = params.height or 512
        steps = params.num_inference_steps or 20
        seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        cfg = params.guidance_scale or 8
        negative_prompt = params.negative_prompt or ""
        scheduler = params.scheduler or "normal"

        graph: dict[str, Any] = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": scheduler,
                    "seed": seed,
                    "steps": steps,
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": height, "width": width},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": params.prompt}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "save_image_websocket_node": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["8", 0]},
            },
        }

        images = await self._execute_graph(graph)
        return images[0] if images else b""

    async def image_to_image(self, image_bytes: bytes, params: ImageToImageParams, context=None) -> ImageBytes:
        log.debug(
            "ComfyProvider.image_to_image called: model_id=%s prompt_len=%d bytes=%d",
            getattr(params.model, "id", ""),
            len(params.prompt) if isinstance(params.prompt, str) else -1,
            len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else -1,
        )

        ckpt = params.model.id
        if not ckpt:
            raise ValueError("ComfyProvider: model.id (checkpoint) is required")

        await self._ensure_server()

        strength = params.strength or 0.65
        steps = params.num_inference_steps or 20
        seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        cfg = params.guidance_scale or 7
        negative_prompt = params.negative_prompt or ""
        scheduler = params.scheduler or "normal"

        image_name = f"nodetool_comfy_input_{uuid.uuid4().hex}.png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        upload_payload = [{"name": image_name, "image": f"data:image/png;base64,{image_b64}"}]

        upload_result = await asyncio.to_thread(upload_images, upload_payload)
        if upload_result.get("status") == "error":
            details = ", ".join(upload_result.get("details", []))
            raise ValueError(f"Failed to upload input image to ComfyUI: {details}")

        graph: dict[str, Any] = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name, "subfolder": "", "type": "input"},
            },
            "2": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["4", 2]}},
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": strength,
                    "latent_image": ["2", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": scheduler,
                    "seed": seed,
                    "steps": steps,
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": params.prompt}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "save_image_websocket_node": {
                "class_type": "SaveImageWebsocket",
                "inputs": {"images": ["8", 0]},
            },
        }

        images = await self._execute_graph(graph)
        return images[0] if images else b""

    async def _ensure_server(self) -> None:
        server_url = f"http://{_server_addr()}/"
        available = await asyncio.to_thread(check_server, server_url, 10, 100)
        if not available:
            raise RuntimeError(f"ComfyUI server {_server_addr()} not reachable")

    async def _execute_graph(self, graph: dict[str, Any]) -> list[bytes]:
        client_id = _ws_client_id()
        ws_url = f"ws://{_server_addr()}/ws?clientId={client_id}"
        log.debug("Comfy WS connecting: %s", ws_url)

        images: list[bytes] = []
        prompt_id: str | None = None

        async with websockets.connect(ws_url) as ws:
            try:
                response = await asyncio.to_thread(queue_workflow, graph, client_id)
            except Exception as exc:  # noqa: BLE001
                log.error("Comfy queue_workflow failed: %s", exc)
                raise

            prompt_id = response.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI did not return prompt_id in response: {response}")

            log.debug("Comfy queued prompt_id=%s", prompt_id)
            images = await self._collect_ws(ws, prompt_id)

        if not images and prompt_id:
            log.debug("No images received via websocket, falling back to history fetch")
            images = await self._fetch_history_images(prompt_id)

        log.debug("Comfy execution complete. Images collected=%d", len(images))
        return images

    async def _collect_ws(self, ws: WebSocketClientProtocol, prompt_id: str) -> list[bytes]:
        images: list[bytes] = []
        current_node = ""

        while True:
            try:
                out = await ws.recv()
            except ConnectionClosedOK:
                log.debug("Comfy WS connection closed normally")
                break
            except ConnectionClosedError as exc:
                log.warning("Comfy WS connection closed with error: %s", exc)
                break

            if isinstance(out, str):
                log.debug("Comfy WS text: %s", out)
                try:
                    message = json.loads(out)
                except json.JSONDecodeError:
                    log.warning("Comfy WS received non-JSON text frame")
                    continue

                msg_type = message.get("type")
                if msg_type == "executing":
                    data = message.get("data", {})
                    node = data.get("node")
                    pid = data.get("prompt_id")
                    log.debug("Comfy WS executing event: node=%s pid=%s", node, pid)
                    if node is None and pid == prompt_id:
                        break
                    current_node = node or ""
                elif msg_type == "status":
                    log.debug("Comfy WS status: %s", message.get("data"))
                elif msg_type == "execution_error":
                    data = message.get("data", {})
                    log.error("Comfy execution error: %s", data)
                    raise RuntimeError(f"ComfyUI execution error: {data}")
            else:
                log.debug("Comfy WS binary: len=%d node=%s", len(out), current_node)
                if current_node == "save_image_websocket_node" and isinstance(out, (bytes, bytearray)):
                    images.append(out[8:])
                    log.debug("Comfy WS image frame received (total=%d)", len(images))

        return images

    async def _fetch_history_images(self, prompt_id: str) -> list[bytes]:
        try:
            history = await asyncio.to_thread(get_history, prompt_id)
        except Exception as exc:  # noqa: BLE001
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
                    data = await asyncio.to_thread(get_image_data, filename, subfolder, image_type)
                except Exception as exc:  # noqa: BLE001
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
