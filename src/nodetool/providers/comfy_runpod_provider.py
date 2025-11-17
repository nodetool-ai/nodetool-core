from __future__ import annotations

import base64
import json
import os
import random
import time
import uuid
from typing import Any, List, Optional

import requests

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ImageModel
from nodetool.metadata.types import Provider as ProviderEnum
from nodetool.providers.base import BaseProvider, register_provider
from nodetool.providers.types import ImageBytes, ImageToImageParams, TextToImageParams

log = get_logger(__name__)


@register_provider(ProviderEnum.ComfyRunpod)
class ComfyRunpodProvider(BaseProvider):
    provider_name = "comfy_runpod"

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["RUNPOD_API_KEY", "RUNPOD_COMFYUI_ENDPOINT_ID"]

    def __init__(self, secrets: dict[str, str] = {}):
        super().__init__(secrets=secrets)
        # Prefer secrets passed in; fall back to env
        self.api_key: str = secrets.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id: str = secrets.get("RUNPOD_COMFYUI_ENDPOINT_ID") or os.environ.get(
            "RUNPOD_COMFYUI_ENDPOINT_ID", ""
        )
        if not self.api_key or not self.endpoint_id:
            log.warning("ComfyRunpodProvider initialized without required secrets.")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _base_url(self) -> str:
        return f"https://api.runpod.ai/v2/{self.endpoint_id}"

    def _poll_status(self, request_id: str, timeout_s: int = 600, poll_interval_s: float = 1.5) -> dict[str, Any]:
        base = self._base_url()
        headers = self._headers()
        start = time.time()
        last_status: Optional[str] = None
        while True:
            status_url = f"{base}/status/{request_id}"
            resp = requests.get(status_url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            status = payload.get("status") or payload.get("state")
            if status != last_status:
                log.debug("RunPod status for %s: %s", request_id, status)
                last_status = status

            if status in {"COMPLETED", "COMPLETED_WITH_ERRORS", "FAILED", "CANCELLED", "ERROR"}:
                return payload

            if time.time() - start > timeout_s:
                raise TimeoutError(f"RunPod request {request_id} timed out after {timeout_s}s")
            time.sleep(poll_interval_s)

    def _execute(self, workflow: dict[str, Any], images: Optional[list[dict[str, str]]] = None) -> list[bytes]:
        base = self._base_url()
        headers = self._headers()
        body: dict[str, Any] = {"input": {"workflow": workflow}}
        if images:
            body["input"]["images"] = images

        run_url = f"{base}/run"
        resp = requests.post(run_url, headers=headers, data=json.dumps(body), timeout=60)
        resp.raise_for_status()
        run_payload = resp.json()
        request_id = run_payload.get("id") or run_payload.get("jobId") or run_payload.get("requestId")
        if not request_id:
            raise RuntimeError(f"Unexpected RunPod /run response: {run_payload}")

        status_payload = self._poll_status(request_id)
        status = status_payload.get("status") or status_payload.get("state")
        if status not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
            err = status_payload.get("error") or status_payload
            raise RuntimeError(f"RunPod job {request_id} failed: {err}")

        output = status_payload.get("output") or {}
        images_out = output.get("images") or []
        results: list[bytes] = []
        for item in images_out:
            try:
                t = item.get("type")
                data = item.get("data")
                if not data:
                    continue
                if t == "base64":
                    b64 = data.split(",", 1)[1] if "," in data else data
                    results.append(base64.b64decode(b64))
                elif t == "s3_url":
                    try:
                        r = requests.get(data, timeout=60)
                        if r.ok:
                            results.append(r.content)
                        else:
                            log.warning("Failed to fetch S3 URL (%s): %s", r.status_code, data)
                    except Exception as exc:
                        log.warning("Error fetching S3 URL %s: %s", data, exc)
                else:
                    log.debug("Ignoring unsupported image type from RunPod: %s", t)
            except Exception as exc:
                log.warning("Skipping invalid image entry from RunPod: %s", exc)
        return results

    async def get_available_image_models(self) -> List[ImageModel]:
        return [
            ImageModel(
                id="flux1-dev-fp8.safetensors",
                name="flux1-dev-fp8.safetensors",
                provider=ProviderEnum.ComfyRunpod,
                supported_tasks=["text_to_image", "image_to_image"],
            )
        ]

    async def text_to_image(
        self,
        params: TextToImageParams,
        timeout_s: int | None = None,
        context=None,
        node_id: str | None = None,
    ) -> ImageBytes:
        ckpt = params.model.id
        if not ckpt:
            raise ValueError("ComfyRunpod: model.id (checkpoint) is required")

        # Flux graph
        if ckpt == "flux1-dev-fp8.safetensors":
            width = params.width or 1024
            height = params.height or 1024
            steps = params.num_inference_steps or 20
            seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
            negative_prompt = params.negative_prompt or ""
            flux_guidance = params.guidance_scale or 3.5

            graph: dict[str, Any] = {
                "6": {
                    "inputs": {"text": params.prompt, "clip": ["30", 1]},
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
                },
                "8": {
                    "inputs": {"samples": ["31", 0], "vae": ["30", 2]},
                    "class_type": "VAEDecode",
                    "_meta": {"title": "VAE Decode"},
                },
                "9": {
                    "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
                    "class_type": "SaveImage",
                    "_meta": {"title": "Save Image"},
                },
                "27": {
                    "inputs": {"width": width, "height": height, "batch_size": 1},
                    "class_type": "EmptySD3LatentImage",
                    "_meta": {"title": "EmptySD3LatentImage"},
                },
                "30": {
                    "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"},
                    "class_type": "CheckpointLoaderSimple",
                    "_meta": {"title": "Load Checkpoint"},
                },
                "31": {
                    "inputs": {
                        "seed": seed,
                        "steps": steps,
                        "cfg": 1,
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "denoise": 1,
                        "model": ["30", 0],
                        "positive": ["35", 0],
                        "negative": ["33", 0],
                        "latent_image": ["27", 0],
                    },
                    "class_type": "KSampler",
                    "_meta": {"title": "KSampler"},
                },
                "33": {
                    "inputs": {"text": negative_prompt, "clip": ["30", 1]},
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
                },
                "35": {
                    "inputs": {"guidance": flux_guidance, "conditioning": ["6", 0]},
                    "class_type": "FluxGuidance",
                    "_meta": {"title": "FluxGuidance"},
                },
            }

            images = await self._to_thread(self._execute, graph)
        else:
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
            }
            images = await self._to_thread(self._execute, graph)

        return images[0] if images else b""

    async def image_to_image(
        self,
        image_bytes: bytes,
        params: ImageToImageParams,
        timeout_s: int | None = None,
        context=None,
        node_id: str | None = None,
    ) -> ImageBytes:
        ckpt = params.model.id
        if not ckpt:
            raise ValueError("ComfyRunpod: model.id (checkpoint) is required")

        strength = params.strength or 0.65
        steps = params.num_inference_steps or 20
        seed = params.seed if params.seed is not None else random.randint(0, 2**31 - 1)
        cfg = params.guidance_scale or 7
        negative_prompt = params.negative_prompt or ""
        scheduler = params.scheduler or "normal"

        image_name = f"nodetool_comfy_input_{uuid.uuid4().hex}.png"
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        upload_payload = [{"name": image_name, "image": f"data:image/png;base64,{image_b64}"}]

        if ckpt == "flux1-dev-fp8.safetensors":
            flux_guidance = params.guidance_scale or 3.5
            graph: dict[str, Any] = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": image_name, "subfolder": "", "type": "input"},
                },
                "2": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["1", 0], "vae": ["30", 2]},
                },
                "30": {
                    "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"},
                    "class_type": "CheckpointLoaderSimple",
                    "_meta": {"title": "Load Checkpoint"},
                },
                "6": {
                    "inputs": {"text": params.prompt, "clip": ["30", 1]},
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
                },
                "33": {
                    "inputs": {"text": negative_prompt, "clip": ["30", 1]},
                    "class_type": "CLIPTextEncode",
                    "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
                },
                "35": {
                    "inputs": {"guidance": flux_guidance, "conditioning": ["6", 0]},
                    "class_type": "FluxGuidance",
                    "_meta": {"title": "FluxGuidance"},
                },
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "cfg": 1,
                        "denoise": strength,
                        "latent_image": ["2", 0],
                        "model": ["30", 0],
                        "negative": ["33", 0],
                        "positive": ["35", 0],
                        "sampler_name": "euler",
                        "scheduler": "simple",
                        "seed": seed,
                        "steps": steps,
                    },
                    "_meta": {"title": "KSampler"},
                },
                "8": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["3", 0], "vae": ["30", 2]},
                },
                "9": {
                    "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
                    "class_type": "SaveImage",
                    "_meta": {"title": "Save Image"},
                },
            }

            images = await self._to_thread(self._execute, graph, upload_payload)
        else:
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
            }
            images = await self._to_thread(self._execute, graph, upload_payload)

        return images[0] if images else b""

    async def _to_thread(self, func, *args, **kwargs):
        # Lazy import to avoid importing asyncio at module top in case of isolated envs
        import asyncio

        return await asyncio.to_thread(func, *args, **kwargs)
