"""
Hyper3D (Rodin) provider implementation for 3D model generation.

Hyper3D API Documentation: https://developer.hyper3d.ai/api-specification/overview

Supported endpoints:
- /api/v2/rodin (Gen-2 Generation) - text-to-3D and image-to-3D
- /api/v2/status (Check Status)
- /api/v2/download (Download Results)
- /api/v2/check_balance (Check Balance)
- /api/v2/bang (Model Segmentation)
- /api/v2/rodin_texture_only (Generate Texture)
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import aiohttp

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Model3DModel,
    Provider,
)
from nodetool.providers.base import BaseProvider, register_provider

if TYPE_CHECKING:
    from nodetool.providers.types import (
        ImageTo3DParams,
        TextTo3DParams,
    )

log = get_logger(__name__)

HYPER3D_API_BASE_URL = "https://api.hyper3d.com/api"

RODIN_3D_MODELS = [
    Model3DModel(
        id="rodin-gen-2",
        name="Rodin Gen-2",
        provider=Provider.Rodin,
        supported_tasks=["text_to_3d", "image_to_3d"],
        output_formats=["glb", "fbx", "obj", "usdz", "stl"],
    ),
]


@register_provider(Provider.Rodin)
class RodinProvider(BaseProvider):
    """Provider for Hyper3D (Rodin) 3D generation services.
    
    Pricing:
    - Gen-2 Generation: 0.5 credits (base), +1 credit for HighPack
    - Bang! (segmentation): 0.5 credits
    - Texture Generation: 0.5 credits
    """

    provider_name = "rodin"
    _poll_interval: float = 5.0
    _max_poll_attempts: int = 120

    @classmethod
    def required_secrets(cls) -> list[str]:
        return ["RODIN_API_KEY"]

    def __init__(self, secrets: dict[str, str] | None = None, **kwargs: Any):
        super().__init__(secrets=secrets)
        self.api_key = secrets.get("RODIN_API_KEY") if secrets else None
        if not self.api_key:
            log.warning("Rodin API key not configured")

    def _get_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_json_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

    # =========================================================================
    # Check Balance API
    # =========================================================================
    async def check_balance(self) -> int:
        """Check remaining credits in account.
        
        Endpoint: GET /api/v2/check_balance
        Cost: Free
        
        Returns:
            Balance of credits
        """
        url = f"{HYPER3D_API_BASE_URL}/v2/check_balance"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self._get_auth_headers()) as response:
                if response.status != 200:
                    raise RuntimeError(f"Check balance failed ({response.status}): {await response.text()}")
                result = await response.json()
                return result.get("balance", 0)

    # =========================================================================
    # Check Status API
    # =========================================================================
    async def _check_status(self, session: aiohttp.ClientSession, subscription_key: str) -> dict[str, Any]:
        """Check task status using subscription_key.
        
        Endpoint: POST /api/v2/status
        Body: {"subscription_key": "..."}
        Cost: Free
        """
        url = f"{HYPER3D_API_BASE_URL}/v2/status"
        payload = {"subscription_key": subscription_key}
        
        async with session.post(url, json=payload, headers=self._get_json_headers()) as response:
            if response.status != 200:
                raise RuntimeError(f"Status check error ({response.status}): {await response.text()}")
            return await response.json()

    async def _poll_task_status(
        self, 
        session: aiohttp.ClientSession, 
        subscription_key: str, 
        poll_interval: float, 
        max_attempts: int
    ) -> None:
        """Poll for task completion."""
        for attempt in range(max_attempts):
            status_data = await self._check_status(session, subscription_key)
            jobs = status_data.get("jobs", [])
            
            if not jobs:
                log.debug(f"Hyper3D: No jobs yet (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(poll_interval)
                continue
            
            all_done = True
            for job in jobs:
                if isinstance(job, dict):
                    status = job.get("status", "").lower()
                else:
                    status = str(job).lower()
                    
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(f"Hyper3D task failed: {job}")
                if status not in ("done",):
                    all_done = False
            
            if all_done and jobs:
                log.debug(f"Hyper3D task completed after {attempt + 1} attempts")
                return
            
            log.debug(f"Hyper3D: In progress (attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(poll_interval)
        
        raise RuntimeError(f"Task timed out after {max_attempts * poll_interval}s")

    # =========================================================================
    # Download Results API
    # =========================================================================
    async def _download_result(self, session: aiohttp.ClientSession, task_uuid: str) -> bytes:
        """Download result using task_uuid.
        
        Endpoint: POST /api/v2/download
        Body: {"task_uuid": "..."}
        Cost: Free
        """
        url = f"{HYPER3D_API_BASE_URL}/v2/download"
        payload = {"task_uuid": task_uuid}
        
        async with session.post(url, json=payload, headers=self._get_json_headers()) as response:
            if response.status != 200:
                raise RuntimeError(f"Download info error ({response.status}): {await response.text()}")
            result = await response.json()
            
            items = result.get("list", [])
            model_url = None
            
            for item in items:
                name = item.get("name", "").lower()
                if any(ext in name for ext in [".glb", ".fbx", ".obj", ".usdz", ".stl"]):
                    model_url = item.get("url")
                    break
            
            if not model_url and items:
                model_url = items[0].get("url")
            
            if not model_url:
                raise RuntimeError(f"No download URL found: {result}")
        
        async with session.get(model_url) as response:
            if response.status != 200:
                raise RuntimeError(f"Download failed ({response.status})")
            return await response.read()

    # =========================================================================
    # Gen-2 Generation API (Text-to-3D / Image-to-3D)
    # =========================================================================
    async def _submit_rodin_task(
        self,
        session: aiohttp.ClientSession,
        images: list[bytes] | None = None,
        prompt: str | None = None,
        seed: int | None = None,
        geometry_file_format: str = "glb",
        tier: str = "Gen-2",
        material: str = "PBR",
        quality: str = "medium",
    ) -> tuple[str, str]:
        """Submit a Rodin Gen-2 generation task.
        
        Endpoint: POST /api/v2/rodin (multipart/form-data)
        Cost: 0.5 credits (base)
        
        Returns:
            Tuple of (task_uuid, subscription_key)
        """
        url = f"{HYPER3D_API_BASE_URL}/v2/rodin"
        data = aiohttp.FormData()
        data.add_field("tier", tier)
        data.add_field("geometry_file_format", geometry_file_format.lower())
        data.add_field("material", material)
        data.add_field("quality", quality)
        
        if prompt:
            data.add_field("prompt", prompt)
        if seed is not None and seed >= 0:
            data.add_field("seed", str(min(seed, 65535)))
        
        if images:
            for i, image_bytes in enumerate(images):
                if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                    content_type, filename = "image/png", f"image_{i}.png"
                elif image_bytes[:3] == b"\xff\xd8\xff":
                    content_type, filename = "image/jpeg", f"image_{i}.jpg"
                else:
                    content_type, filename = "image/png", f"image_{i}.png"
                data.add_field("images", image_bytes, filename=filename, content_type=content_type)

        async with session.post(url, data=data, headers=self._get_auth_headers()) as response:
            response_text = await response.text()
            if response.status not in (200, 202):
                raise RuntimeError(f"Hyper3D API error ({response.status}): {response_text}")
            result = json.loads(response_text)
            if result.get("error"):
                raise RuntimeError(f"Hyper3D API error: {result.get('error')} - {result.get('message')}")
            
            task_uuid = result.get("uuid")
            subscription_key = result.get("jobs", {}).get("subscription_key")
            
            if not task_uuid:
                raise RuntimeError(f"No task UUID returned: {result}")
            if not subscription_key:
                raise RuntimeError(f"No subscription key returned: {result}")
            
            return task_uuid, subscription_key

    # =========================================================================
    # Bang! API (Model Segmentation)
    # =========================================================================
    async def bang_segment_model(
        self,
        asset_id: str | None = None,
        model_bytes: bytes | None = None,
        image_bytes: bytes | None = None,
        prompt: str | None = None,
        strength: int = 5,
        geometry_file_format: str = "glb",
        material: str = "PBR",
        resolution: str = "Basic",
    ) -> bytes:
        """Split a 3D model into multiple submodels using Bang! API.
        
        Endpoint: POST /api/v2/bang (multipart/form-data)
        Cost: 0.5 credits
        
        Args:
            asset_id: UUID of a previous Rodin Gen-2 task (mutually exclusive with model_bytes)
            model_bytes: Custom model file bytes (mutually exclusive with asset_id)
            image_bytes: Optional reference image for texture generation (only with model_bytes)
            prompt: Optional prompt for texture generation (only with model_bytes)
            strength: Splitting strength (2-12, default 5). Higher = more pieces
            geometry_file_format: Output format (glb, fbx, obj, stl, usdz)
            material: Material type (PBR, Shaded, None, All)
            resolution: Texture resolution (Basic=2K, High=4K)
            
        Returns:
            Raw bytes of segmented 3D model
        """
        if not asset_id and not model_bytes:
            raise ValueError("Either asset_id or model_bytes must be provided")
        if asset_id and model_bytes:
            raise ValueError("asset_id and model_bytes are mutually exclusive")
        
        url = f"{HYPER3D_API_BASE_URL}/v2/bang"
        data = aiohttp.FormData()
        data.add_field("strength", str(max(2, min(12, strength))))
        data.add_field("geometry_file_format", geometry_file_format.lower())
        data.add_field("material", material)
        data.add_field("resolution", resolution)
        
        if asset_id:
            data.add_field("asset_id", asset_id)
        
        if model_bytes:
            data.add_field("model", model_bytes, filename="model.glb", content_type="application/octet-stream")
            if image_bytes:
                data.add_field("image", image_bytes, filename="reference.jpg", content_type="image/jpeg")
            if prompt:
                data.add_field("prompt", prompt)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._get_auth_headers()) as response:
                response_text = await response.text()
                if response.status not in (200, 202):
                    raise RuntimeError(f"Bang! API error ({response.status}): {response_text}")
                result = json.loads(response_text)
                if result.get("error"):
                    raise RuntimeError(f"Bang! API error: {result.get('error')} - {result.get('message')}")
                
                task_uuid = result.get("uuid")
                subscription_key = result.get("jobs", {}).get("subscription_key")
                
                if not task_uuid or not subscription_key:
                    raise RuntimeError(f"Invalid response from Bang! API: {result}")
            
            await self._poll_task_status(session, subscription_key, self._poll_interval, self._max_poll_attempts)
            return await self._download_result(session, task_uuid)

    # =========================================================================
    # Generate Texture API
    # =========================================================================
    async def generate_texture(
        self,
        model_bytes: bytes,
        image_bytes: bytes,
        prompt: str | None = None,
        seed: int | None = None,
        reference_scale: float = 1.0,
        geometry_file_format: str = "glb",
        material: str = "PBR",
        resolution: str = "Basic",
    ) -> bytes:
        """Generate textures for a 3D model.
        
        Endpoint: POST /api/v2/rodin_texture_only (multipart/form-data)
        Cost: 0.5 credits
        
        Args:
            model_bytes: 3D model file bytes (max 10MB)
            image_bytes: Reference image for texture generation
            prompt: Optional texture description
            seed: Random seed (0-65535)
            reference_scale: Reference scale for texture generation
            geometry_file_format: Output format (glb, fbx, obj, stl, usdz)
            material: Material type (PBR, Shaded)
            resolution: Texture resolution (Basic=2K, High=4K)
            
        Returns:
            Raw bytes of textured 3D model
        """
        url = f"{HYPER3D_API_BASE_URL}/v2/rodin_texture_only"
        data = aiohttp.FormData()
        data.add_field("model", model_bytes, filename="model.glb", content_type="application/octet-stream")
        data.add_field("image", image_bytes, filename="reference.jpg", content_type="image/jpeg")
        data.add_field("reference_scale", str(reference_scale))
        data.add_field("geometry_file_format", geometry_file_format.lower())
        data.add_field("material", material)
        data.add_field("resolution", resolution)
        
        if prompt:
            data.add_field("prompt", prompt)
        if seed is not None and seed >= 0:
            data.add_field("seed", str(min(seed, 65535)))
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=self._get_auth_headers()) as response:
                response_text = await response.text()
                if response.status not in (200, 202):
                    raise RuntimeError(f"Texture API error ({response.status}): {response_text}")
                result = json.loads(response_text)
                if result.get("error"):
                    raise RuntimeError(f"Texture API error: {result.get('error')} - {result.get('message')}")
                
                task_uuid = result.get("uuid")
                subscription_key = result.get("jobs", {}).get("subscription_key")
                
                if not task_uuid or not subscription_key:
                    raise RuntimeError(f"Invalid response from Texture API: {result}")
            
            await self._poll_task_status(session, subscription_key, self._poll_interval, self._max_poll_attempts)
            return await self._download_result(session, task_uuid)

    # =========================================================================
    # BaseProvider Interface Methods
    # =========================================================================
    async def get_available_3d_models(self) -> list[Model3DModel]:
        if not self.api_key:
            return []
        return RODIN_3D_MODELS

    async def text_to_3d(
        self, 
        params: TextTo3DParams, 
        timeout_s: int | None = None, 
        context: Any = None, 
        node_id: str | None = None
    ) -> bytes:
        """Generate a 3D model from text prompt.
        
        Cost: 0.5 credits
        """
        log.info(f"Generating 3D with Hyper3D Rodin (text-to-3D)")
        
        if not params.prompt:
            raise ValueError("prompt must not be empty")
        if not self.api_key:
            raise ValueError("Rodin API key is not configured")
        
        poll_interval, max_attempts = self._poll_interval, self._max_poll_attempts
        if timeout_s and timeout_s > 0:
            max_attempts = max(1, int(timeout_s / poll_interval))
        
        async with aiohttp.ClientSession() as session:
            task_uuid, subscription_key = await self._submit_rodin_task(
                session, 
                images=None, 
                prompt=params.prompt, 
                seed=params.seed, 
                geometry_file_format=params.output_format,
                tier="Gen-2"
            )
            log.debug(f"Task submitted: uuid={task_uuid}")
            
            await self._poll_task_status(session, subscription_key, poll_interval, max_attempts)
            return await self._download_result(session, task_uuid)

    async def image_to_3d(
        self, 
        image: bytes, 
        params: ImageTo3DParams, 
        timeout_s: int | None = None, 
        context: Any = None, 
        node_id: str | None = None
    ) -> bytes:
        """Generate a 3D model from image.
        
        Cost: 0.5 credits
        """
        log.info(f"Generating 3D with Hyper3D Rodin (image-to-3D)")
        
        if not image:
            raise ValueError("image must not be empty")
        if not self.api_key:
            raise ValueError("Rodin API key is not configured")
        
        poll_interval, max_attempts = self._poll_interval, self._max_poll_attempts
        if timeout_s and timeout_s > 0:
            max_attempts = max(1, int(timeout_s / poll_interval))
        
        async with aiohttp.ClientSession() as session:
            task_uuid, subscription_key = await self._submit_rodin_task(
                session, 
                images=[image], 
                prompt=params.prompt, 
                seed=params.seed, 
                geometry_file_format=params.output_format,
                tier="Gen-2"
            )
            log.debug(f"Task submitted: uuid={task_uuid}")
            
            await self._poll_task_status(session, subscription_key, poll_interval, max_attempts)
            return await self._download_result(session, task_uuid)
