from __future__ import annotations

import asyncio
import base64
import imaplib
import inspect
import json
import os
import queue
import threading
import urllib.parse
import uuid
from contextlib import suppress
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    import builtins

    import numpy as np
    import pandas as pd
    import PIL.Image
    import PIL.ImageOps
    from chromadb.api import ClientAPI
    from pydub import AudioSegment
    from sklearn.base import BaseEstimator

    from nodetool.types.message_types import MessageCreateRequest
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.property import Property
    from nodetool.workflows.types import ProcessingMessage


try:  # Optional dependency used by browser helpers
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright
except ImportError:  # pragma: no cover - playwright is optional
    async_playwright = None  # type: ignore
    Browser = BrowserContext = Page = object  # type: ignore


from io import BytesIO
from pickle import loads
from typing import IO, Any, AsyncGenerator, Callable

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
)
from nodetool.io.uri_utils import create_file_uri as _create_file_uri
from nodetool.media.common.media_constants import (
    DEFAULT_AUDIO_SAMPLE_RATE,
)
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    DataframeRef,
    FontRef,
    ImageRef,
    Model3DRef,
    ModelRef,
    NPArray,
    Provider,
    TextRef,
    VideoRef,
    asset_types,
)
from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.message import Message as DBMessage
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import require_scope
from nodetool.types.prediction import (
    Prediction,
    PredictionResult,
)
from nodetool.workflows.channel import ChannelManager
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_offload import (
    _audio_segment_from_file,
    _audio_segment_to_mp3_bytes,
    _audio_segment_to_numpy,
    _audio_segment_to_wav_bytes,
    _b64decode_to_bytes,
    _b64encode_to_str,
    _in_thread,
    _joblib_dump_to_bytes,
    _joblib_load_from_io,
    _numpy_audio_to_mp3_bytes,
    _numpy_image_to_png_bytes,
    _numpy_video_to_mp4_bytes,
    _open_image_as_rgb,
    _pil_to_jpeg_bytes,
    _pil_to_png_bytes,
    _pil_to_png_bytes_with_exif,
    _read_all_bytes_from_start,
    _read_base64,
    _read_utf8,
)
from nodetool.workflows.torch_support import (
    TORCH_AVAILABLE,
    detach_tensors_recursively,
    is_torch_tensor,
    tensor_from_pil,
    tensor_to_image_array,
    torch_tensor_to_metadata,
)

log = get_logger(__name__)



def _ensure_numpy():
    import numpy as np

    return np


def _ensure_pandas():
    import pandas as pd

    return pd


def _ensure_pil():
    import PIL.Image
    import PIL.ImageOps

    return PIL.Image, PIL.ImageOps


def _ensure_audio_segment():
    from pydub import AudioSegment

    return AudioSegment


def _ensure_joblib():
    import joblib

    return joblib


def _numpy_to_pil_image_util(arr: np.ndarray):
    from nodetool.media.image.image_utils import numpy_to_pil_image

    return numpy_to_pil_image(arr)


def _export_to_video_bytes(
    video_frames,
    fps: int = 10,
    quality: float = 5.0,
    bitrate: int | None = None,
    macro_block_size: int | None = 16,
):
    from nodetool.media.video.video_utils import export_to_video_bytes as _exporter

    return _exporter(
        video_frames,
        fps=fps,
        quality=quality,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    )


def create_file_uri(path: str) -> str:
    """
    Compatibility wrapper delegating to nodetool.io.uri_utils.create_file_uri.
    """
    return _create_file_uri(path)


## AUDIO_CODEC and DEFAULT_AUDIO_SAMPLE_RATE imported from media_constants

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _resolve_default_device(explicit_device: str | None = None) -> str | None:
    """
    Pick the default execution device for workflows.

    Prefers Apple Metal (MPS) when available so that HuggingFace workloads run on
    the GPU by default, otherwise falls back to CUDA (if available) or CPU.
    """
    if explicit_device:
        return explicit_device

    try:
        import torch  # type: ignore

        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        if hasattr(torch, "cuda"):
            try:
                if torch.cuda.is_available():
                    return "cuda"
            except (RuntimeError, AttributeError):
                # CUDA not compiled in or other runtime issue
                pass
    except Exception:
        # torch may be unavailable during installation or CPU-only deployments
        pass

    return "cpu"


class AssetOutputMode(str, Enum):
    """Controls how assets are materialized when emitting workflow messages."""

    PYTHON = "python"
    DATA_URI = "data_uri"
    TEMP_URL = "temp_url"
    STORAGE_URL = "storage_url"
    WORKSPACE = "workspace"
    RAW = "raw"


# 3D model format to MIME type and extension mapping
MODEL_3D_FORMAT_MAPPING: dict[str, tuple[str, str]] = {
    "glb": ("model/gltf-binary", "glb"),
    "gltf": ("model/gltf+json", "gltf"),
    "obj": ("model/obj", "obj"),
    "stl": ("model/stl", "stl"),
    "ply": ("application/x-ply", "ply"),
    "fbx": ("application/octet-stream", "fbx"),
    "usdz": ("model/vnd.usdz+zip", "usdz"),
}


class ProcessingContext:
    """
    The processing context is the workflow's interface to the outside world.
    It maintains the state of the workflow and provides methods for interacting with the environment.

    Initialization and State Management:
    - Initializes the context with user ID, authentication token, workflow ID, graph edges, nodes and a message queue.
    - Manages the results of processed nodes and keeps track of processed nodes.
    - Provides methods for popping and posting messages to the message queue.

    Asset Management:
    - Provides methods for finding, downloading, and creating assets (images, audio, text, video, dataframes, models).
    - Handles conversions between different asset formats (e.g., PIL Image to ImageRef, numpy array to ImageRef).
    - Generates presigned URLs for accessing assets.

    API and Storage Integration:
    - Interacts with the Nodetool API client for asset-related operations.
    - Retrieves and manages asset storage and temporary storage instances.
    - Handles file uploads and downloads to/from storage services.

    Utility Methods:
    - Provides helper methods for converting values for prediction, handling enums, and parsing S3 URLs.
    - Supports data conversion between different formats (e.g., TextRef to string, DataFrame to pandas DataFrame).
    """

    def __init__(
        self,
        user_id: str | None = None,
        auth_token: str | None = None,
        workflow_id: str | None = None,
        job_id: str | None = None,
        graph: Graph | None = None,
        variables: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        message_queue: queue.Queue | None = None,
        device: str | None = None,
        encode_assets_as_base64: bool = False,
        upload_assets_to_s3: bool = False,
        asset_output_mode: AssetOutputMode | None = None,
        chroma_client: ClientAPI | None = None,
        workspace_dir: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        tool_bridge: Any | None = None,
        ui_tool_names: builtins.set[str] | None = None,
        client_tools_manifest: dict[str, dict] | None = None,
    ):
        self.user_id = user_id or "1"
        self.auth_token = auth_token or "local_token"
        self.workflow_id = workflow_id or ""
        self.job_id = job_id
        self.graph = graph or Graph()
        self.message_queue = message_queue if message_queue else queue.Queue()
        self.device = _resolve_default_device(device)
        self.variables: dict[str, Any] = variables if variables else {}
        self.environment: dict[str, str] = Environment.get_environment()
        if environment:
            self.environment.update(environment)
        assert self.auth_token is not None, "Auth token is required"
        self.encode_assets_as_base64 = encode_assets_as_base64
        self.upload_assets_to_s3 = upload_assets_to_s3
        if asset_output_mode is None:
            if encode_assets_as_base64:
                self.asset_output_mode = AssetOutputMode.DATA_URI
            elif upload_assets_to_s3:
                self.asset_output_mode = AssetOutputMode.STORAGE_URL
            else:
                self.asset_output_mode = AssetOutputMode.PYTHON
        else:
            self.asset_output_mode = asset_output_mode
        self.chroma_client = chroma_client
        # HTTP client is now managed by ResourceScope to ensure correct event loop binding
        # Store passed client only as fallback if no scope is available
        if http_client is not None:
            self._http_client = http_client
        self.workspace_dir = workspace_dir  # User-defined workspace only; None if not provided
        self.tool_bridge = tool_bridge
        self.ui_tool_names = ui_tool_names or set()
        self.client_tools_manifest = client_tools_manifest or {}
        # Store current status for each node and edge for reconnection
        self.node_statuses: dict[str, ProcessingMessage] = {}
        self.edge_statuses: dict[str, ProcessingMessage] = {}
        # Streaming channels for named, many-to-many communication
        self.channels = ChannelManager()

    def _numpy_to_pil_image(self, arr: np.ndarray) -> PIL.Image.Image:
        """Delegate to shared numpy_to_pil_image utility for consistent behavior."""
        return _numpy_to_pil_image_util(arr)

    def _memory_get(self, key: str) -> Any | None:
        """
        Retrieve an object from the ResourceScope's memory URI cache.

        Uses the current ResourceScope's memory URI cache for proper
        per-execution isolation.
        """
        import threading

        thread_id = threading.get_ident()
        try:
            value = require_scope().get_memory_uri_cache().get(key)
            log.debug(f"Memory GET '{key}' on thread {thread_id}: {'HIT' if value is not None else 'MISS'}")
            return value
        except RuntimeError:
            # No scope bound - return None
            log.warning(f"Memory GET '{key}' failed: no ResourceScope bound")
            return None

    def _memory_set(self, key: str, value: Any) -> None:
        """
        Store an object under a URI (e.g., memory://<id>) in the ResourceScope's
        memory URI cache.
        """
        import threading

        thread_id = threading.get_ident()
        log.info(f"Setting memory URI cache: {key} on thread {thread_id}")
        try:
            require_scope().get_memory_uri_cache().set(key, value)
        except RuntimeError:
            # No scope bound - log warning
            log.warning(f"Memory SET '{key}' failed: no ResourceScope bound")

    async def _http_request_with_retries(
        self,
        method: str,
        url: str,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        **kwargs,
    ) -> httpx.Response:
        """
        Perform an HTTP request with basic retries for transient transport errors.

        Retries are applied primarily to GET/HEAD requests where it is safe to do so.
        """
        _headers = HTTP_HEADERS.copy()
        _headers.update(kwargs.get("headers", {}))
        kwargs["headers"] = _headers

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await require_scope().get_http_client().request(method, url, **kwargs)
                status = response.status_code
                log.info(f"{method.upper()} {url} {status}")
                # Retry on common transient statuses
                if status in {408, 425, 429, 500, 502, 503, 504}:
                    if attempt == max_retries - 1:
                        response.raise_for_status()
                        return response
                    # Honor Retry-After if present
                    retry_after = response.headers.get("Retry-After")
                    try:
                        header_delay = float(retry_after) if retry_after else None
                    except Exception:
                        header_delay = None
                    delay = header_delay if header_delay is not None else backoff_seconds * (2**attempt)
                    log.warning(
                        f"{method.upper()} {url} got {status}; retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Success and non-retry statuses
                response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.ReadTimeout) as e:
                # Includes RemoteProtocolError (e.g., incomplete chunked read)
                last_exc = e
                if attempt == max_retries - 1:
                    raise
                delay = backoff_seconds * (2**attempt)
                log.warning(
                    f"{method.upper()} {url} transport error: {e}; retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
            except Exception:
                # Non-transport errors: do not retry by default
                raise
        # Should not reach; raise last exception if present
        if last_exc:
            raise last_exc
        raise RuntimeError("HTTP request failed without exception")

    async def get_secret(self, key: str) -> str | None:
        """
        Get a secret value.
        """
        from nodetool.security.secret_helper import get_secret

        return await get_secret(key, self.user_id)

    async def get_secret_required(self, key: str) -> str:
        """
        Get a required secret value.
        """
        from nodetool.security.secret_helper import get_secret_required

        return await get_secret_required(key, self.user_id)

    async def get_provider(self, provider_type: Provider | str):
        """
        Get an AI provider instance.

        Args:
            provider_type (Provider | str): The provider type enum or string

        Returns:
            BaseProvider: A provider instance

        Raises:
            ValueError: If the provider type is not supported
        """
        from nodetool.providers import get_provider

        provider_enum = Provider(provider_type) if isinstance(provider_type, str) else provider_type

        provider = await get_provider(provider_enum, self.user_id)

        # Defensive check: if provider is still awaitable, await it again
        # This handles edge cases where get_provider might return a coroutine
        if inspect.isawaitable(provider):
            log.warning(f"Provider was still awaitable after await, re-awaiting. type={type(provider)}")
            provider = await provider

        if not hasattr(provider, "generate_messages"):
            log.error(
                f"Provider missing generate_messages method. type={type(provider)}, "
                f"attributes={[x for x in dir(provider) if not x.startswith('_')][:10]}"
            )
            raise ValueError(
                f"Provider {type(provider)} does not have generate_messages method. "
                f"This indicates get_provider returned an unexpected type."
            )

        return provider

    def copy(self):
        """
        Creates a copy of the current ProcessingContext with shared references to most properties.

        Returns:
            ProcessingContext: A new ProcessingContext instance with copied properties.
        """
        return ProcessingContext(
            graph=self.graph,
            user_id=self.user_id,
            auth_token=self.auth_token,
            workflow_id=self.workflow_id,
            message_queue=self.message_queue,
            device=self.device,
            variables=self.variables,
            environment=self.environment,
            tool_bridge=self.tool_bridge,
            ui_tool_names=self.ui_tool_names.copy() if self.ui_tool_names else set(),
            client_tools_manifest=(self.client_tools_manifest.copy() if self.client_tools_manifest else {}),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets the value of a variable from the context.
        This is also used to set and retrieve api keys.

        Args:
            key (str): The key of the variable.
            default (Any, optional): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value of the variable.
        """
        return self.variables.get(key, default)

    def set(self, key: str, value: Any):
        """
        Sets the value of a variable in the context.

        Args:
            key (str): The key of the variable.
            value (Any): The value to set.
        """
        self.variables[key] = value
        self._persist_variable_if_needed(key, value)

    def store_step_result(self, key: str, value: Any) -> str:
        """Persist a subtask result to the workspace root and memoize a reference."""

        path = self._workspace_path(f"{key}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(value, fp, ensure_ascii=False, indent=2)
        rel_name = path.name
        self.variables[key] = {"__workspace_result__": rel_name}
        return str(path)

    def load_step_result(self, key: str, default: Any = None) -> Any:
        marker = self.variables.get(key)
        if isinstance(marker, dict) and "__workspace_result__" in marker:
            path = self._workspace_path(marker["__workspace_result__"])
            if path.exists():
                try:
                    with path.open("r", encoding="utf-8") as fp:
                        return json.load(fp)
                except Exception:
                    return default
            return default
        return self.variables.get(key, default)

    def _persist_variable_if_needed(self, key: str, value: Any) -> None:
        marker = self.variables.get(key)
        if isinstance(marker, dict) and "__workspace_result__" in marker:
            return
        if not isinstance(value, dict | list | str | int | float | bool):
            return
        try:
            path = self._workspace_path(f"var_{key}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fp:
                json.dump(value, fp, ensure_ascii=False, indent=2)
        except Exception as e:
            log.debug(f"Failed to persist variable '{key}': {e}")

    def _workspace_path(self, filename: str) -> Path:
        if not self.workspace_dir:
            raise ValueError("workspace_dir is required to store subtask results")
        return Path(self.workspace_dir) / filename

    async def pop_message_async(self) -> ProcessingMessage:
        """
        Retrieves and removes a message from the message queue.
        The message queue is used to communicate updates to upstream
        processing.

        Returns:
            The retrieved message from the message queue.
        """
        return await _in_thread(self.message_queue.get)

    # def pop_message(self) -> ProcessingMessage:
    #     """
    #     Removes and returns the next message from the message queue.

    #     Returns:
    #         The next message from the message queue.
    #     """
    #     assert isinstance(self.message_queue, Queue)
    #     return self.message_queue.get()

    def post_message(self, message: ProcessingMessage):
        """
        Posts a message to the message queue.

        Args:
            message (ProcessingMessage): The message to be posted.
        """
        self.message_queue.put_nowait(message)

        # Store latest status for each node and edge for reconnection replay
        from nodetool.workflows.types import EdgeUpdate, NodeUpdate

        if isinstance(message, NodeUpdate):
            self.node_statuses[message.node_id] = message
        elif isinstance(message, EdgeUpdate):
            self.edge_statuses[message.edge_id] = message

    def has_messages(self) -> bool:
        """
        Checks if the processing context has any messages in the message queue.

        Returns:
            bool: True if the message queue is not empty, False otherwise.
        """
        return self.message_queue.qsize() != 0

    async def asset_storage_url(self, key: str) -> str:
        """
        Returns the URL of an asset in the asset storage.

        Args:
            key (str): The key of the asset.
        """
        return await require_scope().get_asset_storage().get_url(key)

    def generate_node_cache_key(
        self,
        node: BaseNode,
    ) -> str:
        """Generate a cache key for a node based on current user, node type and properties."""
        return f"{self.user_id}:{node.get_node_type()}:{hash(repr(node.model_dump()))}"

    def get_cached_result(self, node: BaseNode) -> Any:
        """
        Get the cached result for a node.

        Args:
            node (BaseNode): The node to get the cached result for.

        Returns:
            Any: The cached result, or None if not found.
        """
        key = self.generate_node_cache_key(node)
        val = require_scope().get_node_cache().get(key)
        return val

    def cache_result(self, node: BaseNode, result: Any, ttl: int = 3600):
        """
        Cache the result for a node.

        Args:
            node (BaseNode): The node to cache the result for.
            result (Any): The result to cache.
            ttl (int, optional): Time to live in seconds. Defaults to 3600 (1 hour).
        """
        all_cacheable = all(out.type.is_cacheable_type() for out in node.outputs())

        if all_cacheable:
            key = self.generate_node_cache_key(node)

            cache_value = detach_tensors_recursively(result)
            require_scope().get_node_cache().set(key, cache_value, ttl)

    async def cache_result_async(self, node: BaseNode, result: Any, ttl: int = 3600) -> None:
        """
        Async variant of cache_result that offloads potentially expensive tensor detaching
        to a thread to avoid blocking the event loop.
        """
        all_cacheable = all(out.type.is_cacheable_type() for out in node.outputs())
        if not all_cacheable:
            return

        key = self.generate_node_cache_key(node)
        cache_value = await _in_thread(detach_tensors_recursively, result)
        require_scope().get_node_cache().set(key, cache_value, ttl)

    async def find_asset(self, asset_id: str):
        """
        Finds an asset by id.

        Args:
            asset_id (str): The ID of the asset.

        Returns:
            Asset: The asset with the given ID.
        """
        return await Asset.find(self.user_id, asset_id)

    async def find_asset_by_filename(self, filename: str):
        """
        Finds an asset by filename.
        """
        from nodetool.models.condition_builder import Field

        assets, _ = await Asset.query(
            Field("user_id").equals(self.user_id).and_(Field("name").equals(filename)),
            limit=1,
        )
        return assets[0] if assets else None

    async def list_assets(
        self,
        parent_id: str | None = None,
        recursive: bool = False,
        content_type: str | None = None,
    ) -> tuple[list[Asset], str | None]:
        """
        Lists assets.
        """
        if recursive:
            result = await Asset.get_assets_recursive(self.user_id, parent_id or self.user_id)
            return result["assets"], None
        else:
            assets, next_cursor = await Asset.paginate(
                user_id=self.user_id,
                parent_id=parent_id,
                content_type=content_type,
                limit=1000,
            )
            return assets, next_cursor

    async def get_asset_url(self, asset_id: str):
        """
        Returns the asset url.

        Args:
            asset_id (str): The ID of the asset.

        Returns:
            str: The URL of the asset.
        """
        asset = await self.find_asset(asset_id)  # type: ignore
        if asset is None:
            raise ValueError(f"Asset with ID {asset_id} not found")
        return await self.asset_storage_url(asset.file_name)

    async def get_workflow(self, workflow_id: str):
        """
        Gets the workflow by ID.

        Args:
            workflow_id (str): The ID of the workflow to retrieve.

        Returns:
            Workflow: The retrieved workflow.
        """
        return await Workflow.find(self.user_id, workflow_id)

    async def _prepare_prediction(
        self,
        node_id: str,
        provider: str,
        model: str,
        params: dict[str, Any] | None = None,
        data: Any = None,
    ) -> Prediction:
        """Common setup for both streaming and non-streaming prediction runs."""
        if params is None:
            params = {}

        return Prediction(
            id="",
            user_id=self.user_id,
            status="",
            provider=provider,
            model=model,
            node_id=node_id,
            workflow_id=self.workflow_id if self.workflow_id else "",
            params=params,
            data=data,
        )

    async def run_prediction(
        self,
        node_id: str,
        provider: str,
        model: str,
        run_prediction_function: Callable[
            [Prediction, dict[str, str]],
            AsyncGenerator[Any, None],
        ],
        params: dict[str, Any] | None = None,
        data: Any = None,
    ) -> Any:
        """
        Run a prediction on a third-party provider and return the final result.

        Args:
            node_id (str): The ID of the node making the prediction.
            provider (Provider): The provider to use for the prediction.
            model (str): The model to use for the prediction.
            run_prediction_function (Callable[[Prediction], AsyncGenerator[PredictionResult | Prediction | ChatResponse, None]]): A function to run the prediction.
            params (dict[str, Any] | None, optional): Parameters for the prediction. Defaults to None.
            data (Any, optional): Data for the prediction. Defaults to None.
        Returns:
            Any: The prediction result.

        Raises:
            ValueError: If the prediction did not return a result.
        """
        from nodetool.models.prediction import Prediction as PredictionModel

        prediction = await self._prepare_prediction(node_id, provider, model, params, data)

        started_at = datetime.now()
        async for msg in run_prediction_function(prediction, self.environment):
            if isinstance(msg, PredictionResult):
                await PredictionModel.create(
                    user_id=self.user_id,
                    node_id=node_id,
                    provider=provider,
                    model=model,
                    workflow_id=self.workflow_id,
                    status="completed",
                    cost=0,
                    hardware="cpu",
                    created_at=started_at,
                    started_at=started_at,
                    completed_at=datetime.now(),
                    duration=(datetime.now() - started_at).total_seconds(),
                )
                return msg.decode_content()
            elif isinstance(msg, Prediction):
                self.post_message(msg)

        raise ValueError("Prediction did not return a result")

    async def stream_prediction(
        self,
        node_id: str,
        provider: Provider,
        model: str,
        run_prediction_function: Callable[
            [Prediction, dict[str, str]],
            AsyncGenerator[Any, None],
        ],
        params: dict[str, Any] | None = None,
        data: Any = None,
    ) -> AsyncGenerator[Any, None]:
        """
        Stream prediction results from a third-party provider.

        Args:
            node_id (str): The ID of the node making the prediction.
            provider (Provider): The provider to use for the prediction.
            model (str): The model to use for the prediction.
            run_prediction_function (Callable[[Prediction], AsyncGenerator[PredictionResult | Prediction | ChatResponse, None]]): A function to run the prediction.
            params (dict[str, Any] | None, optional): Parameters for the prediction. Defaults to None.
            data (Any, optional): Data for the prediction. Defaults to None.

        Returns:
            AsyncGenerator[PredictionResult | Prediction | ChatResponse, None]: An async generator yielding prediction results.
        """
        prediction = await self._prepare_prediction(node_id, provider, model, params, data)

        async for msg in run_prediction_function(prediction, self.environment):
            yield msg

    async def refresh_uri(self, asset: AssetRef):
        """
        Refreshes the URI of the asset.

        Args:
            asset (AssetRef): The asset to refresh.
        """
        if asset.asset_id:
            asset.uri = await self.get_asset_url(asset.asset_id)

    async def get_job(self, job_id: str) -> Job | None:
        """
        Gets the status of a job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            Job: The job status.
        """
        return await Job.find(self.user_id, job_id)

    async def create_asset(
        self,
        name: str,
        content_type: str,
        content: IO | None = None,
        parent_id: str | None = None,
        instructions: IO | None = None,
        node_id: str | None = None,
    ) -> Asset:
        """
        Creates an asset with the given name, content type, content, and optional parent ID.

        Args:
            name (str): The name of the asset.
            content_type (str): The content type of the asset.
            content (IO): The content of the asset.
            parent_id (str | None, optional): The ID of the parent asset. Defaults to None.
            node_id (str | None, optional): The ID of the node that created this asset. Defaults to None.

        Returns:
            Asset: The created asset.

        """
        content = content or instructions
        if content is None:
            raise ValueError("Asset content is required")

        content_bytes = await _in_thread(_read_all_bytes_from_start, content)
        with suppress(Exception):
            content.seek(0)

        # Create the asset record in the database
        asset = await Asset.create(
            user_id=self.user_id,
            name=name,
            content_type=content_type,
            parent_id=parent_id,
            workflow_id=self.workflow_id,
            job_id=self.job_id,
            node_id=node_id,
            size=len(content_bytes),
        )

        # Upload the content to storage
        storage = require_scope().get_asset_storage()
        await storage.upload(asset.file_name, BytesIO(content_bytes))

        return asset

    async def create_message(self, req: MessageCreateRequest):
        """
        Creates a message for a thread.

        Args:
            req (MessageCreateRequest): The message to create.

        Returns:
            Message: The created message.
        """
        if not req.thread_id:
            raise ValueError("Thread ID is required")

        return await DBMessage.create(
            thread_id=req.thread_id,
            user_id=self.user_id,
            role=req.role,
            content=req.content,
            tool_calls=req.tool_calls,
            workflow_id=getattr(req, "workflow_id", None),
            name=getattr(req, "name", None),
            tool_call_id=getattr(req, "tool_call_id", None),
        )

    async def get_messages(
        self,
        thread_id: str,
        limit: int = 10,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        """
        Gets messages for a thread.

        Args:
            thread_id (str): The ID of the thread.
            limit (int, optional): The number of messages to return. Defaults to 10.
            start_key (str, optional): The start key for pagination. Defaults to None.
            reverse (bool, optional): Whether to reverse the order of messages. Defaults to False.

        Returns:
            dict: Dictionary with messages list and next cursor.
        """
        messages, next_cursor = await DBMessage.paginate(
            thread_id=thread_id, limit=limit, start_key=start_key, reverse=reverse
        )
        return {
            "messages": [message.model_dump() for message in messages],
            "next": next_cursor,
        }

    async def download_asset(self, asset_id: str) -> IO:
        """
        Downloads an asset from the asset storage api.

        Args:
            asset_id (str): The ID of the asset to download.

        Returns:
            IO: The downloaded asset.
        """
        asset = await self.find_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset {asset_id} not found")
        io = BytesIO()
        await require_scope().get_asset_storage().download(asset.file_name, io)
        io.seek(0)
        return io

    async def http_get(self, url: str, **kwargs) -> httpx.Response:
        """
        Sends an HTTP GET request to the specified URL.

        Args:
            url (str): The URL to send the request to.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            httpx.Response: The response object.
        """
        return await self._http_request_with_retries("GET", url, **kwargs)

    async def http_post(
        self,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends an HTTP POST request to the specified URL.

        Args:
            url (str): The URL to send the request to.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            httpx.Response: The response object.
        """
        # For providers that support it (e.g., OpenAI), allow callers to pass Idempotency-Key via headers
        return await self._http_request_with_retries("POST", url, **kwargs)

    async def http_patch(
        self,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends an HTTP PATCH request to the specified URL.

        Args:
            url (str): The URL to send the request to.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            httpx.Response: The response object.
        """
        return await self._http_request_with_retries("PATCH", url, **kwargs)

    async def http_put(
        self,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends an HTTP PUT request to the specified URL.

        Args:
            url (str): The URL to send the request to.
            **kwargs: Additional keyword arguments to pass to the request.

        Returns:
            httpx.Response: The response object.
        """
        return await self._http_request_with_retries("PUT", url, **kwargs)

    async def http_delete(
        self,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends an HTTP DELETE request to the specified URL.

        Args:
            url (str): The URL to send the request to.

        Returns:
            bytes: The response content.
        """
        return await self._http_request_with_retries("DELETE", url, **kwargs)

    async def http_head(
        self,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Sends an HTTP HEAD request to the specified URL.

        Args:
            url (str): The URL to send the request to.

        Returns:
            httpx.Response: The response object.
        """
        return await self._http_request_with_retries("HEAD", url, **kwargs)

    async def download_file(self, url: str) -> IO:
        """
        Download a file from URL.

        Args:
            url (str): The URL of the file to download.

        Returns:
            IO: The downloaded file as an IO object.

        Raises:
            FileNotFoundError: If the file does not exist (for local files).
        """
        # Handle local storage URLs that are provided as relative API paths
        if url.startswith("/api/storage/temp/"):
            key = url.split("/api/storage/temp/", 1)[1]
            io = BytesIO()
            await require_scope().get_temp_storage().download(key, io)
            io.seek(0)
            return io
        if url.startswith("/api/storage/"):
            key = url.split("/api/storage/", 1)[1]
            io = BytesIO()
            await require_scope().get_asset_storage().download(key, io)
            io.seek(0)
            return io

        # Handle paths that start with "/" by converting to proper file:// URI
        if url.startswith("/") and not url.startswith("//"):
            url = create_file_uri(url)

        url_parsed = urllib.parse.urlparse(url)

        # Treat empty-scheme inputs as local file paths (supports Windows drive letters)
        if url_parsed.scheme == "" and not url.startswith("data:"):
            local_path = Path(url).expanduser()
            if local_path.exists():
                content = await asyncio.to_thread(local_path.read_bytes)
                return BytesIO(content)

        if url_parsed.scheme == "data":
            fname, data = url.split(",", 1)
            image_bytes = await _in_thread(_b64decode_to_bytes, data)
            file = BytesIO(image_bytes)
            # parse file ext from data uri
            ext = fname.split(";")[0].split("/")[1]
            file.name = f"{uuid.uuid4()}.{ext}"
            return file

        if url_parsed.scheme == "file":
            # Use pathlib to handle file paths cross-platform
            try:
                netloc = url_parsed.netloc
                path_part = urllib.parse.unquote(url_parsed.path)

                # Normalize localhost netloc to empty
                if netloc and netloc.lower() == "localhost":
                    netloc = ""

                if os.name == "nt":
                    # Windows handling: drive letters and UNC paths
                    if netloc:
                        path = Path(netloc + path_part) if ":" in netloc else Path("//" + netloc + path_part)
                    else:
                        # file:///C:/path comes through as path_part="/C:/path"; strip leading slash
                        if len(path_part) >= 3 and path_part[0] == "/" and path_part[2] == ":":
                            path_part = path_part.lstrip("/")
                        path = Path(path_part)
                else:
                    # POSIX: netloc is typically empty or localhost; for others, treat as network path
                    path = Path("//" + netloc + path_part) if netloc else Path(path_part)

                resolved_path = path.expanduser()
                if not resolved_path.exists():
                    raise FileNotFoundError(f"No such file or directory: '{resolved_path}'")

                content = await asyncio.to_thread(resolved_path.read_bytes)
                return BytesIO(content)
            except Exception as e:
                raise FileNotFoundError(f"Failed to access file: {e}") from e

        # Check URI cache first for downloaded content
        try:
            cached = require_scope().get_memory_uri_cache().get(url)
            if isinstance(cached, bytes | bytearray):
                return BytesIO(bytes(cached))
        except Exception as e:
            log.debug(f"Failed to get from URI cache: {e}")

        response = await self.http_get(url)
        content = response.content

        # Store downloaded bytes in URI cache for 5 minutes
        with suppress(Exception):
            require_scope().get_memory_uri_cache().set(url, bytes(content))

        return BytesIO(content)

    def wrap_object(self, obj: Any) -> Any:
        """Wrap raw Python objects into typed refs, storing large media in-memory.

        - Images/Audio: store via memory:// to defer encoding; use asset_to_io for bytes.
        - DataFrames/Numpy/Tensors: use existing typed wrappers.
        """
        pd = _ensure_pandas()
        PIL_Image, _ = _ensure_pil()
        AudioSegment = _ensure_audio_segment()
        np = _ensure_numpy()

        if isinstance(obj, pd.DataFrame):
            return DataframeRef.from_pandas(obj)
        elif isinstance(obj, PIL_Image.Image):
            memory_uri = f"memory://{uuid.uuid4()}"
            self._memory_set(memory_uri, obj)
            return ImageRef(uri=memory_uri)
        elif isinstance(obj, AudioSegment):
            memory_uri = f"memory://{uuid.uuid4()}"
            self._memory_set(memory_uri, obj)
            return AudioRef(uri=memory_uri)
        elif isinstance(obj, np.ndarray):
            return NPArray.from_numpy(obj)
        elif is_torch_tensor(obj):
            return torch_tensor_to_metadata(obj)
        else:
            return obj

    async def asset_to_io(self, asset_ref: AssetRef) -> IO[bytes]:
        """
        Converts an AssetRef object to an IO object.

        Args:
            asset_ref (AssetRef): The AssetRef object to convert.

        Returns:
            IO: The converted IO object.

        Raises:
            ValueError: If the AssetRef is empty or contains unsupported data.
        """
        PIL_Image, _ = _ensure_pil()
        np = _ensure_numpy()
        AudioSegment = _ensure_audio_segment()

        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(asset_ref, "uri") and asset_ref.uri and asset_ref.uri.startswith("memory://"):
            key = asset_ref.uri
            obj = self._memory_get(key)
            if obj is not None:
                # Convert memory object to IO based on asset type and stored format
                if isinstance(obj, bytes):
                    return BytesIO(obj)
                elif isinstance(obj, PIL_Image.Image):
                    # Convert PIL Image to PNG bytes
                    return BytesIO(await _in_thread(_pil_to_png_bytes, obj))
                elif isinstance(obj, AudioSegment):
                    # Convert AudioSegment to MP3 bytes
                    return BytesIO(await _in_thread(_audio_segment_to_mp3_bytes, obj))
                elif isinstance(obj, np.ndarray):
                    # Handle numpy arrays stored in memory depending on the asset type
                    if isinstance(asset_ref, ImageRef):
                        # Encode numpy image array as PNG
                        return BytesIO(await _in_thread(_numpy_image_to_png_bytes, obj))
                    elif isinstance(asset_ref, AudioRef):
                        # Encode numpy audio array as MP3
                        return BytesIO(await _in_thread(_numpy_audio_to_mp3_bytes, obj))
                    elif isinstance(asset_ref, VideoRef):
                        # Encode numpy video array as MP4 using shared utility (T,H,W,C)
                        try:
                            return BytesIO(await _in_thread(_numpy_video_to_mp4_bytes, obj, 30))
                        except Exception as e:
                            raise ValueError(f"Failed to encode numpy video: {e}") from e
                    else:
                        # Generic fallback: return raw bytes
                        return BytesIO(await _in_thread(obj.tobytes))
                elif isinstance(obj, str):
                    # Convert string to UTF-8 bytes
                    return BytesIO(obj.encode("utf-8"))
                elif hasattr(obj, "read"):  # Already an IO object
                    return obj
                else:
                    raise ValueError(f"Unsupported memory object type {type(obj)}")
            else:
                raise ValueError(f"Memory object not found for key {key}")
        # If explicit data is present, normalize it into a consistent byte stream
        elif asset_ref.data is not None:
            data = asset_ref.data
            # Images: always encode to PNG
            if isinstance(asset_ref, ImageRef):
                if isinstance(data, bytes):
                    return BytesIO(data)
                elif isinstance(data, PIL_Image.Image):
                    return BytesIO(await _in_thread(_pil_to_png_bytes_with_exif, data))
                elif isinstance(data, np.ndarray):
                    return BytesIO(await _in_thread(_numpy_image_to_png_bytes, data))
                else:
                    raise ValueError(f"Unsupported ImageRef data type {type(data)}")
            # Audio: always encode to MP3
            elif isinstance(asset_ref, AudioRef):
                if isinstance(data, bytes):
                    return BytesIO(data)
                elif isinstance(data, AudioSegment):
                    return BytesIO(await _in_thread(_audio_segment_to_mp3_bytes, data))
                elif isinstance(data, np.ndarray):
                    return BytesIO(await _in_thread(_numpy_audio_to_mp3_bytes, data))
                else:
                    raise ValueError(f"Unsupported AudioRef data type {type(data)}")
            # Text
            elif isinstance(asset_ref, TextRef):
                if isinstance(data, bytes):
                    return BytesIO(data)
                elif isinstance(data, str):
                    return BytesIO(data.encode("utf-8"))
                else:
                    raise ValueError(f"Unsupported TextRef data type {type(data)}")
            elif isinstance(data, bytes):
                return BytesIO(data)
            elif isinstance(data, str):
                return BytesIO(data.encode("utf-8"))
            elif isinstance(data, list):
                raise ValueError("Unexpected list data type")
            else:
                raise ValueError(f"Unsupported data type {type(data)}")
        # Asset ID takes precedence over URI as the URI could be expired
        elif asset_ref.asset_id is not None:
            return await self.download_asset(asset_ref.asset_id)
        elif asset_ref.uri != "":
            return await self.download_file(asset_ref.uri)
        raise ValueError(f"AssetRef is empty {asset_ref}")

    async def asset_to_bytes(self, asset_ref: AssetRef) -> bytes:
        """
        Converts an AssetRef object to bytes.

        Args:
            asset_ref (AssetRef): The AssetRef object to convert.

        Returns:
            bytes: The asset content as bytes.
        """
        io = await self.asset_to_io(asset_ref)
        return await _in_thread(io.read)

    async def asset_to_base64(self, asset_ref: AssetRef) -> str:
        """
        Converts an AssetRef to a base64-encoded string.
        """
        io = await self.asset_to_io(asset_ref)
        return await _in_thread(_read_base64, io)

    async def asset_to_data_uri(self, asset_ref: AssetRef) -> str:
        """
        Converts an AssetRef to a URI.
        """
        return f"data:image/png;base64,{await self.asset_to_base64(asset_ref)}"

    async def asset_to_data(self, asset_ref: AssetRef) -> AssetRef:
        """
        Converts an AssetRef to a URI with a specific MIME type.
        """
        pd = _ensure_pandas()
        if asset_ref.data is None and asset_ref.uri and asset_ref.uri.startswith("memory://"):
            key = asset_ref.uri
            obj = self._memory_get(key)
            if obj is not None:
                if isinstance(obj, pd.DataFrame):
                    return await self.dataframe_from_pandas(obj)
                # For other asset types use the canonical encoding rules above
                data_bytes = await self.asset_to_bytes(asset_ref)
                return asset_ref.model_copy(update={"data": data_bytes})
            else:
                raise ValueError(f"Memory object not found for key {key}")
        return asset_ref

    async def image_to_pil(self, image_ref: ImageRef) -> PIL.Image.Image:
        """
        Converts an ImageRef to a PIL Image object.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            PIL.Image.Image: The converted PIL Image object.
        """
        PIL_Image, _ = _ensure_pil()
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(image_ref, "uri") and image_ref.uri and image_ref.uri.startswith("memory://"):
            key = image_ref.uri
            obj = self._memory_get(key)
            if obj is not None and isinstance(obj, PIL_Image.Image):
                return await _in_thread(obj.convert, "RGB")
                # Fall through to regular conversion if not a PIL Image

        buffer = await self.asset_to_io(image_ref)
        return await _in_thread(_open_image_as_rgb, buffer)

    async def image_to_numpy(self, image_ref: ImageRef) -> np.ndarray:
        """
        Converts an ImageRef to a numpy array.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            np.ndarray: The image as a numpy array.
        """
        np = _ensure_numpy()
        image = await self.image_to_pil(image_ref)
        return await _in_thread(np.array, image)

    async def image_to_tensor(self, image_ref: ImageRef) -> Any:
        """
        Converts an ImageRef to a tensor.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            Any: The image as a tensor.

        Raises:
            ImportError: If torch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for image_to_tensor")

        image = await self.image_to_pil(image_ref)
        return await _in_thread(tensor_from_pil, image)

    async def image_to_torch_tensor(self, image_ref: ImageRef) -> Any:
        """
        Converts the image to a tensor.

        Args:
            context (ProcessingContext): The processing context.

        Raises:
            ImportError: If torch is not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for image_to_torch_tensor")

        image = await self.image_to_pil(image_ref)
        return await _in_thread(tensor_from_pil, image)

    async def image_to_base64(self, image_ref: ImageRef) -> str:
        """
        Converts the image to a PNG base64-encoded string.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            str: The base64-encoded string representation of the image.
        """
        return await self.asset_to_base64(image_ref)

    async def audio_to_audio_segment(self, audio_ref: AudioRef) -> AudioSegment:
        """
        Converts the audio to an AudioSegment object.

        Args:
            audio_ref (AudioRef): The audio reference to convert.

        Returns:
            AudioSegment: The converted audio segment.
        """
        AudioSegment = _ensure_audio_segment()
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(audio_ref, "uri") and audio_ref.uri and audio_ref.uri.startswith("memory://"):
            key = audio_ref.uri
            obj = self._memory_get(key)
            if obj is not None and isinstance(obj, AudioSegment):
                return obj
                # Fall through to regular conversion if not an AudioSegment

        audio_bytes = await self.asset_to_io(audio_ref)
        return await _in_thread(_audio_segment_from_file, audio_bytes)

    async def audio_to_numpy(
        self,
        audio_ref: AudioRef,
        sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        mono: bool = True,
    ) -> tuple[np.ndarray, int, int]:
        """
        Converts the audio to a np.float32 array.

        Args:
            audio_ref (AudioRef): The audio reference to convert.
            sample_rate (int, optional): The target sample rate. Defaults to DEFAULT_AUDIO_SAMPLE_RATE.
            mono (bool, optional): Whether to convert to mono. Defaults to True.

        Returns:
            tuple[np.ndarray, int, int]: A tuple containing the audio samples as a numpy array,
                the frame rate, and the number of channels.
        """
        segment = await self.audio_to_audio_segment(audio_ref)
        samples, frame_rate, channels = await _in_thread(
            _audio_segment_to_numpy,
            segment,
            sample_rate=sample_rate,
            mono=mono,
        )
        return samples, frame_rate, channels

    async def audio_to_base64(self, audio_ref: AudioRef) -> str:
        """
        Converts the audio to a base64-encoded string.

        Args:
            audio_ref (AudioRef): The audio reference.

        Returns:
            str: The base64-encoded string.
        """
        return await self.asset_to_base64(audio_ref)

    async def audio_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        content_type: str = "audio/mp3",
    ) -> AudioRef:
        """
        Creates an AudioRef from an IO object.

        Args:
            buffer (IO): The IO object.
            name (Optional[str], optional): The name of the asset. Defaults to None
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            content_type (str, optional): The content type of the asset. Defaults to "audio/mp3".

        Returns:
            AudioRef: The AudioRef object.
        """
        if name:
            asset = await self.create_asset(
                name=name,
                content_type=content_type,
                content=buffer,
                parent_id=parent_id,
            )
            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return AudioRef(asset_id=asset.id, uri=url)
        else:
            return AudioRef(data=await _in_thread(buffer.read))

    async def audio_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> AudioRef:
        """
        Creates an AudioRef from a bytes object.

        Args:
            context (ProcessingContext): The processing context.
            b (bytes): The bytes object.
            name (Optional[str], optional): The name of the asset. Defaults to None.

        Returns:
            AudioRef: The AudioRef object.
        """
        return await self.audio_from_io(BytesIO(b), name=name, parent_id=parent_id)

    async def audio_from_base64(self, b64: str, name: str | None = None, parent_id: str | None = None) -> AudioRef:
        """
        Creates an AudioRef from a base64-encoded string.

        Args:
            b64 (str): The base64-encoded string.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.

        Returns:
            AudioRef: The AudioRef object.
        """
        decoded = await _in_thread(_b64decode_to_bytes, b64)
        return await self.audio_from_io(BytesIO(decoded), name=name, parent_id=parent_id)

    async def audio_from_numpy(
        self,
        data: np.ndarray,
        sample_rate: int,
        num_channels: int = 1,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> AudioRef:
        """
        Creates an AudioRef from a numpy array.

        Args:
            context (ProcessingContext): The processing context.
            data (np.ndarray): The numpy array.
            sample_rate (int): The sample rate.
            num_channels (int, optional): The number of channels. Defaults to 1.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
        """
        def _segment_from_numpy() -> Any:
            np = _ensure_numpy()
            AudioSegment = _ensure_audio_segment()
            if data.dtype == np.int16:
                data_bytes = data.tobytes()
            elif data.dtype in (np.float32, np.float64, np.float16):
                data_bytes = (data * (2**14)).astype(np.int16).tobytes()
            else:
                raise ValueError(f"Unsupported dtype {data.dtype}")
            return AudioSegment(
                data=data_bytes,
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit
                channels=num_channels,
            )

        audio_segment = await _in_thread(_segment_from_numpy)
        return await self.audio_from_segment(audio_segment, name=name, parent_id=parent_id)

    async def audio_from_segment(
        self,
        audio_segment: AudioSegment,
        name: str | None = None,
        parent_id: str | None = None,
        **kwargs,
    ) -> AudioRef:
        """
        Converts an audio segment to an AudioRef object.

        Args:
            audio_segment (pydub.AudioSegment): The audio segment to convert.
            name (str, optional): The name of the audio file, will create an asset. Defaults to None.
            parent_id (str, optional): The ID of the parent asset. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            AudioRef: The converted AudioRef object.

        """
        metadata = {
            "sample_rate": audio_segment.frame_rate,
            "channels": audio_segment.channels,
            "format": "wav",
            "duration_seconds": audio_segment.duration_seconds,
        }

        wav_bytes = await _in_thread(_audio_segment_to_wav_bytes, audio_segment)

        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the AudioSegment directly for fast retrieval
            self._memory_set(memory_uri, audio_segment)
            return AudioRef(uri=memory_uri, data=wav_bytes, metadata=metadata)

        ref = await self.audio_from_io(BytesIO(wav_bytes), name=name, parent_id=parent_id, content_type="audio/wav")
        ref.metadata = metadata
        return ref

    async def dataframe_to_pandas(self, df: DataframeRef) -> pd.DataFrame:
        """
        Converts a DataframeRef object to a pandas DataFrame.

        Args:
            df (DataframeRef): The DataframeRef object to convert.

        Returns:
            pd.DataFrame: The converted pandas DataFrame.

        Raises:
            AssertionError: If the deserialized object is not a pandas DataFrame.
        """
        pd = _ensure_pandas()
        # Prefer retrieving from in-memory storage if the DataframeRef uses a memory URI
        if getattr(df, "uri", "").startswith("memory://"):
            key = df.uri
            obj = self._memory_get(key)
            if isinstance(obj, pd.DataFrame):
                return obj
            # If not found in cache, fall back to other representations

        if df.columns:
            column_names = [col.name for col in df.columns]
            return await _in_thread(pd.DataFrame, df.data, columns=column_names)  # type: ignore[arg-type]
        else:
            io = await self.asset_to_io(df)
            raw = await _in_thread(io.read)
            loaded = await _in_thread(loads, raw)
            assert isinstance(loaded, pd.DataFrame), "Is not a dataframe"
            return loaded

    async def dataframe_from_pandas(
        self, data: pd.DataFrame, name: str | None = None, parent_id: str | None = None
    ) -> DataframeRef:
        """
        Converts a pandas DataFrame to a DataframeRef object.

        Args:
            data (pd.DataFrame): The pandas DataFrame to convert.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.

        Returns:
            DataframeRef: The converted DataframeRef object.
        """
        # Always prefer passing DataframeRef as a pure reference via in-memory URI
        memory_uri = f"memory://{uuid.uuid4()}"
        # Store the dataframe directly for fast retrieval
        self._memory_set(memory_uri, data)
        return DataframeRef(uri=memory_uri)

    async def image_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from an IO object.

        Args:
            buffer (IO): The IO object.
            name (Optional[str], optional): The name of the asset. Defaults to None
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        if name:
            asset = await self.create_asset(name=name, content_type="image/png", content=buffer, parent_id=parent_id)
            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return ImageRef(asset_id=asset.id, uri=url, metadata=metadata)
        else:
            data_bytes = await _in_thread(_read_all_bytes_from_start, buffer)
            return ImageRef(data=data_bytes, metadata=metadata)

    async def image_from_url(
        self,
        url: str,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a URL.

        Args:
            context (ProcessingContext): The processing context.
            url (str): The URL.
            name (Optional[str], optional): The name of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        return await self.image_from_io(await self.download_file(url), name=name, parent_id=parent_id)

    async def image_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a bytes object.

        Args:
            b (bytes): The bytes object.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        return await self.image_from_io(BytesIO(b), name=name, parent_id=parent_id, metadata=metadata)

    async def image_from_base64(
        self,
        b64: str,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a base64-encoded string.

        Args:
            b64 (str): The base64-encoded string.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        decoded = await _in_thread(_b64decode_to_bytes, b64)
        return await self.image_from_bytes(decoded, name=name, parent_id=parent_id, metadata=metadata)

    async def image_from_pil(
        self,
        image: PIL.Image.Image,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a PIL Image object.

        Args:
            image (Image.Image): The PIL Image object.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        # Build metadata from PIL Image properties if not provided
        if metadata is None:
            metadata = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": "png",
            }

        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the PIL Image directly for fast retrieval
            self._memory_set(memory_uri, image)
            return ImageRef(uri=memory_uri, metadata=metadata)
        else:
            png_bytes = await _in_thread(_pil_to_png_bytes, image)
            return await self.image_from_io(BytesIO(png_bytes), name=name, parent_id=parent_id, metadata=metadata)

    async def image_from_numpy(
        self,
        image: np.ndarray,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a numpy array.

        Args:
            image (np.ndarray): The numpy array.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        # Build metadata from numpy array shape if not provided
        if metadata is None:
            if image.ndim == 2:
                height, width = image.shape
                channels = 1
            elif image.ndim == 3:
                height, width, channels = image.shape
            else:
                height, width, channels = 0, 0, 0
            metadata = {
                "width": width,
                "height": height,
                "channels": channels,
                "format": "png",
            }

        pil_img = await _in_thread(self._numpy_to_pil_image, image)
        return await self.image_from_pil(pil_img, name=name, metadata=metadata)

    async def image_from_tensor(
        self,
        image_tensor: Any,  # Change type hint to Any since torch.Tensor may not be available
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Creates an ImageRef from a tensor.

        Args:
            image_tensor: The tensor.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.

        Raises:
            ImportError: If torch is not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for image_from_tensor")

        img = await _in_thread(tensor_to_image_array, image_tensor)
        if img.ndim == 5:
            img = img[0]
        if img.shape[0] == 1:
            return await self.image_from_numpy(img[0], name=name, parent_id=parent_id, metadata=metadata)

        def _tensor_batch_to_png_bytes() -> list[bytes]:
            PIL_Image, _ = _ensure_pil()
            return [_pil_to_png_bytes(PIL_Image.fromarray(img[i])) for i in range(img.shape[0])]

        batch = await _in_thread(_tensor_batch_to_png_bytes)
        return ImageRef(data=batch, metadata=metadata)

    async def text_to_str(self, text_ref: TextRef | str) -> str:
        """
        Converts a TextRef to a string.

        Args:
            text_ref (TextRef): The TextRef object.

        Returns:
            str: The string.
        """
        if isinstance(text_ref, TextRef):
            # Check for memory:// protocol URI first (preferred for performance)
            if hasattr(text_ref, "uri") and text_ref.uri and text_ref.uri.startswith("memory://"):
                key = text_ref.uri
                obj = self._memory_get(key)
                if obj is not None and isinstance(obj, str):
                    return obj
                    # Fall through to regular conversion if not a string

            stream = await self.asset_to_io(text_ref)
            return await _in_thread(_read_utf8, stream)
        else:
            return text_ref

    async def text_from_str(
        self,
        s: str,
        name: str | None = None,
        content_type: str = "text/plain",
        parent_id: str | None = None,
    ) -> TextRef:
        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the string directly for fast retrieval
            self._memory_set(memory_uri, s)
            return TextRef(uri=memory_uri)
        else:
            # Create asset when name is provided (persistence needed)
            buffer = BytesIO(s.encode("utf-8"))
            asset = await self.create_asset(name, content_type, buffer, parent_id=parent_id)
            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return TextRef(asset_id=asset.id, uri=url)

    async def video_from_frames(
        self,
        frames: list[PIL.Image.Image] | list[np.ndarray],
        fps: int = 30,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> VideoRef:
        import tempfile

        from nodetool.media.video.video_utils import export_to_video

        # Build metadata from frames
        frame_count = len(frames)
        width, height = 0, 0
        if frame_count > 0:
            first_frame = frames[0]
            PIL_Image, _ = _ensure_pil()
            if isinstance(first_frame, PIL_Image.Image):
                width, height = first_frame.size
            else:
                # numpy array
                if first_frame.ndim >= 2:
                    height, width = first_frame.shape[:2]

        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "format": "mp4",
            "duration_seconds": frame_count / fps if fps > 0 else None,
        }

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
            await _in_thread(export_to_video, frames, temp.name, fps=fps)
            temp.seek(0)
            content = await asyncio.to_thread(temp.read)
            ref = await self.video_from_bytes(content, name=name, parent_id=parent_id)
            ref.metadata = metadata
            return ref

    async def video_from_numpy(
        self,
        video: np.ndarray,
        fps: int = 30,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> VideoRef:
        """
        Creates a VideoRef from a numpy array.

        Args:
            context (ProcessingContext): The processing context.
            video (np.ndarray): The numpy array.
            name (Optional[str], optional): The name of the asset. Defaults to None.

        Returns:
            VideoRef: The VideoRef object.
        """
        # Build metadata from numpy array shape (T, H, W, C)
        frame_count = int(video.shape[0]) if hasattr(video, "shape") and len(video.shape) > 0 else 0
        width, height = 0, 0
        if frame_count > 0 and video.ndim >= 3:
            height, width = video.shape[1], video.shape[2]

        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "format": "mp4",
            "duration_seconds": frame_count / fps if fps > 0 else None,
        }

        # Use shared video utility for consistent behavior
        video_bytes = await _in_thread(_numpy_video_to_mp4_bytes, video, fps)

        # Create BytesIO from the video bytes
        buffer = BytesIO(video_bytes)
        buffer.seek(0)
        ref = await self.video_from_io(buffer, name=name, parent_id=parent_id)
        ref.metadata = metadata
        return ref

    async def url_to_base64(self, url: str) -> str:
        """
        Download a URL and encode its content as base64.

        Args:
            url (str): The URL to download and encode.

        Returns:
            str: The base64-encoded content.
        """
        file_io = await self.download_file(url)
        return await _in_thread(_read_base64, file_io)

    async def urls_to_base64_list(self, urls: list[str]) -> list[str]:
        """
        Convert a list of URLs to base64-encoded strings.

        Args:
            urls (list[str]): List of URLs to download and encode.

        Returns:
            list[str]: List of base64-encoded strings.
        """
        return [await self.url_to_base64(url) for url in urls]

    async def image_ref_to_base64(self, image_ref: ImageRef) -> str:
        """
        Convert an ImageRef to a base64-encoded string.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            str: The base64-encoded image content.
        """
        img = await self.image_to_numpy(image_ref)

        def _encode() -> str:
            PIL_Image, _ = _ensure_pil()
            jpeg_bytes = _pil_to_jpeg_bytes(PIL_Image.fromarray(img))
            return _b64encode_to_str(jpeg_bytes)

        return await _in_thread(_encode)

    async def image_ref_to_data_uri(self, image_ref: ImageRef) -> str:
        """
        Convert an ImageRef to a data URI.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            str: The data URI.
        """
        return f"data:image/jpeg;base64,{await self.image_ref_to_base64(image_ref)}"

    async def audio_ref_to_base64(self, audio_ref: AudioRef) -> str:
        """
        Convert an AudioRef to a base64-encoded string.

        Args:
            audio_ref (AudioRef): The audio reference to convert.

        Returns:
            str: The base64-encoded audio content.
        """
        return await self.asset_to_base64(audio_ref)

    async def audio_ref_to_data_uri(self, audio_ref: AudioRef) -> str:
        """
        Convert an AudioRef to a data URI.

        Args:
            audio_ref (AudioRef): The audio reference to convert.

        Returns:
            str: The data URI.
        """
        return f"data:audio/mpeg;base64,{await self.audio_ref_to_base64(audio_ref)}"

    async def video_ref_to_base64(self, video_ref: VideoRef) -> str:
        """
        Convert a VideoRef to a base64-encoded string.

        Args:
            video_ref (VideoRef): The video reference to convert.

        Returns:
            str: The base64-encoded video content.
        """
        return await self.asset_to_base64(video_ref)

    async def video_ref_to_data_uri(self, video_ref: VideoRef) -> str:
        """
        Convert a VideoRef to a data URI.

        Args:
            video_ref (VideoRef): The video reference to convert.

        Returns:
            str: The data URI.
        """
        return f"data:video/mp4;base64,{await self.video_ref_to_base64(video_ref)}"

    async def video_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoRef:
        """
        Creates an VideoRef from an IO object.

        Args:
            context (ProcessingContext): The processing context.
            buffer (IO): The IO object.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            VideoRef: The VideoRef object.
        """
        if name:
            asset = await self.create_asset(name, "video/mpeg", buffer, parent_id=parent_id)
            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return VideoRef(asset_id=asset.id, uri=url, metadata=metadata)
        else:
            return VideoRef(data=await _in_thread(buffer.read), metadata=metadata)

    async def video_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VideoRef:
        """
        Creates a VideoRef from a bytes object.

        Args:
            b (bytes): The bytes object.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            VideoRef: The VideoRef object.
        """
        return await self.video_from_io(BytesIO(b), name=name, parent_id=parent_id, metadata=metadata)

    async def video_to_frames(self, video: VideoRef, fps: int = 1) -> list[PIL.Image.Image]:
        """
        Convert a video asset to a list of PIL images at a specific FPS.

        Args:
            video: The video asset to convert
            fps: Frames per second to sample. Default is 1.

        Returns:
            List[PIL.Image.Image]: List of PIL images
        """
        from nodetool.media.video.video_utils import extract_video_frames

        if video.is_empty():
            return []

        video_bytes = await self.asset_to_bytes(video)
        return await asyncio.to_thread(extract_video_frames, video_bytes, fps)

    async def model3d_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Model3DRef:
        """
        Creates a Model3DRef from an IO object.

        Args:
            buffer (IO): The IO object containing 3D model data.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            format (Optional[str], optional): The 3D format (glb, gltf, obj, stl, ply, fbx, usdz). Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            Model3DRef: The Model3DRef object.
        """
        # Get content type from shared mapping
        mime_type, _ = MODEL_3D_FORMAT_MAPPING.get(format or "glb", ("model/gltf-binary", "glb"))

        if name:
            asset = await self.create_asset(name, mime_type, buffer, parent_id=parent_id)
            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return Model3DRef(asset_id=asset.id, uri=url, format=format, metadata=metadata)
        else:
            data_bytes = await _in_thread(_read_all_bytes_from_start, buffer)
            return Model3DRef(data=data_bytes, format=format, metadata=metadata)

    async def model3d_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
        format: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Model3DRef:
        """
        Creates a Model3DRef from a bytes object.

        Args:
            b (bytes): The bytes object containing 3D model data.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.
            format (Optional[str], optional): The 3D format (glb, gltf, obj, stl, ply, fbx, usdz). Defaults to None.
            metadata (Dict[str, Any] | None, optional): The metadata of the asset. Defaults to None.

        Returns:
            Model3DRef: The Model3DRef object.
        """
        return await self.model3d_from_io(BytesIO(b), name=name, parent_id=parent_id, format=format, metadata=metadata)

    async def model3d_to_bytes(self, model3d_ref: Model3DRef) -> bytes:
        """
        Converts a Model3DRef to bytes.

        Args:
            model3d_ref (Model3DRef): The 3D model reference to convert.

        Returns:
            bytes: The 3D model data as bytes.
        """
        return await self.asset_to_bytes(model3d_ref)

    async def model3d_to_io(self, model3d_ref: Model3DRef) -> IO[bytes]:
        """
        Converts a Model3DRef to an IO object.

        Args:
            model3d_ref (Model3DRef): The 3D model reference to convert.

        Returns:
            IO[bytes]: The 3D model data as an IO object.
        """
        return await self.asset_to_io(model3d_ref)

    async def model3d_to_base64(self, model3d_ref: Model3DRef) -> str:
        """
        Converts a Model3DRef to a base64-encoded string.

        Args:
            model3d_ref (Model3DRef): The 3D model reference to convert.

        Returns:
            str: The base64-encoded string representation of the 3D model.
        """
        return await self.asset_to_base64(model3d_ref)

    async def model3d_ref_to_data_uri(self, model3d_ref: Model3DRef) -> str:
        """
        Convert a Model3DRef to a data URI.

        Args:
            model3d_ref (Model3DRef): The 3D model reference to convert.

        Returns:
            str: The data URI.
        """
        # Get MIME type from shared mapping
        mime_type, _ = MODEL_3D_FORMAT_MAPPING.get(model3d_ref.format or "glb", ("model/gltf-binary", "glb"))
        return f"data:{mime_type};base64,{await self.model3d_to_base64(model3d_ref)}"

    async def to_estimator(self, model_ref: ModelRef):
        """
        Converts a model reference to an estimator object.

        Args:
            model_ref (ModelRef): The model reference to convert.

        Returns:
            The loaded estimator object.

        Raises:
            ValueError: If the model reference is empty.
        """
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(model_ref, "uri") and model_ref.uri and model_ref.uri.startswith("memory://"):
            key = model_ref.uri
            obj = self._memory_get(key)
            # Return the model object directly if it's already a model
            if obj is not None and (hasattr(obj, "fit") or hasattr(obj, "predict")):
                return obj
            # Fall through to regular conversion if not a model object

        if model_ref.asset_id is None:
            raise ValueError("ModelRef is empty")
        file = await self.asset_to_io(model_ref)
        return await _in_thread(_joblib_load_from_io, file)

    async def from_estimator(self, est: BaseEstimator, name: str | None = None, **kwargs):  # type: ignore
        """
        Create a model asset from an estimator.

        Args:
            est (BaseEstimator): The estimator object to be serialized.
            name (str | None): The name for the model asset. If None, stores in memory.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelRef: A reference to the created model asset.

        """
        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the model object directly for fast retrieval
            self._memory_set(memory_uri, est)
            return ModelRef(uri=memory_uri, **kwargs)
        else:
            # Create asset when name is provided (persistence needed)
            payload = await _in_thread(_joblib_dump_to_bytes, est)
            asset = await self.create_asset(name, "application/model", BytesIO(payload))

            storage = require_scope().get_asset_storage()
            url = await storage.get_url(asset.file_name)
            return ModelRef(uri=url, asset_id=asset.id, **kwargs)

    async def convert_value_for_prediction(
        self,
        property: Property,
        value: Any,
    ):
        """
        Converts the property value for a remote api prediction on replicate or huggingface.

        Args:
            property (Property): The property.
            value: The value to be converted.

        Raises:
            NotImplementedError: If the self type is 'tensor' and not implemented.
            ValueError: If the self value is an invalid enum value.
        """

        if isinstance(value, AssetRef):
            if value.is_empty():
                return None
            io = await self.asset_to_io(value)
            if isinstance(value, TextRef):
                return await _in_thread(_read_utf8, io)
            else:
                b64 = await _in_thread(_read_base64, io)
                if isinstance(value, ImageRef):
                    return "data:image/png;base64," + b64
                elif isinstance(value, AudioRef):
                    return "data:audio/mp3;base64," + b64
                elif isinstance(value, VideoRef):
                    return "data:video/mp4;base64," + b64
                else:
                    return b64
        elif property.type.type == "tensor":
            raise NotImplementedError()
        elif property.type.type == "enum":
            if value is None:
                return None
            elif isinstance(value, str):
                return value
            elif isinstance(value, Enum):
                return value.value
            else:
                raise ValueError(f"Invalid enum value {value} : {type(value)}")
        else:
            return value

    async def get_chroma_client(self):
        """
        Get a ChromaDB client instance for this context.

        Returns:
            ClientAPI: ChromaDB client instance
        """
        if self.chroma_client is None:
            self.chroma_client = await get_async_chroma_client(self.user_id)
        return self.chroma_client

    async def is_huggingface_model_cached(self, repo_id: str):
        """
        Check if a Hugging Face model is already cached locally.

        Args:
            repo_id (str): The repository ID of the model to check.

        Returns:
            bool: True if the model is cached, False otherwise.
        """
        from nodetool.integrations.huggingface.hf_utils import is_model_cached

        return await _in_thread(is_model_cached, repo_id)

    def encode_assets_as_uri(self, value: Any) -> Any:
        """
        Recursively encodes any AssetRef objects found in the given value as URIs.

        Args:
            value: Any Python value that might contain AssetRef objects

        Returns:
            Any: The value with all AssetRef objects encoded as URIs
        """
        from nodetool.io.asset_utils import encode_assets_as_uri

        return encode_assets_as_uri(value)

    def _is_asset_dict(self, value: dict[str, Any]) -> bool:
        """Best-effort detection for dicts shaped like serialized AssetRefs."""
        return "type" in value and value["type"] in asset_types and "uri" in value and "asset_id" in value

    def _guess_asset_mime_ext(self, asset: AssetRef) -> tuple[str, str]:
        """Return (mime, extension) defaults for a given asset type."""
        if isinstance(asset, ImageRef):
            return "image/png", "png"
        if isinstance(asset, AudioRef):
            return "audio/mp3", "mp3"
        if isinstance(asset, VideoRef):
            return "video/mp4", "mp4"
        if isinstance(asset, TextRef):
            return "text/plain", "txt"
        if isinstance(asset, Model3DRef):
            # Use shared format mapping
            return MODEL_3D_FORMAT_MAPPING.get(asset.format or "glb", ("model/gltf-binary", "glb"))
        return "application/octet-stream", "bin"

    async def _asset_to_data_uri(self, asset: AssetRef) -> AssetRef:
        """Convert an AssetRef into a data URI."""
        if isinstance(asset, DataframeRef):
            return asset
        data_bytes = await self.asset_to_bytes(asset)
        mime, _ = self._guess_asset_mime_ext(asset)
        b64 = await _in_thread(_b64encode_to_str, data_bytes)
        uri = f"data:{mime};base64,{b64}"
        return asset.model_copy(update={"uri": uri, "data": None})

    async def _asset_to_storage_url(self, asset: AssetRef) -> dict[str, Any] | AssetRef:
        """Ensure asset is accessible via persistent storage URL."""
        if isinstance(asset, DataframeRef):
            return await self.embed_assets_in_data(asset)

        if asset.asset_id:
            try:
                url = await self.get_asset_url(asset.asset_id)
                return {"type": asset.type, "uri": url, "asset_id": asset.asset_id}
            except Exception:
                log.debug("Falling back to upload for asset_id %s", asset.asset_id)

        if asset.uri and asset.uri.startswith(("http://", "https://")):
            return {"type": asset.type, "uri": asset.uri, "asset_id": asset.asset_id}

        data_bytes = await self.asset_to_bytes(asset)
        storage = require_scope().get_asset_storage()
        _, ext = self._guess_asset_mime_ext(asset)
        key = uuid.uuid4().hex + f".{ext}"
        uri = await storage.upload(key, BytesIO(data_bytes))
        return {"type": asset.type, "uri": uri, "asset_id": asset.asset_id}

    async def _asset_to_temp_url(self, asset: AssetRef) -> dict[str, Any] | AssetRef:
        """Upload an AssetRef to temp storage and return reference dict."""
        if isinstance(asset, DataframeRef):
            return await self.embed_assets_in_data(asset)

        data_bytes = await self.asset_to_bytes(asset)
        storage = require_scope().get_temp_storage()
        _, ext = self._guess_asset_mime_ext(asset)
        key = uuid.uuid4().hex + f".{ext}"
        uri = await storage.upload(key, BytesIO(data_bytes))
        return {"type": asset.type, "uri": uri, "asset_id": asset.asset_id}

    async def _asset_to_workspace_file(self, asset: AssetRef) -> dict[str, Any] | AssetRef:
        """Persist asset to local workspace and return file path reference."""
        if isinstance(asset, DataframeRef):
            return await self.embed_assets_in_data(asset)

        if not self.workspace_dir:
            raise ValueError("workspace_dir is required for workspace asset output")

        data_bytes = await self.asset_to_bytes(asset)
        _, ext = self._guess_asset_mime_ext(asset)
        assets_dir = Path(self.workspace_dir) / "assets"
        await _in_thread(assets_dir.mkdir, parents=True, exist_ok=True)
        file_path = assets_dir / f"{uuid.uuid4().hex}.{ext}"
        await _in_thread(file_path.write_bytes, data_bytes)
        return {
            "type": asset.type,
            "path": str(file_path),
            "asset_id": asset.asset_id,
        }

    async def assets_to_data_uri(self, value: Any) -> Any:
        """Recursively convert AssetRefs within value to data URIs."""
        if isinstance(value, AssetRef):
            return await self._asset_to_data_uri(value)
        elif isinstance(value, dict):
            keys = list(value.keys())
            results = await asyncio.gather(*[self.assets_to_data_uri(value[k]) for k in keys])
            return dict(zip(keys, results, strict=False))
        elif isinstance(value, list):
            results = await asyncio.gather(*[self.assets_to_data_uri(item) for item in value])
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(*[self.assets_to_data_uri(item) for item in value])
            return tuple(results)
        else:
            return value

    async def assets_to_storage_url(self, value: Any) -> Any:
        """Recursively materialize AssetRefs as persistent storage URLs."""
        if isinstance(value, AssetRef):
            return await self._asset_to_storage_url(value)
        elif isinstance(value, dict):
            keys = list(value.keys())
            results = await asyncio.gather(*[self.assets_to_storage_url(value[k]) for k in keys])
            return dict(zip(keys, results, strict=False))
        elif isinstance(value, list):
            results = await asyncio.gather(*[self.assets_to_storage_url(item) for item in value])
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(*[self.assets_to_storage_url(item) for item in value])
            return tuple(results)
        else:
            return value

    async def assets_to_workspace_files(self, value: Any) -> Any:
        """Recursively persist AssetRefs to the workspace directory."""
        if isinstance(value, AssetRef):
            return await self._asset_to_workspace_file(value)
        elif isinstance(value, dict):
            keys = list(value.keys())
            results = await asyncio.gather(*[self.assets_to_workspace_files(value[k]) for k in keys])
            return dict(zip(keys, results, strict=False))
        elif isinstance(value, list):
            results = await asyncio.gather(*[self.assets_to_workspace_files(item) for item in value])
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(*[self.assets_to_workspace_files(item) for item in value])
            return tuple(results)
        else:
            return value

    async def embed_assets_in_data(self, value: Any) -> Any:
        """
        Recursively embeds any memory:// assets in the given value.
        """
        if isinstance(value, AssetRef):
            if isinstance(value, DataframeRef):
                return value
            if (value.uri and value.uri.startswith("memory://")) or value.data is not None:
                data_bytes = await self.asset_to_bytes(value)
                return value.model_copy(update={"uri": None, "data": data_bytes})
            return value
        elif isinstance(value, dict):
            keys = list(value.keys())
            results = await asyncio.gather(*[self.embed_assets_in_data(value[k]) for k in keys])
            return dict(zip(keys, results, strict=False))
        elif isinstance(value, list):
            results = await asyncio.gather(*[self.embed_assets_in_data(item) for item in value])
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(*[self.embed_assets_in_data(item) for item in value])
            return tuple(results)
        else:
            return value

    async def upload_assets_to_temp(self, value: Any) -> Any:
        """
        Recursively uploads any AssetRef objects found in the given value to temp storage.

        Args:
            value: Any Python value that might contain AssetRef objects

        Returns:
            Any: The value with all AssetRef objects uploaded to S3 and replaced with their URLs
        """
        if isinstance(value, AssetRef):
            return await self._asset_to_temp_url(value)
        elif isinstance(value, dict):
            if self._is_asset_dict(value):
                asset_ref = AssetRef(
                    type=value["type"],
                    uri=value.get("uri") or "",
                    asset_id=value.get("asset_id"),
                    data=value.get("data"),
                )
                uploaded = await self._asset_to_temp_url(asset_ref)
                if isinstance(uploaded, dict):
                    merged = dict(value)
                    merged["uri"] = uploaded.get("uri")
                    merged["asset_id"] = uploaded.get("asset_id")
                    merged["type"] = uploaded.get("type", merged.get("type"))
                    merged.pop("data", None)
                    return merged
                else:
                    return uploaded
            keys = list(value.keys())
            results = await asyncio.gather(*[self.upload_assets_to_temp(value[k]) for k in keys])
            return dict(zip(keys, results, strict=False))
        elif isinstance(value, list):
            results = await asyncio.gather(*[self.upload_assets_to_temp(item) for item in value])
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(*[self.upload_assets_to_temp(item) for item in value])
            return tuple(results)
        else:
            return value

    async def normalize_output_value(self, value: Any) -> Any:
        """
        Normalize workflow outputs according to the configured asset output mode.
        """
        mode = self.asset_output_mode
        if mode == AssetOutputMode.PYTHON:
            return value
        if mode == AssetOutputMode.DATA_URI:
            return await self.assets_to_data_uri(value)
        if mode == AssetOutputMode.TEMP_URL:
            return await self.upload_assets_to_temp(value)
        if mode == AssetOutputMode.STORAGE_URL:
            return await self.assets_to_storage_url(value)
        if mode == AssetOutputMode.WORKSPACE:
            return await self.assets_to_workspace_files(value)
        if mode == AssetOutputMode.RAW:
            return await self.embed_assets_in_data(value)
        return value

    def get_system_font_path(self, font_name: str = "Arial.ttf") -> str:
        """
        Get the system path for a font file based on the operating system.

        Args:
            font_name (str, optional): Name of the font file to find. Defaults to "Arial.ttf"

        Returns:
            str: Full path to the font file

        Raises:
            FileNotFoundError: If the font file cannot be found in system locations
        """
        from nodetool.media.image.font_utils import get_system_font_path

        return get_system_font_path(font_name, self.environment)

    def get_font_path(self, font_ref: FontRef) -> str:
        """
        Get the path to a font file, handling both system fonts and web fonts.

        This method supports three font sources:
        - system: Uses the local system font path (default, backwards compatible)
        - google_fonts: Downloads and caches fonts from Google Fonts
        - url: Downloads and caches fonts from a custom URL

        Args:
            font_ref (FontRef): The font reference containing name, source, and optional URL.

        Returns:
            str: Full path to the font file

        Raises:
            FileNotFoundError: If a system font cannot be found
            ValueError: If a Google Font is not in the catalog or URL is invalid
            ConnectionError: If downloading a web font fails
        """
        from nodetool.metadata.types import FontSource

        # Handle backwards compatibility - if no source specified, treat as system font
        if not hasattr(font_ref, "source") or font_ref.source == FontSource.SYSTEM:
            return self.get_system_font_path(font_ref.name)

        # Handle web fonts (Google Fonts or custom URL)
        from nodetool.media.image.web_font_utils import get_web_font_path

        return get_web_font_path(
            font_name=font_ref.name,
            source=font_ref.source.value,
            url=getattr(font_ref, "url", ""),
            weight=getattr(font_ref, "weight", "regular"),
        )

    def resolve_workspace_path(self, path: str) -> str:
        """
        Resolve a path relative to the workspace directory.
        Handles paths starting with '/workspace/', 'workspace/', or absolute paths
        by interpreting them relative to the `workspace_dir`.

        Args:
            path: The path to resolve, which can be:
                - Prefixed with '/workspace/' (e.g., '/workspace/output/file.txt')
                - Prefixed with 'workspace/' (e.g., 'workspace/output/file.txt')
                - An absolute path (e.g., '/input/data.csv') - treated relative to workspace root
                - A relative path (e.g., 'output/file.txt')

        Returns:
            The absolute path in the actual filesystem.

        Raises:
            ValueError: If workspace_dir is not provided or empty.
        """
        from nodetool.io.path_utils import resolve_workspace_path

        return resolve_workspace_path(self.workspace_dir, path)

    async def get_browser(
        self,
    ) -> Browser:
        """
        Initializes a Playwright browser instance (local or remote).

        Returns:
            A Playwright browser instance.
        """
        from urllib.parse import parse_qs, urlencode, urlunparse

        if async_playwright is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "Playwright is required for browser automation. Install optional dependencies "
                "or set BROWSER_URL to disable local browser usage."
            )

        if getattr(self, "_browser", None):
            return self._browser  # type: ignore

        playwright_instance = await async_playwright().start()

        browser_url_env = Environment.get("BROWSER_URL")

        if browser_url_env:
            launch_args_dict = {
                "headless": True,
                "stealth": True,
                "args": ["--window-size=1920,1080", "--force-color-profile=srgb"],
            }

            # Browserless query params
            query_params_to_add = {
                "proxy": "residential",
                "proxyCountry": "us",
                "timeout": "60000",
                "launch": json.dumps(launch_args_dict),
            }

            parsed_url = urlparse(browser_url_env)
            current_query_dict = parse_qs(parsed_url.query, keep_blank_values=True)

            # Update current_query_dict with new params.
            # For each key in query_params_to_add, set it in current_query_dict.
            # parse_qs returns lists for values, so wrap single values in lists.
            for key, value in query_params_to_add.items():
                current_query_dict[key] = [value]

            encoded_query = urlencode(current_query_dict, doseq=True)

            new_url = urlunparse(
                (
                    parsed_url.scheme,
                    parsed_url.netloc,
                    parsed_url.path,
                    parsed_url.params,
                    encoded_query,
                    parsed_url.fragment,
                )
            )
            connection_timeout_ms = int(query_params_to_add.get("timeout", 30000))

            browser = await playwright_instance.chromium.connect_over_cdp(new_url, timeout=connection_timeout_ms)
        else:
            # Logic for local browser launch
            browser = await playwright_instance.chromium.launch(
                headless=True,
                args=[
                    "--window-size=1920,1080",
                    "--force-color-profile=srgb",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--disable-notifications",
                    "--disable-extensions",
                    "--mute-audio",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-dev-shm-usage",
                ],
            )

        self._browser = browser

        return browser

    async def get_browser_context(self):
        """
        Get a browser context for this context.
        """
        if getattr(self, "_browser_context", None):
            return self._browser_context  # type: ignore

        browser = await self.get_browser()
        self._browser_context = await browser.new_context()
        return self._browser_context

    async def get_browser_page(self, url: str):
        """
        Get a browser page for this context.
        """
        if getattr(self, "_browser_pages", None) and url in self._browser_pages:
            return self._browser_pages[url]  # type: ignore

        if not getattr(self, "_browser_pages", None):
            self._browser_pages = {}

        browser_context = await self.get_browser_context()
        page = await browser_context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._browser_pages[url] = page
        return page

    def clear_memory(self, pattern: str | None = None):
        """
        Clear memory objects, optionally matching a pattern.

        Args:
            pattern (str | None): Optional pattern to match memory keys.
                                If None, clears all memory.
        """
        # AbstractNodeCache does not support partial clears by pattern.
        # For now, perform a full clear when requested.
        with suppress(Exception):
            require_scope().get_node_cache().clear()

    def get_memory_stats(self) -> dict[str, int | dict[str, int]]:
        """
        Get statistics about memory usage.

        Returns:
            dict: Statistics including total objects and breakdown by type.
        """
        # Node cache interface does not expose iteration over items.
        # Return an empty summary to avoid leaking implementation details.
        return {"total_objects": 0, "types": {}}

    async def cleanup(self):
        """
        Cleanup the browser context, streaming channels, and memory.
        """
        # Close all streaming channels
        if hasattr(self, "channels"):
            await self.channels.close_all()

        if getattr(self, "_browser", None):
            await self._browser.close()  # type: ignore
            self._browser = None

        # Clear memory to prevent leaks
        self.clear_memory()
