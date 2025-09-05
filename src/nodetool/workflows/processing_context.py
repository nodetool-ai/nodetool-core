from datetime import datetime
from enum import Enum
import asyncio
import imaplib
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from playwright.async_api import async_playwright, Browser
import io
import json
import os
import queue
import urllib.parse
import uuid
from pathlib import Path
import httpx
import joblib
import base64
import PIL.Image
import PIL.ImageOps
import numpy as np
import pandas as pd
from pydub import AudioSegment

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.workflow import Workflow
from nodetool.models.message import Message as DBMessage
from nodetool.types.chat import (
    MessageCreateRequest,
)
from nodetool.types.prediction import (
    Prediction,
    PredictionResult,
)
from nodetool.chat.workspace_manager import WorkspaceManager
from nodetool.metadata.types import (
    NPArray,
    Provider,
    TorchTensor,
    asset_types,
)
from nodetool.workflows.graph import Graph
from nodetool.workflows.types import (
    ProcessingMessage,
)
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    DataframeRef,
    ImageRef,
    ModelRef,
    TextRef,
    VideoRef,
)
from nodetool.config.environment import Environment
import logging
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.property import Property
from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
    get_async_chroma_client,
)


from io import BytesIO
from typing import IO, Any, AsyncGenerator, Callable
from chromadb.api import ClientAPI
from pickle import loads
from nodetool.media.common.media_constants import (
    DEFAULT_AUDIO_SAMPLE_RATE,
)
from nodetool.io.uri_utils import create_file_uri as _create_file_uri
from nodetool.media.image.image_utils import (
    numpy_to_pil_image as _numpy_to_pil_image_util,
)
from nodetool.media.image.font_utils import (
    get_system_font_path as _get_system_font_path_util,
)
from nodetool.media.video.video_utils import export_to_video_bytes


log = logging.getLogger(__name__)


def create_file_uri(path: str) -> str:
    """
    Compatibility wrapper delegating to nodetool.io.uri_utils.create_file_uri.
    """
    return _create_file_uri(path)


## AUDIO_CODEC and DEFAULT_AUDIO_SAMPLE_RATE imported from media_constants

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/1"
}

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
        chroma_client: ClientAPI | None = None,
        workspace_dir: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        tool_bridge: Any | None = None,
        ui_tool_names: set[str] | None = None,
        client_tools_manifest: dict[str, dict] | None = None,
    ):
        self.user_id = user_id or "1"
        self.auth_token = auth_token or "local_token"
        self.workflow_id = workflow_id or ""
        self.job_id = job_id
        self.graph = graph or Graph()
        self.message_queue = message_queue if message_queue else queue.Queue()
        self.device = device
        self.variables: dict[str, Any] = variables if variables else {}
        self.environment: dict[str, str] = Environment.get_environment()
        if environment:
            self.environment.update(environment)
        assert self.auth_token is not None, "Auth token is required"
        self.encode_assets_as_base64 = encode_assets_as_base64
        self.upload_assets_to_s3 = upload_assets_to_s3
        self.chroma_client = chroma_client
        if http_client is not None:
            self._http_client = http_client
        self.workspace_dir = workspace_dir or WorkspaceManager().get_current_directory()
        self.tool_bridge = tool_bridge
        self.ui_tool_names = ui_tool_names or set()
        self.client_tools_manifest = client_tools_manifest or {}
        # Use global node_cache for memory:// storage to enable portability

    def _numpy_to_pil_image(self, arr: np.ndarray) -> PIL.Image.Image:
        """Delegate to shared numpy_to_pil_image utility for consistent behavior."""
        return _numpy_to_pil_image_util(arr)

    def _memory_get(self, key: str) -> Any | None:
        """
        Retrieve an object stored under a memory:// key from the global node cache.
        Local per-instance memory is avoided for portability across processes.
        """
        try:
            return Environment.get_node_cache().get(key)
        except Exception:
            return None

    def _memory_set(self, key: str, value: Any) -> None:
        """
        Store an object under a memory:// key in the global node cache only.
        Use ttl=0 (no expiration) for persistence based on backend policy.
        """
        try:
            Environment.get_node_cache().set(key, value, ttl=0)
        except Exception:
            pass

    def get_http_client(self):
        if not hasattr(self, "_http_client"):
            self._http_client = httpx.AsyncClient(
                follow_redirects=True, timeout=600, verify=False
            )
        return self._http_client

    def get_gmail_connection(self) -> imaplib.IMAP4_SSL:
        """
        Creates a Gmail connection configuration.

        Args:
            email_address: Gmail address to connect to
            app_password: Google App Password for authentication

        Returns:
            IMAPConnection configured for Gmail

        Raises:
            ValueError: If email_address or app_password is empty
        """
        if hasattr(self, "_gmail_connection"):
            return self._gmail_connection

        email_address = self.environment.get("GOOGLE_MAIL_USER")
        app_password = self.environment.get("GOOGLE_APP_PASSWORD")
        if not email_address:
            raise ValueError("GOOGLE_MAIL_USER is not set")
        if not app_password:
            raise ValueError("GOOGLE_APP_PASSWORD is not set")

        if not email_address:
            raise ValueError("Email address is required")
        if not app_password:
            raise ValueError("App password is required")

        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(email_address, app_password)
        self._gmail_connection = imap
        return imap

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
            client_tools_manifest=(
                self.client_tools_manifest.copy() if self.client_tools_manifest else {}
            ),
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

    async def pop_message_async(self) -> ProcessingMessage:
        """
        Retrieves and removes a message from the message queue.
        The message queue is used to communicate updates to upstream
        processing.

        Returns:
            The retrieved message from the message queue.
        """
        return self.message_queue.get()

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

    def has_messages(self) -> bool:
        """
        Checks if the processing context has any messages in the message queue.

        Returns:
            bool: True if the message queue is not empty, False otherwise.
        """
        return not self.message_queue.empty()

    def asset_storage_url(self, key: str) -> str:
        """
        Returns the URL of an asset in the asset storage.

        Args:
            key (str): The key of the asset.
        """
        return Environment.get_asset_storage().get_url(key)

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
        val = Environment.get_node_cache().get(key)
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

            # Move torch tensors to CPU before caching if torch is available
            if TORCH_AVAILABLE:
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.cpu().detach()
                elif isinstance(result, torch.Tensor):
                    result = result.cpu().detach()

            Environment.get_node_cache().set(key, result, ttl)

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
            result = await Asset.get_assets_recursive(
                self.user_id, parent_id or self.user_id
            )
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
        return self.asset_storage_url(asset.file_name)

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

        prediction = await self._prepare_prediction(
            node_id, provider, model, params, data
        )

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
        prediction = await self._prepare_prediction(
            node_id, provider, model, params, data
        )

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
        content: IO,
        parent_id: str | None = None,
    ) -> Asset:
        """
        Creates an asset with the given name, content type, content, and optional parent ID.

        Args:
            name (str): The name of the asset.
            content_type (str): The content type of the asset.
            content (IO): The content of the asset.
            parent_id (str | None, optional): The ID of the parent asset. Defaults to None.

        Returns:
            Asset: The created asset.

        """
        content.seek(0)
        content_bytes = content.read()
        content.seek(0)

        # Create the asset record in the database
        asset = await Asset.create(
            user_id=self.user_id,
            name=name,
            content_type=content_type,
            parent_id=parent_id,
            workflow_id=self.workflow_id,
            size=len(content_bytes),
        )

        # Upload the content to storage
        from nodetool.config.environment import Environment

        storage = Environment.get_asset_storage()
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
        await Environment.get_asset_storage().download(asset.file_name, io)
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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().get(url, **kwargs)
        log.info(f"GET {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().post(url, **kwargs)
        log.info(f"POST {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().patch(url, **kwargs)
        log.info(f"PATCH {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().put(url, **kwargs)
        log.info(f"PUT {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().delete(url, **kwargs)
        log.info(f"DELETE {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        _headers = HTTP_HEADERS.copy()
        kwargs["headers"] = _headers.update(kwargs.get("headers", {}))
        response = await self.get_http_client().head(url, **kwargs)
        log.info(f"HEAD {url} {response.status_code}")
        response.raise_for_status()
        return response

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
        # Handle paths that start with "/" by converting to proper file:// URI
        if url.startswith("/") and not url.startswith("//"):
            url = create_file_uri(url)

        url_parsed = urllib.parse.urlparse(url)

        # Treat empty-scheme inputs as local file paths (supports Windows drive letters)
        if url_parsed.scheme == "" and not url.startswith("data:"):
            local_path = Path(url).expanduser()
            if local_path.exists():
                return open(local_path, "rb")

        if url_parsed.scheme == "data":
            fname, data = url.split(",", 1)
            image_bytes = base64.b64decode(data)
            file = io.BytesIO(image_bytes)
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
                        if ":" in netloc:
                            # Drive letter path: file://C:/path
                            path = Path(netloc + path_part)
                        else:
                            # UNC path: file://server/share
                            path = Path("//" + netloc + path_part)
                    else:
                        # file:///C:/path comes through as path_part="/C:/path"; strip leading slash
                        if (
                            len(path_part) >= 3
                            and path_part[0] == "/"
                            and path_part[2] == ":"
                        ):
                            path_part = path_part.lstrip("/")
                        path = Path(path_part)
                else:
                    # POSIX: netloc is typically empty or localhost; for others, treat as network path
                    if netloc:
                        path = Path("//" + netloc + path_part)
                    else:
                        path = Path(path_part)

                resolved_path = path.expanduser()
                if not resolved_path.exists():
                    raise FileNotFoundError(
                        f"No such file or directory: '{resolved_path}'"
                    )

                return open(resolved_path, "rb")
            except Exception as e:
                raise FileNotFoundError(f"Failed to access file: {e}")

        response = await self.http_get(url)
        return BytesIO(response.content)

    def wrap_object(self, obj: Any) -> Any:
        """Wrap raw Python objects into typed refs, storing large media in-memory.

        - Images/Audio: store via memory:// to defer encoding; use asset_to_io for bytes.
        - DataFrames/Numpy/Tensors: use existing typed wrappers.
        """
        if isinstance(obj, pd.DataFrame):
            return DataframeRef.from_pandas(obj)
        elif isinstance(obj, PIL.Image.Image):
            memory_uri = f"memory://{uuid.uuid4()}"
            self._memory_set(memory_uri, obj)
            return ImageRef(uri=memory_uri)
        elif isinstance(obj, AudioSegment):
            memory_uri = f"memory://{uuid.uuid4()}"
            self._memory_set(memory_uri, obj)
            return AudioRef(uri=memory_uri)
        elif isinstance(obj, np.ndarray):
            return NPArray.from_numpy(obj)
        elif TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            return TorchTensor.from_tensor(obj)
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
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(asset_ref, "uri") and asset_ref.uri.startswith("memory://"):
            key = asset_ref.uri
            obj = self._memory_get(key)
            if obj is not None:
                # Convert memory object to IO based on asset type and stored format
                if isinstance(obj, bytes):
                    return BytesIO(obj)
                elif isinstance(obj, PIL.Image.Image):
                    # Convert PIL Image to PNG bytes
                    buffer = BytesIO()
                    obj.save(buffer, format="PNG")
                    buffer.seek(0)
                    return buffer
                elif isinstance(obj, AudioSegment):
                    # Convert AudioSegment to MP3 bytes
                    buffer = BytesIO()
                    obj.export(buffer, format="mp3")
                    buffer.seek(0)
                    return buffer
                elif isinstance(obj, np.ndarray):
                    # Handle numpy arrays stored in memory depending on the asset type
                    if isinstance(asset_ref, ImageRef):
                        # Encode numpy image array as PNG
                        img = self._numpy_to_pil_image(obj)
                        buf = BytesIO()
                        img.convert("RGB").save(buf, format="PNG")
                        buf.seek(0)
                        return buf
                    elif isinstance(asset_ref, AudioRef):
                        # Encode numpy audio array as MP3
                        # Infer channels: (samples,) -> 1, (samples, channels) -> channels
                        channels = 1
                        audio_arr = obj
                        if audio_arr.ndim == 2:
                            # pydub expects interleaved samples for multi-channel when building from raw bytes.
                            # If provided in shape (samples, channels), interleave by reshaping C-order.
                            channels = audio_arr.shape[1]
                        # Normalize/convert dtype similarly to audio_from_numpy
                        if audio_arr.dtype == np.int16:
                            raw = audio_arr.tobytes()
                        elif audio_arr.dtype in (np.float32, np.float64, np.float16):
                            raw = (audio_arr * (2**14)).astype(np.int16).tobytes()
                        else:
                            raise ValueError(
                                f"Unsupported audio ndarray dtype {audio_arr.dtype}"
                            )
                        seg = AudioSegment(
                            data=raw,
                            frame_rate=DEFAULT_AUDIO_SAMPLE_RATE,  # default sample rate
                            sample_width=2,  # 16-bit
                            channels=int(channels),
                        )
                        out = BytesIO()
                        seg.export(out, format="mp3")
                        out.seek(0)
                        return out
                    elif isinstance(asset_ref, VideoRef):
                        # Encode numpy video array as MP4 using shared utility (T,H,W,C)
                        try:
                            # Convert numpy array to list of frames for the utility function
                            video_frames = [frame for frame in obj]

                            # Use shared video utility for consistent behavior
                            video_bytes = export_to_video_bytes(video_frames, fps=30)
                            out = BytesIO(video_bytes)
                            out.seek(0)
                            return out
                        except Exception as e:
                            raise ValueError(f"Failed to encode numpy video: {e}")
                    else:
                        # Generic fallback: return raw bytes
                        return BytesIO(obj.tobytes())
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
                elif isinstance(data, PIL.Image.Image):
                    buf = BytesIO()
                    PIL.ImageOps.exif_transpose(data).convert("RGB").save(
                        buf, format="PNG"
                    )
                    buf.seek(0)
                    return buf
                elif isinstance(data, np.ndarray):
                    img = self._numpy_to_pil_image(data)
                    buf = BytesIO()
                    img.convert("RGB").save(buf, format="PNG")
                    buf.seek(0)
                    return buf
                else:
                    raise ValueError(f"Unsupported ImageRef data type {type(data)}")
            # Audio: always encode to MP3
            elif isinstance(asset_ref, AudioRef):
                if isinstance(data, bytes):
                    return BytesIO(data)
                elif isinstance(data, AudioSegment):
                    buf = BytesIO()
                    data.export(buf, format="mp3")
                    buf.seek(0)
                    return buf
                elif isinstance(data, np.ndarray):
                    # Convert numpy audio to MP3 bytes
                    channels = 1
                    audio_arr = data
                    if audio_arr.ndim == 2:
                        channels = audio_arr.shape[1]
                    if audio_arr.dtype == np.int16:
                        raw = audio_arr.tobytes()
                    elif audio_arr.dtype in (np.float32, np.float64, np.float16):
                        raw = (audio_arr * (2**14)).astype(np.int16).tobytes()
                    else:
                        raise ValueError(
                            f"Unsupported AudioRef ndarray dtype {audio_arr.dtype}"
                        )
                    seg = AudioSegment(
                        data=raw,
                        frame_rate=DEFAULT_AUDIO_SAMPLE_RATE,
                        sample_width=2,
                        channels=int(channels),
                    )
                    out = BytesIO()
                    seg.export(out, format="mp3")
                    out.seek(0)
                    return out
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
            # Video and generic assets: assume data is already encoded bytes
            elif isinstance(data, bytes):
                return BytesIO(data)
            elif isinstance(data, str):
                return BytesIO(data.encode("utf-8"))
            elif isinstance(data, list):
                raise ValueError(
                    "Batched data must be converted to list using BatchToList node"
                )
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
        return io.read()

    async def asset_to_base64(self, asset_ref: AssetRef) -> str:
        """
        Converts an AssetRef to a base64-encoded string.
        """
        io = await self.asset_to_io(asset_ref)
        return base64.b64encode(io.read()).decode("utf-8")

    async def asset_to_data_uri(self, asset_ref: AssetRef) -> str:
        """
        Converts an AssetRef to a URI.
        """
        return f"data:image/png;base64,{await self.asset_to_base64(asset_ref)}"

    async def asset_to_data(self, asset_ref: AssetRef) -> AssetRef:
        """
        Converts an AssetRef to a URI with a specific MIME type.
        """
        if asset_ref.data is None and asset_ref.uri.startswith("memory://"):
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
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(image_ref, "uri") and image_ref.uri.startswith("memory://"):
            key = image_ref.uri
            obj = self._memory_get(key)
            if obj is not None:
                if isinstance(obj, PIL.Image.Image):
                    return obj.convert("RGB")
                # Fall through to regular conversion if not a PIL Image

        buffer = await self.asset_to_io(image_ref)
        image = PIL.Image.open(buffer)

        # Apply EXIF orientation if present
        try:
            # Use PIL's built-in method to handle EXIF orientation
            rotated_image = PIL.ImageOps.exif_transpose(image)
            # exif_transpose can return None in some cases, so fallback to original
            image = rotated_image if rotated_image is not None else image
        except (AttributeError, KeyError, TypeError):
            # If EXIF data is not available or malformed, continue without rotation
            pass

        return image.convert("RGB")

    async def image_to_numpy(self, image_ref: ImageRef) -> np.ndarray:
        """
        Converts an ImageRef to a numpy array.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            np.ndarray: The image as a numpy array.
        """
        image = await self.image_to_pil(image_ref)
        return np.array(image)

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
        return torch.tensor(np.array(image)).float() / 255.0

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
        return torch.tensor(np.array(image)).float() / 255.0

    async def image_to_base64(self, image_ref: ImageRef) -> str:
        """
        Converts the image to a PNG base64-encoded string.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            str: The base64-encoded string representation of the image.
        """
        buffer = await self.asset_to_io(image_ref)
        return base64.b64encode(buffer.read()).decode("utf-8")

    async def audio_to_audio_segment(self, audio_ref: AudioRef) -> AudioSegment:
        """
        Converts the audio to an AudioSegment object.

        Args:
            audio_ref (AudioRef): The audio reference to convert.

        Returns:
            AudioSegment: The converted audio segment.
        """
        # Check for memory:// protocol URI first (preferred for performance)
        if hasattr(audio_ref, "uri") and audio_ref.uri.startswith("memory://"):
            key = audio_ref.uri
            obj = self._memory_get(key)
            if obj is not None:
                if isinstance(obj, AudioSegment):
                    return obj
                # Fall through to regular conversion if not an AudioSegment

        import pydub

        audio_bytes = await self.asset_to_io(audio_ref)
        return pydub.AudioSegment.from_file(audio_bytes)

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
        segment = segment.set_frame_rate(sample_rate)
        if mono and segment.channels > 1:
            segment = segment.set_channels(1)
        samples = np.array(segment.get_array_of_samples())
        max_value = float(2 ** (8 * segment.sample_width - 1))
        samples = samples.astype(np.float32) / max_value

        return samples, segment.frame_rate, segment.channels

    async def audio_to_base64(self, audio_ref: AudioRef) -> str:
        """
        Converts the audio to a base64-encoded string.

        Args:
            audio_ref (AudioRef): The audio reference.

        Returns:
            str: The base64-encoded string.
        """
        audio_bytes = await self.asset_to_io(audio_ref)
        audio_bytes.seek(0)
        return base64.b64encode(audio_bytes.read()).decode("utf-8")

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
            storage = Environment.get_asset_storage()
            url = storage.get_url(asset.file_name)
            return AudioRef(asset_id=asset.id, uri=url)
        else:
            return AudioRef(data=buffer.read())

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

    async def audio_from_base64(
        self, b64: str, name: str | None = None, parent_id: str | None = None
    ) -> AudioRef:
        """
        Creates an AudioRef from a base64-encoded string.

        Args:
            b64 (str): The base64-encoded string.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.

        Returns:
            AudioRef: The AudioRef object.
        """
        return await self.audio_from_io(
            BytesIO(base64.b64decode(b64)), name=name, parent_id=parent_id
        )

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
        if data.dtype == np.int16:
            data_bytes = data.tobytes()
        elif (
            data.dtype == np.float32
            or data.dtype == np.float64
            or data.dtype == np.float16
        ):
            data_bytes = (data * (2**14)).astype(np.int16).tobytes()
        else:
            raise ValueError(f"Unsupported dtype {data.dtype}")

        audio_segment = AudioSegment(
            data=data_bytes,
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=num_channels,
        )
        return await self.audio_from_segment(
            audio_segment, name=name, parent_id=parent_id
        )

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
        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the AudioSegment directly for fast retrieval
            self._memory_set(memory_uri, audio_segment)
            # Also populate data field with binary representation for consistency
            buffer = BytesIO()
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)
            return AudioRef(uri=memory_uri, data=buffer.read())
        else:
            # Create asset when name is provided (persistence needed)
            buffer = BytesIO()
            audio_segment.export(buffer, format="mp3")
            buffer.seek(0)
            return await self.audio_from_io(buffer, name=name, parent_id=parent_id)

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
        # Prefer retrieving from in-memory storage if the DataframeRef uses a memory URI
        if getattr(df, "uri", "").startswith("memory://"):
            key = df.uri
            obj = self._memory_get(key)
            if isinstance(obj, pd.DataFrame):
                return obj
            # If not found in cache, fall back to other representations

        if df.columns:
            column_names = [col.name for col in df.columns]
            return pd.DataFrame(df.data, columns=column_names)  # type: ignore
        else:
            io = await self.asset_to_io(df)
            df = loads(io.read())
            assert isinstance(df, pd.DataFrame), "Is not a dataframe"
            return df

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
    ) -> ImageRef:
        """
        Creates an ImageRef from an IO object.

        Args:
            buffer (IO): The IO object.
            name (Optional[str], optional): The name of the asset. Defaults to None
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        if name:
            asset = await self.create_asset(
                name=name, content_type="image/png", content=buffer, parent_id=parent_id
            )
            storage = Environment.get_asset_storage()
            url = storage.get_url(asset.file_name)
            return ImageRef(asset_id=asset.id, uri=url)
        else:
            buffer.seek(0)
            return ImageRef(data=buffer.read())

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
        return await self.image_from_io(
            await self.download_file(url), name=name, parent_id=parent_id
        )

    async def image_from_bytes(
        self,
        b: bytes,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> ImageRef:
        """
        Creates an ImageRef from a bytes object.

        Args:
            b (bytes): The bytes object.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        return await self.image_from_io(BytesIO(b), name=name, parent_id=parent_id)

    async def image_from_base64(
        self,
        b64: str,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> "ImageRef":
        """
        Creates an ImageRef from a base64-encoded string.

        Args:
            b64 (str): The base64-encoded string.
            name (str | None, optional): The name of the asset. Defaults to None.
            parent_id (str | None, optional): The parent ID of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        return await self.image_from_bytes(
            base64.b64decode(b64), name=name, parent_id=parent_id
        )

    async def image_from_pil(
        self,
        image: PIL.Image.Image,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> "ImageRef":
        """
        Creates an ImageRef from a PIL Image object.

        Args:
            image (Image.Image): The PIL Image object.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the PIL Image directly for fast retrieval
            self._memory_set(memory_uri, image)
            return ImageRef(uri=memory_uri)
        else:
            # Create asset when name is provided (persistence needed)
            buffer = BytesIO()
            image.save(buffer, format="png")
            buffer.seek(0)
            return await self.image_from_io(buffer, name=name, parent_id=parent_id)

    async def image_from_numpy(
        self, image: np.ndarray, name: str | None = None, parent_id: str | None = None
    ) -> "ImageRef":
        """
        Creates an ImageRef from a numpy array.

        Args:
            image (np.ndarray): The numpy array.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.

        Returns:
            ImageRef: The ImageRef object.
        """
        pil_img = self._numpy_to_pil_image(image)
        return await self.image_from_pil(pil_img, name=name)

    async def image_from_tensor(
        self,
        image_tensor: Any,  # Change type hint to Any since torch.Tensor may not be available
    ):
        """
        Creates an ImageRef from a tensor.

        Args:
            image_tensor: The tensor.

        Returns:
            ImageRef: The ImageRef object.

        Raises:
            ImportError: If torch is not installed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for image_from_tensor")

        img = np.clip(255.0 * image_tensor.cpu().detach().numpy(), 0, 255).astype(
            np.uint8
        )
        if img.ndim == 5:
            img = img[0]
        if img.shape[0] == 1:
            return await self.image_from_numpy(img[0])

        batch = []
        for i in range(img.shape[0]):
            buffer = BytesIO()
            PIL.Image.fromarray(img[i]).save(buffer, format="png")
            batch.append(buffer.getvalue())

        return ImageRef(data=batch)

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
            if hasattr(text_ref, "uri") and text_ref.uri.startswith("memory://"):
                key = text_ref.uri
                obj = self._memory_get(key)
                if obj is not None:
                    if isinstance(obj, str):
                        return obj
                    # Fall through to regular conversion if not a string

            stream = await self.asset_to_io(text_ref)
            return stream.read().decode("utf-8")
        else:
            return text_ref

    async def text_from_str(
        self,
        s: str,
        name: str | None = None,
        content_type: str = "text/plain",
        parent_id: str | None = None,
    ) -> "TextRef":
        # Prefer memory representation when no name is provided (no persistence needed)
        if name is None:
            memory_uri = f"memory://{uuid.uuid4()}"
            # Store the string directly for fast retrieval
            self._memory_set(memory_uri, s)
            return TextRef(uri=memory_uri)
        else:
            # Create asset when name is provided (persistence needed)
            buffer = BytesIO(s.encode("utf-8"))
            asset = await self.create_asset(
                name, content_type, buffer, parent_id=parent_id
            )
            storage = Environment.get_asset_storage()
            url = storage.get_url(asset.file_name)
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

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp:
            export_to_video(frames, temp.name, fps=fps)
            return await self.video_from_io(open(temp.name, "rb"))

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
        # Convert numpy array to list of frames for the utility function
        video_frames = [frame for frame in video]

        # Use shared video utility for consistent behavior
        video_bytes = export_to_video_bytes(video_frames, fps=fps)

        # Create BytesIO from the video bytes
        buffer = BytesIO(video_bytes)
        buffer.seek(0)
        return await self.video_from_io(buffer, name=name, parent_id=parent_id)

    async def video_from_io(
        self,
        buffer: IO,
        name: str | None = None,
        parent_id: str | None = None,
    ):
        """
        Creates an VideoRef from an IO object.

        Args:
            context (ProcessingContext): The processing context.
            buffer (IO): The IO object.
            name (Optional[str], optional): The name of the asset. Defaults to None.

        Returns:
            VideoRef: The VideoRef object.
        """
        if name:
            asset = await self.create_asset(
                name, "video/mpeg", buffer, parent_id=parent_id
            )
            storage = Environment.get_asset_storage()
            url = storage.get_url(asset.file_name)
            return VideoRef(asset_id=asset.id, uri=url)
        else:
            return VideoRef(data=buffer.read())

    async def video_from_bytes(
        self, b: bytes, name: str | None = None, parent_id: str | None = None
    ) -> VideoRef:
        """
        Creates a VideoRef from a bytes object.

        Args:
            b (bytes): The bytes object.
            name (Optional[str], optional): The name of the asset. Defaults to None.
            parent_id (Optional[str], optional): The parent ID of the asset. Defaults to None.

        Returns:
            VideoRef: The VideoRef object.
        """
        return await self.video_from_io(BytesIO(b), name=name, parent_id=parent_id)

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
        if hasattr(model_ref, "uri") and model_ref.uri.startswith("memory://"):
            key = model_ref.uri
            obj = self._memory_get(key)
            # Return the model object directly if it's already a model
            if obj is not None and (hasattr(obj, "fit") or hasattr(obj, "predict")):
                return obj
            # Fall through to regular conversion if not a model object

        if model_ref.asset_id is None:
            raise ValueError("ModelRef is empty")
        file = await self.asset_to_io(model_ref)
        return joblib.load(file)

    async def from_estimator(self, est: "BaseEstimator", name: str | None = None, **kwargs):  # type: ignore
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
            stream = BytesIO()
            joblib.dump(est, stream)
            stream.seek(0)
            asset = await self.create_asset(name, "application/model", stream)

            storage = Environment.get_asset_storage()
            url = storage.get_url(asset.file_name)
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
                return io.read().decode("utf-8")
            else:
                img_bytes = io.read()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
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

        return is_model_cached(repo_id)

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

    async def embed_assets_in_data(self, value: Any) -> Any:
        """
        Recursively embeds any memory:// assets in the given value.
        """
        if isinstance(value, AssetRef):
            if value.uri.startswith("memory://") or value.data is not None:
                data_bytes = await self.asset_to_bytes(value)
                return value.model_copy(update={"uri": None, "data": data_bytes})
            return value
        elif isinstance(value, dict):
            keys = list(value.keys())
            results = await asyncio.gather(
                *[self.embed_assets_in_data(value[k]) for k in keys]
            )
            return {k: r for k, r in zip(keys, results)}
        elif isinstance(value, list):
            results = await asyncio.gather(
                *[self.embed_assets_in_data(item) for item in value]
            )
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(
                *[self.embed_assets_in_data(item) for item in value]
            )
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

        def get_ext(value: Any) -> str:
            if isinstance(value, ImageRef):
                return "png"
            elif isinstance(value, AudioRef):
                return "mp3"
            elif isinstance(value, VideoRef):
                return "mp4"
            else:
                return "bin"

        async def upload_asset(value: AssetRef) -> dict[str, Any]:
            log.info(f"Uploading asset {value.uri} to temp storage")
            # Always normalize to bytes via the canonical encoder
            data = await self.asset_to_bytes(value)

            if data is not None:
                storage = Environment.get_temp_storage()
                ext = get_ext(value)
                key = uuid.uuid4().hex + "." + ext
                uri = await storage.upload(
                    key,
                    BytesIO(data),
                )
                log.info(f"Uploaded {len(data)} bytes to {uri}")
                return {
                    "type": value.type,
                    "uri": uri,
                    "asset_id": value.asset_id,
                }
            else:
                return {
                    "type": value.type,
                    "uri": value.uri,
                    "asset_id": value.asset_id,
                }

        if isinstance(value, AssetRef):
            return await upload_asset(value)
        elif isinstance(value, dict):
            if (
                "type" in value
                and "uri" in value
                and "asset_id" in value
                and "data" in value
                and value["type"] in asset_types
            ):
                asset_ref = AssetRef(
                    type=value["type"],
                    uri=value["uri"],
                    asset_id=value["asset_id"],
                    data=value["data"],
                )
                return await upload_asset(asset_ref)
            keys = list(value.keys())
            tasks = [self.upload_assets_to_temp(value[k]) for k in keys]
            results = await asyncio.gather(*tasks)
            return {k: r for k, r in zip(keys, results)}
        elif isinstance(value, list):
            results = await asyncio.gather(
                *[self.upload_assets_to_temp(item) for item in value]
            )
            return list(results)
        elif isinstance(value, tuple):
            results = await asyncio.gather(
                *[self.upload_assets_to_temp(item) for item in value]
            )
            return tuple(results)
        else:
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
        return _get_system_font_path_util(font_name, self.environment)

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
        from urllib.parse import urlunparse, parse_qs, urlencode

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

            browser = await playwright_instance.chromium.connect_over_cdp(
                new_url, timeout=connection_timeout_ms
            )
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
        try:
            Environment.get_node_cache().clear()
        except Exception:
            pass

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
        Cleanup the browser context and pages.
        """
        if getattr(self, "_browser", None):
            await self._browser.close()  # type: ignore
            self._browser = None

        # Clear memory to prevent leaks
        self.clear_memory()
