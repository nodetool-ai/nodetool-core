import asyncio
from enum import Enum
from urllib.parse import urlparse
import io
import json
import multiprocessing
import os
import queue
import urllib.parse
import uuid
import httpx
import joblib
import base64
import PIL.Image
import numpy as np
from ollama import ChatResponse
import pandas as pd
from pydub import AudioSegment
from starlette.datastructures import URL

from huggingface_hub.file_download import try_to_load_from_cache
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.asset import Asset, AssetCreateRequest, AssetList
from nodetool.types.chat import (
    MessageList,
    MessageCreateRequest,
    TaskCreateRequest,
    TaskList,
    TaskUpdateRequest,
)
from nodetool.types.graph import Node
from nodetool.types.job import Job, JobUpdate
from nodetool.types.prediction import (
    Prediction,
    PredictionResult,
)
from nodetool.types.workflow import Workflow
from nodetool.common.nodetool_api_client import NodetoolAPIClient, Response
from nodetool.metadata.types import ComfyModel, Message, Provider, Task
from nodetool.workflows.graph import Graph
from nodetool.workflows.types import (
    NodeProgress,
    NodeUpdate,
    ProcessingMessage,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    ColumnDef,
    DataframeRef,
    ImageRef,
    ModelRef,
    TextRef,
    VideoRef,
    dtype_name,
)
from nodetool.common.environment import Environment
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.property import Property
from nodetool.metadata.types import ImageRef
from nodetool.common.environment import Environment
from nodetool.common.chroma_client import get_chroma_client


from io import BytesIO
from typing import IO, Any, AsyncGenerator, Literal, Union
from chromadb.api import ClientAPI
from pickle import dumps, loads
from chromadb.config import Settings
import platform


log = Environment.get_logger()


AUDIO_CODEC = "mp3"

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

    Workflow Execution:
    - Runs the workflow by sending a RunJobRequest to a remote worker.
    - Processes and handles various types of messages received from the worker (node progress, updates, errors).

    Utility Methods:
    - Provides helper methods for converting values for prediction, handling enums, and parsing S3 URLs.
    - Supports data conversion between different formats (e.g., TextRef to string, DataFrame to pandas DataFrame).
    """

    def __init__(
        self,
        user_id: str,
        auth_token: str,
        workflow_id: str = "",
        graph: Graph = Graph(),
        variables: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        results: dict[str, Any] | None = None,
        message_queue: Union[
            queue.Queue, asyncio.Queue, multiprocessing.Queue, None
        ] = None,
        http_client: httpx.AsyncClient | None = None,
        device: str | None = None,
        endpoint_url: URL | None = None,
        encode_assets_as_base64: bool = False,
        upload_assets_to_s3: bool = False,
        chroma_client: ClientAPI | None = None,
    ):
        self.user_id = user_id
        self.auth_token = auth_token
        self.workflow_id = workflow_id
        self.graph = graph
        self.results = results if results else {}
        self.message_queue = message_queue if message_queue else asyncio.Queue()
        self.device = device
        self.variables: dict[str, Any] = variables if variables else {}
        self.nodes: dict[str, BaseNode] = {}
        self.environment: dict[str, str] = (
            environment if environment else dict(os.environ)
        )
        self.endpoint_url = endpoint_url
        self.http_client = (
            httpx.AsyncClient(follow_redirects=True, timeout=600, verify=False)
            if http_client is None
            else http_client
        )
        assert self.auth_token is not None, "Auth token is required"
        self.encode_assets_as_base64 = encode_assets_as_base64
        self.upload_assets_to_s3 = upload_assets_to_s3
        self.chroma_client = chroma_client
        env = Environment.get_environment()
        self.environment.update(env)

    def copy(self):
        """
        Creates a copy of the current ProcessingContext with shared references to most properties.

        Returns:
            ProcessingContext: A new ProcessingContext instance with copied properties.
        """
        return ProcessingContext(
            graph=self.graph,
            results=self.results.copy(),
            user_id=self.user_id,
            auth_token=self.auth_token,
            workflow_id=self.workflow_id,
            message_queue=self.message_queue,
            device=self.device,
            variables=self.variables,
            environment=self.environment,
            http_client=self.http_client,
        )

    @property
    def api_client(self) -> NodetoolAPIClient:
        """
        Lazily initializes and returns the Nodetool API client.

        Returns:
            NodetoolAPIClient: The API client for interacting with the Nodetool API.
        """
        if not hasattr(self, "_api_client"):
            self._api_client = Environment.get_nodetool_api_client(
                self.user_id,
                self.auth_token,
            )
        return self._api_client

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
        assert isinstance(self.message_queue, asyncio.Queue)
        return await self.message_queue.get()

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

    def load_nodes(self, nodes: list[Node]):
        """
        Loads nodes into the runner.

        Args:
            nodes (list[Node]): The list of nodes to load.

        Returns:
            list[BaseNode]: The list of loaded nodes.
        """
        result = []
        for node in nodes:
            if node.id in self.nodes:
                result.append(self.nodes[node.id])
            else:
                self.nodes[node.id] = BaseNode.from_dict(node.model_dump())
                result.append(self.nodes[node.id])
        return result

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
        res = await self.api_client.get(f"api/assets/{asset_id}")
        return Asset(**res.json())

    async def get_asset_url(self, asset_id: str):
        """
        Returns the asset url.

        Args:
            asset_id (str): The ID of the asset.

        Returns:
            str: The URL of the asset.
        """
        asset = await self.find_asset(asset_id)  # type: ignore

        return self.asset_storage_url(asset.file_name)

    def get_result(self, node_id: str, slot: str) -> Any:
        """
        Get the result of a node.

        Results are stored in the context's results dictionary after a node is processed.

        Args:
            node_id (str): The ID of the node.
            slot (str): The slot name.

        Returns:
            Any: The result of the node.
        """
        res = self.results.get(node_id, {})

        if res:
            if self.encode_assets_as_base64:
                res = self.encode_assets_as_uri(res)
            if self.upload_assets_to_s3:
                res = self.upload_assets_to_temp(res)
            return res.get(slot, None)
        else:
            return None

    def set_result(self, node_id: str, res: dict[str, Any]):
        """
        Set the result of a node.

        Results are stored in the context's results dictionary after a node is processed.

        Args:
            node_id (str): The ID of the node.
            res (dict[str, Any]): The result of the node.
        """
        self.results[node_id] = res

    def get_node_input_types(self, node_id: str) -> dict[str, TypeMetadata | None]:
        """
        Retrieves the input types for a given node, inferred from the output types of the source nodes.

        Args:
            node_id (str): The ID of the node.

        Returns:
            dict[str, str]: A dictionary containing the input types for the node, where the keys are the input slot names
            and the values are the types of the corresponding source nodes.
        """

        def output_type(node_id: str, slot: str):
            node = self.graph.find_node(node_id)
            if node is None:
                return None
            for output in node.outputs():
                if output.name == slot:
                    return output.type
            return None

        return {
            edge.targetHandle: output_type(edge.source, edge.sourceHandle)
            for edge in self.graph.edges
            if edge.target == node_id
        }

    def get_node_inputs(self, node_id: str) -> dict[str, Any]:
        """
        Retrieves the inputs for a given node.

        Args:
            node_id (str): The ID of the node.

        Returns:
            dict[str, Any]: A dictionary containing the inputs for the node, where the keys are the input slot names
            and the values are the results from the corresponding source nodes.
        """
        return {
            edge.targetHandle: self.get_result(edge.source, edge.sourceHandle)
            for edge in self.graph.edges
            if edge.target == node_id
        }

    def find_node(self, node_id: str) -> BaseNode:
        """
        Finds a node by its ID.

        Args:
            node_id (str): The ID of the node to be found.

        Returns:
            BaseNode: The node with the given ID.

        Raises:
            ValueError: If the node with the given ID does not exist.
        """
        node = self.graph.find_node(node_id)
        if node is None:
            raise ValueError(f"Node with ID {node_id} does not exist")
        return node

    async def get_workflow(self, workflow_id: str):
        """
        Gets the workflow by ID.

        Args:
            workflow_id (str): The ID of the workflow to retrieve.

        Returns:
            Workflow: The retrieved workflow.
        """
        res = await self.api_client.get(f"api/workflows/{workflow_id}")
        return Workflow(**res.json())

    async def _prepare_prediction(
        self,
        node_id: str,
        provider: Provider,
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
        provider: Provider,
        model: str,
        run_prediction_function: Callable[
            [Prediction],
            AsyncGenerator[PredictionResult | Prediction | ChatResponse, None],
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
        prediction = await self._prepare_prediction(
            node_id, provider, model, params, data
        )

        async for msg in run_prediction(prediction, self.environment):
            if isinstance(msg, PredictionResult):
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
            [Prediction],
            AsyncGenerator[PredictionResult | Prediction | ChatResponse, None],
        ],
        params: dict[str, Any] | None = None,
        data: Any = None,
    ) -> AsyncGenerator[PredictionResult | Prediction | ChatResponse, None]:
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

    async def paginate_assets(
        self,
        parent_id: str | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> AssetList:
        """
        Lists children assets for a given parent asset.
        Lists top level assets if parent_id is None.

        Args:
            parent_id (str | None, optional): The ID of the parent asset. Defaults to None.
            page_size (int, optional): The number of assets to return. Defaults to 100.
            cursor (str | None, optional): The cursor for pagination. Defaults to None.

        Returns:
            AssetList: The list of assets.
        """
        res = await self.api_client.get(
            "api/assets/",
            params={"parent_id": parent_id, page_size: page_size, "cursor": cursor},
        )
        return AssetList(**res.json())

    async def refresh_uri(self, asset: AssetRef):
        """
        Refreshes the URI of the asset.

        Args:
            asset (AssetRef): The asset to refresh.
        """
        if asset.asset_id:
            asset.uri = await self.get_asset_url(asset.asset_id)

    async def create_job(
        self,
        req: RunJobRequest,
    ) -> Job:
        """
        Creates a job to run a workflow but does not execute it.

        Args:
            req (RunJobRequest): The job request.

        Returns:
            Job: The created job.
        """
        res = await self.api_client.post(
            "api/jobs/", json=req.model_dump(), params={"execute": "false"}
        )
        return Job(**res.json())

    async def get_job(self, job_id: str) -> Job:
        """
        Gets the status of a job.

        Args:
            job_id (str): The ID of the job.

        Returns:
            Job: The job status.
        """
        res = await self.api_client.get(f"api/jobs/{job_id}")
        return Job(**res.json())

    async def update_job(self, job_id: str, req: JobUpdate) -> Job:
        """
        Updates the status of a job.

        Args:
            job_id (str): The ID of the job.
            req (JobUpdate): The job update request.

        Returns:
            Job: The updated job.
        """
        res = await self.api_client.put(f"api/jobs/{job_id}", json=req.model_dump())
        return Job(**res.json())

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

        req = AssetCreateRequest(
            workflow_id=self.workflow_id,
            name=name,
            content_type=content_type,
            parent_id=parent_id,
        )
        res = await self.api_client.post(
            "api/assets/",
            data={"json": req.model_dump_json()},
            files={"file": (name, content, content_type)},
        )

        return Asset(**res.json())

    async def create_message(self, req: MessageCreateRequest):
        """
        Creates a message for a thread.

        Args:
            req (MessageCreateRequest): The message to create.

        Returns:
            Message: The created message.
        """
        res = await self.api_client.post("api/messages/", json=req.model_dump())
        return Message(**res.json())

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
            MessageList: The list of messages.
        """
        res = await self.api_client.get(
            "api/messages/",
            params={"thread_id": thread_id, "limit": limit, "cursor": start_key},
        )
        return MessageList(**res.json())

    async def get_tasks(self, thread_id: str):
        """
        Gets tasks for a thread.

        Args:
            thread_id (str): The ID of the thread.

        Returns:
            TaskList: The list of tasks.
        """
        res = await self.api_client.get("api/tasks/", params={"thread_id": thread_id})
        return TaskList(**res.json())

    async def download_asset(self, asset_id: str) -> IO:
        """
        Downloads an asset from the asset storage api.

        Args:
            asset_id (str): The ID of the asset to download.

        Returns:
            IO: The downloaded asset.
        """
        asset = await self.find_asset(asset_id)
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
        response = await self.http_client.get(url, **kwargs)
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
        response = await self.http_client.post(url, **kwargs)
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
        response = await self.http_client.patch(url, **kwargs)
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
        response = await self.http_client.put(url, **kwargs)
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
        response = await self.http_client.delete(url, **kwargs)
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
        response = await self.http_client.head(url, **kwargs)
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
        if url.startswith("/"):
            url = f"file://{url}"

        # replace backslashes with forward slashes
        url = url.replace("\\", "/")

        url_parsed = urllib.parse.urlparse(url)

        if url_parsed.scheme == "data":
            fname, data = url.split(",", 1)
            image_bytes = base64.b64decode(data)
            file = io.BytesIO(image_bytes)
            # parse file ext from data uri
            ext = fname.split(";")[0].split("/")[1]
            file.name = f"{uuid.uuid4()}.{ext}"
            return file

        if url_parsed.scheme == "file":
            # Handle Windows paths
            if os.name == "nt":  # Windows system
                if url_parsed.netloc:
                    # Handle drive letter paths (file://C:/path) and UNC paths (file://server/share)
                    if ":" in url_parsed.netloc:
                        # Drive letter path
                        path = url_parsed.netloc + url_parsed.path
                    else:
                        # UNC path
                        path = "//" + url_parsed.netloc + url_parsed.path
                else:
                    # Direct path
                    path = url_parsed.path
                    if path.startswith("/"):
                        path = path[1:]  # Remove leading slash for Windows
            else:
                # Unix path
                path = url_parsed.path

            # Replace URL-encoded characters and normalize slashes
            path = urllib.parse.unquote(path)
            path = os.path.normpath(path)

            if not os.path.exists(path):
                raise FileNotFoundError(f"No such file or directory: '{path}'")

            return open(path, "rb")

        response = await self.http_get(url)
        return BytesIO(response.content)

    async def asset_to_io(self, asset_ref: AssetRef) -> IO:
        """
        Converts an AssetRef object to an IO object.

        Args:
            asset_ref (AssetRef): The AssetRef object to convert.

        Returns:
            IO: The converted IO object.

        Raises:
            ValueError: If the AssetRef is empty or contains unsupported data.
        """
        # Date takes precedence over anything else as it is the most up-to-date
        # and already in memory
        if asset_ref.data:
            if isinstance(asset_ref.data, bytes):
                return BytesIO(asset_ref.data)
            elif isinstance(asset_ref.data, list):
                raise ValueError(
                    "Batched data must be converted to list using BatchToList node"
                )
            else:
                raise ValueError(f"Unsupported data type {type(asset_ref.data)}")
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

    async def upload_tmp_asset(self, asset: AssetRef):
        if asset.uri:
            return asset.uri

        assert asset.data
        assert isinstance(asset.data, bytes)

        tmp_id = uuid.uuid4()

        return await Environment.get_asset_storage().upload(
            f"tmp/{tmp_id}", BytesIO(asset.data)
        )

    async def image_to_pil(self, image_ref: ImageRef) -> PIL.Image.Image:
        """
        Converts an ImageRef to a PIL Image object.

        Args:
            image_ref (ImageRef): The image reference to convert.

        Returns:
            PIL.Image.Image: The converted PIL Image object.
        """
        buffer = await self.asset_to_io(image_ref)
        return PIL.Image.open(buffer).convert("RGB")

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
        image = PIL.Image.open(buffer).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def audio_to_audio_segment(self, audio_ref: AudioRef) -> AudioSegment:
        """
        Converts the audio to an AudioSegment object.

        Args:
            audio_ref (AudioRef): The audio reference to convert.

        Returns:
            AudioSegment: The converted audio segment.
        """
        import pydub

        audio_bytes = await self.asset_to_io(audio_ref)
        return pydub.AudioSegment.from_file(audio_bytes)

    async def audio_to_numpy(
        self, audio_ref: AudioRef, sample_rate: int = 32_000, mono: bool = True
    ) -> tuple[np.ndarray, int, int]:
        """
        Converts the audio to a np.float32 array.

        Args:
            audio_ref (AudioRef): The audio reference to convert.
            sample_rate (int, optional): The target sample rate. Defaults to 32_000.
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
            return AudioRef(asset_id=asset.id, uri=asset.get_url or "")
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
        if df.columns:
            column_names = [col.name for col in df.columns]
            return pd.DataFrame(df.data, columns=column_names)
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
        buffer = BytesIO(dumps(data))
        if name:
            asset = await self.create_asset(
                name, "application/octet-stream", buffer, parent_id=parent_id
            )
            return DataframeRef(asset_id=asset.id, uri=asset.get_url or "")
        else:
            # TODO: avoid for large tables
            rows = data.values.tolist()
            column_defs = [
                ColumnDef(name=name, data_type=dtype_name(dtype.name))
                for name, dtype in zip(data.columns, data.dtypes)
            ]
            return DataframeRef(columns=column_defs, data=rows)

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
            return ImageRef(asset_id=asset.id, uri=asset.get_url or "")
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
        return await self.image_from_pil(PIL.Image.fromarray(image), name=name)

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
        buffer = BytesIO(s.encode("utf-8"))
        if name:
            asset = await self.create_asset(
                name, content_type, buffer, parent_id=parent_id
            )
            return TextRef(asset_id=asset.id, uri=asset.get_url or "")
        else:
            return TextRef(data=s.encode("utf-8"))

    async def video_from_frames(
        self,
        frames: list[PIL.Image.Image] | list[np.ndarray],
        fps: int = 30,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> VideoRef:
        import tempfile
        from diffusers.utils.export_utils import export_to_video

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
        import imageio

        buffer = BytesIO()
        imageio.mimwrite(buffer, video, format="mp4", fps=fps)  # type: ignore
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
            return VideoRef(asset_id=asset.id, uri=asset.get_url or "")
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
        if model_ref.asset_id is None:
            raise ValueError("ModelRef is empty")
        file = await self.asset_to_io(model_ref)
        return joblib.load(file)

    async def from_estimator(self, est: "BaseEstimator", **kwargs):  # type: ignore
        """
        Create a model asset from an estimator.

        Args:
            est (BaseEstimator): The estimator object to be serialized.
            **kwargs: Additional keyword arguments.

        Returns:
            ModelRef: A reference to the created model asset.

        """
        stream = BytesIO()
        joblib.dump(est, stream)
        stream.seek(0)
        asset = await self.create_asset("model", "application/model", stream)

        return ModelRef(uri=asset.get_url or "", asset_id=asset.id, **kwargs)

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

    async def run_worker(self, req: RunJobRequest) -> dict[str, Any]:
        """
        Runs the workflow using a remote worker.

        Args:
            req (RunJobRequest): The job request containing workflow details.

        Returns:
            dict[str, Any]: The result of running the workflow.

        Raises:
            Exception: If an error occurs during workflow execution.
        """

        # TODO: logic to determine which worker to use for given workflow

        url = Environment.get_worker_url()
        assert url is not None, "Worker URL is required"
        assert self.auth_token is not None, "Auth token is required"

        req.auth_token = self.auth_token
        req.user_id = self.user_id

        headers = {
            "Content-Type": "application/json",
        }

        if req.env is None:
            req.env = {}

        log.info("===== Run remote worker ====")
        log.info(url)
        # log.info(json.dumps(req.model_dump(), indent=2))
        result = {}

        async with self.http_client.stream(
            "POST",
            url,
            headers=headers,
            json=req.model_dump(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                try:
                    message = json.loads(line)
                    print(message)

                    if isinstance(message, dict) and "type" in message:
                        if message["type"] == "node_progress":
                            self.post_message(NodeProgress(**message))

                        elif message["type"] == "node_update":
                            self.post_message(NodeUpdate(**message))

                        elif message["type"] == "job_update":
                            self.post_message(JobUpdate(**message))

                        elif message["type"] == "error":
                            raise Exception(message["error"])

                except json.JSONDecodeError as e:
                    log.error("Error decoding message: " + str(e))

        return result

    def get_chroma_client(self):
        """
        Get a ChromaDB client instance for this context.

        Returns:
            ClientAPI: ChromaDB client instance
        """
        if self.chroma_client is None:
            self.chroma_client = get_chroma_client(self.user_id)
        return self.chroma_client

    async def is_huggingface_model_cached(self, repo_id: str):
        """
        Check if a Hugging Face model is already cached locally.

        Args:
            repo_id (str): The repository ID of the model to check.

        Returns:
            bool: True if the model is cached, False otherwise.
        """
        cache_path = try_to_load_from_cache(repo_id, "config.json")
        return cache_path is not None

    def encode_assets_as_uri(self, value: Any) -> Any:
        """
        Recursively encodes any AssetRef objects found in the given value as URIs.

        Args:
            value: Any Python value that might contain AssetRef objects

        Returns:
            Any: The value with all AssetRef objects encoded as URIs
        """
        if isinstance(value, AssetRef):
            return value.encode_data_to_uri()
        elif isinstance(value, dict):
            return {k: self.encode_assets_as_uri(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.encode_assets_as_uri(item) for item in value]
        elif isinstance(value, tuple):
            items = [self.encode_assets_as_uri(item) for item in value]
            return tuple(items)
        else:
            return value

    def upload_assets_to_temp(self, value: Any) -> Any:
        """
        Recursively uploads any AssetRef objects found in the given value to S3.

        Args:
            value: Any Python value that might contain AssetRef objects

        Returns:
            Any: The value with all AssetRef objects uploaded to S3 and replaced with their URLs
        """
        if isinstance(value, AssetRef):
            log.info(f"Uploading asset {value.uri} to S3")
            # Upload the asset data to S3 and return the URL
            if value.data is not None:
                storage = Environment.get_asset_temp_storage()
                key = uuid.uuid4().hex
                uri = storage.upload_sync(
                    key,
                    BytesIO(
                        value.data[0] if isinstance(value.data, list) else value.data
                    ),
                )
                log.info(f"Uploaded to {uri}")
                return value.__class__(uri=uri, asset_id=value.asset_id)
            else:
                return value
        elif isinstance(value, dict):
            return {k: self.upload_assets_to_temp(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.upload_assets_to_temp(item) for item in value]
        elif isinstance(value, tuple):
            items = [self.upload_assets_to_temp(item) for item in value]
            return tuple(items)
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
        # First check FONT_PATH environment variable if it exists
        if "FONT_PATH" in self.environment:
            font_path = self.environment["FONT_PATH"]
            if font_path and os.path.exists(font_path):
                # If FONT_PATH points directly to a file
                if os.path.isfile(font_path):
                    return font_path
                # If FONT_PATH is a directory, search for the font file
                for root, _, files in os.walk(font_path):
                    if font_name.lower() in [f.lower() for f in files]:
                        return os.path.join(root, font_name)

        home_dir = os.path.expanduser("~")

        # Common font locations by OS
        font_locations = {
            "Windows": [
                "C:\\Windows\\Fonts",
            ],
            "Darwin": [  # macOS
                "/System/Library/Fonts",
                "/Library/Fonts",
                f"{home_dir}/Library/Fonts",
            ],
            "Linux": [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                f"{home_dir}/.fonts",
                f"{home_dir}/.local/share/fonts",
            ],
        }

        # Get paths for current OS
        current_os = platform.system()
        search_paths = font_locations.get(current_os, [])

        log.info(f"Searching for font {font_name} in {search_paths}")

        # Search for the font file
        for base_path in search_paths:
            if os.path.exists(base_path):
                # Walk through all subdirectories
                for root, _, files in os.walk(base_path):
                    if font_name.lower() in [f.lower() for f in files]:
                        return os.path.join(root, font_name)

        raise FileNotFoundError(
            f"Could not find font '{font_name}' in system locations"
        )
