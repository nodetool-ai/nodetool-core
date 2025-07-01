import dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nodetool.api.model import RepoPath
from nodetool.api.utils import flatten_models
from nodetool.common.worker_api_client import WorkerAPIClient
from nodetool.common.huggingface_file import (
    HFFileInfo,
    HFFileRequest,
    get_huggingface_file_infos,
)
from nodetool.common.huggingface_cache import try_to_load_from_cache
from nodetool.common.system_stats import get_system_stats, SystemStats
from nodetool.common.websocket_runner import WebSocketRunner
from nodetool.common.chat_websocket_runner import ChatWebSocketRunner

from nodetool.common.environment import Environment
from nodetool.common.huggingface_cache import huggingface_download_endpoint
from nodetool.common.huggingface_models import (
    CachedModel,
    delete_cached_hf_model,
    read_cached_hf_models,
)
from nodetool.metadata.types import HuggingFaceModel
from typing import List

try:
    from nodes import init_extra_nodes  # type: ignore
    import comfy.cli_args  # type: ignore

    comfy.cli_args.args.force_fp16 = True

    init_extra_nodes()
except ImportError:
    pass


env_file = dotenv.find_dotenv(usecwd=True)

if env_file != "":
    print(f"Loading environment from {env_file}")
    dotenv.load_dotenv(env_file)

Environment.initialize_sentry()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.websocket("/predict")
async def predict_websocket_endpoint(websocket: WebSocket):
    await WebSocketRunner().run(websocket)


@app.websocket("/chat")
async def chat_websocket_endpoint(websocket: WebSocket):
    await ChatWebSocketRunner().run(websocket)


@app.websocket("/download")
async def download(websocket: WebSocket):
    await huggingface_download_endpoint(websocket)


@app.get("/system_stats")
async def system_stats() -> SystemStats:
    return get_system_stats()


# Simple in-memory cache
_cached_recommended_models: List[HuggingFaceModel] | None = None
_cached_huggingface_models: List[CachedModel] | None = None


def get_worker_app(worker: WorkerAPIClient) -> FastAPI:
    global _cached_recommended_models
    global _cached_huggingface_models

    app = FastAPI()

    if Environment.is_development():
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/models/recommended", response_model=List[HuggingFaceModel])
    async def get_recommended_models_endpoint() -> List[HuggingFaceModel]:
        global _cached_recommended_models
        if _cached_recommended_models is None:
            recommended = await worker.get_recommended_models()
            _cached_recommended_models = flatten_models(recommended)
        return _cached_recommended_models

    @app.get("/models/huggingface", response_model=List[CachedModel])
    async def get_huggingface_models_endpoint() -> List[CachedModel]:
        global _cached_huggingface_models
        if _cached_huggingface_models is None:
            _cached_huggingface_models = await worker.get_installed_models()
        return _cached_huggingface_models

    @app.get("/models/huggingface/{repo_id:path}", response_model=CachedModel)
    async def get_huggingface_model_endpoint(repo_id: str) -> CachedModel:
        models = await read_cached_hf_models()
        for model in models:
            if model.repo_id == repo_id:
                return model
        # If model not found, raise HTTPException
        raise HTTPException(status_code=404, detail=f"Model {repo_id} not found in cache")


    @app.delete("/huggingface_model")
    async def delete_huggingface_model(repo_id: str) -> bool:
        return delete_cached_hf_model(repo_id)

    @app.post("/huggingface_file_info")
    async def get_huggingface_file_info(
        requests: list[HFFileRequest],
    ) -> list[HFFileInfo]:
        return get_huggingface_file_infos(requests)

    @app.post("/huggingface/try_cache_files")
    async def try_cache_files(
        paths: list[RepoPath],
    ) -> list[RepoPath]:
        def check_path(path: RepoPath) -> bool:
            return try_to_load_from_cache(path.repo_id, path.path) is not None

        return [
            RepoPath(repo_id=path.repo_id, path=path.path, downloaded=check_path(path))
            for path in paths
        ]

    return app
