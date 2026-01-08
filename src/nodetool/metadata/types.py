import base64
import enum
import json
from datetime import UTC, date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import NoneType
from typing import Any, Dict, List, Literal, Optional, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.api_graph import Graph

#######################
# Type Name Mappings
#######################
# Maps Python types to their string representations and vice versa

TypeToName = {}
NameToType = {}


def add_type_name(type: Type, name: str):
    """
    Adds a type name to the TypeToEnum and EnumToType mappings.
    """
    TypeToName[type] = name
    NameToType[name] = type


def add_type_names(types):
    """
    Add type names to the TypeToEnum and EnumToType mappings.
    """
    for type, name in types.items():
        add_type_name(type, name)


# Add the default type names
add_type_names(
    {
        Any: "any",
        NoneType: "none",
        list: "list",
        dict: "dict",
        object: "object",
        tuple: "tuple",
        int: "int",
        float: "float",
        bool: "bool",
        str: "str",
        bytes: "bytes",
        Enum: "enum",
        Union: "union",
    }
)


#######################
# Base Types
#######################
# Core base classes that other types inherit from


class BaseType(BaseModel):
    """
    This is the base class for all Nodetool types.

    It is used to create a mapping of type names to their corresponding classes.
    """

    type: Any

    @classmethod
    def __init_subclass__(cls):
        """
        This method is called when a subclass of BaseType is created.
        We remember the mapping of the subclass to its type name,
        so that we can use it later to create instances of the subclass from the type name.
        """
        super().__init_subclass__()
        if hasattr(cls, "type"):
            add_type_name(cls, cls.type)

    @classmethod
    def from_dict(cls, data):
        """
        Create an instance of the class from a dictionary.

        Args:
            data (dict): The dictionary to create the instance from.

        Returns:
            BaseType: The instance of the class.
        """
        type_name = data.get("type")
        if type_name is None:
            raise ValueError("Type name is missing. Types must derive from BaseType")
        if type_name not in NameToType:
            raise ValueError(f"Unknown type name: {type_name}. Types must derive from BaseType. Data: {data}")
        return NameToType[type_name](**data)


class Collection(BaseType):
    type: Literal["collection"] = "collection"
    name: str = ""


class FaissIndex(BaseType):
    type: Literal["faiss_index"] = "faiss_index"
    index: Any = None


#######################
# Date and Time Types
#######################
# Types for handling dates, times, and timestamps


class Date(BaseType):
    type: Literal["date"] = "date"
    year: int = 0
    month: int = 0
    day: int = 0

    @classmethod
    def from_date(cls, data: date):
        return cls(year=data.year, month=data.month, day=data.day)

    def to_date(self):
        return date(self.year, self.month, self.day)


class Datetime(BaseType):
    type: Literal["datetime"] = "datetime"
    year: int = 0
    month: int = 0
    day: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    microsecond: int = 0
    tzinfo: str = "UTC"
    utc_offset: float = 0

    @staticmethod
    def from_timestamp(timestamp: float):
        return Datetime.from_datetime(datetime.fromtimestamp(timestamp))

    def to_datetime(self):
        return datetime(
            year=self.year,
            month=self.month,
            day=self.day,
            hour=self.hour,
            minute=self.minute,
            second=self.second,
            microsecond=self.microsecond,
            tzinfo=(timezone(timedelta(seconds=self.utc_offset), self.tzinfo) if self.utc_offset else UTC),
        )

    @staticmethod
    def from_datetime(dt: datetime):
        utc_offset = dt.utcoffset()
        return Datetime(
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour=dt.hour,
            minute=dt.minute,
            second=dt.second,
            microsecond=dt.microsecond,
            tzinfo=dt.tzinfo.tzname(dt) or "UTC" if dt.tzinfo else "UTC",
            utc_offset=utc_offset.total_seconds() if utc_offset else 0,
        )


#######################
# Asset Reference Types
#######################
# Types for referencing different kinds of assets (files, models, etc.)

asset_types = set()


class AssetRef(BaseType):
    """
    Base class for asset references in the workflow system.

    Asset references can point to data in multiple ways:
    - uri: A URI pointing to the asset location (memory://, data:, file://, http(s)://, asset://)
    - asset_id: Database ID for persistent assets
    - data: Direct byte content (with canonical encoding per asset type)
    - metadata: Optional metadata dict

    Data Field Canonical Encodings:
    --------------------------------
    The data field, when populated, must use standardized encodings to ensure
    consistent processing by the frontend and other consumers:

    - **ImageRef**: PNG bytes (image/png)
      * Decoded from: PIL.Image objects, numpy arrays, raw image bytes
      * Frontend expects: Can be converted to data:image/png;base64,{data}

    - **AudioRef**: MP3 bytes (audio/mp3 or audio/mpeg)
      * Decoded from: AudioSegment objects, numpy audio arrays, raw audio bytes
      * Frontend expects: Can be converted to data:audio/mp3;base64,{data}

    - **VideoRef**: MP4 bytes (video/mp4)
      * Decoded from: Video file bytes
      * Frontend expects: Can be converted to data:video/mp4;base64,{data}

    - **TextRef**: UTF-8 encoded bytes (text/plain)
      * Decoded from: Python str objects
      * Frontend expects: Can be decoded as text or converted to data URI

    - **Generic AssetRef**: Raw bytes (application/octet-stream)
      * No transformation applied
      * Frontend must handle based on context

    Memory URIs (memory://):
    -----------------------
    - Used for temporary storage of Python objects during workflow execution
    - Objects are stored in ResourceScope's MemoryUriCache with 5-minute TTL
    - When serializing for client (e.g., in result_for_client), memory URIs are
      resolved to populate the data field with canonical byte encoding
    - Supports: PIL.Image, AudioSegment, str, pd.DataFrame, bytes, and more

    Data URIs (data:):
    -----------------
    - Embed data directly in the URI using base64 encoding
    - Format: data:{mime};base64,{base64_data}
    - Common for small assets or when network requests are undesirable

    File URIs (file://):
    -------------------
    - Point to local filesystem paths

    Asset IDs (asset://):
    --------------------
    - Reference persistent assets stored in the database
    - Require database lookup to retrieve content
    """

    type: Any = "asset"
    uri: str = ""
    asset_id: str | None = None
    data: Any = None
    metadata: Dict[str, Any] | None = None

    @staticmethod
    def from_file(path: str):
        # Accept already-formed file URIs
        if isinstance(path, str) and path.startswith("file://"):
            return AssetRef(uri=path)

        try:
            resolved_path = Path(path).expanduser().resolve(strict=False)
            return AssetRef(uri=resolved_path.as_uri())
        except Exception:
            # Fallback: best-effort POSIX-style URI
            posix_path = Path(path).as_posix()
            prefix = "file:///" if not posix_path.startswith("/") else "file://"
            return AssetRef(uri=f"{prefix}{posix_path}")

    def to_dict(self):
        res = {
            "uri": self.uri,
        }
        if self.asset_id:
            res["asset_id"] = self.asset_id
        return res

    def is_empty(self):
        return self.uri == "" and self.asset_id is None and self.data is None

    def is_set(self):
        return not self.is_empty()

    def encode_data_to_uri(self):
        if self.data:
            new_ref = self.__class__(
                uri=f"data:application/octet-stream;base64,{base64.b64encode(self.data[0] if isinstance(self.data, list) else self.data).decode('utf-8')}",
                asset_id=self.asset_id,
            )
            return new_ref
        return self

    @property
    def document_id(self):
        if self.asset_id:
            return "asset://" + self.asset_id
        return self.uri

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if hasattr(cls, "type"):
            asset_types.add(cls.type)


class FilePath(BaseType):
    type: Literal["file_path"] = "file_path"
    path: str = ""


class FolderPath(BaseType):
    type: Literal["folder_path"] = "folder_path"
    path: str = ""


class FolderRef(AssetRef):
    type: Literal["folder"] = "folder"


class ModelRef(AssetRef):
    type: Literal["model_ref"] = "model_ref"


class VideoRef(AssetRef):
    """A reference to a video asset."""

    type: Literal["video"] = "video"
    duration: Optional[float] = None  # Duration in seconds
    format: Optional[str] = None


class TextRef(AssetRef):
    type: Literal["text"] = "text"


class AudioRef(AssetRef):
    """A reference to an audio asset."""

    type: Literal["audio"] = "audio"


class ImageRef(AssetRef):
    """A reference to an image asset."""

    type: Literal["image"] = "image"


class DocumentRef(AssetRef):
    """
    A reference to a document asset.
    Can be a PDF, DOCX, etc.
    """

    type: Literal["document"] = "document"


class Model3DRef(AssetRef):
    """
    A reference to a 3D model asset.
    Supports common 3D formats like GLB, GLTF, OBJ, FBX, STL, PLY, USDZ.
    """

    type: Literal["model_3d"] = "model_3d"
    format: Optional[str] = None  # The 3D format (glb, gltf, obj, fbx, stl, ply, usdz)


class RSSEntry(BaseType):
    type: Literal["rss_entry"] = "rss_entry"
    title: str = ""
    link: str = ""
    published: "Datetime" = Datetime()
    summary: str = ""
    author: str = ""


class WorkflowRef(BaseType):
    type: Literal["workflow"] = "workflow"
    id: str = ""


class NodeRef(BaseType):
    type: Literal["node"] = "node"
    id: str = ""


class FontRef(BaseType):
    type: Literal["font"] = "font"
    name: str = ""


class Provider(str, enum.Enum):
    AIME = "aime"
    OpenAI = "openai"
    OpenRouter = "openrouter"
    Anthropic = "anthropic"
    MiniMax = "minimax"
    Replicate = "replicate"
    Ollama = "ollama"
    LMStudio = "lmstudio"
    KIE = "kie"
    # Comfy providers (two explicit entries)
    ComfyLocal = "comfy_local"
    ComfyRunpod = "comfy_runpod"
    Local = "local"
    LlamaCpp = "llama_cpp"
    Gemini = "gemini"
    VLLM = "vllm"
    Empty = "empty"
    MLX = "mlx"
    FalAI = "fal_ai"
    HuggingFace = "huggingface"  # local hf models
    # Providers for HuggingFace Inference Providers
    HuggingFaceCohere = "huggingface_cohere"
    HuggingFaceFalAI = "huggingface_fal_ai"
    HuggingFaceFeatherlessAI = "huggingface_featherless_ai"
    HuggingFaceFireworksAI = "huggingface_fireworks_ai"
    HuggingFaceGroq = "huggingface_groq"
    HuggingFaceCerebras = "huggingface_cerebras"
    HuggingFaceHFInference = "huggingface_hf_inference"
    HuggingFaceHyperbolic = "huggingface_hyperbolic"
    HuggingFaceNebius = "huggingface_nebius"
    HuggingFaceNovita = "huggingface_novita"
    HuggingFaceNscale = "huggingface_nscale"
    HuggingFaceOpenAI = "huggingface_openai"
    HuggingFaceReplicate = "huggingface_replicate"
    HuggingFaceSambanova = "huggingface_sambanova"
    HuggingFaceScaleway = "huggingface_scaleway"
    HuggingFaceTogether = "huggingface_together"
    HuggingFaceZAI = "huggingface_zai"


class InferenceProvider(str, Enum):
    cerebras = "cerebras"
    cohere = "cohere"
    fal_ai = "fal-ai"
    featherless_ai = "featherless-ai"
    fireworks_ai = "fireworks-ai"
    groq = "groq"
    hf_inference = "hf-inference"
    hyperbolic = "hyperbolic"
    nebius = "nebius"
    novita = "novita"
    nscale = "nscale"
    openai = "openai"
    replicate = "replicate"
    sambanova = "sambanova"
    scaleway = "scaleway"
    together = "together"
    zai = "zai-org"


class InferenceProviderAudioClassificationModel(BaseType):
    type: Literal["inference_provider_audio_classification_model"] = "inference_provider_audio_classification_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderAutomaticSpeechRecognitionModel(BaseType):
    type: Literal["inference_provider_automatic_speech_recognition_model"] = (
        "inference_provider_automatic_speech_recognition_model"
    )
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderImageClassificationModel(BaseType):
    type: Literal["inference_provider_image_classification_model"] = "inference_provider_image_classification_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderImageToImageModel(BaseType):
    type: Literal["inference_provider_image_to_image_model"] = "inference_provider_image_to_image_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderImageSegmentationModel(BaseType):
    type: Literal["inference_provider_image_segmentation_model"] = "inference_provider_image_segmentation_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextClassificationModel(BaseType):
    type: Literal["inference_provider_text_classification_model"] = "inference_provider_text_classification_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderSummarizationModel(BaseType):
    type: Literal["inference_provider_summarization_model"] = "inference_provider_summarization_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextToImageModel(BaseType):
    type: Literal["inference_provider_text_to_image_model"] = "inference_provider_text_to_image_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTranslationModel(BaseType):
    type: Literal["inference_provider_translation_model"] = "inference_provider_translation_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextToTextModel(BaseType):
    type: Literal["inference_provider_text_to_text_model"] = "inference_provider_text_to_text_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextToSpeechModel(BaseType):
    type: Literal["inference_provider_text_to_speech_model"] = "inference_provider_text_to_speech_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextToAudioModel(BaseType):
    type: Literal["inference_provider_text_to_audio_model"] = "inference_provider_text_to_audio_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class InferenceProviderTextGenerationModel(BaseType):
    type: Literal["inference_provider_text_generation_model"] = "inference_provider_text_generation_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""


class OpenAIEmbeddingModel(str, enum.Enum):
    ADA_002 = "text-embedding-ada-002"
    SMALL = "text-embedding-3-small"
    LARGE = "text-embedding-3-large"


class LanguageModel(BaseType):
    type: Literal["language_model"] = "language_model"
    provider: Provider = Provider.Empty
    id: str = ""
    name: str = ""
    path: str | None = None
    supported_tasks: list[str] = Field(default_factory=list)


class ImageModel(BaseType):
    type: Literal["image_model"] = "image_model"
    provider: Provider = Provider.Empty
    id: str = ""
    name: str = ""
    path: str | None = None
    supported_tasks: list[str] = Field(default_factory=list)


class TTSModel(BaseType):
    type: Literal["tts_model"] = "tts_model"
    provider: Provider = Provider.Empty
    id: str = ""
    name: str = ""
    path: str | None = None
    voices: list[str] = Field(default_factory=list)
    selected_voice: str = ""


class ASRModel(BaseType):
    type: Literal["asr_model"] = "asr_model"
    provider: Provider = Provider.Empty
    id: str = ""
    name: str = ""
    path: str | None = None


class VideoModel(BaseType):
    type: Literal["video_model"] = "video_model"
    provider: Provider = Provider.Empty
    id: str = ""
    name: str = ""
    path: str | None = None
    supported_tasks: list[str] = Field(default_factory=list)


class LlamaModel(BaseType):
    type: Literal["llama_model"] = "llama_model"
    name: str = ""
    repo_id: str = ""
    modified_at: str = ""
    size: int = 0
    digest: str = ""
    details: dict = Field(default_factory=dict)

    def is_set(self) -> bool:
        return self.repo_id != ""


class LlamaCppModel(BaseType):
    """Model type for llama.cpp GGUF models.

    These models are stored in the llama.cpp native cache directories:
    - Linux: ~/.cache/llama.cpp/hf/
    - macOS: ~/Library/Caches/llama.cpp/hf/
    - Windows: %LOCALAPPDATA%/llama.cpp/hf/
    """

    type: Literal["llama_cpp_model"] = "llama_cpp_model"
    name: str = ""
    repo_id: str = ""
    filename: str = ""
    size: int = 0

    def is_set(self) -> bool:
        return self.repo_id != ""


class OpenAIModel(BaseType):
    type: Literal["openai_model"] = "openai_model"
    id: str = ""
    object: str = ""
    created: int = 0
    owned_by: str = ""


#######################
# Hugging Face Models
#######################


class HuggingFaceModel(BaseType):
    type: Any = "hf.model"
    repo_id: str = ""
    path: str | None = None
    variant: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None

    def is_set(self) -> bool:
        return self.repo_id != ""

    def is_empty(self) -> bool:
        return self.repo_id == ""


class HFImageTextToText(HuggingFaceModel):
    type: Literal["hf.image_text_to_text"] = "hf.image_text_to_text"


class HFVisualQuestionAnswering(HuggingFaceModel):
    type: Literal["hf.visual_question_answering"] = "hf.visual_question_answering"


class HFMiniCPM(HuggingFaceModel):
    type: Literal["hf.minicpm"] = "hf.minicpm"


class HFGOTOCR(HuggingFaceModel):
    type: Literal["hf.gotocr"] = "hf.gotocr"


class HFDocumentQuestionAnswering(HuggingFaceModel):
    type: Literal["hf.document_question_answering"] = "hf.document_question_answering"


class HFVideoTextToText(HuggingFaceModel):
    type: Literal["hf.video_text_to_text"] = "hf.video_text_to_text"


class HFComputerVision(HuggingFaceModel):
    type: Literal["hf.computer_vision"] = "hf.computer_vision"


class HFDepthEstimation(HuggingFaceModel):
    type: Literal["hf.depth_estimation"] = "hf.depth_estimation"


class HFImageClassification(HuggingFaceModel):
    type: Literal["hf.image_classification"] = "hf.image_classification"


class HFObjectDetection(HuggingFaceModel):
    type: Literal["hf.object_detection"] = "hf.object_detection"


class HFImageSegmentation(HuggingFaceModel):
    type: Literal["hf.image_segmentation"] = "hf.image_segmentation"


class HFTextToImage(HuggingFaceModel):
    type: Literal["hf.text_to_image"] = "hf.text_to_image"


class HFStableDiffusion(HuggingFaceModel):
    type: Literal["hf.stable_diffusion"] = "hf.stable_diffusion"


class HFStableDiffusionCheckpoint(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_checkpoint"] = "hf.stable_diffusion_checkpoint"


class HFStableDiffusionXL(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_xl"] = "hf.stable_diffusion_xl"


class HFStableDiffusionXLCheckpoint(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_xl_checkpoint"] = "hf.stable_diffusion_xl_checkpoint"


class HFStableDiffusion3(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_3"] = "hf.stable_diffusion_3"


class HFStableDiffusion3Checkpoint(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_3_checkpoint"] = "hf.stable_diffusion_3_checkpoint"


class HFStableDiffusionXLRefiner(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_xl_refiner"] = "hf.stable_diffusion_xl_refiner"


class HFStableDiffusionXLRefinerCheckpoint(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_xl_refiner_checkpoint"] = "hf.stable_diffusion_xl_refiner_checkpoint"


class HFFlux(HuggingFaceModel):
    type: Literal["hf.flux"] = "hf.flux"


class HFFluxCheckpoint(HuggingFaceModel):
    type: Literal["hf.flux_checkpoint"] = "hf.flux_checkpoint"


class HFFluxFP8(HuggingFaceModel):
    type: Literal["hf.flux_fp8"] = "hf.flux_fp8"


class HFFluxFP8Checkpoint(HuggingFaceModel):
    type: Literal["hf.flux_fp8_checkpoint"] = "hf.flux_fp8_checkpoint"


class HFFluxKontext(HuggingFaceModel):
    type: Literal["hf.flux_kontext"] = "hf.flux_kontext"


class HFFluxKontextCheckpoint(HuggingFaceModel):
    type: Literal["hf.flux_kontext_checkpoint"] = "hf.flux_kontext_checkpoint"


class HFFluxDepth(HuggingFaceModel):
    type: Literal["hf.flux_depth"] = "hf.flux_depth"


class HFFluxDepthCheckpoint(HuggingFaceModel):
    type: Literal["hf.flux_depth_checkpoint"] = "hf.flux_depth_checkpoint"


class HFFluxRedux(HuggingFaceModel):
    type: Literal["hf.flux_redux"] = "hf.flux_redux"


class HFFluxReduxCheckpoint(HuggingFaceModel):
    type: Literal["hf.flux_redux_checkpoint"] = "hf.flux_redux_checkpoint"


class HFFluxFill(HuggingFaceModel):
    type: Literal["hf.inpainting"] = "hf.inpainting"


class HFFluxFillCheckpoint(HuggingFaceModel):
    type: Literal["hf.inpainting_checkpoint"] = "hf.inpainting_checkpoint"


class HFQwenImage(HuggingFaceModel):
    type: Literal["hf.qwen_image"] = "hf.qwen_image"


class HFQwen2_5_VL(HuggingFaceModel):
    type: Literal["hf.qwen2_5_vl"] = "hf.qwen2_5_vl"


class HFQwen3_VL(HuggingFaceModel):
    type: Literal["hf.qwen3_vl"] = "hf.qwen3_vl"


class HFQwenImageCheckpoint(HuggingFaceModel):
    type: Literal["hf.qwen_image_checkpoint"] = "hf.qwen_image_checkpoint"


class HFQwenImageEdit(HuggingFaceModel):
    type: Literal["hf.qwen_image_edit"] = "hf.qwen_image_edit"


class HFQwenImageEditCheckpoint(HuggingFaceModel):
    type: Literal["hf.qwen_image_edit_checkpoint"] = "hf.qwen_image_edit_checkpoint"


class HFControlNet(HuggingFaceModel):
    type: Literal["hf.controlnet"] = "hf.controlnet"


class HFControlNetSDXL(HuggingFaceModel):
    type: Literal["hf.controlnet_sdxl"] = "hf.controlnet_sdxl"


class HFControlNetFlux(HuggingFaceModel):
    type: Literal["hf.controlnet_flux"] = "hf.controlnet_flux"


class HFIPAdapter(HuggingFaceModel):
    type: Literal["hf.ip_adapter"] = "hf.ip_adapter"


class HFStyleModel(HuggingFaceModel):
    type: Literal["hf.style_model"] = "hf.style_model"


class HFLoraSD(HuggingFaceModel):
    type: Literal["hf.lora_sd"] = "hf.lora_sd"


class HFLoraSDXL(HuggingFaceModel):
    type: Literal["hf.lora_sdxl"] = "hf.lora_sdxl"


class HFLoraQwenImage(HuggingFaceModel):
    type: Literal["hf.lora_qwen_image"] = "hf.lora_qwen_image"


class HFImageToText(HuggingFaceModel):
    type: Literal["hf.image_to_text"] = "hf.image_to_text"


class HFImageToImage(HuggingFaceModel):
    type: Literal["hf.image_to_image"] = "hf.image_to_image"


class HFImageToVideo(HuggingFaceModel):
    type: Literal["hf.image_to_video"] = "hf.image_to_video"


class HFUnconditionalImageGeneration(HuggingFaceModel):
    type: Literal["hf.unconditional_image_generation"] = "hf.unconditional_image_generation"


class HFUnet(HuggingFaceModel):
    type: Literal["hf.unet"] = "hf.unet"


class HFVAE(HuggingFaceModel):
    type: Literal["hf.vae"] = "hf.vae"


class HFCLIP(HuggingFaceModel):
    type: Literal["hf.clip"] = "hf.clip"


class HFT5(HuggingFaceModel):
    type: Literal["hf.t5"] = "hf.t5"


class HFQwenVL(HuggingFaceModel):
    type: Literal["hf.qwen_vl"] = "hf.qwen_vl"


class HFCLIPVision(HuggingFaceModel):
    type: Literal["hf.clip_vision"] = "hf.clip_vision"


class HFVideoClassification(HuggingFaceModel):
    type: Literal["hf.video_classification"] = "hf.video_classification"


class HFTextToVideo(HuggingFaceModel):
    type: Literal["hf.text_to_video"] = "hf.text_to_video"


class HFZeroShotImageClassification(HuggingFaceModel):
    type: Literal["hf.zero_shot_image_classification"] = "hf.zero_shot_image_classification"


class HFMaskGeneration(HuggingFaceModel):
    type: Literal["hf.mask_generation"] = "hf.mask_generation"


class HFZeroShotObjectDetection(HuggingFaceModel):
    type: Literal["hf.zero_shot_object_detection"] = "hf.zero_shot_object_detection"


class HFTextTo3D(HuggingFaceModel):
    type: Literal["hf.text_to_3d"] = "hf.text_to_3d"


class HFImageTo3D(HuggingFaceModel):
    type: Literal["hf.image_to_3d"] = "hf.image_to_3d"


class HFImageFeatureExtraction(HuggingFaceModel):
    type: Literal["hf.image_feature_extraction"] = "hf.image_feature_extraction"


class HFNaturalLanguageProcessing(HuggingFaceModel):
    type: Literal["hf.natural_language_processing"] = "hf.natural_language_processing"


class HFTextClassification(HuggingFaceModel):
    type: Literal["hf.text_classification"] = "hf.text_classification"


class HFTokenClassification(HuggingFaceModel):
    type: Literal["hf.token_classification"] = "hf.token_classification"


class HFTableQuestionAnswering(HuggingFaceModel):
    type: Literal["hf.table_question_answering"] = "hf.table_question_answering"


class HFQuestionAnswering(HuggingFaceModel):
    type: Literal["hf.question_answering"] = "hf.question_answering"


class HFZeroShotClassification(HuggingFaceModel):
    type: Literal["hf.zero_shot_classification"] = "hf.zero_shot_classification"


class HFTranslation(HuggingFaceModel):
    type: Literal["hf.translation"] = "hf.translation"


class HFSummarization(HuggingFaceModel):
    type: Literal["hf.summarization"] = "hf.summarization"


class HFFeatureExtraction(HuggingFaceModel):
    type: Literal["hf.feature_extraction"] = "hf.feature_extraction"


class HFTextGeneration(HuggingFaceModel):
    type: Literal["hf.text_generation"] = "hf.text_generation"


class HFText2TextGeneration(HuggingFaceModel):
    type: Literal["hf.text2text_generation"] = "hf.text2text_generation"


class HFFillMask(HuggingFaceModel):
    type: Literal["hf.fill_mask"] = "hf.fill_mask"


class HFSentenceSimilarity(HuggingFaceModel):
    type: Literal["hf.sentence_similarity"] = "hf.sentence_similarity"


class HFReranker(HuggingFaceModel):
    type: Literal["hf.reranker"] = "hf.reranker"


class HFTextToSpeech(HuggingFaceModel):
    type: Literal["hf.text_to_speech"] = "hf.text_to_speech"


class HFTextToAudio(HuggingFaceModel):
    type: Literal["hf.text_to_audio"] = "hf.text_to_audio"


class HFAutomaticSpeechRecognition(HuggingFaceModel):
    type: Literal["hf.automatic_speech_recognition"] = "hf.automatic_speech_recognition"


class HFAudioToAudio(HuggingFaceModel):
    type: Literal["hf.audio_to_audio"] = "hf.audio_to_audio"


class HFAudioClassification(HuggingFaceModel):
    type: Literal["hf.audio_classification"] = "hf.audio_classification"


class HFZeroShotAudioClassification(HuggingFaceModel):
    type: Literal["hf.zero_shot_audio_classification"] = "hf.zero_shot_audio_classification"


class HFRealESRGAN(HuggingFaceModel):
    type: Literal["hf.real_esrgan"] = "hf.real_esrgan"


class HFVoiceActivityDetection(HuggingFaceModel):
    type: Literal["hf.voice_activity_detection"] = "hf.voice_activity_detection"


class HFLoraSDConfig(BaseType):
    type: Literal["hf.lora_sd_config"] = "hf.lora_sd_config"
    lora: HFLoraSD = Field(default=HFLoraSD(), description="The LoRA model to use.")
    strength: float = Field(default=0.5, ge=0.0, le=3.0, description="LoRA strength")


class HFLoraSDXLConfig(BaseType):
    type: Literal["hf.lora_sdxl_config"] = "hf.lora_sdxl_config"
    lora: HFLoraSDXL = Field(default=HFLoraSDXL(), description="The LoRA model to use.")
    strength: float = Field(default=0.5, ge=0.0, le=3.0, description="LoRA strength")


CLASSNAME_TO_MODEL_TYPE = {
    "StableDiffusionPipeline": "hf.stable_diffusion",
    "StableDiffusionImg2ImgPipeline": "hf.stable_diffusion",
    "StableDiffusionInpaintPipeline": "hf.inpainting",
    "StableDiffusionXLPipeline": "hf.stable_diffusion_xl",
    "StableDiffusionXLImg2ImgPipeline": "hf.stable_diffusion_xl",
    "StableDiffusionXLInpaintPipeline": "hf.inpainting",
    "StableDiffusionXLRefinerPipeline": "hf.stable_diffusion_xl_refiner",
    "StableDiffusionXLControlNetPipeline": "hf.stable_diffusion_xl",
    "StableDiffusionUpscalePipeline": "hf.stable_diffusion_upscale",
    "StableDiffusion3Pipeline": "hf.stable_diffusion_3",
    "StableDiffusion3Img2ImgPipeline": "hf.stable_diffusion_3",
    "StableDiffusion3InpaintPipeline": "hf.inpainting",
    "PixArtAlphaPipeline": "hf.pixart_alpha",
    "FluxPipeline": "hf.flux",
    "FluxKontextPipeline": "hf.flux_kontext",
    "FluxDepthPipeline": "hf.flux_depth",
    "FluxReduxPipeline": "hf.flux_redux",
    "FluxFillPipeline": "hf.inpainting",
    "QwenImagePipeline": "hf.qwen_image",
    "QwenImageEditPlusPipeline": "hf.qwen_image_edit",
    "NunchakuQwenImageTransformer2DModel": "hf.qwen_image",
}


#######################
# ComfyUI Types
#######################
# Types for handling ComfyUI models


model_file_types = set()


class ModelFile(BaseType):
    name: str = ""

    def is_set(self) -> bool:
        return self.name != ""

    def is_empty(self) -> bool:
        return self.name == ""

    def __init_subclass__(cls):
        super().__init_subclass__()
        if hasattr(cls, "type"):
            model_file_types.add(cls.type)


class CheckpointFile(ModelFile):
    type: Literal["comfy.checkpoint_file"] = "comfy.checkpoint_file"


class UNetFile(ModelFile):
    type: Literal["comfy.unet_file"] = "comfy.unet_file"


class VAEFile(ModelFile):
    type: Literal["comfy.vae_file"] = "comfy.vae_file"


class CLIPFile(ModelFile):
    type: Literal["comfy.clip_file"] = "comfy.clip_file"


class unCLIPFile(ModelFile):
    type: Literal["comfy.unclip_file"] = "comfy.unclip_file"


class GLIGENFile(ModelFile):
    type: Literal["comfy.gligen_file"] = "comfy.gligen_file"


class CLIPVisionFile(ModelFile):
    type: Literal["comfy.clip_vision_file"] = "comfy.clip_vision_file"


class ControlNetFile(ModelFile):
    type: Literal["comfy.control_net_file"] = "comfy.control_net_file"


class IPAdapterFile(ModelFile):
    type: Literal["comfy.ip_adapter_file"] = "comfy.ip_adapter_file"


class LORAFile(ModelFile):
    type: Literal["comfy.lora_file"] = "comfy.lora_file"


class UpscaleModelFile(ModelFile):
    type: Literal["comfy.upscale_model_file"] = "comfy.upscale_model_file"


class InstantIDFile(ModelFile):
    type: Literal["comfy.instant_id_file"] = "comfy.instant_id_file"


class StyleModelFile(ModelFile):
    type: Literal["comfy.style_model_file"] = "comfy.style_model_file"


def comfy_model_to_folder(type_name: str) -> str:
    folder_mapping = {
        "comfy.checkpoint_file": "checkpoints",
        "comfy.vae_file": "vae",
        "comfy.clip": "clip",
        "comfy.clip_vision": "clip_vision",
        "comfy.control_net": "controlnet",
        "comfy.ip_adapter": "ipadapter",
        "comfy.gligen": "gligen",
        "comfy.upscale_model": "upscale_models",
        "comfy.lora": "loras",
        "comfy.unet": "unet",
        "comfy.instant_id_file": "instantid",
    }
    return folder_mapping.get(type_name, type_name)


comfy_model_types = set()


class ComfyModel(BaseType):
    name: str = ""
    model: Any = None

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if hasattr(cls, "type"):
            comfy_model_types.add(cls.type)


class CLIP(ComfyModel):
    type: Literal["comfy.clip"] = "comfy.clip"


class CLIPVision(ComfyModel):
    type: Literal["comfy.clip_vision"] = "comfy.clip_vision"


class GLIGEN(ComfyModel):
    type: Literal["comfy.gligen"] = "comfy.gligen"


class ControlNet(ComfyModel):
    type: Literal["comfy.control_net"] = "comfy.control_net"


class VAE(ComfyModel):
    type: Literal["comfy.vae"] = "comfy.vae"


class UNet(ComfyModel):
    type: Literal["comfy.unet"] = "comfy.unet"


class InstantID(ComfyModel):
    type: Literal["comfy.instant_id"] = "comfy.instant_id"


class UpscaleModel(ComfyModel):
    type: Literal["comfy.upscale_model"] = "comfy.upscale_model"


class LORA(ComfyModel):
    type: Literal["comfy.lora"] = "comfy.lora"


class IPAdapter(ComfyModel):
    type: Literal["comfy.ip_adapter"] = "comfy.ip_adapter"


class StyleModel(ComfyModel):
    type: Literal["comfy.style_model"] = "comfy.style_model"


comfy_data_types = set()


class ComfyData(BaseType):
    data: Any = None

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        if hasattr(cls, "type"):
            comfy_data_types.add(cls.type)

    def serialize(self):
        return None


class LoRAConfig(BaseType):
    type: Literal["comfy.lora_config"] = "comfy.lora_config"
    lora: LORAFile = Field(default=LORAFile(), description="The LoRA model to use.")
    strength: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA strength")


class Conditioning(ComfyData):
    type: Literal["comfy.conditioning"] = "comfy.conditioning"


class Noise(ComfyData):
    type: Literal["comfy.noise"] = "comfy.noise"


class CLIPVisionOutput(ComfyData):
    type: Literal["comfy.clip_vision_output"] = "comfy.clip_vision_output"


class Guider(ComfyData):
    type: Literal["comfy.guider"] = "comfy.guider"


class Latent(ComfyData):
    type: Literal["comfy.latent"] = "comfy.latent"


class ImageTensor(ComfyData):
    type: Literal["comfy.image_tensor"] = "comfy.image_tensor"


class Mask(ComfyData):
    type: Literal["comfy.mask"] = "comfy.mask"


class Sigmas(ComfyData):
    type: Literal["comfy.sigmas"] = "comfy.sigmas"


class Sampler(ComfyData):
    type: Literal["comfy.sampler"] = "comfy.sampler"


class Embeds(ComfyData):
    type: Literal["comfy.embeds"] = "comfy.embeds"


class FaceAnalysis(ComfyData):
    type: Literal["comfy.face_analysis"] = "comfy.face_analysis"


class FaceEmbeds(ComfyData):
    type: Literal["comfy.face_embeds"] = "comfy.face_embeds"


class REMBGSession(ComfyData):
    type: Literal["comfy.rembg_session"] = "comfy.rembg_session"


#######################
# Output and Data Types
#######################
# Types for handling various kinds of output data and results


class OutputType(BaseModel):
    """
    This is the base class for all strucutred output types when a node
    wants to return more than one output.
    """

    pass


class ToolName(BaseType):
    """
    A name for an LLM tool.
    """

    type: Literal["tool_name"] = "tool_name"
    name: str = Field(default="", description="The name of the tool")


class LogEntry(BaseType):
    """
    A log entry for a step.
    """

    type: Literal["log_entry"] = "log_entry"
    message: str = Field(default="", description="The message of the log entry")
    level: Literal["debug", "info", "warning", "error"] = Field(
        default="info", description="The level of the log entry"
    )
    timestamp: int = Field(default=0, description="The timestamp of the log entry")


class Step(BaseType):
    """A step item with completion status, dependencies, and tools."""

    type: Literal["step"] = "step"
    id: str = Field(
        default="",
        description="Unique identifier for the step",
    )

    instructions: str = Field(description="Instructions for the step to execute")
    logs: list[LogEntry] = Field(default=[], description="The logs of the step")
    completed: bool = Field(default=False, description="Whether the step is completed")
    start_time: int = Field(default=0, description="The start time of the step")
    end_time: int = Field(default=0, description="The end time of the step")
    depends_on: list[str] = Field(default=[], description="The IDs of steps this step depends on")
    tools: list[str] | None = Field(
        default=None,
        description="Optional list of allowed tool names for this step (None = no restriction).",
    )
    tool_name: str | None = Field(
        default=None,
        description="Optional deterministic tool name for tool-only steps.",
    )
    output_schema: str = Field(
        default="",
        description="The JSON schema of the output of the step",
    )

    def to_markdown(self) -> str:
        """Convert the step to markdown format."""
        checkbox = "[x]" if self.completed else "[*]" if self.is_running() else "[ ]"
        deps_str = f" (depends on {', '.join(self.depends_on)})" if self.depends_on else ""
        output_schema_str = f" (output schema: {self.output_schema})" if self.output_schema else ""
        return f"- {checkbox} {self.instructions}{deps_str}{output_schema_str}"

    def is_running(self) -> bool:
        """
        Check if the step is currently running.

        A step is considered running if:
        1. It has a non-zero start time (execution has begun)
        2. It has a zero end time (execution has not completed)
        3. It is not marked as completed

        Returns:
            bool: True if the step is currently running, False otherwise
        """
        return self.start_time > 0 and not self.completed


class Task(BaseType):
    """A task containing a title, description, and list of steps."""

    id: str = Field(
        default="",
        description="Unique identifier for the task",
    )

    type: Literal["task"] = "task"

    title: str = Field(default="", description="The title of the task")
    description: str = Field(default="", description="A description of the task, not used for execution")
    steps: list[Step] = Field(default=[], description="The steps of the task, a list of step IDs")

    def is_completed(self) -> bool:
        """Returns True if all steps are marked as completed."""
        return all(step.completed for step in self.steps)

    def to_markdown(self) -> str:
        """Converts task and steps to markdown format with headings and checkboxes."""
        lines = f"# Task: {self.title}\n"
        lines += f"Description: {self.description}\n"
        if self.description:
            lines += f"{self.description}\n"
        if self.steps:
            for step in self.steps:
                lines += f"{step.to_markdown()}\n"
        return lines


class TaskPlan(BaseType):
    """
    A plan for an agent to achieve a specific objective.
    The plan is a list of tasks that are executed in order.
    The tasks are a list of steps that are executed in order.
    Each task has a title, description, and list of steps.
    """

    type: Literal["task_plan"] = "task_plan"
    title: str = Field(default="", description="The title of the task list")
    tasks: list[Task] = Field(default=[], description="The tasks of the task list")

    def to_markdown(self) -> str:
        """Convert all tasks to a markdown string."""
        lines = f"# Task Plan - {self.title}\n"
        for task in self.tasks:
            lines += f"{task.to_markdown()}\n"
        return lines


class TorchTensor(BaseType):
    type: Literal["torch_tensor"] = "torch_tensor"
    value: Optional[bytes] = None  # raw bytes in row-major order
    dtype: str = "<i8"  # NumPy dtype string, includes endianness
    shape: tuple[int, ...] = (1,)  # logical shape (row-major)

    def is_set(self) -> bool:
        return self.value is not None

    def is_empty(self) -> bool:
        return self.value is None or len(self.value) == 0

    def _validate_nbytes(self) -> None:
        assert self.value is not None, "No bytes stored"
        itemsize = np.dtype(self.dtype).itemsize
        expected = int(np.prod(self.shape)) * itemsize
        actual = len(self.value)
        assert actual == expected, f"Byte length {actual} != expected {expected}"

    def to_tensor(self) -> Any:
        """
        Reconstruct as a CPU tensor and then (optionally) move to `self.device`.
        """
        import torch

        assert self.value is not None, "No bytes stored"
        self._validate_nbytes()

        # Interpret bytes with the recorded NumPy dtype (including endianness)
        arr = np.frombuffer(self.value, dtype=np.dtype(self.dtype))

        # Ensure native byte order for PyTorch memory sharing
        if arr.dtype.byteorder not in (
            "=",
            "|",
        ):  # not native (and not byte-order-less)
            arr = arr.newbyteorder("=")  # type: ignore

        # Reshape and wrap without extra copy if possible
        arr = arr.reshape(self.shape)
        return torch.from_numpy(arr)

    @staticmethod
    def from_tensor(tensor: Any, **kwargs) -> "TorchTensor":
        """
        Stores raw bytes + NumPy dtype string + shape (+ device).
        """
        import torch

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)

        # Always serialize from CPU to ensure contiguity and stable .numpy()
        cpu = tensor.detach().contiguous().to("cpu")
        np_arr = cpu.numpy()

        return TorchTensor(
            value=np_arr.tobytes(order="C"),
            dtype=np_arr.dtype.str,  # e.g. '<f4', '<i8'
            shape=tuple(np_arr.shape),
            **kwargs,
        )

    @staticmethod
    def from_numpy(arr: np.ndarray, **kwargs) -> "TorchTensor":
        import torch

        t = torch.from_numpy(arr)
        return TorchTensor.from_tensor(t, **kwargs)

    @staticmethod
    def from_list(arr: list, **kwargs) -> "TorchTensor":
        np_arr = np.array(arr)
        return TorchTensor.from_numpy(np_arr, **kwargs)


class NPArray(BaseType):
    type: Literal["np_array"] = "np_array"
    value: bytes | None = None
    dtype: str = "<i8"
    shape: tuple[int, ...] = (1,)

    def is_set(self) -> bool:
        return self.value is not None

    def is_empty(self):
        return self.value is None or len(self.value) == 0

    def to_numpy(self) -> np.ndarray:
        assert self.value is not None
        return np.frombuffer(self.value, dtype=np.dtype(self.dtype)).reshape(self.shape)

    def to_list(self) -> list:
        return self.to_numpy().tolist()

    @staticmethod
    def from_numpy(arr: np.ndarray, **kwargs):
        return NPArray(value=arr.tobytes(), dtype=arr.dtype.str, shape=arr.shape, **kwargs)

    @staticmethod
    def from_list(arr: list, **_kwargs):
        return NPArray.from_numpy(np.array(arr))


def to_numpy(num: float | int | NPArray) -> np.ndarray:
    if type(num) in (float, int, list):
        return np.array(num)
    elif type(num) is NPArray:
        return num.to_numpy()
    else:
        raise ValueError()


ColumnType = Literal["int"] | Literal["float"] | Literal["datetime"] | Literal["string"] | Literal["object"]


class ColumnDef(BaseModel):
    name: str
    data_type: ColumnType
    description: str = ""


def dtype_name(dtype: str):
    if dtype.startswith("int"):
        return "int"
    if dtype.startswith("float"):
        return "float"
    if dtype.startswith("datetime"):
        return "datetime"
    return "object"


class RecordType(BaseType):
    type: Literal["record_type"] = "record_type"
    columns: list[ColumnDef] = []


class DataframeRef(AssetRef):
    type: Literal["dataframe"] = "dataframe"
    columns: list[ColumnDef] | None = None
    data: list[list[Any]] | None = None

    @staticmethod
    def from_pandas(data: pd.DataFrame):
        rows = data.values.tolist()
        column_defs = [
            ColumnDef(name=name, data_type=dtype_name(dtype.name))
            for name, dtype in zip(data.columns, data.dtypes, strict=False)
        ]
        return DataframeRef(columns=column_defs, data=rows)


#######################
# ML Types
#######################
# Types for handling different kinds of models


class SKLearnModel(BaseType):
    type: Literal["sklearn_model"] = "sklearn_model"
    model: bytes | None = None


class StatsModelsModel(BaseType):
    type: Literal["statsmodels_model"] = "statsmodels_model"
    model: bytes | None = None


class ExcelRef(AssetRef):
    type: Literal["excel"] = "excel"


class RankingResult(BaseType):
    type: Literal["ranking_result"] = "ranking_result"
    score: float
    text: str


class ImageSegmentationResult(BaseType):
    type: Literal["image_segmentation_result"] = "image_segmentation_result"
    label: str
    mask: ImageRef


class BoundingBox(BaseType):
    type: Literal["bounding_box"] = "bounding_box"
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class ObjectDetectionResult(BaseType):
    type: Literal["object_detection_result"] = "object_detection_result"
    label: str
    score: float
    box: BoundingBox


class JSONRef(AssetRef):
    type: Literal["json"] = "json"
    data: str | None = None


class SVGRef(AssetRef):
    """A reference to an SVG asset."""

    type: Literal["svg"] = "svg"
    data: bytes | None = None


class OutputSlot(BaseModel):
    """
    An output slot is a slot that can be connected to an input slot.
    """

    type: TypeMetadata
    name: str
    stream: bool = False


class ToolCallEvent(BaseType):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any]


class ToolResultEvent(BaseType):
    type: Literal["tool_result"] = "tool_result"
    id: str
    result: Any
    error: str | None = None


#######################
# Chat Types
#######################
# Types for handling chat messages


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Any


class ToolCall(BaseModel):
    id: str = ""
    name: str = ""
    args: dict[str, Any] = {}
    result: Any = None
    step_id: str | None = None
    message: str | None = None


class MessageTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class MessageImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image: ImageRef = ImageRef()


class MessageAudioContent(BaseModel):
    type: Literal["audio"] = "audio"
    audio: AudioRef = AudioRef()


class MessageVideoContent(BaseModel):
    type: Literal["video"] = "video"
    video: VideoRef = VideoRef()


class MessageDocumentContent(BaseModel):
    type: Literal["document"] = "document"
    document: DocumentRef = DocumentRef()


MessageContent = (
    MessageTextContent | MessageImageContent | MessageAudioContent | MessageVideoContent | MessageDocumentContent
)


class Chunk(BaseType):
    """
    A message representing a chunk of streamed content from a provider.

    Used for streaming partial results in text generation, audio processing,
    or other operations where results are produced incrementally.
    """

    type: Literal["chunk"] = "chunk"
    node_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    content_type: Literal["text", "audio", "image", "video", "document"] = "text"
    content: str = ""
    content_metadata: dict[str, Any] = {}
    done: bool = False


class MessageFile(BaseModel):
    type: Literal["file"] = "file"
    content: bytes
    mime_type: str


class Message(BaseType):
    """
    Abstract representation for a chat message.
    Independent of the underlying chat system, such as OpenAI or Anthropic.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["message"] = "message"
    id: str | None = None
    """
    The unique identifier of the message.
    """

    workflow_id: str | None = None
    """
    The unique identifier of the workflow the message should be processed within.
    """

    graph: Graph | None = None
    """
    For unsaved workflows, the whole graph needs to be provided.
    """

    thread_id: str | None = None
    """
    The unique identifier of the thread the message belongs to.
    """

    tools: list[str] | None = None
    """
    The list of tools that the user has selected to use.
    """

    tool_call_id: str | None = None
    """
    The unique identifier of the tool call associated with the message.
    """

    role: str = ""
    """
    One of "user", "assistant", "system", or "tool".
    """

    name: str | None = None
    """
    The name of the tool that sent the message (aka tool result).
    """

    content: str | dict[str, Any] | list[MessageContent] | None = None
    """
    Text content or a list of message content objects, which can be text, images, or other types of content.
    """

    tool_calls: list[ToolCall] | None = None
    """
    The list of tool calls returned by the model.
    """

    collections: list[str] | None = None
    """
    The list of collections to query for this message.
    """

    input_files: list[MessageFile] | None = None
    """
    The list of input files for the message.
    """

    created_at: str | None = None
    """
    The timestamp when the message was created.
    It is represented as a string in ISO 8601 format.
    """

    provider: Provider | None = None
    """
    The provider that was used to generate the message.
    """

    model: str | None = None
    """
    The model that was used to generate the message.
    """

    agent_mode: bool | None = None
    """
    Whether to use agent mode for processing this message.
    """

    help_mode: bool | None = None
    """
    Whether to use help mode for processing this message.
    """

    agent_execution_id: str | None = None
    """
    Identifier for grouping agent execution trace messages.
    """

    execution_event_type: str | None = None
    """
    Type of agent execution event (planning_update, task_update, step_result, log_update).
    """

    workflow_target: str | None = None
    """
    Target routing for the message. If set to "workflow", the message will be sent directly
    to the workflow processor (bypassing normal chat routing). If null or "chat", uses normal routing.
    """

    @model_validator(mode="before")
    @classmethod
    def _coerce_instructions_to_content(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        content = data.get("content")
        instructions = data.get("instructions")

        if content is None and isinstance(instructions, str | list):
            data = dict(data)
            data["content"] = instructions
            return data

        if instructions is None and isinstance(content, str | list):
            data = dict(data)
            data["instructions"] = content
            return data

        return data

    def is_empty(self) -> bool:
        """Check if the message is empty (has no meaningful content).

        A message is considered empty if:
        - It has no role (None, empty string, or falsy)
        - AND it has no content (None, empty string, empty list, or falsy)

        This is used to detect default-initialized Message objects that
        should not be pushed as input values in workflows.

        Returns:
            bool: True if the message has no meaningful content, False otherwise.
        """
        # Check if content is empty (handles None, "", [], and other falsy values)
        content_empty = not self.content
        if isinstance(self.content, list):
            content_empty = len(self.content) == 0

        # Check if role is empty (handles None, "", and other falsy values)
        role_empty = not self.role

        # Message is empty if both role and content are empty
        return content_empty and role_empty

    def is_set(self) -> bool:
        """Check if the message has meaningful content.

        Returns:
            bool: True if the message has content, False otherwise.
        """
        return not self.is_empty()

    @staticmethod
    def from_model(message: Any):
        """
        Convert a Model object to a Message object.

        Args:
            message (Message): The Message object to convert.

        Returns:
            Message: The abstract Message object.
        """
        content = message.content
        if message.role == "agent_execution" and isinstance(content, str):
            try:
                content = json.loads(content)
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except Exception:
                        pass
            except Exception:
                pass
        execution_event_type = getattr(message, "execution_event_type", None)
        agent_execution_id = getattr(message, "agent_execution_id", None)
        if message.role == "agent_execution" and isinstance(content, dict):
            if not execution_event_type:
                execution_event_type = content.get("type")
            if not agent_execution_id:
                agent_execution_id = message.id

        return Message(
            id=message.id,
            thread_id=message.thread_id,
            tool_call_id=message.tool_call_id,
            role=message.role,
            name=message.name,
            content=content,
            tool_calls=message.tool_calls,
            created_at=message.created_at.isoformat() if message.created_at else None,
            provider=message.provider,
            model=message.model,
            agent_mode=message.agent_mode,
            help_mode=message.help_mode,
            agent_execution_id=agent_execution_id,
            execution_event_type=execution_event_type,
        )


#######################
# Result Types
#######################
# Types for handling results


class AudioChunk(BaseType):
    """Represents a chunk of audio with metadata about its source"""

    type: Literal["audio_chunk"] = "audio_chunk"
    timestamp: tuple[float, float] = (0, 0)
    text: str = ""


class TextChunk(BaseType):
    """Represents a chunk of text with metadata about its source"""

    type: Literal["text_chunk"] = "text_chunk"
    text: str = ""
    source_id: str = ""
    start_index: int = 0

    def get_document_id(self):
        return f"{self.source_id}:{self.start_index}"


class OCRResult(BaseType):
    type: Literal["ocr_result"] = "ocr_result"
    text: str
    score: float
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]


#######################
# Visualization Types
#######################
# Types for handling visualizations


class SVGElement(BaseType):
    """Base type for SVG elements that can be combined."""

    type: Literal["svg_element"] = "svg_element"
    name: str = ""
    attributes: dict[str, str] = {}
    content: str | None = None
    children: list["SVGElement"] = Field(default_factory=list)

    def render_attributes(self) -> str:
        return " ".join([f'{key}="{value}"' for key, value in self.attributes.items()])

    def __str__(self) -> str:
        children_content = "".join(str(child) for child in self.children)
        inner_content = f"{self.content}{children_content}"
        return f"<{self.name} {self.render_attributes()}>{inner_content}</{self.name}>"


class SeabornPlotType(str, Enum):
    # Relational plots
    SCATTER = "scatter"
    LINE = "line"
    RELPLOT = "relplot"

    # Distribution plots
    HISTPLOT = "histplot"
    KDEPLOT = "kdeplot"
    ECDFPLOT = "ecdfplot"
    RUGPLOT = "rugplot"
    DISTPLOT = "distplot"

    # Categorical plots
    STRIPPLOT = "stripplot"
    SWARMPLOT = "swarmplot"
    BOXPLOT = "boxplot"
    VIOLINPLOT = "violinplot"
    BOXENPLOT = "boxenplot"
    POINTPLOT = "pointplot"
    BARPLOT = "barplot"
    COUNTPLOT = "countplot"

    # Regression plots
    REGPLOT = "regplot"
    LMPLOT = "lmplot"
    RESIDPLOT = "residplot"

    # Matrix plots
    HEATMAP = "heatmap"
    CLUSTERMAP = "clustermap"

    # Joint plots
    JOINTPLOT = "jointplot"

    # Pair plots
    PAIRPLOT = "pairplot"

    # Facet plots
    FACETGRID = "facetgrid"


class SeabornEstimator(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    VAR = "var"
    STD = "std"


class SeabornStatistic(str, Enum):
    COUNT = "count"
    FREQUENCY = "frequency"
    PROBABILITY = "probability"
    PERCENT = "percent"
    DENSITY = "density"


class DataSeries(BaseType):
    type: Literal["data_series"] = "data_series"
    name: str = ""
    x: str = ""
    y: str | None = None  # Optional for some plot types
    hue: str | None = None  # For color encoding
    size: str | None = None  # For size encoding
    style: str | None = None  # For style encoding
    weight: str | None = None  # For weighted plots
    color: str | None = None
    plot_type: SeabornPlotType = Field(default=SeabornPlotType.LINE)

    # Common plot parameters
    estimator: SeabornEstimator | None = None
    ci: float | None = None
    n_boot: int = 1000
    units: str | None = None
    seed: int | None = None

    # Distribution plot parameters
    stat: SeabornStatistic | None = None
    bins: int | str | None = None
    binwidth: float | None = None
    binrange: tuple[float, float] | None = None
    discrete: bool | None = None

    # Appearance parameters
    line_style: str = Field(default="solid")
    marker: str = Field(default=".")
    alpha: float = Field(default=1.0)
    orient: Literal["v", "h"] | None = None


class PlotlySeries(BaseType):
    """
    Configuration for a single Plotly Express data series.
    """

    type: Literal["plotly_series"] = "plotly_series"

    name: str = Field(description="Name of the data series")
    x: str = Field(description="Column name for x-axis")
    y: str | None = Field(
        default=None,
        description="Column name for y-axis (optional for some charts like histogram)",
    )
    color: str | None = Field(default=None, description="Column name for color encoding")
    size: str | None = Field(default=None, description="Column name for size encoding")
    symbol: str | None = Field(default=None, description="Column name for symbol encoding")
    line_dash: str | None = Field(default=None, description="Column name for line dash pattern encoding")
    chart_type: str = Field(description="The type of chart to create (scatter, line, bar, histogram, box, violin)")


class PlotlyConfig(BaseType):
    """
    Configuration for Plotly Express charts.
    Captures essential visualization parameters while maintaining simplicity.
    """

    type: Literal["plotly_config"] = "plotly_config"

    config: dict[str, Any] = {}


class ChartData(BaseType):
    type: Literal["chart_data"] = "chart_data"
    series: list[DataSeries] = []

    # Additional data-wide parameters
    row: str | None = None  # For FacetGrid
    col: str | None = None  # For FacetGrid
    col_wrap: int | None = None  # For FacetGrid


class ChartConfig(BaseType):
    type: Literal["chart_config"] = "chart_config"
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    legend: bool = True
    data: ChartData = Field(default=ChartData())

    # Figure parameters
    height: float | None = None
    aspect: float | None = None

    # Axis parameters
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    x_scale: Literal["linear", "log"] | None = None
    y_scale: Literal["linear", "log"] | None = None

    # Legend parameters
    legend_position: Literal["auto", "right", "left", "top", "bottom"] = "auto"

    # Additional styling
    palette: str | None = None
    hue_order: list[str] | None = None
    hue_norm: tuple[float, float] | None = None
    sizes: tuple[float, float] | None = None
    size_order: list[str] | None = None
    size_norm: tuple[float, float] | None = None

    # Joint plot specific
    marginal_kws: dict | None = None
    joint_kws: dict | None = None

    # Pair plot specific
    diag_kind: Literal["auto", "hist", "kde"] | None = None
    corner: bool = False

    # Matrix plot specific
    center: float | None = None
    vmin: float | None = None
    vmax: float | None = None
    cmap: str | None = None
    annot: bool = False
    fmt: str = ".2g"
    square: bool = False


class ColorRef(BaseType):
    """A reference to a color value."""

    type: Literal["color"] = "color"
    value: str | None = None

    def __str__(self) -> str:
        return self.value or ""


class DataSeriesSchema(BaseModel):
    name: str
    x: str
    y: Optional[str] = None
    hue: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    weight: Optional[str] = None
    color: Optional[str] = None
    plot_type: SeabornPlotType
    estimator: Optional[SeabornEstimator] = None
    ci: Optional[float] = None
    n_boot: int = 1000
    units: Optional[str] = None
    seed: Optional[int] = None
    stat: Optional[SeabornStatistic] = None
    bins: Optional[int] = None
    binwidth: Optional[float] = None
    binrange: Optional[tuple[float, float]] = None
    discrete: Optional[bool] = None
    line_style: str = "solid"
    marker: str = "."
    alpha: float = 1.0
    orient: Optional[Literal["v", "h"]] = None


class ChartDataSchema(BaseModel):
    series: list[DataSeriesSchema] = []
    row: Optional[str] = None
    col: Optional[str] = None
    col_wrap: Optional[int] = None


class ChartConfigSchema(BaseModel):
    title: str
    x_label: str
    y_label: str
    legend: bool = True
    data: ChartDataSchema = Field(default=ChartDataSchema())
    height: Optional[float] = None
    aspect: Optional[float] = None
    x_lim: Optional[tuple[float, float]] = None
    y_lim: Optional[tuple[float, float]] = None
    x_scale: Optional[Literal["linear", "log"]] = None
    y_scale: Optional[Literal["linear", "log"]] = None
    legend_position: Optional[Literal["auto", "right", "left", "top", "bottom"]] = None
    palette: Optional[str] = None
    hue_order: Optional[list[str]] = None
    hue_norm: Optional[tuple[float, float]] = None
    sizes: Optional[tuple[float, float]] = None
    size_order: Optional[list[str]] = None
    size_norm: Optional[tuple[float, float]] = None
    marginal_kws: Optional[dict] = None
    joint_kws: Optional[dict] = None
    diag_kind: Optional[Literal["auto", "hist", "kde"]] = None
    corner: bool = False
    center: Optional[float] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cmap: Optional[str] = None
    annot: bool = False
    fmt: str = ".2g"
    square: bool = False


#######################
# Email Types
#######################
# Types for handling email data


class Email(BaseType):
    type: Literal["email"] = "email"
    id: str = Field(default="", description="Message ID")
    sender: str = Field(default="", description="Sender email address")
    subject: str = Field(default="", description="Email subject line")
    date: Datetime = Field(default=Datetime(), description="Email date")
    body: str | TextRef = Field(default="", description="Email body content")


class EmailFlag(str, Enum):
    SEEN = "SEEN"
    UNSEEN = "UNSEEN"
    ANSWERED = "ANSWERED"
    UNANSWERED = "UNANSWERED"
    FLAGGED = "FLAGGED"
    UNFLAGGED = "UNFLAGGED"


class DateCriteria(str, Enum):
    BEFORE = "BEFORE"
    SINCE = "SINCE"
    ON = "ON"


class DateSearchCondition(BaseType):
    type: Literal["date_search_condition"] = "date_search_condition"
    criteria: DateCriteria
    date: Datetime


class EmailSearchCriteria(BaseType):
    type: Literal["email_search_criteria"] = "email_search_criteria"
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    cc: Optional[str] = None
    bcc: Optional[str] = None
    date_condition: Optional[DateSearchCondition] = None
    flags: list[EmailFlag] = []
    keywords: list[str] = []
    folder: Optional[str] = None
    text: Optional[str] = None


class IMAPConnection(BaseType):
    """Configuration for an IMAP email connection."""

    type: Literal["imap_connection"] = "imap_connection"
    host: str = ""
    port: int = 993
    username: str = ""
    password: str = ""
    use_ssl: bool = True

    def is_configured(self) -> bool:
        """Check if the connection has all required fields set."""
        return bool(self.host and self.username and self.password)


class LoraWeight(BaseType):
    """A weight for a LoRA model."""

    type: Literal["lora_weight"] = "lora_weight"
    url: str = ""
    scale: float = 1.0


class CalendarEvent(BaseType):
    """Represents a calendar event with its properties."""

    type: Literal["calendar_event"] = "calendar_event"
    title: str = ""
    start_date: Datetime = Datetime()
    end_date: Datetime = Datetime()
    calendar: str = ""
    location: str = ""
    notes: str = ""


class Source(BaseType):
    type: Literal["source"] = "source"
    title: str = ""
    url: str = ""


#######################
# Search Result Types
#######################
# Types for handling search engine results (SERP)


class OrganicResult(BaseType):
    """Search engine organic result"""

    type: Literal["organic_result"] = "organic_result"
    position: int
    title: str
    link: str
    redirect_link: Optional[str] = None
    displayed_link: str
    date: Optional[str] = None
    snippet: str
    snippet_highlighted_words: Optional[list[str]] = None
    thumbnail: Optional[str] = None


class NewsResult(BaseType):
    """News search result"""

    type: Literal["news_result"] = "news_result"
    position: int
    title: str | None = None
    link: str
    thumbnail: str | None = None
    date: str


class ImageResult(BaseType):
    """Image search result"""

    type: Literal["image_result"] = "image_result"
    position: int
    thumbnail: str
    original: str
    original_width: int
    original_height: int
    is_product: bool
    source: str
    title: str
    link: str


class JobResult(BaseType):
    """Job listing search result"""

    type: Literal["job_result"] = "job_result"
    title: str | None = None
    company_name: str | None = None
    location: str | None = None
    via: str | None = None
    share_link: str | None = None
    thumbnail: str | None = None
    extensions: list[str] | None = None


class VisualMatchResult(BaseType):
    """Visual/image match result"""

    type: Literal["visual_match_result"] = "visual_match_result"
    position: int
    title: str | None = None
    link: str | None = None
    thumbnail: str | None = None
    thumbnail_width: int | None = None
    thumbnail_height: int | None = None
    image: str | None = None
    image_width: int | None = None
    image_height: int | None = None


class LocalResult(BaseType):
    """Local/maps search result"""

    type: Literal["local_result"] = "local_result"
    position: int
    title: str | None = None
    place_id: str | None = None
    data_id: str | None = None
    data_cid: str | None = None
    reviews_link: str | None = None
    photos_link: str | None = None
    gps_coordinates: dict[str, float] | None = None
    place_id_search: str | None = None
    provider_id: str | None = None
    rating: float | None = None
    reviews: int | None = None
    price: str | None = None
    types: list[str] | None = None
    address: str | None = None
    open_state: str | None = None
    hours: str | None = None
    operating_hours: dict[str, str] | None = None
    phone: str | None = None
    website: str | None = None
    description: str | None = None
    thumbnail: str | None = None


class ShoppingResult(BaseType):
    """Shopping/product search result"""

    type: Literal["shopping_result"] = "shopping_result"
    position: int
    title: str | None = None
    link: str | None = None
    product_link: str | None = None
    product_id: str | None = None
    source: str | None = None
    source_icon: str | None = None
    extensions: list[str] | None = None
    badge: str | None = None
    thumbnail: str | None = None
    tag: str | None = None
    delivery: str | None = None
    price: str | None = None
    extracted_price: float | None = None
    old_price: str | None = None
    extracted_old_price: float | None = None
    rating: float | None = None
    reviews: int | None = None
    store_rating: float | None = None
    store_reviews: int | None = None
