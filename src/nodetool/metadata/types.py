from datetime import date, datetime, timedelta, timezone
from enum import Enum
import enum
from types import NoneType
from nodetool.types.workflow import Workflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional, Type, Union
import base64

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.types.graph import Graph


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

    type: str

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
            raise ValueError(
                f"Unknown type name: {type_name}. Types must derive from BaseType"
            )
        return NameToType[type_name](**data)


class Collection(BaseType):
    type: Literal["collection"] = "collection"
    name: str = ""


class Event(BaseType):
    """
    An event is a special object in Nodetool.
    It can be dispatched by a node async.
    Nodes can received events async.
    """

    type: Literal["event"] = "event"
    name: str = ""
    payload: dict[str, Any] = {}


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
            tzinfo=(
                timezone(timedelta(seconds=self.utc_offset), self.tzinfo)
                if self.utc_offset
                else timezone.utc
            ),
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
    type: str = "asset"
    uri: str = ""
    asset_id: str | None = None
    data: Any = None

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
            return self.asset_id
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
    Anthropic = "anthropic"
    Replicate = "replicate"
    HuggingFace = "huggingface"
    HuggingFaceGroq = "huggingface_groq"
    HuggingFaceCerebras = "huggingface_cerebras"
    Ollama = "ollama"
    Comfy = "comfy"
    Local = "local"
    Gemini = "gemini"
    Empty = "empty"


class InferenceProvider(str, Enum):
    none = ""
    black_forest_labs = "black-forest-labs"
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
    together = "together"
    

class InferenceProviderAudioClassificationModel(BaseType):
    type: Literal["inference_provider_audio_classification_model"] = "inference_provider_audio_classification_model"
    provider: InferenceProvider = InferenceProvider.hf_inference
    model_id: str = ""

class InferenceProviderAutomaticSpeechRecognitionModel(BaseType):
    type: Literal["inference_provider_automatic_speech_recognition_model"] = "inference_provider_automatic_speech_recognition_model"
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
    type: str = "hf.model"
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


class HFCheckpointModel(HuggingFaceModel):
    type: Literal["hf.checkpoint_model"] = "hf.checkpoint_model"


class HFStableDiffusion(HFCheckpointModel):
    type: Literal["hf.stable_diffusion"] = "hf.stable_diffusion"


class HFStableDiffusionXL(HFCheckpointModel):
    type: Literal["hf.stable_diffusion_xl"] = "hf.stable_diffusion_xl"


class HFStableDiffusion3(HFCheckpointModel):
    type: Literal["hf.stable_diffusion_3"] = "hf.stable_diffusion_3"


class HFFlux(HFCheckpointModel):
    type: Literal["hf.flux"] = "hf.flux"


class HFLTXV(HFCheckpointModel):
    type: Literal["hf.ltxv"] = "hf.ltxv"


class HFControlNet(HuggingFaceModel):
    type: Literal["hf.controlnet"] = "hf.controlnet"


class HFControlNetSDXL(HuggingFaceModel):
    type: Literal["hf.controlnet_sdxl"] = "hf.controlnet_sdxl"


class HFIPAdapter(HuggingFaceModel):
    type: Literal["hf.ip_adapter"] = "hf.ip_adapter"


class HFStyleModel(HuggingFaceModel):
    type: Literal["hf.style_model"] = "hf.style_model"


class HFLoraSD(HuggingFaceModel):
    type: Literal["hf.lora_sd"] = "hf.lora_sd"


class HFLoraSDXL(HuggingFaceModel):
    type: Literal["hf.lora_sdxl"] = "hf.lora_sdxl"


class HFStableDiffusionXLTurbo(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_xl_turbo"] = "hf.stable_diffusion_xl_turbo"


class HFStableDiffusionUpscale(HuggingFaceModel):
    type: Literal["hf.stable_diffusion_upscale"] = "hf.stable_diffusion_upscale"


class HFImageToText(HuggingFaceModel):
    type: Literal["hf.image_to_text"] = "hf.image_to_text"


class HFImageToImage(HuggingFaceModel):
    type: Literal["hf.image_to_image"] = "hf.image_to_image"


class HFImageToVideo(HuggingFaceModel):
    type: Literal["hf.image_to_video"] = "hf.image_to_video"


class HFUnconditionalImageGeneration(HuggingFaceModel):
    type: Literal["hf.unconditional_image_generation"] = (
        "hf.unconditional_image_generation"
    )


class HFUnet(HuggingFaceModel):
    type: Literal["hf.unet"] = "hf.unet"


class HFVAE(HuggingFaceModel):
    type: Literal["hf.vae"] = "hf.vae"


class HFCLIP(HuggingFaceModel):
    type: Literal["hf.clip"] = "hf.clip"


class HFCLIPVision(HuggingFaceModel):
    type: Literal["hf.clip_vision"] = "hf.clip_vision"


class HFVideoClassification(HuggingFaceModel):
    type: Literal["hf.video_classification"] = "hf.video_classification"


class HFTextToVideo(HuggingFaceModel):
    type: Literal["hf.text_to_video"] = "hf.text_to_video"


class HFZeroShotImageClassification(HuggingFaceModel):
    type: Literal["hf.zero_shot_image_classification"] = (
        "hf.zero_shot_image_classification"
    )


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
    type: Literal["hf.zero_shot_audio_classification"] = (
        "hf.zero_shot_audio_classification"
    )


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
    "StableDiffusionXLPipeline": "hf.stable_diffusion_xl",
    "StableDiffusionXLControlNetPipeline": "hf.stable_diffusion_xl",
    "StableDiffusionUpscalePipeline": "hf.stable_diffusion_upscale",
    "PixArtAlphaPipeline": "hf.pixart_alpha",
}


PIPELINE_TAGS = {
    "hf.audio_classification": ["audio-classification"],
    "hf.audio_to_audio": ["audio-to-audio"],
    "hf.automatic_speech_recognition": ["automatic-speech-recognition"],
    "hf.computer_vision": ["computer-vision"],
    "hf.depth_estimation": ["depth-estimation"],
    "hf.document_question_answering": ["document-question-answering"],
    "hf.feature_extraction": ["feature-extraction"],
    "hf.fill_mask": ["fill-mask"],
    "hf.image_classification": ["image-classification"],
    "hf.image_feature_extraction": ["image-feature-extraction"],
    "hf.image_segmentation": ["image-segmentation"],
    "hf.image_text_to_text": ["image-text-to-text"],
    "hf.image_to_3d": ["image-to-3d"],
    "hf.image_to_image": ["image-to-image"],
    "hf.image_to_video": ["image-to-video"],
    "hf.mask_generation": ["mask-generation"],
    "hf.natural_language_processing": ["natural-language-processing"],
    "hf.object_detection": ["object-detection"],
    "hf.question_answering": ["question-answering"],
    "hf.sentence_similarity": ["sentence-similarity"],
    "hf.stable_diffusion_xl": ["stable-diffusion-xl"],
    "hf.stable_diffusion": ["stable-diffusion"],
    "hf.summarization": ["summarization"],
    "hf.table_question_answering": ["table-question-answering"],
    "hf.text_classification": ["text-classification"],
    "hf.text_generation": ["text-generation"],
    "hf.text_to_3d": ["text-to-3d"],
    "hf.text_to_audio": ["text-to-audio"],
    "hf.text_to_image": ["text-to-image"],
    "hf.text_to_speech": ["text-to-speech"],
    "hf.text_to_video": ["text-to-video"],
    "hf.text2text_generation": ["text2text-generation"],
    "hf.token_classification": ["token-classification"],
    "hf.translation": ["translation"],
    "hf.unconditional_image_generation": ["unconditional-image-generation"],
    "hf.video_classification": ["video-classification"],
    "hf.video_text_to_text": ["video-text-to-text"],
    "hf.visual_question_answering": ["visual-question-answering"],
    "hf.voice_activity_detection": ["voice-activity-detection"],
    "hf.zero_shot_audio_classification": ["zero-shot-audio-classification"],
    "hf.zero_shot_classification": ["zero-shot-classification"],
    "hf.zero_shot_image_classification": ["zero-shot-image-classification"],
    "hf.zero_shot_object_detection": ["zero-shot-object-detection"],
}


def pipeline_tag_to_model_type(tag: str) -> str | None:
    for model_type, tags in PIPELINE_TAGS.items():
        if tag in tags:
            return model_type
    return None


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


class ChatConversation(OutputType):
    """
    The result of a chat conversation.
    """

    messages: list[str] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    response: str = Field(default="", description="The response from the chat system")


class ToolName(BaseType):
    """
    A name for an LLM tool.
    """

    type: Literal["tool_name"] = "tool_name"
    name: str = Field(default="", description="The name of the tool")


class LogEntry(BaseType):
    """
    A log entry for a subtask.
    """

    type: Literal["log_entry"] = "log_entry"
    message: str = Field(default="", description="The message of the log entry")
    level: Literal["debug", "info", "warning", "error"] = Field(
        default="info", description="The level of the log entry"
    )
    timestamp: int = Field(default=0, description="The timestamp of the log entry")


class SubTask(BaseType):
    """A subtask item with completion status, dependencies, and tools."""

    type: Literal["subtask"] = "subtask"
    id: str = Field(
        default="",
        description="Unique identifier for the subtask",
    )

    model: str | None = Field(
        default=None,
        description="The model to use for the subtask",
    )
    content: str = Field(description="Instructions for the subtask")
    logs: list[LogEntry] = Field(default=[], description="The logs of the subtask")
    max_iterations: int = Field(
        default=10,
        description="The maximum number of iterations for the subtask",
    )
    max_tool_calls: int = Field(
        default=10,
        description="The maximum number of tool calls for the subtask",
    )
    completed: bool = Field(
        default=False, description="Whether the subtask is completed"
    )
    start_time: int = Field(default=0, description="The start time of the subtask")
    end_time: int = Field(default=0, description="The end time of the subtask")
    input_tasks: list[str] = Field(
        default=[], description="The input tasks for the subtask"
    )
    input_files: list[str] = Field(
        default=[], description="The input files required for the subtask"
    )
    output_file: str = Field(
        default="", description="The output file produced by the subtask"
    )
    output_schema: str = Field(
        default="",
        description="The JSON schema of the output of the subtask",
    )
    is_intermediate_result: bool = Field(
        default=False,
        description="Whether the subtask is an intermediate result of a task",
    )

    def to_markdown(self) -> str:
        """Convert the subtask to markdown format."""
        checkbox = "[x]" if self.completed else "[*]" if self.is_running() else "[ ]"
        deps_str = (
            f" (depends on {', '.join(self.input_tasks)})" if self.input_tasks else ""
        )
        output_schema_str = (
            f" (output schema: {self.output_schema})" if self.output_schema else ""
        )
        return f"- {checkbox} {self.content}{deps_str}{output_schema_str}"

    def is_running(self) -> bool:
        """
        Check if the subtask is currently running.

        A subtask is considered running if:
        1. It has a non-zero start time (execution has begun)
        2. It has a zero end time (execution has not completed)
        3. It is not marked as completed

        Returns:
            bool: True if the subtask is currently running, False otherwise
        """
        return self.start_time > 0 and not self.completed


class Task(BaseType):
    """A task containing a title, description, and list of subtasks."""

    id: str = Field(
        default="",
        description="Unique identifier for the task",
    )

    type: Literal["task"] = "task"

    title: str = Field(default="", description="The title of the task")
    description: str = Field(
        default="", description="A description of the task, not used for execution"
    )
    subtasks: list[SubTask] = Field(
        default=[], description="The subtasks of the task, a list of subtask IDs"
    )

    def is_completed(self) -> bool:
        """Returns True if all subtasks are marked as completed."""
        return all(subtask.completed for subtask in self.subtasks)

    def to_markdown(self) -> str:
        """Converts task and subtasks to markdown format with headings and checkboxes."""
        lines = f"# Task: {self.title}\n"
        if self.description:
            lines += f"{self.description}\n"
        if self.subtasks:
            for subtask in self.subtasks:
                lines += f"{subtask.to_markdown()}\n"
        return lines


class TaskPlan(BaseType):
    """
    A plan for an agent to achieve a specific objective.
    The plan is a list of tasks that are executed in order.
    The tasks are a list of subtasks that are executed in order.
    Each task has a title, description, and list of subtasks.
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
        return NPArray(
            value=arr.tobytes(), dtype=arr.dtype.str, shape=arr.shape, **kwargs
        )

    @staticmethod
    def from_list(arr: list, **kwargs):
        return NPArray.from_numpy(np.array(arr))


def to_numpy(num: float | int | NPArray) -> np.ndarray:
    if type(num) in (float, int, list):
        return np.array(num)
    elif type(num) is NPArray:
        return num.to_numpy()
    else:
        raise ValueError()


ColumnType = Union[
    Literal["int"],
    Literal["float"],
    Literal["datetime"],
    Literal["string"],
    Literal["object"],
]


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
            for name, dtype in zip(data.columns, data.dtypes)
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


class Dataset(OutputType):
    """
    This class represents a dataset, which includes a dataframe of features and a dataframe of targets.
    """

    data: DataframeRef = DataframeRef()
    target: DataframeRef = DataframeRef()


class JSONRef(AssetRef):
    type: Literal["json"] = "json"
    data: str | None = None


class SVGRef(AssetRef):
    """A reference to an SVG asset."""

    type: Literal["svg"] = "svg"
    data: bytes | None = None


def is_output_type(type):
    try:
        return issubclass(type, OutputType)
    except Exception:
        return False


class OutputSlot(BaseModel):
    """
    An output slot is a slot that can be connected to an input slot.
    """

    type: TypeMetadata
    name: str
    stream: bool = False


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
    subtask_id: str | None = None
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
    MessageTextContent
    | MessageImageContent
    | MessageAudioContent
    | MessageVideoContent
    | MessageDocumentContent
)


class MessageFile(BaseModel):
    type: Literal["file"] = "file"
    content: bytes
    mime_type: str


class Message(BaseType):
    """
    Abstract representation for a chat message.
    Independent of the underlying chat system, such as OpenAI or Anthropic.
    """

    type: str = "message"
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

    content: str | list[MessageContent] | None = None
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

    output_files: list[MessageFile] | None = None
    """
    The list of output files for the message.
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

    workflow_assistant: bool | None = None
    """
    Whether to use workflow assistant mode for processing this message.
    """

    help_mode: bool | None = None
    """
    Whether to use help mode for processing this message.
    """

    @staticmethod
    def from_model(message: Any):
        """
        Convert a Model object to a Message object.

        Args:
            message (Message): The Message object to convert.

        Returns:
            Message: The abstract Message object.
        """
        return Message(
            id=message.id,
            thread_id=message.thread_id,
            tool_call_id=message.tool_call_id,
            role=message.role,
            name=message.name,
            content=message.content,
            tool_calls=message.tool_calls,
            created_at=message.created_at.isoformat(),
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
    color: str | None = Field(
        default=None, description="Column name for color encoding"
    )
    size: str | None = Field(default=None, description="Column name for size encoding")
    symbol: str | None = Field(
        default=None, description="Column name for symbol encoding"
    )
    line_dash: str | None = Field(
        default=None, description="Column name for line dash pattern encoding"
    )
    chart_type: str = Field(
        description="The type of chart to create (scatter, line, bar, histogram, box, violin)"
    )


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


class RSSEntry(BaseType):
    """Represents an RSS entry."""

    type: Literal["rss_entry"] = "rss_entry"
    title: str = ""
    link: str = ""
    published: Datetime = Datetime()
    summary: str = ""
    author: str = ""


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
