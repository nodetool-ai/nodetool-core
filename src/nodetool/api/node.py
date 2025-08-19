#!/usr/bin/env python

from fastapi import APIRouter, HTTPException
import httpx
from nodetool.common.system_stats import SystemStats
from nodetool.types.job import (
    JobUpdate,
)
from nodetool.types.prediction import Prediction
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.metadata.types import (
    AssetRef,
    DocumentRef,
    HuggingFaceModel,
    InferenceProvider,
    LanguageModel,
    NodeRef,
    Email,
    FilePath,
    PlotlyConfig,
    SVGElement,
    TaskPlan,
    WorkflowRef,
    AudioRef,
    DataframeRef,
    FolderRef,
    FontRef,
    ImageRef,
    ModelRef,
    NPArray,
    TextRef,
    VideoRef,
    HFImageTextToText,
    HFVisualQuestionAnswering,
    HFDocumentQuestionAnswering,
    HFVideoTextToText,
    HFComputerVision,
    HFDepthEstimation,
    HFImageClassification,
    HFObjectDetection,
    HFImageSegmentation,
    HFTextToImage,
    HFStableDiffusion,
    HFStableDiffusionXL,
    HFImageToText,
    HFImageToImage,
    HFImageToVideo,
    HFUnconditionalImageGeneration,
    HFVideoClassification,
    HFTextToVideo,
    HFZeroShotImageClassification,
    HFMaskGeneration,
    HFZeroShotObjectDetection,
    HFTextTo3D,
    HFImageTo3D,
    HFImageFeatureExtraction,
    HFNaturalLanguageProcessing,
    HFTextClassification,
    HFTokenClassification,
    HFTableQuestionAnswering,
    HFQuestionAnswering,
    HFZeroShotClassification,
    HFTranslation,
    HFSummarization,
    HFFeatureExtraction,
    HFTextGeneration,
    HFText2TextGeneration,
    HFFillMask,
    HFSentenceSimilarity,
    HFTextToSpeech,
    HFTextToAudio,
    HFAutomaticSpeechRecognition,
    HFAudioToAudio,
    HFAudioClassification,
    HFZeroShotAudioClassification,
    HFVoiceActivityDetection,
    InferenceProviderAutomaticSpeechRecognitionModel,
    InferenceProviderAudioClassificationModel,
    InferenceProviderImageClassificationModel,
    InferenceProviderTextClassificationModel,
    InferenceProviderSummarizationModel,
    InferenceProviderTextToImageModel,
    InferenceProviderTranslationModel,
    InferenceProviderTextToTextModel,
    InferenceProviderTextToSpeechModel,
    InferenceProviderTextToAudioModel,
    InferenceProviderTextGenerationModel,
    InferenceProviderImageToImageModel,
    InferenceProviderImageSegmentationModel,
)
from nodetool.workflows.base_node import get_node_class
from nodetool.common.environment import Environment
from nodetool.workflows.types import ProcessingMessage
from nodetool.packages.registry import Registry
import asyncio

log = Environment.get_logger()
router = APIRouter(prefix="/api/nodes", tags=["nodes"])


# This is a dummy type that contains all property types and Websocket types.
UnionType = (
    AssetRef
    | AudioRef
    | DataframeRef
    | Email
    | FilePath
    | FolderRef
    | ImageRef
    | NPArray
    | VideoRef
    | ModelRef
    | DocumentRef
    | FontRef
    | TextRef
    | WorkflowRef
    | NodeRef
    | Prediction
    | JobUpdate
    | LanguageModel
    | HuggingFaceModel
    | HFImageTextToText
    | HFVisualQuestionAnswering
    | HFDocumentQuestionAnswering
    | HFVideoTextToText
    | HFComputerVision
    | HFDepthEstimation
    | HFImageClassification
    | HFObjectDetection
    | HFImageSegmentation
    | HFTextToImage
    | HFStableDiffusion
    | HFStableDiffusionXL
    | HFImageToText
    | HFImageToImage
    | HFImageToVideo
    | HFUnconditionalImageGeneration
    | HFVideoClassification
    | HFTextToVideo
    | HFZeroShotImageClassification
    | HFMaskGeneration
    | HFZeroShotObjectDetection
    | HFTextTo3D
    | HFImageTo3D
    | HFImageFeatureExtraction
    | HFNaturalLanguageProcessing
    | HFTextClassification
    | HFTokenClassification
    | HFTableQuestionAnswering
    | HFQuestionAnswering
    | HFZeroShotClassification
    | HFTranslation
    | HFSummarization
    | HFFeatureExtraction
    | HFTextGeneration
    | HFText2TextGeneration
    | HFFillMask
    | HFSentenceSimilarity
    | HFTextToSpeech
    | HFTextToAudio
    | HFAutomaticSpeechRecognition
    | HFAudioToAudio
    | HFAudioClassification
    | HFZeroShotAudioClassification
    | HFVoiceActivityDetection
    | SVGElement
    | SystemStats
    | TaskPlan
    | PlotlyConfig
    | dict
    | InferenceProvider
    | InferenceProviderAutomaticSpeechRecognitionModel
    | InferenceProviderAudioClassificationModel
    | InferenceProviderImageClassificationModel
    | InferenceProviderTextClassificationModel
    | InferenceProviderSummarizationModel
    | InferenceProviderTextToImageModel
    | InferenceProviderTranslationModel
    | InferenceProviderTextToTextModel
    | InferenceProviderTextToSpeechModel
    | InferenceProviderTextToAudioModel
    | InferenceProviderTextGenerationModel
    | InferenceProviderImageToImageModel
    | InferenceProviderImageSegmentationModel
    | ProcessingMessage
)


@router.get("/dummy")
async def dummy() -> UnionType:
    """
    Returns a dummy node.
    """
    return {"hello": "world"}


@router.get("/metadata")
async def metadata() -> list[NodeMetadata]:
    """
    Returns a list of installed nodes.
    """
    registry = Registry()
    installed_packages = await asyncio.to_thread(registry.list_installed_packages)
    nodes = []
    for package in installed_packages:
        if package.nodes:
            nodes.extend(package.nodes)
    return nodes


@router.get("/replicate_status")
async def replicate_status(node_type: str) -> str:
    """
    Returns the status of the Replicate model.
    """
    node_class = get_node_class(node_type)
    if node_class:
        url = node_class.get_model_info().get("url")
        if url:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/status")
                return response.json()["status"]
        else:
            raise HTTPException(status_code=404, detail="Node type not found")
    else:
        raise HTTPException(status_code=404, detail="Node type not found")
