#!/usr/bin/env python

import asyncio

import httpx
from fastapi import APIRouter, HTTPException

from nodetool.config.logging_config import get_logger
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    CalendarEvent,
    DataframeRef,
    DocumentRef,
    Email,
    FilePath,
    FolderRef,
    FontRef,
    HFAudioClassification,
    HFAudioToAudio,
    HFAutomaticSpeechRecognition,
    HFComputerVision,
    HFDepthEstimation,
    HFDocumentQuestionAnswering,
    HFFeatureExtraction,
    HFFillMask,
    HFFlux,
    HFFluxDepth,
    HFFluxFill,
    HFFluxKontext,
    HFFluxRedux,
    HFImageClassification,
    HFImageFeatureExtraction,
    HFImageSegmentation,
    HFImageTextToText,
    HFImageTo3D,
    HFImageToImage,
    HFImageToText,
    HFImageToVideo,
    HFMaskGeneration,
    HFNaturalLanguageProcessing,
    HFObjectDetection,
    HFQuestionAnswering,
    HFSentenceSimilarity,
    HFSummarization,
    HFTableQuestionAnswering,
    HFText2TextGeneration,
    HFTextClassification,
    HFTextGeneration,
    HFTextTo3D,
    HFTextToAudio,
    HFTextToImage,
    HFTextToSpeech,
    HFTextToVideo,
    HFTokenClassification,
    HFTranslation,
    HFUnconditionalImageGeneration,
    HFVideoClassification,
    HFVideoTextToText,
    HFVisualQuestionAnswering,
    HFVoiceActivityDetection,
    HFZeroShotAudioClassification,
    HFZeroShotClassification,
    HFZeroShotImageClassification,
    HFZeroShotObjectDetection,
    HuggingFaceModel,
    ImageRef,
    InferenceProvider,
    InferenceProviderAudioClassificationModel,
    InferenceProviderAutomaticSpeechRecognitionModel,
    InferenceProviderImageClassificationModel,
    InferenceProviderImageSegmentationModel,
    InferenceProviderImageToImageModel,
    InferenceProviderSummarizationModel,
    InferenceProviderTextClassificationModel,
    InferenceProviderTextGenerationModel,
    InferenceProviderTextToAudioModel,
    InferenceProviderTextToImageModel,
    InferenceProviderTextToSpeechModel,
    InferenceProviderTextToTextModel,
    InferenceProviderTranslationModel,
    LanguageModel,
    ModelRef,
    NodeRef,
    NPArray,
    PlotlyConfig,
    SVGElement,
    TaskPlan,
    TextRef,
    VideoRef,
    WorkflowRef,
)
from nodetool.packages.registry import Registry
from nodetool.system.system_stats import SystemStats
from nodetool.types.job import (
    JobUpdate,
)
from nodetool.types.prediction import Prediction
from nodetool.workflows.base_node import ToolResultNode, get_node_class
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.types import ProcessingMessage

log = get_logger(__name__)
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
    | CalendarEvent
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
    | HFFlux
    | HFFluxKontext
    | HFFluxRedux
    | HFFluxDepth
    | HFFluxFill
    | HFTextToImage
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
    | RunJobRequest
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
    nodes = [
        ToolResultNode.get_metadata(),
    ]
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
