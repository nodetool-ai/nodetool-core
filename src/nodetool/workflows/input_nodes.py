"""
Input nodes for workflows.

This module provides input nodes that can be used in workflows to accept
various types of input values. These nodes are registered automatically
when imported.
"""

from typing import TypedDict

from pydantic import Field

from nodetool.metadata.types import (
    AudioRef,
    ColorRef,
    DataframeRef,
    DocumentRef,
    FolderRef,
    HuggingFaceModel,
    ImageModel,
    ImageRef,
    LanguageModel,
    VideoRef,
)
from nodetool.workflows.base_node import InputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class FloatInput(InputNode):
    """
    Accepts a floating-point number as a parameter for workflows, typically constrained by a minimum and maximum value.  This input allows for precise numeric settings, such as adjustments, scores, or any value requiring decimal precision.
    input, parameter, float, number, decimal, range

    Use cases:
    - Specify a numeric value within a defined range (e.g., 0.0 to 1.0).
    - Set thresholds, confidence scores, or scaling factors.
    - Configure continuous parameters like opacity, volume, or temperature.
    """

    value: float = 0.0
    min: float = 0
    max: float = 100

    @classmethod
    def return_type(cls):
        return float

    async def process(self, context: ProcessingContext) -> float:
        return self.value


class BooleanInput(InputNode):
    """
    Accepts a boolean (true/false) value as a parameter for workflows.  This input is used for binary choices, enabling or disabling features, or controlling conditional logic paths.
    input, parameter, boolean, bool, toggle, switch, flag

    Use cases:
    - Toggle features or settings on or off.
    - Set binary flags to control workflow behavior.
    - Make conditional choices within a workflow (e.g., proceed if true).
    """

    value: bool = False

    @classmethod
    def return_type(cls):
        return bool

    async def process(self, context: ProcessingContext) -> bool:
        return self.value


class IntegerInput(InputNode):
    """
    Accepts an integer (whole number) as a parameter for workflows, typically constrained by a minimum and maximum value.  This input is used for discrete numeric values like counts, indices, or iteration limits.
    input, parameter, integer, number, count, index, whole_number

    Use cases:
    - Specify counts or quantities (e.g., number of items, iterations).
    - Set index values for accessing elements in a list or array.
    - Configure discrete numeric parameters like age, steps, or quantity.
    """

    value: int = 0
    min: int = 0
    max: int = 100

    @classmethod
    def return_type(cls):
        return int

    async def process(self, context: ProcessingContext) -> int:
        return self.value


class StringInput(InputNode):
    """
    Accepts a string value as a parameter for workflows.
    input, parameter, string, text, label, name, value

    Use cases:
    - Define a name for an entity or process.
    - Specify a label for a component or output.
    - Enter a short keyword or search term.
    - Provide a simple configuration value (e.g., an API key, a model name).
    - If you need to input multi-line text or the content of a file, use 'DocumentFileInput'.
    """

    value: str = ""

    @classmethod
    def return_type(cls):
        return str

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class StringListInput(InputNode):
    """
    Accepts a list of strings as a parameter for workflows.
    input, parameter, string, text, label, name, value
    """

    value: list[str] = Field(
        default=[], description="The list of strings to use as input."
    )

    @classmethod
    def return_type(cls):
        return list[str]


class FolderPathInput(InputNode):
    """
    Accepts a folder path as a parameter for workflows.
    input, parameter, folder, path, folderpath, local_folder, filesystem
    """

    value: str = Field(
        "",
        description="The folder path to use as input.",
        json_schema_extra={"type": "folder_path"},
    )

    @classmethod
    def return_type(cls):
        return str


class HuggingFaceModelInput(InputNode):
    """
    Accepts a Hugging Face model as a parameter for workflows.
    input, parameter, model, huggingface, hugging_face, model_name
    """

    value: HuggingFaceModel = Field(
        HuggingFaceModel(), description="The Hugging Face model to use as input."
    )

    @classmethod
    def return_type(cls):
        return HuggingFaceModel


class ColorInput(InputNode):
    """
    Accepts a color value as a parameter for workflows.
    input, parameter, color, color_picker, color_input
    """

    value: ColorRef = Field(ColorRef(), description="The color to use as input.")

    @classmethod
    def return_type(cls):
        return ColorRef


class LanguageModelInput(InputNode):
    """
    Accepts a language model as a parameter for workflows.
    input, parameter, model, language, model_name
    """

    value: LanguageModel = Field(
        LanguageModel(), description="The language model to use as input."
    )

    @classmethod
    def return_type(cls):
        return LanguageModel


class ImageModelInput(InputNode):
    """
    Accepts an image generation model as a parameter for workflows.
    input, parameter, model, image, generation
    """

    value: ImageModel = Field(
        ImageModel(), description="The image generation model to use as input."
    )

    @classmethod
    def return_type(cls):
        return ImageModel


class DataframeInput(InputNode):
    """
    Accepts a reference to a dataframe asset for workflows.
    input, parameter, dataframe, table, data
    """

    value: DataframeRef = Field(
        DataframeRef(), description="The dataframe to use as input."
    )

    @classmethod
    def return_type(cls):
        return DataframeRef


class DocumentInput(InputNode):
    """
    Accepts a reference to a document asset for workflows, specified by a 'DocumentRef'.  A 'DocumentRef' points to a structured document (e.g., PDF, DOCX, TXT) which can be processed or analyzed. This node is used when the workflow needs to operate on a document as a whole entity, potentially including its structure and metadata, rather than just raw text.
    input, parameter, document, file, asset, reference

    Use cases:
    - Load a specific document (e.g., PDF, Word, text file) for content extraction or analysis.
    - Pass a document to models that are designed to process specific document formats.
    - Manage documents as distinct assets within a workflow.
    - If you have a local file path and need to convert it to a 'DocumentRef', consider using 'DocumentFileInput'.
    """

    value: DocumentRef = Field(
        DocumentRef(), description="The document to use as input."
    )

    @classmethod
    def return_type(cls):
        return DocumentRef


class ImageInput(InputNode):
    """
    Accepts a reference to an image asset for workflows, specified by an 'ImageRef'.  An 'ImageRef' points to image data that can be used for display, analysis, or processing by vision models.
    input, parameter, image, picture, graphic, visual, asset

    Use cases:
    - Load an image for visual processing or analysis.
    - Provide an image as input to computer vision models (e.g., object detection, image classification).
    - Select an image for manipulation, enhancement, or inclusion in a document.
    - Display an image within a workflow interface.
    """

    value: ImageRef = Field(ImageRef(), description="The image to use as input.")

    @classmethod
    def return_type(cls):
        return ImageRef

    async def process(self, context: ProcessingContext) -> ImageRef:
        return self.value


class VideoInput(InputNode):
    """
    Accepts a reference to a video asset for workflows, specified by a 'VideoRef'.  A 'VideoRef' points to video data that can be used for playback, analysis, frame extraction, or processing by video-capable models.
    input, parameter, video, movie, clip, visual, asset

    Use cases:
    - Load a video file for processing or content analysis.
    - Analyze video content for events, objects, or speech.
    - Extract frames or audio tracks from a video.
    - Provide video input to models that understand video data.
    """

    value: VideoRef = Field(VideoRef(), description="The video to use as input.")

    @classmethod
    def return_type(cls):
        return VideoRef

    async def process(self, context: ProcessingContext) -> VideoRef:
        return self.value


class AudioInput(InputNode):
    """
    Accepts a reference to an audio asset for workflows, specified by an 'AudioRef'.  An 'AudioRef' points to audio data that can be used for playback, transcription, analysis, or processing by audio-capable models.
    input, parameter, audio, sound, voice, speech, asset

    Use cases:
    - Load an audio file for speech-to-text transcription.
    - Analyze sound for specific events or characteristics.
    - Provide audio input to models for tasks like voice recognition or music generation.
    - Process audio for enhancement or feature extraction.
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    @classmethod
    def return_type(cls):
        return AudioRef

    async def process(self, context: ProcessingContext) -> AudioRef:
        return self.value


class RealtimeAudioInput(InputNode):
    """
    Accepts streaming audio data for workflows.
    input, parameter, audio, sound, voice, speech, asset
    """

    value: AudioRef = Field(AudioRef(), description="The audio to use as input.")

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        basic_fields = super().get_basic_fields()
        return basic_fields + ["value"]

    @classmethod
    def is_streaming_output(cls) -> bool:
        return True

    class OutputType(TypedDict):
        chunk: Chunk

    @classmethod
    def return_type(cls):
        return cls.OutputType


class AssetFolderInput(InputNode):
    """
    Accepts an asset folder as a parameter for workflows.
    input, parameter, folder, path, folderpath, local_folder, filesystem
    """

    value: FolderRef = Field(FolderRef(), description="The folder to use as input.")

    @classmethod
    def return_type(cls):
        return FolderRef


class FilePathInput(InputNode):
    """
    Accepts a local filesystem path (to a file or directory) as input for workflows.
    input, parameter, path, filepath, directory, local_file, filesystem

    Use cases:
    - Provide a local path to a specific file or directory for processing.
    - Specify an input or output location on the local filesystem for a development task.
    - Load local datasets or configuration files not managed as assets.
    - Not available in production: raises an error if used in a production environment.
    """

    value: str = Field(
        "",
        description="The path to use as input.",
        json_schema_extra={"type": "file_path"},
    )

    @classmethod
    def return_type(cls):
        return str


class DocumentFileInput(InputNode):
    """
    Accepts a local file path pointing to a document and converts it into a 'DocumentRef'.
    input, parameter, document, file, path, local_file, load

    Use cases:
    - Directly load a document (e.g., PDF, TXT, DOCX) from a specified local file path.
    - Convert a local file path into a 'DocumentRef' that can be consumed by other document-processing nodes.
    - Useful for development or workflows that have legitimate access to the local filesystem.
    - To provide an existing 'DocumentRef', use 'DocumentInput'.
    """

    value: str = Field(
        "",
        description="The path to the document file.",
        json_schema_extra={"type": "file_path"},
    )

    class OutputType(TypedDict):
        document: DocumentRef
        path: str

    @classmethod
    def return_type(cls):
        return cls.OutputType
