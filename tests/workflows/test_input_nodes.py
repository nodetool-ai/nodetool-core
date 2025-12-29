"""Tests for input node classes."""

import pytest

from nodetool.workflows.input_nodes import (
    AudioInput,
    AssetFolderInput,
    BooleanInput,
    ColorInput,
    DataframeInput,
    DocumentFileInput,
    DocumentInput,
    FilePathInput,
    FloatInput,
    FolderPathInput,
    HuggingFaceModelInput,
    ImageInput,
    ImageModelInput,
    IntegerInput,
    LanguageModelInput,
    RealtimeAudioInput,
    StringInput,
    StringListInput,
    VideoInput,
)
from nodetool.workflows.base_node import NODE_BY_TYPE
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


class TestInputNodeRegistration:
    """Tests that input nodes are properly registered in the node registry."""

    def test_float_input_registered(self):
        """FloatInput should be registered in NODE_BY_TYPE."""
        assert any(
            cls.__name__ == "FloatInput" for cls in NODE_BY_TYPE.values()
        )

    def test_boolean_input_registered(self):
        """BooleanInput should be registered in NODE_BY_TYPE."""
        assert any(
            cls.__name__ == "BooleanInput" for cls in NODE_BY_TYPE.values()
        )

    def test_integer_input_registered(self):
        """IntegerInput should be registered in NODE_BY_TYPE."""
        assert any(
            cls.__name__ == "IntegerInput" for cls in NODE_BY_TYPE.values()
        )

    def test_string_input_registered(self):
        """StringInput should be registered in NODE_BY_TYPE."""
        assert any(
            cls.__name__ == "StringInput" for cls in NODE_BY_TYPE.values()
        )

    def test_all_input_nodes_registered(self):
        """All input nodes should be registered in NODE_BY_TYPE."""
        expected_nodes = [
            "FloatInput",
            "BooleanInput",
            "IntegerInput",
            "StringInput",
            "StringListInput",
            "FolderPathInput",
            "HuggingFaceModelInput",
            "ColorInput",
            "LanguageModelInput",
            "ImageModelInput",
            "DataframeInput",
            "DocumentInput",
            "ImageInput",
            "VideoInput",
            "AudioInput",
            "RealtimeAudioInput",
            "AssetFolderInput",
            "FilePathInput",
            "DocumentFileInput",
        ]
        registered_names = {cls.__name__ for cls in NODE_BY_TYPE.values()}
        for node_name in expected_nodes:
            assert node_name in registered_names, f"{node_name} not registered"


class TestInputNodeReturnTypes:
    """Tests that input nodes return the correct types."""

    def test_float_input_return_type(self):
        """FloatInput should return float type."""
        assert FloatInput.return_type() == float

    def test_boolean_input_return_type(self):
        """BooleanInput should return bool type."""
        assert BooleanInput.return_type() == bool

    def test_integer_input_return_type(self):
        """IntegerInput should return int type."""
        assert IntegerInput.return_type() == int

    def test_string_input_return_type(self):
        """StringInput should return str type."""
        assert StringInput.return_type() == str

    def test_string_list_input_return_type(self):
        """StringListInput should return list[str] type."""
        assert StringListInput.return_type() == list[str]

    def test_folder_path_input_return_type(self):
        """FolderPathInput should return str type."""
        assert FolderPathInput.return_type() == str

    def test_huggingface_model_input_return_type(self):
        """HuggingFaceModelInput should return HuggingFaceModel type."""
        assert HuggingFaceModelInput.return_type() == HuggingFaceModel

    def test_color_input_return_type(self):
        """ColorInput should return ColorRef type."""
        assert ColorInput.return_type() == ColorRef

    def test_language_model_input_return_type(self):
        """LanguageModelInput should return LanguageModel type."""
        assert LanguageModelInput.return_type() == LanguageModel

    def test_image_model_input_return_type(self):
        """ImageModelInput should return ImageModel type."""
        assert ImageModelInput.return_type() == ImageModel

    def test_dataframe_input_return_type(self):
        """DataframeInput should return DataframeRef type."""
        assert DataframeInput.return_type() == DataframeRef

    def test_document_input_return_type(self):
        """DocumentInput should return DocumentRef type."""
        assert DocumentInput.return_type() == DocumentRef

    def test_image_input_return_type(self):
        """ImageInput should return ImageRef type."""
        assert ImageInput.return_type() == ImageRef

    def test_video_input_return_type(self):
        """VideoInput should return VideoRef type."""
        assert VideoInput.return_type() == VideoRef

    def test_audio_input_return_type(self):
        """AudioInput should return AudioRef type."""
        assert AudioInput.return_type() == AudioRef

    def test_asset_folder_input_return_type(self):
        """AssetFolderInput should return FolderRef type."""
        assert AssetFolderInput.return_type() == FolderRef

    def test_file_path_input_return_type(self):
        """FilePathInput should return str type."""
        assert FilePathInput.return_type() == str


class TestInputNodeProcessing:
    """Tests that input nodes process values correctly."""

    @pytest.mark.asyncio
    async def test_float_input_process(self):
        """FloatInput should return its value when processed."""
        node = FloatInput(id="test", value=3.14)
        result = await node.process(None)
        assert result == 3.14

    @pytest.mark.asyncio
    async def test_boolean_input_process(self):
        """BooleanInput should return its value when processed."""
        node = BooleanInput(id="test", value=True)
        result = await node.process(None)
        assert result is True

    @pytest.mark.asyncio
    async def test_integer_input_process(self):
        """IntegerInput should return its value when processed."""
        node = IntegerInput(id="test", value=42)
        result = await node.process(None)
        assert result == 42

    @pytest.mark.asyncio
    async def test_string_input_process(self):
        """StringInput should return its value when processed."""
        node = StringInput(id="test", value="hello")
        result = await node.process(None)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_image_input_process(self):
        """ImageInput should return its value when processed."""
        image_ref = ImageRef(uri="test://image.png")
        node = ImageInput(id="test", value=image_ref)
        result = await node.process(None)
        assert result == image_ref

    @pytest.mark.asyncio
    async def test_video_input_process(self):
        """VideoInput should return its value when processed."""
        video_ref = VideoRef(uri="test://video.mp4")
        node = VideoInput(id="test", value=video_ref)
        result = await node.process(None)
        assert result == video_ref

    @pytest.mark.asyncio
    async def test_audio_input_process(self):
        """AudioInput should return its value when processed."""
        audio_ref = AudioRef(uri="test://audio.mp3")
        node = AudioInput(id="test", value=audio_ref)
        result = await node.process(None)
        assert result == audio_ref


class TestInputNodeDefaults:
    """Tests that input nodes have correct default values."""

    def test_float_input_defaults(self):
        """FloatInput should have correct default values."""
        node = FloatInput(id="test")
        assert node.value == 0.0
        assert node.min == 0
        assert node.max == 100

    def test_boolean_input_defaults(self):
        """BooleanInput should have correct default values."""
        node = BooleanInput(id="test")
        assert node.value is False

    def test_integer_input_defaults(self):
        """IntegerInput should have correct default values."""
        node = IntegerInput(id="test")
        assert node.value == 0
        assert node.min == 0
        assert node.max == 100

    def test_string_input_defaults(self):
        """StringInput should have correct default values."""
        node = StringInput(id="test")
        assert node.value == ""

    def test_string_list_input_defaults(self):
        """StringListInput should have correct default values."""
        node = StringListInput(id="test")
        assert node.value == []


class TestInputNodeProperties:
    """Tests that input nodes have correct properties."""

    def test_input_nodes_not_cacheable(self):
        """Input nodes should not be cacheable."""
        assert FloatInput.is_cacheable() is False
        assert BooleanInput.is_cacheable() is False
        assert IntegerInput.is_cacheable() is False
        assert StringInput.is_cacheable() is False

    def test_input_nodes_basic_fields(self):
        """Input nodes should have 'name' and 'value' as basic fields."""
        assert "name" in FloatInput.get_basic_fields()
        assert "value" in FloatInput.get_basic_fields()

    def test_realtime_audio_input_streaming(self):
        """RealtimeAudioInput should be a streaming output node."""
        assert RealtimeAudioInput.is_streaming_output() is True
