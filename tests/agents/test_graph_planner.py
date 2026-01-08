"""Unit tests for GraphPlanner"""

import tempfile
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import Field

from nodetool.agents.graph_planner import (
    GraphInput,
    GraphOutput,
    GraphPlanner,
    _is_type_compatible,
    get_node_type_for_metadata,
)
from nodetool.metadata.types import TypeMetadata
from nodetool.workflows.base_node import InputNode, OutputNode
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_types import Chunk


class StringInputNode(InputNode):
    value: str = Field(default="", description="The value of the input.")

    async def process(self, context: ProcessingContext) -> str:
        return self.value


class StringOutputNode(OutputNode):
    value: str = Field(default="", description="The value of the output.")

    async def process(self, context: ProcessingContext) -> str:
        return self.value


@pytest.mark.asyncio
class TestGraphPlanner:
    """Test the GraphPlanner class"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock chat provider"""
        provider = Mock()
        provider.generate_message = AsyncMock()
        provider.generate_messages = AsyncMock()
        return provider

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def processing_context(self, temp_workspace):
        """Create a processing context for tests"""
        return ProcessingContext(user_id="test_user", auth_token="test_token", workspace_dir=temp_workspace)

    def test_initialization(self, mock_provider, temp_workspace):
        """Test GraphPlanner initialization"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test objective",
            verbose=False,
        )

        assert planner.provider == mock_provider
        assert planner.model == "test-model"
        assert planner.objective == "Test objective"
        assert planner.input_schema == []
        assert planner.output_schema == []
        assert planner.graph is None
        assert planner.existing_graph is None
        assert planner.max_tokens == 8192

    def test_type_compatibility(self):
        """Test type compatibility checking"""
        # Test compatible types
        str_type = TypeMetadata(type="str")
        assert _is_type_compatible(str_type, str_type) is True

        # Test numeric conversions
        int_type = TypeMetadata(type="int")
        float_type = TypeMetadata(type="float")
        assert _is_type_compatible(int_type, float_type) is True
        assert _is_type_compatible(float_type, int_type) is True

        # Test incompatible types
        image_type = TypeMetadata(type="image")
        assert _is_type_compatible(image_type, str_type) is False
        assert _is_type_compatible(str_type, image_type) is False

        # Test any type
        any_type = TypeMetadata(type="any")
        assert _is_type_compatible(any_type, str_type) is True
        assert _is_type_compatible(str_type, any_type) is True

    def test_explicit_input_schema(self, mock_provider, temp_workspace):
        """Test that GraphPlanner can work with explicit input schema"""
        # Create explicit schema
        input_schema = [
            GraphInput(name="text", type=TypeMetadata(type="str"), description="Text input"),
            GraphInput(name="number", type=TypeMetadata(type="int"), description="Number input"),
            GraphInput(
                name="float_num",
                type=TypeMetadata(type="float"),
                description="Float input",
            ),
            GraphInput(name="flag", type=TypeMetadata(type="bool"), description="Boolean flag"),
            GraphInput(
                name="items",
                type=TypeMetadata(type="list", type_args=[TypeMetadata(type="str")]),
                description="List of strings",
            ),
            GraphInput(
                name="mapping",
                type=TypeMetadata(
                    type="dict",
                    type_args=[TypeMetadata(type="str"), TypeMetadata(type="str")],
                ),
                description="String to string mapping",
            ),
        ]

        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Process various data types",
            input_schema=input_schema,
            verbose=False,
        )

        # Check that input_schema was set
        assert len(planner.input_schema) == len(input_schema)

        # Check specific types
        schema_by_name = {inp.name: inp for inp in planner.input_schema}

        assert schema_by_name["text"].type.type == "str"
        assert schema_by_name["number"].type.type == "int"
        assert schema_by_name["float_num"].type.type == "float"
        assert schema_by_name["flag"].type.type == "bool"

        # Check list type
        assert schema_by_name["items"].type.type == "list"
        assert schema_by_name["items"].type.type_args[0].type == "str"

        # Check dict type
        assert schema_by_name["mapping"].type.type == "dict"
        assert schema_by_name["mapping"].type.type_args[0].type == "str"
        assert schema_by_name["mapping"].type.type_args[1].type == "str"

    def test_output_schema(self, mock_provider, temp_workspace):
        """Test that GraphPlanner handles output schema correctly"""
        # Create explicit output schema
        output_schema = [
            GraphOutput(name="result", type=TypeMetadata(type="str"), description="Result text"),
            GraphOutput(
                name="score",
                type=TypeMetadata(type="float"),
                description="Result score",
            ),
        ]

        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test with output schema",
            output_schema=output_schema,
            verbose=False,
        )

        # Check that the output schema was set
        assert len(planner.output_schema) == 2
        schema_by_name = {out.name: out for out in planner.output_schema}

        # Should use the explicit types
        assert schema_by_name["result"].type.type == "str"
        assert schema_by_name["score"].type.type == "float"

    def test_mixed_schemas(self, mock_provider, temp_workspace):
        """Test GraphPlanner with both input and output schemas"""
        # Create schemas
        input_schema = [
            GraphInput(
                name="input_text",
                type=TypeMetadata(type="str"),
                description="Input text",
            ),
            GraphInput(
                name="threshold",
                type=TypeMetadata(type="float"),
                description="Processing threshold",
            ),
        ]

        output_schema = [
            GraphOutput(
                name="processed_text",
                type=TypeMetadata(type="str"),
                description="Processed text output",
            ),
            GraphOutput(
                name="confidence",
                type=TypeMetadata(type="float"),
                description="Processing confidence",
            ),
        ]

        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Process text with confidence scoring",
            input_schema=input_schema,
            output_schema=output_schema,
            verbose=False,
        )

        # Check both schemas were set correctly
        assert len(planner.input_schema) == 2
        assert len(planner.output_schema) == 2

        # Verify input schema
        input_by_name = {inp.name: inp for inp in planner.input_schema}
        assert input_by_name["input_text"].type.type == "str"
        assert input_by_name["threshold"].type.type == "float"

        # Verify output schema
        output_by_name = {out.name: out for out in planner.output_schema}
        assert output_by_name["processed_text"].type.type == "str"
        assert output_by_name["confidence"].type.type == "float"

    def test_create_graph_method_exists(self, mock_provider):
        """Test that create_graph method exists"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Simple test",
            verbose=False,
        )

        # Test that the method exists and is callable
        assert hasattr(planner, "create_graph")
        assert callable(planner.create_graph)

    @pytest.mark.asyncio
    async def test_create_graph_failure(self, mock_provider, processing_context):
        """Test create_graph handles failures gracefully"""

        # Mock generate_messages to return chunks with invalid JSON content
        async def mock_generate_messages(*args, **kwargs):
            yield Chunk(content="Invalid JSON response")

        mock_provider.generate_messages = mock_generate_messages

        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test failure handling",
            verbose=False,
        )

        with pytest.raises(ValueError) as exc_info:
            async for _update in planner.create_graph(processing_context):
                pass

        assert "Failed to produce valid workflow design" in str(exc_info.value)

    def test_node_type_mapping_exists(self):
        """Test that get_node_type_for_metadata function exists and is importable"""
        # Simple test to verify the function can be imported and called
        # without doing actual node type resolution which requires full registry
        str_meta = TypeMetadata(type="str")

        # Just test that the function is callable - we can't easily test actual
        # functionality without the full node registry setup
        try:
            # This will likely fail but we're just testing it's callable
            get_node_type_for_metadata(str_meta, InputNode)
        except (ValueError, Exception):
            # Expected to fail without proper registry setup
            pass

        # If we get here, the function exists and is callable
        assert callable(get_node_type_for_metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
