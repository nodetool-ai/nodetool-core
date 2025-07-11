"""Unit tests for GraphPlanner"""

import pytest
import tempfile
from pydantic import Field
from unittest.mock import Mock, AsyncMock, patch

from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.graph_planner import (
    GraphPlanner,
    GraphInput,
    GraphOutput,
    get_node_type_for_metadata,
    _is_type_compatible,
)
from nodetool.metadata.types import TypeMetadata
from nodetool.workflows.base_node import InputNode, OutputNode
from nodetool.workflows.types import Chunk


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
        return ProcessingContext(
            user_id="test_user",
            auth_token="test_token", 
            workspace_dir=temp_workspace
        )

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

    def test_type_inference_from_values(self, mock_provider, temp_workspace):
        """Test that GraphPlanner can infer input schema from provided values"""
        # Test with various input types
        test_inputs = {
            "text": "Hello world",
            "number": 42,
            "float_num": 3.14,
            "flag": True,
            "items": ["a", "b", "c"],
            "mapping": {"key1": "value1", "key2": "value2"},
            "empty_list": [],
            "empty_dict": {},
            "nested": {"data": [1, 2, 3]},
        }
        
        with patch(
            "nodetool.agents.graph_planner.get_node_type_for_metadata",
            return_value="dummy.input_node",
        ):
            planner = GraphPlanner(
                provider=mock_provider,
                model="test-model",
                objective="Process various data types",
                inputs=test_inputs,
                verbose=False,
            )

        # Check that input_schema was inferred
        assert len(planner.input_schema) == len(test_inputs)

        # Check specific type inferences
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

        # Check empty collections
        assert schema_by_name["empty_list"].type.type == "list"
        assert schema_by_name["empty_list"].type.type_args[0].type == "any"

        assert schema_by_name["empty_dict"].type.type == "dict"
        assert schema_by_name["empty_dict"].type.type_args[0].type == "str"
        assert schema_by_name["empty_dict"].type.type_args[1].type == "any"

    def test_type_inference_with_existing_schema(self, mock_provider, temp_workspace):
        """Test that existing input_schema takes precedence over inferred types"""
        test_inputs = {"text": "Hello world", "number": 42}

        # Create explicit schema
        input_schema = [
            GraphInput(
                name="text", type=TypeMetadata(type="str"), description="Text input"
            ),
            GraphInput(
                name="number",
                type=TypeMetadata(
                    type="float"
                ),  # Intentionally different from actual value type
                description="Number input",
            ),
        ]
        
        with patch(
            "nodetool.agents.graph_planner.get_node_type_for_metadata",
            return_value="dummy.input_node",
        ):
            planner = GraphPlanner(
                provider=mock_provider,
                model="test-model",
                objective="Test with explicit schema",
                inputs=test_inputs,
                input_schema=input_schema,
                verbose=False,
            )

        # Check that the explicit schema was used
        assert len(planner.input_schema) == 2
        schema_by_name = {inp.name: inp for inp in planner.input_schema}

        # Should use the explicit types, not inferred ones
        assert schema_by_name["text"].type.type == "str"
        assert schema_by_name["number"].type.type == "float"  # Not "int"

    def test_type_inference_special_types(self, mock_provider, temp_workspace):
        """Test type inference for special types like None and asset references"""
        test_inputs = {
            "null_value": None,
            "image_ref": {"type": "image", "uri": "path/to/image.jpg"},
            "tuple_data": (1, "two", 3.0),
        }
        
        with patch(
            "nodetool.agents.graph_planner.get_node_type_for_metadata",
            return_value="dummy.input_node",
        ):
            planner = GraphPlanner(
                provider=mock_provider,
                model="test-model",
                objective="Test special types",
                inputs=test_inputs,
                verbose=False,
            )

        schema_by_name = {inp.name: inp for inp in planner.input_schema}

        # Check None type
        assert schema_by_name["null_value"].type.type == "none"
        assert schema_by_name["null_value"].type.optional is True

        # Check asset reference
        assert schema_by_name["image_ref"].type.type == "image"

        # Check tuple type
        assert schema_by_name["tuple_data"].type.type == "tuple"
        assert len(schema_by_name["tuple_data"].type.type_args) == 3
        assert schema_by_name["tuple_data"].type.type_args[0].type == "int"
        assert schema_by_name["tuple_data"].type.type_args[1].type == "str"
        assert schema_by_name["tuple_data"].type.type_args[2].type == "float"

    def test_create_graph_method_exists(self, mock_provider):
        """Test that create_graph method exists"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Simple test",
            verbose=False,
        )

        # Test that the method exists and is callable
        assert hasattr(planner, 'create_graph')
        assert callable(planner.create_graph)

    def test_schema_handling(self, mock_provider):
        """Test that GraphPlanner handles input and output schemas correctly"""
        input_schema = [
            GraphInput(
                name="name",
                type=TypeMetadata(type="str"),
                description="Person's name"
            )
        ]
        
        output_schema = [
            GraphOutput(
                name="greeting",
                type=TypeMetadata(type="str"),
                description="Greeting message"
            )
        ]

        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Generate greeting",
            inputs={"name": "Alice"},
            input_schema=input_schema,
            output_schema=output_schema,
            verbose=False,
        )

        # Verify the planner used the provided schemas
        assert len(planner.input_schema) == 1
        assert planner.input_schema[0].name == "name"
        assert len(planner.output_schema) == 1
        assert planner.output_schema[0].name == "greeting"

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
            async for update in planner.create_graph(processing_context):
                pass

        assert "failed to produce valid result" in str(exc_info.value)

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