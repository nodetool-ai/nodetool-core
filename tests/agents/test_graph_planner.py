"""Unit tests for GraphPlanner"""

from nodetool.workflows.processing_context import ProcessingContext
import pytest
import tempfile
from unittest.mock import Mock, AsyncMock, patch 

from nodetool.agents.graph_planner import (
    GraphPlanner,
    GraphInput,
)
from nodetool.metadata.types import DocumentRef, DataframeRef
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.workflows.base_node import BaseNode, InputNode, OutputNode

# Add test node classes at the module level
class TestDocumentInputNode(InputNode):
    """Test node that outputs a document"""
    
    @classmethod
    def get_namespace(cls):
        return "test.input"
    
    async def process(self, context: ProcessingContext) -> DocumentRef:
        return DocumentRef(uri="test_document")


class TestStringInputNode(InputNode):
    """Test node that outputs a string - for compatible edge testing"""
    
    @classmethod
    def get_namespace(cls):
        return "test.input"
    
    async def process(self, context: ProcessingContext) -> str:
        return "test_string"


class TestCSVImportNode(BaseNode):
    """Test node that imports CSV with string input"""
    
    csv_data: str = ""
    
    @classmethod 
    def get_namespace(cls):
        return "test.data"
    
    async def process(self, context: ProcessingContext) -> DataframeRef:
        return DataframeRef(uri="test_dataframe")


class TestSumArrayNode(BaseNode):
    """Test node that sums an array"""
    
    array: list = []
    
    @classmethod
    def get_namespace(cls):
        return "test.math"
    
    async def process(self, context: ProcessingContext) -> float:
        return 42.0


class TestGraphPlanner:
    """Test the GraphPlanner class"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock chat provider"""
        provider = Mock()
        provider.generate_message = AsyncMock()
        return provider

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_prompt_context(self, mock_provider, temp_workspace):
        """Test prompt context generation"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test objective",
            verbose=False,
        )

        context = planner._get_prompt_context()
        assert context["objective"] == "Test objective"

    def test_is_edge_type_compatible(self, mock_provider, temp_workspace):
        """Test edge type compatibility checking"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test objective",
            verbose=False,
        )

        from nodetool.metadata.type_metadata import TypeMetadata

        # Test compatible types
        str_type = TypeMetadata(type="str")
        assert planner._is_edge_type_compatible_enhanced(str_type, str_type) is True

        # Test numeric conversions
        int_type = TypeMetadata(type="int")
        float_type = TypeMetadata(type="float")
        assert planner._is_edge_type_compatible_enhanced(int_type, float_type) is True
        assert planner._is_edge_type_compatible_enhanced(float_type, int_type) is True

        # Test incompatible types
        image_type = TypeMetadata(type="image")
        assert planner._is_edge_type_compatible_enhanced(image_type, str_type) is False
        assert planner._is_edge_type_compatible_enhanced(str_type, image_type) is False

        # Test any type
        any_type = TypeMetadata(type="any")
        assert planner._is_edge_type_compatible_enhanced(any_type, str_type) is True
        assert planner._is_edge_type_compatible_enhanced(str_type, any_type) is True

    def test_validate_graph_edge_types(self, mock_provider, temp_workspace):
        """Test graph edge type validation catches type mismatches"""
        planner = GraphPlanner(
            provider=mock_provider,
            model="test-model",
            objective="Test objective",
            verbose=False,
        )

        # Create test node instances
        doc_input_node = {
            "node_id": "doc_input",
            "node_type": TestDocumentInputNode.get_node_type(),
            "properties": "{}"
        }
        
        csv_import_node = {
            "node_id": "csv_import",
            "node_type": TestCSVImportNode.get_node_type(),
            "properties": "{\"csv_data\": {\"type\": \"edge\", \"source\": \"doc_input\", \"sourceHandle\": \"output\"}}"
        }

        sum_array_node = {
            "node_id": "sum_array",
            "node_type": TestSumArrayNode.get_node_type(),
            "properties": "{}"
        }

        # Test incompatible edge (document to string)
        node_specs = {
            "node_specifications": [doc_input_node, csv_import_node, sum_array_node],
        }

        error = planner._validate_graph_edge_types(node_specs)
        assert "Type mismatch on edge" in error or "Graph edge type validation errors:" in error
        assert "doc_input.output" in error and "csv_import.csv_data" in error

        # Test compatible edge (use string input node instead)
        string_input_node = {
            "node_id": "string_input",
            "node_type": TestStringInputNode.get_node_type(),
            "properties": "{}"
        }
        csv_import_node_compatible = {
            "node_id": "csv_import_compatible",
            "node_type": TestCSVImportNode.get_node_type(),
            "properties": "{\"csv_data\": {\"type\": \"edge\", \"source\": \"string_input\", \"sourceHandle\": \"output\"}}"
        }
        
        node_specs_compatible = {
            "node_specifications": [string_input_node, csv_import_node_compatible],
        }
        
        error = planner._validate_graph_edge_types(node_specs_compatible)
        assert error == ""  # No error for compatible types


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
