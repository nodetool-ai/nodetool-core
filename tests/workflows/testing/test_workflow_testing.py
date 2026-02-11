"""
Tests for the Workflow Testing Framework
=========================================

These tests validate the workflow testing framework itself,
demonstrating its usage patterns and ensuring reliability.
"""

import pytest

from nodetool.workflows.testing import (
    MockProcessingContext,
    WorkflowTestContext,
    assert_node_executed,
    assert_output,
    assert_output_contains,
    assert_output_type,
    assert_output_value,
    run_node_test,
    run_workflow_test,
)

# ------------------------------------------------------------------
# Test Nodes for framework tests
# ------------------------------------------------------------------


class TestMockProcessingContext:
    """Tests for MockProcessingContext."""

    def test_create_default(self):
        """Test creating a MockProcessingContext with defaults."""
        ctx = MockProcessingContext()
        assert ctx.user_id == "test-user"
        assert ctx.auth_token == "test-token"
        assert ctx.workflow_id != ""

    def test_create_with_custom_values(self):
        """Test creating with custom values."""
        ctx = MockProcessingContext(
            user_id="custom-user",
            auth_token="custom-token",
        )
        assert ctx.user_id == "custom-user"
        assert ctx.auth_token == "custom-token"

    @pytest.mark.asyncio
    async def test_mock_secret(self):
        """Test mocking secrets."""
        ctx = MockProcessingContext()
        ctx.set_mock_secret("API_KEY", "test-api-key")

        value = await ctx.get_secret("API_KEY")
        assert value == "test-api-key"

    @pytest.mark.asyncio
    async def test_mock_secret_required(self):
        """Test mocking required secrets."""
        ctx = MockProcessingContext()
        ctx.set_mock_secret("API_KEY", "test-api-key")

        value = await ctx.get_secret_required("API_KEY")
        assert value == "test-api-key"

    @pytest.mark.asyncio
    async def test_mock_secret_required_missing(self):
        """Test that missing required secret raises error."""
        ctx = MockProcessingContext()

        with pytest.raises(ValueError, match=r"Mock secret.*not configured"):
            await ctx.get_secret_required("MISSING_KEY")

    @pytest.mark.asyncio
    async def test_mock_http_response(self):
        """Test mocking HTTP responses."""
        ctx = MockProcessingContext()
        ctx.set_mock_http_response(
            "https://api.example.com/data",
            json_data={"result": "success"},
        )

        response = await ctx.http_get("https://api.example.com/data")
        assert response.status_code == 200
        assert response.json() == {"result": "success"}

    @pytest.mark.asyncio
    async def test_mock_asset(self):
        """Test mocking assets."""
        ctx = MockProcessingContext()
        ctx.set_mock_asset(
            "asset-123",
            "test.txt",
            "text/plain",
            b"Hello, World!",
        )

        asset = await ctx.find_asset("asset-123")
        assert asset is not None
        assert asset.name == "test.txt"

        io_obj = await ctx.download_asset("asset-123")
        assert io_obj.read() == b"Hello, World!"

    @pytest.mark.asyncio
    async def test_create_and_download_asset(self):
        """Test creating and downloading assets."""
        from io import BytesIO

        ctx = MockProcessingContext()

        asset = await ctx.create_asset(
            name="new-file.txt",
            content_type="text/plain",
            content=BytesIO(b"New content"),
        )

        assert asset.name == "new-file.txt"

        io_obj = await ctx.download_asset(asset.id)
        assert io_obj.read() == b"New content"


class TestWorkflowTestContext:
    """Tests for WorkflowTestContext configuration."""

    def test_create(self):
        """Test creating a WorkflowTestContext."""
        ctx = WorkflowTestContext()
        assert ctx._secrets == {}
        assert ctx._http_responses == {}

    def test_chain_configuration(self):
        """Test method chaining for configuration."""
        ctx = (
            WorkflowTestContext()
            .mock_secret("KEY1", "value1")
            .mock_secret("KEY2", "value2")
            .mock_http_response("https://api.example.com", json_data={"ok": True})
        )

        assert ctx._secrets["KEY1"] == "value1"
        assert ctx._secrets["KEY2"] == "value2"
        assert "https://api.example.com" in ctx._http_responses

    def test_convert_to_mock_context(self):
        """Test converting to MockProcessingContext."""
        test_ctx = WorkflowTestContext()
        test_ctx.mock_secret("API_KEY", "test-key")
        test_ctx.set_variable("my_var", 42)

        mock_ctx = MockProcessingContext.from_test_context(test_ctx)

        assert mock_ctx._mock_secrets["API_KEY"] == "test-key"
        assert mock_ctx.variables["my_var"] == 42


class TestRunNodeTest:
    """Tests for run_node_test function."""

    @pytest.mark.asyncio
    async def test_simple_node(self):
        """Test running a simple node."""
        from nodetool.workflows.test_nodes import Add

        node = Add(a=5.0, b=3.0)
        result = await run_node_test(node)

        assert result == 8.0

    @pytest.mark.asyncio
    async def test_node_with_mock_context(self):
        """Test running a node with mock context."""
        from nodetool.workflows.test_nodes import Add

        ctx = MockProcessingContext()
        node = Add(a=10.0, b=5.0)
        result = await run_node_test(node, context=ctx)

        assert result == 15.0

    @pytest.mark.asyncio
    async def test_multiply_node(self):
        """Test multiply node."""
        from nodetool.workflows.test_nodes import Multiply

        node = Multiply(a=4.0, b=3.0)
        result = await run_node_test(node)

        assert result == 12.0


class TestRunWorkflowTest:
    """Tests for run_workflow_test function."""

    @pytest.mark.asyncio
    async def test_simple_workflow_dsl(self):
        """Test running a simple workflow using DSL."""
        # Import DSL node wrappers
        from nodetool.workflows.test_helper import FormatTextDSL, StringInputDSL

        # Create a simple workflow
        input_node = StringInputDSL(value="World")
        format_node = FormatTextDSL(
            template="Hello, {{ text }}!",
            text=input_node.output,
        )

        result = await run_workflow_test(format_node)

        # Node names are title-cased with spaces
        assert "Format Text" in result
        assert result["Format Text"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_workflow_with_test_context(self):
        """Test running workflow with pre-configured test context."""
        from nodetool.workflows.test_helper import FormatTextDSL, StringInputDSL

        test_ctx = WorkflowTestContext()
        test_ctx.set_variable("test_mode", True)

        # Need to add a consuming node since InputNode alone doesn't emit results
        input_node = StringInputDSL(value="Test")
        format_node = FormatTextDSL(template="{{ text }}", text=input_node.output)

        result = await run_workflow_test(format_node, context=test_ctx)

        # Check that the Format Text node received the input
        assert "Format Text" in result
        assert result["Format Text"] == "Test"


class TestAssertions:
    """Tests for assertion helpers."""

    def test_assert_output_success(self):
        """Test successful output assertion."""
        result = {"Add": 8.0, "Multiply": 15.0}
        assert_output(result, "Add", 8.0)

    def test_assert_output_failure_wrong_value(self):
        """Test output assertion failure with wrong value."""
        result = {"Add": 8.0}
        with pytest.raises(AssertionError, match="Output mismatch"):
            assert_output(result, "Add", 10.0)

    def test_assert_output_failure_missing_node(self):
        """Test output assertion failure with missing node."""
        result = {"Add": 8.0}
        with pytest.raises(AssertionError, match="not found"):
            assert_output(result, "Missing", 0)

    def test_assert_output_type_success(self):
        """Test successful type assertion."""
        result = {"FormatText": "Hello"}
        assert_output_type(result, "FormatText", str)

    def test_assert_output_type_failure(self):
        """Test type assertion failure."""
        result = {"Add": 8.0}
        with pytest.raises(AssertionError, match="Type mismatch"):
            assert_output_type(result, "Add", str)

    def test_assert_output_value_success(self):
        """Test successful value check assertion."""
        result = {"Add": 8.0}
        assert_output_value(result, "Add", lambda x: x > 0)

    def test_assert_output_value_failure(self):
        """Test value check assertion failure."""
        result = {"Add": -5.0}
        with pytest.raises(AssertionError, match="Output check failed"):
            assert_output_value(result, "Add", lambda x: x > 0)

    def test_assert_node_executed_success(self):
        """Test successful node execution assertion."""
        result = {"Add": 8.0}
        assert_node_executed(result, "Add")

    def test_assert_node_executed_failure(self):
        """Test node execution assertion failure."""
        result = {"Add": 8.0}
        with pytest.raises(AssertionError, match="was not executed"):
            assert_node_executed(result, "Missing")

    def test_assert_output_contains_success(self):
        """Test successful contains assertion."""
        result = {"FormatText": "Hello, World!"}
        assert_output_contains(result, "FormatText", "World")

    def test_assert_output_contains_failure(self):
        """Test contains assertion failure."""
        result = {"FormatText": "Hello, World!"}
        with pytest.raises(AssertionError, match="does not contain"):
            assert_output_contains(result, "FormatText", "Universe")


class TestIntegration:
    """Integration tests for the testing framework."""

    @pytest.mark.asyncio
    async def test_complete_workflow_pattern(self):
        """Test a complete workflow testing pattern."""
        from nodetool.workflows.test_helper import FormatTextDSL, StringInputDSL

        # 1. Set up test context with mocks
        test_ctx = WorkflowTestContext()
        test_ctx.mock_secret("API_KEY", "test-key-123")

        # 2. Create workflow using DSL
        name_input = StringInputDSL(value="Tester")
        greeting = FormatTextDSL(
            template="Hello, {{ text }}!",
            text=name_input.output,
        )

        # 3. Run the test
        result = await run_workflow_test(greeting, context=test_ctx)

        # 4. Make assertions (node names are title-cased with spaces)
        assert_node_executed(result, "Format Text")
        assert_output(result, "Format Text", "Hello, Tester!")
        assert_output_type(result, "Format Text", str)
        assert_output_contains(result, "Format Text", "Hello")

    @pytest.mark.asyncio
    async def test_multiple_node_workflow(self):
        """Test workflow with multiple connected nodes."""
        from nodetool.workflows.test_helper import FormatTextDSL, StringInputDSL

        # Build a chain of nodes with unique input names
        input1 = StringInputDSL(name="name1", value="Alice")

        # Single format operation for cleaner test
        format1 = FormatTextDSL(template="Hello {{ text }}", text=input1.output)

        # Run
        result = await run_workflow_test(format1)

        # Check output
        assert "Format Text" in result
        assert result["Format Text"] == "Hello Alice"
