"""
Tests for GraphPlanner Evaluation System

Basic validation tests to ensure the evaluation framework works correctly.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

from nodetool.agents.graph_planner_evaluator import (
    GraphPlannerEvaluator, 
    TestCase, 
    EvaluationResult,
    MetricType,
    EvaluationMetric,
    PromptOptimizer
)
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode


class TestGraphPlannerEvaluator:
    """Test suite for GraphPlannerEvaluator"""
    
    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def evaluator(self, temp_results_dir):
        """Create evaluator instance for testing"""
        return GraphPlannerEvaluator(
            results_dir=temp_results_dir,
            enable_logging=False  # Disable logging for tests
        )
    
    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph for testing"""
        nodes = [
            APINode(
                id="input_1",
                type="InputNode",
                data={"name": "sales_data"}
            ),
            APINode(
                id="process_1", 
                type="GroupByNode",
                data={
                    "group_by": ["date", "product"],
                    "data": {"type": "edge", "source": "input_1", "sourceHandle": "output"}
                }
            ),
            APINode(
                id="output_1",
                type="OutputNode",
                data={
                    "name": "results",
                    "data": {"type": "edge", "source": "process_1", "sourceHandle": "output"}
                }
            )
        ]
        
        return APIGraph(nodes=nodes, edges=[])
    
    @pytest.fixture
    def mock_planner(self, mock_graph):
        """Create a mock GraphPlanner"""
        planner = Mock()
        planner.graph = mock_graph
        planner.model = "test-model"
        
        # Mock the create_graph method to be async
        async def mock_create_graph(context):
            yield Mock()  # Yield a mock update
        
        planner.create_graph = mock_create_graph
        return planner
    
    def test_evaluator_initialization(self, temp_results_dir):
        """Test evaluator initializes correctly"""
        evaluator = GraphPlannerEvaluator(
            results_dir=temp_results_dir,
            enable_logging=False
        )
        
        assert evaluator.results_dir == Path(temp_results_dir)
        assert "sales_analysis" in evaluator.test_cases
        assert evaluator.test_cases["sales_analysis"].complexity_score == 7
    
    def test_test_case_structure(self, evaluator):
        """Test that test cases are properly structured"""
        test_case = evaluator.test_cases["sales_analysis"]
        
        assert isinstance(test_case, TestCase)
        assert test_case.name == "sales_analysis"
        assert len(test_case.input_schema) == 1
        assert len(test_case.output_schema) == 2
        assert len(test_case.expected_nodes) > 0
        assert test_case.complexity_score >= 1
    
    def test_graph_validity_check(self, evaluator, mock_graph):
        """Test graph validity checking"""
        score = evaluator._check_graph_validity(mock_graph)
        assert 0.0 <= score <= 1.0
        
        # Test empty graph
        empty_graph = APIGraph(nodes=[], edges=[])
        empty_score = evaluator._check_graph_validity(empty_graph)
        assert empty_score == 0.0
        
        # Test None graph
        none_score = evaluator._check_graph_validity(None)
        assert none_score == 0.0
    
    def test_node_type_accuracy(self, evaluator, mock_graph):
        """Test node type accuracy checking"""
        test_case = evaluator.test_cases["sales_analysis"]
        score = evaluator._check_node_types(test_case, mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_edge_consistency(self, evaluator, mock_graph):
        """Test edge consistency checking"""
        score = evaluator._check_edge_consistency(mock_graph)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Our mock graph has proper edge format
    
    def test_schema_compliance(self, evaluator, mock_graph):
        """Test schema compliance checking"""
        test_case = evaluator.test_cases["sales_analysis"]
        score = evaluator._check_schema_compliance(test_case, mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_objective_achievement(self, evaluator, mock_graph):
        """Test objective achievement checking"""
        test_case = evaluator.test_cases["sales_analysis"]
        score = evaluator._check_objective_achievement(test_case, mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_completeness_check(self, evaluator, mock_graph):
        """Test completeness checking"""
        test_case = evaluator.test_cases["sales_analysis"]
        score = evaluator._check_completeness(test_case, mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_efficiency_check(self, evaluator, mock_graph):
        """Test efficiency checking"""
        score = evaluator._check_efficiency(mock_graph)
        assert 0.0 <= score <= 1.0
        
        # Test with too many nodes
        many_nodes = APIGraph(nodes=[
            APINode(id=f"node_{i}", type="TestNode", data={})
            for i in range(20)
        ], edges=[])
        efficiency_score = evaluator._check_efficiency(many_nodes)
        assert efficiency_score < 1.0
    
    def test_property_validation(self, evaluator, mock_graph):
        """Test property validation checking"""
        test_case = evaluator.test_cases["sales_analysis"]
        score = evaluator._check_property_validation(test_case, mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_best_practices_check(self, evaluator, mock_graph):
        """Test best practices checking"""
        score = evaluator._check_best_practices(mock_graph)
        assert 0.0 <= score <= 1.0
    
    def test_overall_score_calculation(self, evaluator):
        """Test overall score calculation"""
        metrics = [
            EvaluationMetric(
                name="test1",
                category=MetricType.STRUCTURAL,
                weight=0.5,
                score=0.8,
                max_score=1.0,
                details={},
                passed=True
            ),
            EvaluationMetric(
                name="test2", 
                category=MetricType.FUNCTIONAL,
                weight=0.3,
                score=0.6,
                max_score=1.0,
                details={},
                passed=False
            ),
            EvaluationMetric(
                name="test3",
                category=MetricType.QUALITY,
                weight=0.2,
                score=1.0,
                max_score=1.0,
                details={},
                passed=True
            )
        ]
        
        overall_score = evaluator._calculate_overall_score(metrics)
        expected_score = (0.8 * 0.5 + 0.6 * 0.3 + 1.0 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(overall_score - expected_score) < 0.001
    
    def test_recommendations_generation(self, evaluator):
        """Test recommendations generation"""
        test_case = evaluator.test_cases["sales_analysis"]
        
        # Create metrics with some failures
        metrics = [
            EvaluationMetric(
                name="graph_validity",
                category=MetricType.STRUCTURAL,
                weight=0.1,
                score=0.5,
                max_score=1.0,
                details={},
                passed=False
            ),
            EvaluationMetric(
                name="objective_achievement",
                category=MetricType.FUNCTIONAL,
                weight=0.2,
                score=0.6,
                max_score=1.0,
                details={},
                passed=False
            )
        ]
        
        recommendations = evaluator._generate_recommendations(metrics, test_case)
        assert len(recommendations) > 0
        assert any("graph structure" in rec.lower() for rec in recommendations)
        assert any("objective" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_evaluate_graph_planner(self, evaluator, mock_planner):
        """Test complete evaluation workflow"""
        result = await evaluator.evaluate_graph_planner(
            planner=mock_planner,
            test_case_name="sales_analysis",
            prompt_version="test"
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.test_case == "sales_analysis"
        assert result.prompt_version == "test"
        assert result.model_used == "test-model"
        assert 0.0 <= result.overall_score <= 1.0
        assert len(result.metrics) > 0
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_evaluate_with_invalid_test_case(self, evaluator, mock_planner):
        """Test evaluation with invalid test case name"""
        with pytest.raises(ValueError, match="Test case 'invalid' not found"):
            await evaluator.evaluate_graph_planner(
                planner=mock_planner,
                test_case_name="invalid",
                prompt_version="test"
            )


class TestPromptOptimizer:
    """Test suite for PromptOptimizer"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for prompt optimizer"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield GraphPlannerEvaluator(
                results_dir=temp_dir,
                enable_logging=False
            )
    
    @pytest.fixture
    def optimizer(self, evaluator):
        """Create prompt optimizer instance"""
        return PromptOptimizer(evaluator)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly"""
        assert optimizer.evaluator is not None
        assert len(optimizer.prompt_variants) == 0
    
    def test_add_prompt_variant(self, optimizer):
        """Test adding prompt variants"""
        optimizer.add_prompt_variant("test1", "Test prompt 1")
        optimizer.add_prompt_variant("test2", "Test prompt 2")
        
        assert len(optimizer.prompt_variants) == 2
        assert optimizer.prompt_variants["test1"] == "Test prompt 1"
        assert optimizer.prompt_variants["test2"] == "Test prompt 2"
    
    def test_analyze_ab_results(self, optimizer):
        """Test A/B results analysis"""
        # Create mock results
        mock_results = {
            "variant1": [
                Mock(overall_score=0.8),
                Mock(overall_score=0.75),
                Mock(overall_score=0.85)
            ],
            "variant2": [
                Mock(overall_score=0.9),
                Mock(overall_score=0.88),
                Mock(overall_score=0.92)
            ]
        }
        
        analysis = optimizer.analyze_ab_results(mock_results)
        
        assert "variant1" in analysis
        assert "variant2" in analysis
        assert "best_variant" in analysis
        
        assert analysis["best_variant"] == "variant2"
        assert abs(analysis["variant1"]["mean_score"] - 0.8) < 0.001
        assert abs(analysis["variant2"]["mean_score"] - 0.9) < 0.001
        assert analysis["variant1"]["sample_size"] == 3


# Integration test function (runs if executed directly)
async def run_integration_test():
    """Run a simple integration test"""
    print("🧪 Running GraphPlanner Evaluation Integration Test")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create evaluator
        evaluator = GraphPlannerEvaluator(
            results_dir=temp_dir,
            enable_logging=True
        )
        
        # Create a simple mock planner that generates a basic graph
        mock_planner = Mock()
        mock_planner.model = "integration-test-model"
        
        # Create a simple test graph
        test_graph = APIGraph(nodes=[
            APINode(id="input", type="InputNode", data={"name": "data"}),
            APINode(id="output", type="OutputNode", data={"name": "result"})
        ], edges=[])
        mock_planner.graph = test_graph
        
        async def mock_create_graph(context):
            yield Mock()
        
        mock_planner.create_graph = mock_create_graph
        
        # Run evaluation
        result = await evaluator.evaluate_graph_planner(
            planner=mock_planner,
            test_case_name="sales_analysis",
            prompt_version="integration_test"
        )
        
        print(f"✅ Integration test completed successfully!")
        print(f"   - Overall Score: {result.overall_score:.2%}")
        print(f"   - Metrics Count: {len(result.metrics)}")
        print(f"   - Execution Time: {result.execution_time:.2f}s")
        print(f"   - Results saved to: {temp_dir}")
        
        return True


if __name__ == "__main__":
    # Run integration test when executed directly
    asyncio.run(run_integration_test())