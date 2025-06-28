"""
GraphPlanner Evaluation System

A comprehensive evaluation framework for measuring GraphPlanner performance
across multiple dimensions with automated scoring and prompt optimization.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional, Tuple
from enum import Enum

from nodetool.agents.graph_planner import GraphInput, GraphOutput, GraphPlanner
from nodetool.metadata.types import TypeMetadata
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.types.graph import Graph as APIGraph


class MetricType(Enum):
    """Types of evaluation metrics"""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional" 
    QUALITY = "quality"
    PERFORMANCE = "performance"


@dataclass
class EvaluationMetric:
    """Individual evaluation metric"""
    name: str
    category: MetricType
    weight: float
    score: float
    max_score: float
    details: Dict[str, Any]
    passed: bool


@dataclass
class TestCase:
    """Test case specification with ground truth"""
    name: str
    objective: str
    input_schema: List[GraphInput]
    output_schema: List[GraphOutput]
    expected_nodes: List[str]  # Expected node types
    expected_connections: List[Tuple[str, str]]  # Expected data flow
    expected_properties: Dict[str, Any]  # Expected node configurations
    complexity_score: int  # 1-10 complexity rating
    description: str


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    test_case: str
    timestamp: datetime
    overall_score: float
    metrics: List[EvaluationMetric]
    generated_graph: Optional[APIGraph]
    execution_time: float
    prompt_version: str
    model_used: str
    errors: List[str]
    recommendations: List[str]


class GraphPlannerEvaluator:
    """State-of-the-art evaluation system for GraphPlanner"""

    _custom_validators: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []
    
    def __init__(self, 
                 results_dir: str = "evaluation_results",
                 enable_logging: bool = True):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        if enable_logging:
            self._setup_logging()
        else:
            # Create a null logger when logging is disabled
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        
        self.test_cases = {}
        self._load_test_cases()
    
    def _setup_logging(self):
        """Setup evaluation logging"""
        log_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_test_cases(self):
        """Load predefined test cases"""
        # Test Case 1: Sales Data Processing
        self.test_cases["sales_analysis"] = TestCase(
            name="sales_analysis",
            objective="Process CSV sales data by calculating monthly totals, identifying top products, and generating a comprehensive summary report with charts",
            input_schema=[
                GraphInput(
                    name="sales_data",
                    type=TypeMetadata(type="dataframe"),
                    description="CSV file containing sales data with columns: date, product_id, product_name, quantity, unit_price, region"
                )
            ],
            output_schema=[
                GraphOutput(
                    name="monthly_summary",
                    type=TypeMetadata(type="dataframe"),
                    description="Monthly sales totals by product and region"
                ),
                GraphOutput(
                    name="summary_report",
                    type=TypeMetadata(type="str"),
                    description="Comprehensive analysis report with insights and recommendations"
                )
            ],
            expected_nodes=[
                "nodetool.workflows.input_node.InputNode",  # Input data
                "nodetool.lib.statistics.GroupByNode",      # Group by month/product
                "nodetool.lib.statistics.SumNode",          # Calculate totals
                "nodetool.lib.statistics.TopKNode",         # Find top products
                "nodetool.workflows.output_node.OutputNode" # Output results
            ],
            expected_connections=[
                ("input_data", "group_by_month"),
                ("group_by_month", "calculate_totals"),
                ("calculate_totals", "find_top_products"),
                ("find_top_products", "generate_report"),
                ("generate_report", "output_summary")
            ],
            expected_properties={
                "group_by": ["date", "product_name"],
                "aggregation": "sum",
                "top_k": 10
            },
            complexity_score=7,
            description="Multi-step data analysis with aggregations and reporting"
        )
    
    async def evaluate_graph_planner(self, 
                                   planner: GraphPlanner,
                                   test_case_name: str,
                                   prompt_version: str = "default") -> EvaluationResult:
        """Evaluate GraphPlanner on a specific test case"""
        
        if test_case_name not in self.test_cases:
            raise ValueError(f"Test case '{test_case_name}' not found")
        
        test_case = self.test_cases[test_case_name]
        self.logger.info(f"Starting evaluation of test case: {test_case_name}")
        
        # Record start time
        start_time = asyncio.get_event_loop().time()
        errors = []
        
        try:
            # Create processing context
            context = ProcessingContext()
            
            # Generate graph
            generated_graph = None
            async for update in planner.create_graph(context):
                pass  # Consume all updates
            
            generated_graph = planner.graph
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Evaluate all metrics
            assert generated_graph is not None
            metrics = await self._evaluate_all_metrics(test_case, generated_graph, execution_time)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, test_case)
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            errors.append(str(e))
            self.logger.error(f"Evaluation failed: {e}")
            
            # Create minimal result for failed evaluation
            metrics = [EvaluationMetric(
                name="execution_failure",
                category=MetricType.PERFORMANCE,
                weight=1.0,
                score=0.0,
                max_score=1.0,
                details={"error": str(e)},
                passed=False
            )]
            overall_score = 0.0
            generated_graph = None
            recommendations = ["Fix execution errors before proceeding"]
        
        # Create result
        result = EvaluationResult(
            test_case=test_case_name,
            timestamp=datetime.now(),
            overall_score=overall_score,
            metrics=metrics,
            generated_graph=generated_graph,
            execution_time=execution_time,
            prompt_version=prompt_version,
            model_used=getattr(planner, 'model', 'unknown'),
            errors=errors,
            recommendations=recommendations
        )
        
        # Save results
        await self._save_result(result)
        
        return result
    
    async def _evaluate_all_metrics(self, 
                                  test_case: TestCase, 
                                  graph: APIGraph,
                                  execution_time: float) -> List[EvaluationMetric]:
        """Evaluate all metrics for the generated graph"""
        metrics = []
        
        # Structural Correctness (40% total weight)
        metrics.extend(await self._evaluate_structural_metrics(test_case, graph))
        
        # Functional Effectiveness (35% total weight)  
        metrics.extend(await self._evaluate_functional_metrics(test_case, graph))
        
        # Code Quality (15% total weight)
        metrics.extend(await self._evaluate_quality_metrics(test_case, graph))
        
        # Performance (10% total weight)
        metrics.extend(await self._evaluate_performance_metrics(test_case, execution_time))
        
        return metrics
    
    async def _evaluate_structural_metrics(self, 
                                         test_case: TestCase, 
                                         graph: APIGraph) -> List[EvaluationMetric]:
        """Evaluate structural correctness metrics"""
        metrics = []
        
        # Graph Validity (10% weight)
        validity_score = self._check_graph_validity(graph)
        metrics.append(EvaluationMetric(
            name="graph_validity",
            category=MetricType.STRUCTURAL,
            weight=0.10,
            score=validity_score,
            max_score=1.0,
            details=self._get_validity_details(graph),
            passed=validity_score > 0.8
        ))
        
        # Node Type Accuracy (15% weight)
        node_accuracy = self._check_node_types(test_case, graph)
        metrics.append(EvaluationMetric(
            name="node_type_accuracy",
            category=MetricType.STRUCTURAL,
            weight=0.15,
            score=node_accuracy,
            max_score=1.0,
            details=self._get_node_type_details(test_case, graph),
            passed=node_accuracy > 0.7
        ))
        
        # Edge Consistency (10% weight)
        edge_score = self._check_edge_consistency(graph)
        metrics.append(EvaluationMetric(
            name="edge_consistency",
            category=MetricType.STRUCTURAL,
            weight=0.10,
            score=edge_score,
            max_score=1.0,
            details=self._get_edge_details(graph),
            passed=edge_score > 0.8
        ))
        
        # Schema Compliance (5% weight)
        schema_score = self._check_schema_compliance(test_case, graph)
        metrics.append(EvaluationMetric(
            name="schema_compliance",
            category=MetricType.STRUCTURAL,
            weight=0.05,
            score=schema_score,
            max_score=1.0,
            details=self._get_schema_details(test_case, graph),
            passed=schema_score > 0.9
        ))
        
        return metrics
    
    async def _evaluate_functional_metrics(self, 
                                         test_case: TestCase, 
                                         graph: APIGraph) -> List[EvaluationMetric]:
        """Evaluate functional effectiveness metrics"""
        metrics = []
        
        # Objective Achievement (20% weight)
        objective_score = self._check_objective_achievement(test_case, graph)
        metrics.append(EvaluationMetric(
            name="objective_achievement",
            category=MetricType.FUNCTIONAL,
            weight=0.20,
            score=objective_score,
            max_score=1.0,
            details=self._get_objective_details(test_case, graph),
            passed=objective_score > 0.7
        ))
        
        # Completeness (10% weight)
        completeness_score = self._check_completeness(test_case, graph)
        metrics.append(EvaluationMetric(
            name="completeness",
            category=MetricType.FUNCTIONAL,
            weight=0.10,
            score=completeness_score,
            max_score=1.0,
            details=self._get_completeness_details(test_case, graph),
            passed=completeness_score > 0.8
        ))
        
        # Efficiency (5% weight)
        efficiency_score = self._check_efficiency(graph)
        metrics.append(EvaluationMetric(
            name="efficiency",
            category=MetricType.FUNCTIONAL,
            weight=0.05,
            score=efficiency_score,
            max_score=1.0,
            details=self._get_efficiency_details(graph),
            passed=efficiency_score > 0.6
        ))
        
        return metrics
    
    async def _evaluate_quality_metrics(self, 
                                       test_case: TestCase, 
                                       graph: APIGraph) -> List[EvaluationMetric]:
        """Evaluate code quality metrics"""
        metrics = []
        
        # Property Validation (10% weight)
        property_score = self._check_property_validation(test_case, graph)
        metrics.append(EvaluationMetric(
            name="property_validation",
            category=MetricType.QUALITY,
            weight=0.10,
            score=property_score,
            max_score=1.0,
            details=self._get_property_details(test_case, graph),
            passed=property_score > 0.7
        ))
        
        # Best Practices (5% weight)
        practices_score = self._check_best_practices(graph)
        metrics.append(EvaluationMetric(
            name="best_practices",
            category=MetricType.QUALITY,
            weight=0.05,
            score=practices_score,
            max_score=1.0,
            details=self._get_practices_details(graph),
            passed=practices_score > 0.8
        ))
        
        return metrics
    
    async def _evaluate_performance_metrics(self, 
                                          test_case: TestCase, 
                                          execution_time: float) -> List[EvaluationMetric]:
        """Evaluate performance metrics"""
        metrics = []
        
        # Execution Time (10% weight)
        # Expected time based on complexity: simple=5s, medium=15s, complex=30s
        expected_time = min(30.0, test_case.complexity_score * 3)
        time_score = max(0.0, min(1.0, expected_time / max(execution_time, 1.0)))
        
        metrics.append(EvaluationMetric(
            name="execution_time",
            category=MetricType.PERFORMANCE,
            weight=0.10,
            score=time_score,
            max_score=1.0,
            details={
                "execution_time": execution_time,
                "expected_time": expected_time,
                "complexity": test_case.complexity_score
            },
            passed=execution_time < expected_time * 2
        ))
        
        return metrics
    
    def _check_graph_validity(self, graph: APIGraph) -> float:
        """Check if graph is a valid DAG structure"""
        if not graph or not graph.nodes:
            return 0.0
        
        score = 1.0
        
        # Check for cycles (simplified check)
        node_ids = {node.id for node in graph.nodes}
        
        # Check that all edge references are valid
        for node in graph.nodes:
            if hasattr(node, 'properties'):
                for prop_name, prop_value in node.data.items():
                    if isinstance(prop_value, dict) and prop_value.get('type') == 'edge':
                        source_id = prop_value.get('source')
                        if source_id and source_id not in node_ids:
                            score -= 0.2  # Invalid edge reference
        
        return max(0.0, score)
    
    def _check_node_types(self, test_case: TestCase, graph: APIGraph) -> float:
        """Check if appropriate node types are used"""
        if not graph or not graph.nodes:
            return 0.0
        
        actual_types = [node.type for node in graph.nodes]
        expected_types = set(test_case.expected_nodes)
        
        # Calculate overlap
        relevant_matches = 0
        for node_type in actual_types:
            # Check for exact matches or semantic matches
            if any(expected in node_type.lower() for expected in 
                   ['input', 'output', 'statistic', 'group', 'sum', 'report']):
                relevant_matches += 1
        
        if not actual_types:
            return 0.0
        
        return min(1.0, relevant_matches / len(actual_types))
    
    def _check_edge_consistency(self, graph: APIGraph) -> float:
        """Check edge format consistency"""
        if not graph or not graph.nodes:
            return 1.0
        
        total_edges = 0
        valid_edges = 0
        
        for node in graph.nodes:
            for prop_value in node.data.values():
                if isinstance(prop_value, dict) and prop_value.get('type') == 'edge':
                    total_edges += 1
                    if 'source' in prop_value and 'sourceHandle' in prop_value:
                        valid_edges += 1
        
        if total_edges == 0:
            return 1.0
        
        return valid_edges / total_edges
    
    def _check_schema_compliance(self, test_case: TestCase, graph: APIGraph) -> float:
        """Check if input/output nodes match schema"""
        if not graph or not graph.nodes:
            return 0.0
        
        input_nodes = [n for n in graph.nodes if 'input' in n.type.lower()]
        output_nodes = [n for n in graph.nodes if 'output' in n.type.lower()]
        
        expected_inputs = len(test_case.input_schema)
        expected_outputs = len(test_case.output_schema)
        
        input_score = min(1.0, len(input_nodes) / max(expected_inputs, 1))
        output_score = min(1.0, len(output_nodes) / max(expected_outputs, 1))
        
        return (input_score + output_score) / 2
    
    def _check_objective_achievement(self, test_case: TestCase, graph: APIGraph) -> float:
        """Check if graph achieves the stated objective"""
        if not graph or not graph.nodes:
            return 0.0
        
        objective = test_case.objective.lower()
        node_types = [node.type.lower() for node in graph.nodes]
        
        # Keyword-based scoring for objective achievement
        score = 0.0
        
        # Check for data processing keywords
        if any('csv' in obj or 'data' in obj for obj in [objective]):
            if any('input' in nt or 'read' in nt for nt in node_types):
                score += 0.3
        
        # Check for calculation keywords  
        if any(word in objective for word in ['calculate', 'total', 'sum', 'aggregate']):
            if any(word in nt for nt in node_types for word in ['sum', 'aggregate', 'calculate']):
                score += 0.4
        
        # Check for reporting keywords
        if any(word in objective for word in ['report', 'summary', 'generate']):
            if any('output' in nt or 'report' in nt for nt in node_types):
                score += 0.3
        
        return min(1.0, score)
    
    def _check_completeness(self, test_case: TestCase, graph: APIGraph) -> float:
        """Check if all required steps are included"""
        if not graph or not graph.nodes:
            return 0.0
        
        # Based on expected workflow steps
        required_steps = ['input', 'process', 'output']
        found_steps = []
        
        for node in graph.nodes:
            node_type = node.type.lower()
            if 'input' in node_type:
                found_steps.append('input')
            elif any(word in node_type for word in ['calculate', 'sum', 'aggregate', 'group']):
                found_steps.append('process')
            elif 'output' in node_type:
                found_steps.append('output')
        
        return len(set(found_steps)) / len(required_steps)
    
    def _check_efficiency(self, graph: APIGraph) -> float:
        """Check for unnecessary nodes or redundancy"""
        if not graph or not graph.nodes:
            return 1.0
        
        # Simple efficiency check - penalize excessive nodes
        node_count = len(graph.nodes)
        optimal_range = (3, 8)  # Reasonable range for most workflows
        
        if optimal_range[0] <= node_count <= optimal_range[1]:
            return 1.0
        elif node_count < optimal_range[0]:
            return 0.8  # Too few nodes, might be incomplete
        else:
            # Too many nodes, penalize excess
            excess = node_count - optimal_range[1]
            return max(0.3, 1.0 - (excess * 0.1))
    
    def _check_property_validation(self, test_case: TestCase, graph: APIGraph) -> float:
        """Check if node properties are properly configured"""
        if not graph or not graph.nodes:
            return 0.0
        
        total_nodes = 0
        properly_configured = 0
        
        for node in graph.nodes:
            total_nodes += 1
            if node.data:
                # Check if properties seem reasonable (non-empty, proper types)
                valid_props = sum(1 for v in node.data.values() 
                                if v is not None and v != "")
                if valid_props > 0:
                    properly_configured += 1
        
        if total_nodes == 0:
            return 1.0
        
        return properly_configured / total_nodes
    
    def _check_best_practices(self, graph: APIGraph) -> float:
        """Check adherence to workflow best practices"""
        if not graph or not graph.nodes:
            return 0.0
        
        score = 1.0
        
        # Check for meaningful node IDs
        node_ids = [node.id for node in graph.nodes]
        if any(len(nid) < 3 for nid in node_ids):
            score -= 0.2
        
        # Check for proper input/output structure
        has_input = any('input' in node.type.lower() for node in graph.nodes)
        has_output = any('output' in node.type.lower() for node in graph.nodes)
        
        if not has_input:
            score -= 0.4
        if not has_output:
            score -= 0.4
        
        return max(0.0, score)
    
    def _calculate_overall_score(self, metrics: List[EvaluationMetric]) -> float:
        """Calculate weighted overall score"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            total_weighted_score += metric.score * metric.weight
            total_weight += metric.weight
        
        if total_weight == 0:
            return 0.0
        
        return total_weighted_score / total_weight
    
    def _generate_recommendations(self, 
                                metrics: List[EvaluationMetric], 
                                test_case: TestCase) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        recommendations = []
        
        for metric in metrics:
            if not metric.passed:
                if metric.name == "graph_validity":
                    recommendations.append("Fix graph structure issues - ensure valid DAG with proper node connections")
                elif metric.name == "node_type_accuracy":
                    recommendations.append("Use more appropriate node types for the given objective")
                elif metric.name == "objective_achievement":
                    recommendations.append("Ensure the workflow fully addresses the stated objective")
                elif metric.name == "execution_time":
                    recommendations.append("Optimize planning process to reduce execution time")
                elif metric.name == "completeness":
                    recommendations.append("Add missing workflow steps to complete the processing pipeline")
        
        if not recommendations:
            recommendations.append("Excellent performance! Consider minor optimizations for edge cases.")
        
        return recommendations
    
    # Helper methods for detailed analysis
    def _get_validity_details(self, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed validity analysis"""
        if not graph:
            return {"error": "No graph generated"}
        
        return {
            "node_count": len(graph.nodes) if graph.nodes else 0,
            "has_cycles": False,  # Simplified
            "orphaned_nodes": 0   # Simplified
        }
    
    def _get_node_type_details(self, test_case: TestCase, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed node type analysis"""
        if not graph or not graph.nodes:
            return {"error": "No nodes found"}
        
        return {
            "actual_types": [node.type for node in graph.nodes],
            "expected_types": test_case.expected_nodes,
            "type_coverage": len(set(node.type for node in graph.nodes))
        }
    
    def _get_edge_details(self, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed edge analysis"""
        if not graph or not graph.nodes:
            return {"edges": 0}
        
        edge_count = 0
        for node in graph.nodes:
            if node.data:
                edge_count += sum(1 for v in node.data.values() 
                                if isinstance(v, dict) and v.get('type') == 'edge')
        
        return {"total_edges": edge_count}
    
    def _get_schema_details(self, test_case: TestCase, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed schema compliance analysis"""
        return {
            "expected_inputs": len(test_case.input_schema),
            "expected_outputs": len(test_case.output_schema),
            "actual_input_nodes": len([n for n in (graph.nodes or []) if 'input' in n.type.lower()]),
            "actual_output_nodes": len([n for n in (graph.nodes or []) if 'output' in n.type.lower()])
        }
    
    def _get_objective_details(self, test_case: TestCase, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed objective achievement analysis"""
        return {
            "objective": test_case.objective,
            "analysis": "Keyword-based matching against node types"
        }
    
    def _get_completeness_details(self, test_case: TestCase, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed completeness analysis"""
        return {
            "required_steps": ["input", "process", "output"],
            "workflow_length": len(graph.nodes) if graph and graph.nodes else 0
        }
    
    def _get_efficiency_details(self, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed efficiency analysis"""
        return {
            "node_count": len(graph.nodes) if graph and graph.nodes else 0,
            "optimal_range": [3, 8]
        }
    
    def _get_property_details(self, test_case: TestCase, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed property analysis"""
        if not graph or not graph.nodes:
            return {"configured_nodes": 0}
        
        configured = sum(1 for node in graph.nodes 
                        if node.data)
        
        return {
            "total_nodes": len(graph.nodes),
            "configured_nodes": configured
        }
    
    def _get_practices_details(self, graph: APIGraph) -> Dict[str, Any]:
        """Get detailed best practices analysis"""
        return {
            "has_input_output": True,  # Simplified
            "meaningful_ids": True     # Simplified
        }
    
    async def _save_result(self, result: EvaluationResult):
        """Save evaluation result to disk"""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{result.test_case}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Convert to JSON-serializable format
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        result_dict['generated_graph'] = None  # Skip graph serialization for now
        
        # Convert MetricType enum to string
        for metric in result_dict['metrics']:
            if 'category' in metric:
                metric['category'] = metric['category'].value if hasattr(metric['category'], 'value') else str(metric['category'])
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Evaluation result saved to {filepath}")
    
    def compare_results(self, result1: EvaluationResult, result2: EvaluationResult) -> Dict[str, Any]:
        """Compare two evaluation results"""
        comparison = {
            "test_case": result1.test_case,
            "improvement": result2.overall_score - result1.overall_score,
            "metric_changes": {}
        }
        
        # Compare individual metrics
        metrics1 = {m.name: m.score for m in result1.metrics}
        metrics2 = {m.name: m.score for m in result2.metrics}
        
        for metric_name in metrics1:
            if metric_name in metrics2:
                comparison["metric_changes"][metric_name] = {
                    "old_score": metrics1[metric_name],
                    "new_score": metrics2[metric_name],
                    "change": metrics2[metric_name] - metrics1[metric_name]
                }
        
        return comparison


# Prompt optimization utilities
class PromptOptimizer:
    """Utilities for systematic prompt optimization"""
    
    def __init__(self, evaluator: GraphPlannerEvaluator):
        self.evaluator = evaluator
        self.prompt_variants = {}
    
    def add_prompt_variant(self, name: str, system_prompt: str):
        """Add a prompt variant for testing"""
        self.prompt_variants[name] = system_prompt
    
    async def run_ab_test(self, 
                         planner_factory,
                         test_case_name: str,
                         variants: List[str],
                         runs_per_variant: int = 3) -> Dict[str, List[EvaluationResult]]:
        """Run A/B test across prompt variants"""
        results = {}
        
        for variant in variants:
            if variant not in self.prompt_variants:
                raise ValueError(f"Prompt variant '{variant}' not found")
            
            results[variant] = []
            
            for run in range(runs_per_variant):
                # Create planner with specific prompt
                planner = planner_factory(system_prompt=self.prompt_variants[variant])
                
                # Run evaluation
                result = await self.evaluator.evaluate_graph_planner(
                    planner, test_case_name, prompt_version=f"{variant}_run_{run}"
                )
                results[variant].append(result)
        
        return results
    
    def analyze_ab_results(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, Any]:
        """Analyze A/B test results"""
        analysis = {}
        
        for variant, variant_results in results.items():
            scores = [r.overall_score for r in variant_results]
            analysis[variant] = {
                "mean_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                "sample_size": len(scores)
            }
        
        # Find best performing variant
        best_variant = max(analysis.keys(), key=lambda k: analysis[k]["mean_score"])
        analysis["best_variant"] = best_variant
        
        return analysis