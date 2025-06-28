# GraphPlanner Evaluation System

A comprehensive, state-of-the-art evaluation framework for measuring and optimizing GraphPlanner performance across multiple dimensions.

## Overview

The GraphPlanner Evaluation System provides automated, quantitative assessment of workflow graph generation quality with built-in prompt optimization capabilities. It evaluates generated graphs across four key dimensions: structural correctness, functional effectiveness, code quality, and performance.

## Features

### 🎯 Multi-Dimensional Evaluation
- **Structural Correctness (40%)**: Graph validity, node types, edge consistency, schema compliance
- **Functional Effectiveness (35%)**: Objective achievement, completeness, efficiency
- **Code Quality (15%)**: Property validation, best practices adherence
- **Performance (10%)**: Execution time relative to complexity

### 🧪 Prompt Optimization
- A/B testing framework for prompt variants
- Statistical analysis of performance differences
- Automated recommendation generation

### 📊 Comprehensive Reporting
- Detailed metric breakdowns with pass/fail status
- Performance comparisons across prompt variants
- Actionable improvement recommendations

### 🔧 Extensible Architecture
- Easy addition of new test cases and metrics
- Configurable evaluation weights
- Pluggable prompt optimization strategies

## Quick Start

### Basic Evaluation

```python
from nodetool.agents.graph_planner_evaluator import GraphPlannerEvaluator
from nodetool.agents.graph_planner import GraphPlanner
from nodetool.chat.anthropic import AnthropicProvider

# Create evaluator
evaluator = GraphPlannerEvaluator(results_dir="evaluation_results")

# Create planner
planner = GraphPlanner(
    provider=AnthropicProvider(),
    model="claude-3-5-sonnet-20241022",
    objective="Process CSV sales data and generate summary report",
    input_schema=[...],
    output_schema=[...]
)

# Run evaluation
result = await evaluator.evaluate_graph_planner(
    planner=planner,
    test_case_name="sales_analysis",
    prompt_version="baseline"
)

print(f"Overall Score: {result.overall_score:.2%}")
```

### Prompt Optimization

```python
from nodetool.agents.graph_planner_evaluator import PromptOptimizer

# Create optimizer
optimizer = PromptOptimizer(evaluator)

# Add prompt variants
optimizer.add_prompt_variant("baseline", "Standard system prompt...")
optimizer.add_prompt_variant("detailed", "Detailed system prompt...")

# Run A/B test
results = await optimizer.run_ab_test(
    planner_factory=lambda prompt: create_planner(system_prompt=prompt),
    test_case_name="sales_analysis", 
    variants=["baseline", "detailed"],
    runs_per_variant=3
)

# Analyze results
analysis = optimizer.analyze_ab_results(results)
print(f"Best variant: {analysis['best_variant']}")
```

### Command Line Interface

```bash
# Run single evaluation
python examples/evaluate_graph_planner.py --prompt-variant baseline

# Run A/B test across all variants
python examples/evaluate_graph_planner.py --mode ab-test

# Run both single and A/B test
python examples/evaluate_graph_planner.py --mode both --output-dir my_results
```

## Test Cases

### Sales Analysis Test Case

**Objective**: "Process CSV sales data by calculating monthly totals, identifying top products, and generating a comprehensive summary report with charts"

**Expected Workflow**:
1. Input CSV data with sales records
2. Group data by month and product
3. Calculate aggregated totals
4. Identify top-performing products
5. Generate formatted summary report

**Evaluation Criteria**:
- Proper data ingestion nodes
- Appropriate aggregation operations
- Correct data flow connections
- Output formatting compliance

## Evaluation Metrics

### Structural Correctness (40% total)

| Metric | Weight | Description |
|--------|--------|-------------|
| Graph Validity | 10% | Valid DAG structure, no cycles |
| Node Type Accuracy | 15% | Appropriate node types for tasks |
| Edge Consistency | 10% | Proper edge format and connections |
| Schema Compliance | 5% | Input/output nodes match requirements |

### Functional Effectiveness (35% total)

| Metric | Weight | Description |
|--------|--------|-------------|
| Objective Achievement | 20% | Workflow solves stated problem |
| Completeness | 10% | All required steps included |
| Efficiency | 5% | Minimal unnecessary operations |

### Code Quality (15% total)

| Metric | Weight | Description |
|--------|--------|-------------|
| Property Validation | 10% | Correct parameter configurations |
| Best Practices | 5% | Adherence to workflow conventions |

### Performance (10% total)

| Metric | Weight | Description |
|--------|--------|-------------|
| Execution Time | 10% | Planning time relative to complexity |

## Scoring System

- **Score Range**: 0.0 - 1.0 (0% - 100%)
- **Pass Threshold**: Varies by metric (typically 70-90%)
- **Weighted Calculation**: Metrics combined using specified weights
- **Grade Scale**:
  - 90-100%: Excellent
  - 80-89%: Good 
  - 70-79%: Satisfactory
  - 60-69%: Needs Improvement
  - Below 60%: Poor

## Adding Custom Test Cases

```python
from nodetool.agents.graph_planner_evaluator import TestCase
from nodetool.api.types.workflow import GraphInput, GraphOutput
from nodetool.metadata.types import TypeMetadata

# Define custom test case
custom_test = TestCase(
    name="custom_analysis",
    objective="Your custom objective here",
    input_schema=[
        GraphInput(
            name="input_data",
            type=TypeMetadata(type="dataframe"),
            description="Input description"
        )
    ],
    output_schema=[
        GraphOutput(
            name="results",
            type=TypeMetadata(type="string"), 
            description="Output description"
        )
    ],
    expected_nodes=["InputNode", "ProcessingNode", "OutputNode"],
    expected_connections=[("input", "process"), ("process", "output")],
    expected_properties={"param": "value"},
    complexity_score=5,
    description="Test case description"
)

# Add to evaluator
evaluator.test_cases["custom_analysis"] = custom_test
```

## Extending Evaluation Metrics

```python
async def _evaluate_custom_metrics(self, test_case, graph, execution_time):
    """Add custom evaluation metrics"""
    metrics = []
    
    # Custom metric example
    custom_score = self._check_custom_requirement(graph)
    metrics.append(EvaluationMetric(
        name="custom_requirement",
        category=MetricType.FUNCTIONAL,
        weight=0.05,
        score=custom_score,
        max_score=1.0,
        details={"custom_info": "details"},
        passed=custom_score > 0.8
    ))
    
    return metrics
```

## Results and Reporting

### Evaluation Results Structure

```python
@dataclass
class EvaluationResult:
    test_case: str                    # Test case name
    timestamp: datetime               # Evaluation timestamp
    overall_score: float              # Weighted overall score
    metrics: List[EvaluationMetric]   # Individual metric results
    generated_graph: APIGraph         # Generated workflow graph
    execution_time: float             # Planning execution time
    prompt_version: str               # Prompt variant used
    model_used: str                   # LLM model identifier
    errors: List[str]                 # Any execution errors
    recommendations: List[str]        # Improvement suggestions
```

### Output Files

The evaluation system generates several output files:

- `eval_{test_case}_{timestamp}.json`: Individual evaluation results
- `evaluation_report.md`: Comprehensive analysis report
- `raw_results.json`: Raw A/B test data
- `evaluation_{timestamp}.log`: Detailed execution logs

### Report Sections

1. **Executive Summary**: Key findings and best-performing variants
2. **Detailed Metrics**: Per-metric breakdowns and analysis
3. **Prompt Comparison**: Performance differences across variants
4. **Recommendations**: Specific improvement suggestions
5. **Raw Data**: Complete evaluation results for further analysis

## Best Practices

### For Effective Evaluation

1. **Run Multiple Evaluations**: Use several runs per prompt variant for statistical significance
2. **Test Edge Cases**: Include complex scenarios that stress-test the system
3. **Monitor Trends**: Track performance over time as prompts evolve
4. **Validate Manually**: Spot-check automated scores with human evaluation

### For Prompt Optimization

1. **Start Simple**: Begin with clear, straightforward prompts
2. **Add Specificity**: Include domain-specific instructions gradually
3. **Test Incrementally**: Make small changes and measure impact
4. **Document Changes**: Keep detailed records of prompt modifications

### For Custom Test Cases

1. **Clear Objectives**: Write specific, measurable objectives
2. **Realistic Complexity**: Match complexity scores to actual difficulty
3. **Comprehensive Coverage**: Include diverse workflow patterns
4. **Ground Truth**: Define expected outcomes precisely

## Troubleshooting

### Common Issues

**Low Structural Scores**:
- Check node registry availability
- Verify edge format consistency
- Ensure proper input/output nodes

**Poor Functional Scores**:
- Review objective clarity and specificity
- Check for missing workflow steps
- Validate data flow logic

**Execution Failures**:
- Verify provider configuration
- Check model availability
- Review input schema validity

### Performance Optimization

**Slow Evaluations**:
- Reduce runs per variant for initial testing
- Use simpler test cases for development
- Cache provider connections

**Memory Issues**:
- Process results incrementally
- Avoid storing large graphs in memory
- Use streaming evaluation for bulk tests

## Contributing

### Adding New Metrics

1. Implement metric calculation method
2. Add to appropriate evaluation phase
3. Define weight and threshold
4. Update documentation

### Improving Test Cases

1. Analyze real-world workflow patterns
2. Create diverse complexity scenarios
3. Validate expected outcomes
4. Test with multiple prompt variants

### Extending Prompt Optimization

1. Implement new optimization strategies
2. Add statistical analysis methods
3. Create visualization tools
4. Integrate with CI/CD pipelines

## API Reference

See the complete API documentation in the source code docstrings:
- `GraphPlannerEvaluator`: Main evaluation class
- `PromptOptimizer`: Prompt optimization utilities
- `TestCase`: Test case definition structure
- `EvaluationResult`: Results data structure
- `EvaluationMetric`: Individual metric structure

## License

This evaluation system is part of the nodetool-core project and follows the same licensing terms.