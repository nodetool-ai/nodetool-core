#!/usr/bin/env python3
"""
GraphPlanner Evaluation Demo

Demonstrates the state-of-the-art evaluation system for GraphPlanner
with a comprehensive test case and prompt optimization.

Usage:
    python evaluate_graph_planner.py [--prompt-variant VARIANT] [--output-dir DIR]
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any

from nodetool.agents.graph_planner import GraphPlanner
from nodetool.agents.graph_planner_evaluator import GraphPlannerEvaluator, PromptOptimizer
from nodetool.chat.providers.anthropic_provider import AnthropicProvider
from typing import List, Dict


# Default system prompts for A/B testing
DEFAULT_PROMPTS = {
    "baseline": """You are an expert workflow designer specializing in creating efficient data processing graphs.

Your task is to design a workflow graph that accomplishes the given objective using the available node types.

Key principles:
1. Use appropriate node types for each step
2. Ensure proper data flow between nodes
3. Include all necessary processing steps
4. Optimize for clarity and efficiency

Create a complete workflow that transforms the input data according to the objective.""",

    "detailed": """You are a senior data engineering consultant with expertise in workflow automation and graph-based processing systems.

Your mission is to design a comprehensive, production-ready workflow graph that efficiently accomplishes the stated objective.

Follow these detailed guidelines:

ANALYSIS PHASE:
- Carefully analyze the objective to identify all required processing steps
- Consider data transformations, aggregations, and output requirements
- Plan the most logical sequence of operations

DESIGN PHASE:
- Select the most appropriate node types for each operation
- Ensure type compatibility between connected nodes
- Minimize unnecessary processing steps while maintaining completeness
- Design for scalability and maintainability

VALIDATION PHASE:
- Verify that all input requirements are met
- Confirm that all output specifications will be satisfied
- Check for potential data flow bottlenecks or issues

Create a workflow that demonstrates best practices in data processing pipeline design.""",

    "concise": """Design an efficient workflow graph for the given objective.

Requirements:
- Use minimal necessary nodes
- Ensure proper data flow
- Meet all input/output specifications
- Focus on core functionality

Create a clean, streamlined processing pipeline."""
}


class GraphPlannerDemo:
    """Demonstration of GraphPlanner evaluation system"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.evaluator = GraphPlannerEvaluator(results_dir=output_dir)
        self.optimizer = PromptOptimizer(self.evaluator)
        
        # Valid node types from nodetool-base metadata
        self.valid_node_types = {
            # Calendly
            "calendly.events.ListScheduledEvents", "calendly.events.ScheduledEventFields",
            
            # Chroma
            "chroma.chroma_node.Chroma", "chroma.collections.Collection", "chroma.collections.Count",
            "chroma.collections.GetDocuments", "chroma.collections.Peek", "chroma.index.IndexAggregatedText",
            "chroma.index.IndexEmbedding", "chroma.index.IndexImage", "chroma.index.IndexImages",
            "chroma.index.IndexString", "chroma.index.IndexTextChunk", "chroma.index.IndexTextChunks",
            "chroma.query.HybridSearch", "chroma.query.QueryImage", "chroma.query.QueryText",
            "chroma.query.RemoveOverlap",
            
            # Google
            "google.image_generation.ImageGeneration",
            
            # OpenAI
            "openai.audio.TextToSpeech", "openai.audio.Transcribe", "openai.audio.Translate",
            "openai.image.CreateImage", "openai.text.Embedding", "openai.text.WebSearch",
            
            # Core nodetool types (sample of key ones)
            "nodetool.agents.agent_node.Agent", "nodetool.audio.analysis.AudioAnalysis",
            "nodetool.audio.convert.ConvertAudio", "nodetool.constant.boolean.Boolean",
            "nodetool.constant.float.Float", "nodetool.constant.integer.Integer",
            "nodetool.constant.string.String", "nodetool.image.enhance.Enhance",
            "nodetool.image.generate.CreateImage", "nodetool.input.image.ImageInput",
            "nodetool.input.text.TextInput", "nodetool.llms.anthropic.text.AnthropicText",
            "nodetool.llms.openai.text.OpenAIText", "nodetool.math.binary_ops.Add",
            "nodetool.math.binary_ops.Subtract", "nodetool.math.binary_ops.Multiply",
            "nodetool.output.image.ImageOutput", "nodetool.output.text.TextOutput",
            "nodetool.text.combine.CombineText", "nodetool.text.split.SplitText"
        }
        
        # Add prompt variants
        for name, prompt in DEFAULT_PROMPTS.items():
            self.optimizer.add_prompt_variant(name, prompt)
        
        # Add node type validation
        self._add_node_type_validation()
    
    def _add_node_type_validation(self):
        """Add node type validation to the evaluator"""
        def validate_node_types(graph_data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate that all node types in the graph are valid"""
            invalid_types = []
            valid_types = []
            
            for node in graph_data.get("nodes", []):
                node_type = node.get("type")
                if node_type:
                    if node_type in self.valid_node_types:
                        valid_types.append(node_type)
                    else:
                        invalid_types.append(node_type)
            
            total_nodes = len(valid_types) + len(invalid_types)
            validity_score = len(valid_types) / total_nodes if total_nodes > 0 else 0.0
            
            return {
                "valid_types": valid_types,
                "invalid_types": invalid_types,
                "validity_score": validity_score,
                "total_nodes": total_nodes
            }
        
        # Add custom validation to evaluator
        if hasattr(self.evaluator, '_custom_validators'):
            self.evaluator._custom_validators.append(validate_node_types)
        else:
            self.evaluator._custom_validators = [validate_node_types]
    
    def create_planner(self, 
                      model: str = "claude-sonnet-4-20250514",
                      system_prompt: str | None = None) -> GraphPlanner:
        """Create a GraphPlanner instance"""
        
        # Use the sales analysis test case
        test_case = self.evaluator.test_cases["sales_analysis"]
        
        return GraphPlanner(
            provider=AnthropicProvider(),
            model=model,
            objective=test_case.objective,
            input_schema=test_case.input_schema,
            output_schema=test_case.output_schema,
            system_prompt=system_prompt,
            verbose=True
        )
    
    async def run_single_evaluation(self, prompt_variant: str = "baseline") -> Dict[str, Any]:
        """Run a single evaluation with specified prompt variant"""
        
        print(f"\n🔍 Running evaluation with prompt variant: {prompt_variant}")
        print("=" * 60)
        
        # Create planner with specific prompt
        system_prompt = DEFAULT_PROMPTS.get(prompt_variant)
        planner = self.create_planner(system_prompt=system_prompt)
        
        # Run evaluation
        try:
            result = await self.evaluator.evaluate_graph_planner(
                planner=planner,
                test_case_name="sales_analysis",
                prompt_version=prompt_variant
            )
            
            # Display results
            self._display_result(result)
            
            return {
                "success": True,
                "result": result,
                "overall_score": result.overall_score
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "overall_score": 0.0
            }
    
    async def run_ab_test(self) -> Dict[str, Any]:
        """Run A/B test across all prompt variants"""
        
        print("\n🧪 Running A/B Test Across Prompt Variants")
        print("=" * 60)
        
        variants = list(DEFAULT_PROMPTS.keys())
        
        try:
            # Run A/B test
            results = await self.optimizer.run_ab_test(
                planner_factory=lambda system_prompt: self.create_planner(system_prompt=system_prompt),
                test_case_name="sales_analysis",
                variants=variants,
                runs_per_variant=2  # Reduced for demo
            )
            
            # Analyze results
            analysis = self.optimizer.analyze_ab_results(results)
            
            # Display analysis
            self._display_ab_analysis(analysis, results)
            
            return {
                "success": True,
                "analysis": analysis,
                "raw_results": results
            }
            
        except Exception as e:
            print(f"❌ A/B test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _display_result(self, result):
        """Display evaluation result in a readable format"""
        
        print(f"\n📊 Evaluation Results for: {result.test_case}")
        print(f"📅 Timestamp: {result.timestamp}")
        print(f"🎯 Overall Score: {result.overall_score:.2%}")
        print(f"⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"🤖 Model: {result.model_used}")
        print(f"📝 Prompt Version: {result.prompt_version}")
        
        if result.errors:
            print(f"\n❌ Errors:")
            for error in result.errors:
                print(f"   • {error}")
        
        # Group metrics by category
        metric_categories = {}
        for metric in result.metrics:
            category = metric.category.value
            if category not in metric_categories:
                metric_categories[category] = []
            metric_categories[category].append(metric)
        
        print(f"\n📈 Detailed Metrics:")
        for category, metrics in metric_categories.items():
            print(f"\n  {category.upper()}:")
            for metric in metrics:
                status = "✅" if metric.passed else "❌"
                score_pct = metric.score / metric.max_score * 100
                weight_pct = metric.weight * 100
                print(f"    {status} {metric.name}: {score_pct:.1f}% (weight: {weight_pct:.1f}%)")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"   {i}. {rec}")
    
    def _display_ab_analysis(self, analysis: Dict[str, Any], raw_results: Dict[str, Any]):
        """Display A/B test analysis"""
        
        print(f"\n🏆 Best Performing Variant: {analysis['best_variant']}")
        
        print(f"\n📊 Performance Summary:")
        print(f"{'Variant':<12} {'Mean Score':<12} {'Std Dev':<10} {'Min':<8} {'Max':<8} {'Runs':<6}")
        print("-" * 60)
        
        for variant, stats in analysis.items():
            if variant == "best_variant":
                continue
            
            print(f"{variant:<12} {stats['mean_score']:<12.3f} {stats['std_dev']:<10.3f} "
                  f"{stats['min_score']:<8.3f} {stats['max_score']:<8.3f} {stats['sample_size']:<6}")
        
        # Show improvement over baseline
        if "baseline" in analysis:
            baseline_score = analysis["baseline"]["mean_score"]
            print(f"\n📈 Performance vs Baseline:")
            
            for variant, stats in analysis.items():
                if variant in ["best_variant", "baseline"]:
                    continue
                
                improvement = stats["mean_score"] - baseline_score
                improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
                
                direction = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"   {direction} {variant}: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    
    async def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report_lines = [
            "# GraphPlanner Evaluation Report",
            f"Generated: {asyncio.get_event_loop().time()}",
            "",
            "## Executive Summary",
            ""
        ]
        
        if "analysis" in results:
            analysis = results["analysis"]
            best_variant = analysis["best_variant"]
            best_score = analysis[best_variant]["mean_score"]
            
            report_lines.extend([
                f"- **Best Performing Prompt**: {best_variant}",
                f"- **Peak Performance Score**: {best_score:.2%}",
                f"- **Evaluation Framework**: Multi-dimensional (Structural, Functional, Quality, Performance)",
                ""
            ])
        
        report_lines.extend([
            "## Key Findings",
            "",
            "### Prompt Optimization Results",
            ""
        ])
        
        if "analysis" in results:
            for variant, stats in analysis.items():
                if variant == "best_variant":
                    continue
                
                report_lines.extend([
                    f"**{variant.title()} Prompt**:",
                    f"- Mean Score: {stats['mean_score']:.2%}",
                    f"- Performance Range: {stats['min_score']:.2%} - {stats['max_score']:.2%}",
                    f"- Consistency (Std Dev): {stats['std_dev']:.3f}",
                    ""
                ])
        
        report_lines.extend([
            "## Recommendations",
            "",
            "### For Prompt Engineering:",
            "1. Use the best-performing prompt variant as baseline",
            "2. Focus on detailed instruction clarity for complex workflows", 
            "3. Include explicit validation steps in system prompts",
            "",
            "### For System Improvement:",
            "1. Monitor structural correctness metrics closely",
            "2. Enhance node type selection accuracy",
            "3. Optimize execution time for complex workflows",
            "",
            "---",
            "*Report generated by GraphPlanner Evaluation System*"
        ])
        
        return "\n".join(report_lines)


async def main():
    """Main evaluation demonstration"""
    
    parser = argparse.ArgumentParser(description="GraphPlanner Evaluation Demo")
    parser.add_argument("--prompt-variant", 
                       choices=list(DEFAULT_PROMPTS.keys()),
                       default="baseline",
                       help="Prompt variant to test")
    parser.add_argument("--output-dir", 
                       default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--mode",
                       choices=["single", "ab-test", "both"],
                       default="single",
                       help="Evaluation mode")
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = GraphPlannerDemo(output_dir=args.output_dir)
    
    print("🚀 GraphPlanner Evaluation System Demo")
    print("=" * 60)
    print(f"Output Directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    
    results = {}
    
    # Run evaluations based on mode
    if args.mode in ["single", "both"]:
        print(f"\n📍 Running single evaluation with {args.prompt_variant} prompt...")
        single_result = await demo.run_single_evaluation(args.prompt_variant)
        results["single"] = single_result
    
    if args.mode in ["ab-test", "both"]:
        print(f"\n📍 Running A/B test across all prompt variants...")
        ab_result = await demo.run_ab_test()
        results["ab_test"] = ab_result
    
    # Generate comprehensive report
    if results:
        print(f"\n📄 Generating evaluation report...")
        report_content = await demo.generate_report(results)
        
        # Save report
        report_path = Path(args.output_dir) / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        print(f"✅ Report saved to: {report_path}")
        
        # Save raw results
        results_path = Path(args.output_dir) / "raw_results.json"
        with open(results_path, "w") as f:
            # Convert non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Raw results saved to: {results_path}")
    
    print(f"\n🎉 Evaluation completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    # Set up event loop for asyncio
    asyncio.run(main())