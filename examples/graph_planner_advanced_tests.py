"""Advanced integration examples for GraphPlanner with complex workflows"""

import asyncio
from typing import Any

from nodetool.agents.graph_planner import GraphPlanner, print_visual_graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.types import Chunk, PlanningUpdate
from nodetool.providers.huggingface_provider import HuggingFaceProvider

# Set up logging
from nodetool.config.logging_config import get_logger

logger = get_logger(__name__)

provider = HuggingFaceProvider("cerebras")
model = "openai/gpt-oss-120b"


async def create_and_execute_workflow(
    objective: str,
    inputs: dict[str, Any],
    max_planning_attempts: int = 3,
):
    """Create and execute a workflow graph for the given objective"""

    # Create GraphPlanner
    graph_planner = GraphPlanner(
        provider=provider,
        model=model,
        objective=objective,
        verbose=True,
    )

    # Plan the graph
    logger.info(f"Planning workflow for: {objective}")
    context = ProcessingContext()

    attempt = 0
    while attempt < max_planning_attempts:
        try:
            async for update in graph_planner.create_graph(context):
                if isinstance(update, PlanningUpdate):
                    logger.info(f"Planning: {update.phase} - {update.status}")
                elif isinstance(update, Chunk):
                    logger.debug(f"Received chunk: {update.content}")

            if not graph_planner.graph:
                raise ValueError("Failed to create workflow graph")

            break
        except Exception as e:
            attempt += 1
            logger.warning(f"Planning attempt {attempt} failed: {e}")
            if attempt >= max_planning_attempts:
                raise ValueError(
                    f"Failed to create workflow after {max_planning_attempts} attempts: {e}"
                )

    assert graph_planner.graph is not None
    print_visual_graph(graph_planner.graph)
    graph = graph_planner.graph
    logger.info(f"Generated workflow has {len(graph.nodes)} nodes")

    req = RunJobRequest(
        graph=graph,
        params=inputs,
    )

    logger.info("Executing workflow")
    async for msg in run_workflow(req, context=context):
        logger.info(f"Workflow message: {msg}")


async def vector_search_and_analysis_workflow():
    """Example: Create a vector database workflow with semantic search and analysis"""

    objective = """
    Create a workflow to build and query a vector database:
    1. Take a list of text documents as input
    2. Create a Chroma collection called "documents"
    3. Index all the documents into the collection with embeddings
    4. Take a search query and perform hybrid search (semantic + keyword)
    5. Extract the top 3 most relevant documents
    6. Deduplicate any overlapping results
    7. Count the total number of unique results found
    8. Output both the search results and the count
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={
                "documents": [
                    "Python is a high-level programming language known for its simplicity and readability.",
                    "Machine learning algorithms require large datasets to train effectively.",
                    "Vector databases enable semantic search capabilities for AI applications.",
                    "Natural language processing helps computers understand human text.",
                    "Deep learning models use neural networks with multiple layers.",
                    "Python libraries like NumPy and Pandas are essential for data science.",
                    "Transformer models revolutionized natural language understanding.",
                    "Vector embeddings represent text as numerical vectors in high-dimensional space.",
                ],
                "search_query": "What programming languages are good for AI and machine learning?",
            },
        )
    except Exception as e:
        logger.error(f"Vector search workflow failed: {e}", exc_info=True)


async def data_processing_pipeline_workflow():
    """Example: Complex data processing with filtering, chunking, and aggregation"""

    objective = """
    Create a comprehensive data processing pipeline:
    1. Take a large list of product data (dictionaries with name, price, category, rating)
    2. Filter products with rating >= 4.0
    3. Filter products with price between $10 and $100
    4. Group the filtered products by category using list chunking
    5. Calculate the average price for each category group
    6. Find products that exist in "electronics" category but not in "books" category
    7. Sort the final results and remove any duplicates
    8. Output the processed data with category averages and difference analysis
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={
                "products": [
                    {
                        "name": "Laptop",
                        "price": 899.99,
                        "category": "electronics",
                        "rating": 4.5,
                    },
                    {
                        "name": "Mouse",
                        "price": 25.99,
                        "category": "electronics",
                        "rating": 4.2,
                    },
                    {
                        "name": "Book: Python Guide",
                        "price": 29.99,
                        "category": "books",
                        "rating": 4.7,
                    },
                    {
                        "name": "Headphones",
                        "price": 79.99,
                        "category": "electronics",
                        "rating": 4.1,
                    },
                    {
                        "name": "Tablet",
                        "price": 299.99,
                        "category": "electronics",
                        "rating": 3.9,
                    },
                    {
                        "name": "Keyboard",
                        "price": 45.99,
                        "category": "electronics",
                        "rating": 4.3,
                    },
                    {
                        "name": "Book: AI Basics",
                        "price": 39.99,
                        "category": "books",
                        "rating": 4.6,
                    },
                    {
                        "name": "Monitor",
                        "price": 199.99,
                        "category": "electronics",
                        "rating": 4.4,
                    },
                    {
                        "name": "Phone Case",
                        "price": 15.99,
                        "category": "electronics",
                        "rating": 3.8,
                    },
                    {
                        "name": "Book: Data Science",
                        "price": 49.99,
                        "category": "books",
                        "rating": 4.8,
                    },
                ]
            },
        )
    except Exception as e:
        logger.error(f"Data processing pipeline failed: {e}", exc_info=True)


async def agent_research_workflow():
    """Example: Agent-based research and content generation"""

    objective = """
    Create an autonomous research agent workflow:
    1. Take a research topic as input
    2. Use an Agent node to research the topic comprehensively
    3. Configure the agent with appropriate tools for web search and analysis
    4. Set the agent's objective to gather key facts, recent developments, and expert opinions
    5. Have the agent output structured findings in JSON format
    6. Take the agent's JSON output and extract key insights using text processing
    7. Generate a formatted summary report combining all findings
    8. Output both the raw research data and the formatted summary
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={
                "research_topic": "Latest developments in large language models and their practical applications in 2024",
                "max_agent_steps": 15,
                "output_format": "json",
            },
        )
    except Exception as e:
        logger.error(f"Agent research workflow failed: {e}", exc_info=True)


async def conditional_logic_workflow():
    """Example: Complex conditional logic with boolean operations"""

    objective = """
    Create a conditional logic processing workflow:
    1. Take multiple boolean conditions and numeric thresholds as input
    2. Evaluate if ALL conditions in a group are True using boolean logic
    3. Check if ANY condition in another group is True
    4. Combine results using logical AND and OR operations
    5. Based on the boolean results, filter a list of items conditionally
    6. Apply different processing paths depending on the boolean outcomes
    7. Use conditional branching to determine final output format
    8. Output results with decision reasoning included
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={
                "primary_conditions": [True, True, False, True],
                "secondary_conditions": [False, False, True],
                "data_items": ["item1", "item2", "item3", "item4", "item5"],
                "numeric_threshold": 75.0,
                "test_values": [80.5, 65.2, 90.1, 45.8, 82.7],
                "require_all_primary": True,
                "require_any_secondary": True,
            },
        )
    except Exception as e:
        logger.error(f"Conditional logic workflow failed: {e}", exc_info=True)


async def text_analysis_and_generation_workflow():
    """Example: Advanced text processing, analysis, and generation"""

    objective = """
    Create an advanced text analysis and generation workflow:
    1. Take multiple text documents as input
    2. Split each document into smaller text chunks for processing
    3. Analyze each chunk for key themes and sentiment
    4. Filter chunks based on relevance criteria using regex patterns
    5. Concatenate the most relevant chunks together
    6. Use the filtered content to generate a comprehensive summary
    7. Create multiple format outputs (markdown, plain text, structured data)
    8. Validate the generated content meets quality criteria
    9. Output the analysis results, summary, and formatted versions
    """

    try:
        await create_and_execute_workflow(
            objective=objective,
            inputs={
                "documents": [
                    "Artificial intelligence is transforming industries across the globe. From healthcare to finance, AI applications are becoming more sophisticated and widespread. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy.",
                    "The future of work is being reshaped by automation and AI technologies. While some jobs may become obsolete, new opportunities are emerging in AI development, data science, and human-AI collaboration. Organizations must adapt their workforce strategies to remain competitive.",
                    "Climate change represents one of the most pressing challenges of our time. Technological solutions, including AI-powered climate modeling and renewable energy optimization, offer hope for mitigation and adaptation strategies. Urgent action is needed across all sectors of society.",
                ],
                "relevance_keywords": [
                    "AI",
                    "technology",
                    "future",
                    "innovation",
                    "transformation",
                ],
                "chunk_size": 50,
                "min_relevance_score": 0.6,
                "output_formats": ["markdown", "json", "txt"],
            },
        )
    except Exception as e:
        logger.error(f"Text analysis workflow failed: {e}", exc_info=True)


async def run_all_advanced_tests():
    """Run all advanced test workflows"""

    test_workflows = [
        ("Vector Search and Analysis", vector_search_and_analysis_workflow),
        ("Data Processing Pipeline", data_processing_pipeline_workflow),
        # ("Agent Research", agent_research_workflow),
        ("Conditional Logic", conditional_logic_workflow),
        ("Text Analysis and Generation", text_analysis_and_generation_workflow),
    ]

    for test_name, test_func in test_workflows:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Workflow")
        print(f"{'='*60}")

        try:
            await test_func()
            print(f"✅ {test_name} completed successfully")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            logger.error(f"{test_name} workflow failed", exc_info=True)

        print(f"{'='*60}\n")

    print("🎉 All advanced test workflows completed!")


if __name__ == "__main__":
    # You can run a specific test or all tests
    # asyncio.run(vector_search_and_analysis_workflow())
    # asyncio.run(data_processing_pipeline_workflow())
    # asyncio.run(agent_research_workflow())
    asyncio.run(run_all_advanced_tests())
