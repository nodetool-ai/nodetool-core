#!/usr/bin/env python3
"""
Product Hunt AI Product Extractor Agent

This script creates an agent that:
1. Takes a Product Hunt monthly leaderboard URL.
2. Uses BrowserTool to visit the leaderboard and then individual product pages.
3. Analyzes product descriptions and tags to identify AI-powered products.
4. Extracts information about these AI products.
5. Organizes and saves the results as a structured Markdown report.

"""

import asyncio

# import json # Retained if complex JSON manipulation were needed later

from nodetool.agents.agent import Agent
from nodetool.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.ui.console import AgentConsole
from nodetool.runtime.resources import ResourceScope


async def test_product_hunt_ai_extractor_agent(
    provider: BaseProvider,
    model: str,
    product_hunt_archive_url: str,
):
    """
    Initializes and runs the Product Hunt AI Product Extractor agent.

    Args:
        provider: The chat provider (e.g., OpenAI, Gemini).
        model: The primary model for generation.
        product_hunt_archive_url: The URL of the Product Hunt monthly leaderboard to scan.
    """
    context = ProcessingContext()
    # The product_hunt_archive_url is injected into the agent's objective.

    ai_product_extractor_agent = Agent(
        name="ProductHuntAIProductExtractor",
        objective=f"""
        You are an expert data analyst specializing in identifying AI-powered products from Product Hunt leaderboards.
        Your mission is to:

        1.  Start with the given Product Hunt monthly leaderboard URL: {product_hunt_archive_url}
            Use the BrowserTool to load and parse this main leaderboard page. Your goal is to identify all listed products and extract the links to their individual Product Hunt pages.

        2.  For each unique product URL identified on the leaderboard:
            a.  Use the BrowserTool to visit the product's individual Product Hunt page.
            b.  Extract key information: Product Name, Tagline, the main Description (usually found near the top or under a 'Description' heading), and any Tags associated with the product. Focus on the primary content areas.
            c.  Analyze the extracted text (description, tagline, tags) to determine if the product is an "AI product" or heavily utilizes AI.
                Look for explicit mentions of AI technologies (e.g., "AI", "Artificial Intelligence", "Machine Learning", "Deep Learning", "LLM", "GPT", "Generative AI", "Neural Network", "Computer Vision", "NLP", "transformers")
                OR descriptions of features that are clearly AI-driven (e.g., intelligent automation, smart recommendations, pattern recognition, natural language understanding, image generation from text, automated content creation, predictive analytics).
            d.  If the Product Hunt page is unavailable or the content is very sparse and doesn't allow for a confident AI assessment, you may perform a single, targeted Google search for "[Product Name] AI features" or "[Product Name] Product Hunt" to gather more context. Prioritize information directly from the Product Hunt ecosystem if available. Note if external sources were used.

        3.  If a product is determined to be AI-related based on the analysis:
            a.  Gather the Product Name.
            b.  Gather the Product Hunt URL for the individual product.
            c.  Create a concise "AI Focus Summary" (1-2 sentences maximum, explaining how AI is used or what AI features it has, based on the gathered information).
            d.  List any "AI Keywords/Tags" found or inferred that confirm its AI nature (e.g., AI, Machine Learning, NLP, Generative AI).
            e.  Determine the "Archive Source Month/Year" by parsing it from the initial URL ({product_hunt_archive_url}). For example, if the URL contains '/2025/4', the source is 'April 2025'.

        4.  Generate a single, consolidated Markdown report listing ALL identified AI products from the given archive URL.
            Each AI product should be a distinct section in the report. The report must strictly follow the format specified in the output_schema.
            If no AI products are found after thoroughly checking all products on the page, the report should clearly state this as per the schema.

        # Output Schema:
        A Markdown report. Each identified AI product should follow this approximate structure:

        ## AI Product: [Product Name]
        **Product Hunt URL:** [Link to Product Hunt Page]
        **AI Focus Summary:** [Brief description of its AI capabilities or how AI is central to the product. Max 2 sentences.]
        **AI Keywords/Tags:**
        * Keyword/Tag 1
        * Keyword/Tag 2
        **Archive Source Month/Year:** [e.g., April 2025]
        ---

        If no AI products are found, the report should state *exactly*:
        No AI products were identified in the [Month Year] Product Hunt archive.
        (Replace [Month Year] with the actual month and year from the source URL)
        """,
        provider=provider,
        model=model,
        enable_analysis_phase=True,
        enable_data_contracts_phase=True,  # Crucial for structured output
        display_manager=AgentConsole(),
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
    )

    print(f"Starting agent: {ai_product_extractor_agent.name}")
    print(f"Processing Product Hunt Archive URL: {product_hunt_archive_url}")
    print("Objective: To identify AI products and generate a Markdown report.\n")

    # Stream the output as it's generated
    async for item in ai_product_extractor_agent.execute(context):
        pass

    print(ai_product_extractor_agent.get_results())

    print("\n\n--- Agent execution finished ---")
    print(
        f"Workspace Directory for any artifacts (logs, intermediate files): {context.workspace_dir}"
    )
    # The full report should have been printed to stdout.
    # If the agent saves the final report to a file, it would typically be in the workspace_dir.


async def main():
    # Ensure you have your API key (e.g., OPENAI_API_KEY) set in your environment variables
    # or configure the provider appropriately within the nodetool library.

    # The user requested URL: https://www.producthunt.com/leaderboard/monthly/2025/4
    target_ph_url = "https://www.producthunt.com/leaderboard/monthly/2025/4"

    # To test with a page that likely has content:
    # target_ph_url = "https://www.producthunt.com/leaderboard/monthly/2024/04" # April 2024
    # target_ph_url = "https://www.producthunt.com/leaderboard/monthly/2024/01" # January 2024

    print(f"Attempting to process: {target_ph_url}")
    print("Please ensure your API keys for the chosen provider are correctly set.")

    async with ResourceScope():
        await test_product_hunt_ai_extractor_agent(
            provider=await get_provider(
                Provider.HuggingFaceCerebras  # pyright: ignore[reportCallIssue]
            ),  # Specify your provider: Provider.OpenAI, Provider.Gemini, Provider.Anthropic
            model="openai/gpt-oss-120b",
            product_hunt_archive_url=target_ph_url,
        )


if __name__ == "__main__":
    asyncio.run(main())
