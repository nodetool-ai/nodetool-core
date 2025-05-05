import os
import asyncio
import json
import dotenv
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools import GoogleNewsTool

# Load environment variables from .env file
dotenv.load_dotenv()


async def test_google_news_search():
    """
    Simple test function to demonstrate the GoogleNewsTool functionality.
    Run this directly to test the tool with a sample query.
    Requires DATA_FOR_SEO_LOGIN and DATA_FOR_SEO_PASSWORD in the environment or .env file.
    """
    # Instantiate the tool
    news_tool = GoogleNewsTool()

    # Create a processing context
    # Workspace is not strictly needed for this tool but context is required
    context = ProcessingContext()

    # Define search parameters
    # Ensure required location and language parameters are set
    params = {
        "keyword": "latest AI advancements",
        "location_name": "United States",  # Example: Search in the US
        "language_name": "English",  # Example: Search for English news
        "sort_by": "date",  # Optional: Sort by date
        # "output_file": "ai_news_results.json" # Optional: save to file
    }

    print(
        f"Searching Google News for: '{params['keyword']}' in {params['location_name']} ({params['language_name']})"
    )

    # Execute the search
    try:
        results = await news_tool.process(context, params)

        # Print the results
        print("\n=== NEWS SEARCH RESULTS ===")
        # Use json.dumps for pretty printing the dictionary
        print(json.dumps(results, indent=2))

        if results.get("success"):
            print(
                f"\nSuccessfully retrieved {len(results.get('results', []))} news items."
            )
        else:
            print(f"\nSearch failed: {results.get('error')}")
            if "details" in results:
                print(f"Details: {results['details']}")

    except Exception as e:
        print(f"\nError during search: {e}")


if __name__ == "__main__":
    asyncio.run(test_google_news_search())
