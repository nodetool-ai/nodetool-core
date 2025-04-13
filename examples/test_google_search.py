import os
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.google import GoogleGroundedSearchTool


async def test_google_search():
    """
    Simple test function to demonstrate the GoogleGroundedSearchTool functionality.
    Run this directly to test the tool with a sample query.
    """
    # Create a workspace directory
    workspace_dir = "./workspace"
    os.makedirs(workspace_dir, exist_ok=True)

    # Initialize the tool
    search_tool = GoogleGroundedSearchTool(workspace_dir)

    # Create a simple processing context
    context = ProcessingContext()

    # Define a search query
    search_params = {"query": "What are the latest developments in AI?"}

    print(f"Searching for: {search_params['query']}")

    # Execute the search
    try:
        results = await search_tool.process(context, search_params)

        # Print the results
        print("\n=== SEARCH RESULTS ===")
        print(results["results"])

        print("\n=== SOURCES ===")
        for source in results["sources"]:
            print(f"- {source['title']}: {source['url']}")

    except Exception as e:
        print(f"Error during search: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_google_search())
