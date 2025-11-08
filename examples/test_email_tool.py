import asyncio
from nodetool.chat.tools.email import SearchEmailTool
from nodetool.workflows.processing_context import ProcessingContext
import tiktoken
from nodetool.runtime.resources import ResourceScope


async def test_email_search():
    # Create a processing context with required environment variables
    context = ProcessingContext()

    # Initialize the search tool
    search_tool = SearchEmailTool(context.workspace_dir)

    # Set search parameters
    params = {"subject": "AINews", "since_hours_ago": 24, "max_results": 5}

    try:
        # Execute the search
        results = await search_tool.process(context, params)

        # Print results
        if isinstance(results, list):
            print(f"Found {len(results)} emails:")
            for mail in results:
                print("\n-------------------")
                print(f"Subject: {mail['subject']}")
                print(f"From: {mail['sender']}")
                print(f"Message ID: {mail['message_id']}")
                print(mail["body"])
                print(
                    f"Token count: {len(tiktoken.encoding_for_model('gpt-4o').encode(mail['body']))}"
                )
        else:
            print("Error:", results.get("error", "Unknown error occurred"))

    except Exception as e:
        print(f"An error occurred: {str(e)}")


async def main():
    async with ResourceScope():
        await test_email_search()


if __name__ == "__main__":
    asyncio.run(main())
