import asyncio
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
import json
from pathlib import Path


async def test_instagram_scraper_task(
    provider: ChatProvider,
    model: str,
):
    # 1. Set up workspace directory
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # 3. Set up browser and search tools
    tools = [
        BrowserTool(),
        GoogleSearchTool(),
    ]

    # 4. Create a sample task
    task = Task(
        title="Instagram Trends Collection",
        description="Collect current trends, hashtags, and viral content from Instagram",
        subtasks=[],  # We'll add the subtask directly to the SubTaskContext
    )

    # 5. Create a sample subtask
    subtask = SubTask(
        content="""
        Use google and browser tools to search for Instagram trends.
        Find example posts for each trend.
        Create a summary of the trends, hashtags, and viral content.
        Return all trends you can find.
        """,
        output_file="instagram_trends_detailed.json",
        input_files=[],
        output_type="json",
        output_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "trends": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "hashtag": {"type": "string"},
                                "description": {"type": "string"},
                                "popularity_score": {
                                    "type": "string",
                                    "description": "e.g., High, Medium, Low or a numeric score",
                                },
                                "example_posts": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "post_url": {"type": "string"},
                                            "caption": {"type": "string"},
                                            "like_count": {"type": "integer"},
                                            "comment_count": {"type": "integer"},
                                        },
                                        "required": ["post_url"],
                                    },
                                },
                            },
                            "required": ["hashtag", "description"],
                        },
                    },
                },
            },
        ),
    )

    # Add the subtask to the task
    task.subtasks = [subtask]

    # 6. Create the SubTaskContext
    subtask_context = SubTaskContext(
        task=task,
        subtask=subtask,
        processing_context=ProcessingContext(workspace_dir=workspace_dir),
        tools=tools,
        model=model,
        provider=provider,
        max_iterations=20,
    )

    # 7. Execute the subtask
    async for event in subtask_context.execute():
        if isinstance(event, Chunk):
            print(event.content, end="")
        elif isinstance(event, TaskUpdate):
            print(f"Task Update: {event.event}")

    # Check if output file was created
    output_path = Path(workspace_dir) / subtask.output_file
    if output_path.exists():
        with open(output_path, "r") as f:
            result = json.load(f)
        print("\nSubTask Execution Successful!")
        print("\nOutput File Content:")
        print(json.dumps(result, indent=2))
    else:
        print("\nOutput file was not created!")


if __name__ == "__main__":
    asyncio.run(
        test_instagram_scraper_task(
            provider=get_provider(Provider.OpenAI), model="gpt-4o-mini"
        )
    )
    # asyncio.run(
    #     test_instagram_scraper_task(
    #         provider=get_provider(Provider.Gemini), model="gemini-2.0-flash"
    #     )
    # )

    # asyncio.run(
    #     test_instagram_scraper_task(
    #         provider=get_provider(Provider.Anthropic),
    #         model="claude-3-5-sonnet-20241022",
    #     )
    # )
