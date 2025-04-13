import asyncio
import json
from nodetool.chat.providers.gemini_provider import GeminiProvider
from nodetool.metadata.types import Message
from nodetool.agents.task_planner import CreateTaskTool, DEFAULT_PLANNING_SYSTEM_PROMPT


async def main():
    provider = GeminiProvider()

    messages = [
        Message(
            role="user",
            content="""
            Create the US GDP for 1950-2000
            """,
        ),
    ]

    # # Get final response using the provider
    # response = await provider.generate_message(
    #     messages=messages,
    #     model="gemini-2.0-flash",
    #     response_format={
    #         "type": "json_schema",
    #         "json_schema": {
    #             "name": "GDP",
    #             "strict": True,
    #             "schema": {
    #                 "type": "object",
    #                 "properties": {
    #                     "data": {
    #                         "type": "array",
    #                         "items": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "year": {"type": "integer"},
    #                                 "gdp": {"type": "number"},
    #                             },
    #                             "required": ["year", "gdp"],
    #                         },
    #                     }
    #                 },
    #                 "required": ["data"],
    #             },
    #         },
    #     },
    # )
    # assert response.content
    # print(json.loads(str(response.content)))  # type: ignore

    create_task_tool = CreateTaskTool(workspace_dir="")
    response_schema = create_task_tool.input_schema
    messages = [
        Message(
            role="system",
            content=DEFAULT_PLANNING_SYSTEM_PROMPT,
        ),
        Message(
            role="user",
            content="""
            Create a task plan to create a website for a company that sells AI tools
            """,
        ),
    ]

    response = await provider.generate_message(
        messages=messages,
        model="gemini-2.0-flash",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TaskPlan",
                "strict": True,
                "schema": response_schema,
            },
        },
    )
    assert response.content
    print(json.loads(str(response.content)))  # type: ignore


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
