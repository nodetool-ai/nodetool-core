import asyncio
from nodetool.chat.providers import get_provider

# Assuming ReasoningTool is available alongside other standard tools
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, Task, SubTask, ToolCall
from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
import json
from pathlib import Path


async def test_logical_puzzle_task(
    provider: ChatProvider,
    model: str,
):
    # 1. Set up workspace directory
    context = ProcessingContext()
    workspace_dir = context.workspace_dir

    # 3. Create a sample task focused on reasoning
    task = Task(
        title="Logical Puzzle Solving with ReasoningTool",
        subtasks=[],
    )

    # 4. Create a sample subtask for logical reasoning
    puzzle_statement = (
        "Use step by step reasoning to solve the following logical puzzle: "
        "Five friends (Alex, Ben, Clara, David, Emily) each own a different exotic pet (Chameleon, Fennec Fox, Axolotl, Sugar Glider, Hedgehog). "
        "Each pet has a unique name (Zephyr, Luna, Sparky, Bubbles, Quill). "
        "Each friend also has a favorite color (Red, Blue, Green, Yellow, Purple), and no two friends share the same favorite color. "
        "Use the clues below to determine which pet each friend owns, the pet's name, and each friend's favorite color: "
        "1. The Fennec Fox, which is not named Sparky, belongs to someone whose favorite color is Green. "
        "2. Alex's favorite color is not Red, and he does not own the Hedgehog. "
        "3. Bubbles is not the Chameleon, which is owned by David. "
        "4. The person who likes Yellow owns the Axolotl named Luna. "
        "5. Ben owns the Sugar Glider, and his favorite color is Purple. "
        "6. Emily's pet is named Quill, and it is not the Fennec Fox. "
        "7. The Hedgehog's owner does not like Blue. "
        "8. Zephyr is not owned by the person whose favorite color is Red. "
        "What are the details for each friend (pet, pet name, favorite color)?"
    )

    subtask = SubTask(
        content=puzzle_statement,
        output_file="solutions.json",
        input_files=[],
        output_type="json",
        output_schema=json.dumps(
            {
                "type": "object",
                "properties": {
                    "solution": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "friend_name": {"type": "string"},
                                "pet_type": {"type": "string"},
                                "pet_name": {"type": "string"},
                                "favorite_color": {"type": "string"},
                            },
                            "required": [
                                "friend_name",
                                "pet_type",
                                "pet_name",
                                "favorite_color",
                            ],
                        },
                    }
                },
                "required": ["solution"],
            }
        ),
    )

    # Add the subtask to the task
    task.subtasks = [subtask]

    # 5. Create the SubTaskContext
    subtask_context = SubTaskContext(
        task=task,
        subtask=subtask,
        processing_context=ProcessingContext(),
        tools=[],
        model=model,
        provider=provider,
        max_iterations=5,
    )

    # 6. Execute the subtask
    print(f"\n--- Starting Logical Puzzle Task using {model} ---")
    async for event in subtask_context.execute():
        if isinstance(event, Chunk):
            print(event.content, end="")
        elif isinstance(event, ToolCall):
            print(event.message, end="")
        elif isinstance(event, TaskUpdate):
            event_details = event.event
            # Ensure event_details are printable
            if not isinstance(
                event_details, (str, dict, list, int, float, bool, type(None))
            ):
                event_details = str(event_details)
            print(
                f"\nTask Update: {json.dumps(event_details, indent=2) if isinstance(event_details, (dict, list)) else event_details}"
            )

    # 7. Check if output file was created and print its content
    output_path = context.resolve_workspace_path(subtask.output_file)
    if Path(output_path).exists():
        with open(output_path, "r") as f:
            try:
                result = json.load(f)
                print("\n\n--- SubTask Execution Successful! ---")
                print("Output File Content:")
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                # If JSON parsing fails, print raw content for debugging
                f.seek(0)  # Go to the beginning of the file
                raw_content = f.read()
                print("\n\n--- SubTask Execution Generated Non-JSON Output ---")
                print(f"Content of {output_path}:")
                print(raw_content)

    else:
        print(f"\n\n--- Output file '{subtask.output_file}' was not created! ---")
        print(f"Contents of workspace directory '{workspace_dir}':")
        for item in Path(workspace_dir).iterdir():
            print(f"- {item.name}")


if __name__ == "__main__":
    # Example usage with OpenAI
    asyncio.run(
        test_logical_puzzle_task(
            provider=get_provider(Provider.OpenAI), model="o4-mini"
        )
    )
    asyncio.run(
        test_logical_puzzle_task(
            provider=get_provider(Provider.Ollama), model="qwen3:4b"
        )
    )

    # Example usage with Gemini (uncomment to run)
    # print("\n\n--------------------------------------------------\n")
    # asyncio.run(
    #     test_logical_puzzle_task(
    #         provider=get_provider(Provider.Gemini), model="gemini-1.5-flash-latest" # Or your preferred Gemini model
    #     )
    # )

    # Example usage with Anthropic (uncomment to run)
    # print("\n\n--------------------------------------------------\n")
    # asyncio.run(
    #    test_logical_puzzle_task(
    #        provider=get_provider(Provider.Anthropic),
    #        model="claude-3-5-sonnet-20240620", # Or your preferred Anthropic model
    #    )
    # )
