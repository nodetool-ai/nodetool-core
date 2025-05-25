import asyncio
from nodetool.agents.simple_agent import SimpleAgent
from nodetool.chat.providers import get_provider
from nodetool.agents.tools import GoogleSearchTool, BrowserTool
from nodetool.chat.providers.base import ChatProvider
from nodetool.metadata.types import Provider, ToolCall, SubTask
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk, TaskUpdate
import dotenv

dotenv.load_dotenv()


async def run_simple_agent_test(provider: ChatProvider, model: str):
    """
    Tests the SimpleAgent with a basic objective.
    """
    context = ProcessingContext()
    objective = "Find out what the current weather in London is and provide a short description."

    print(f"Objective: {objective}")

    simple_agent = SimpleAgent(
        name="Weather Reporter Agent",
        objective=objective,
        provider=provider,
        model=model,
        tools=[
            GoogleSearchTool(),
            BrowserTool(),
        ],
        output_type="json",
        output_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "string"},
                "description": {"type": "string"},
                "source_url": {
                    "type": "string",
                    "description": "URL where the weather info was found",
                },
            },
            "required": ["location", "temperature", "description"],
        },
        input_files=[],
        max_iterations=10,
        system_prompt="You are a helpful assistant. When searching for weather, try to find a reliable source and extract the key information: location, temperature, and a brief description. Also, provide the URL of the source.",
    )

    print(f"--- Running SimpleAgent: {simple_agent.name} ---")
    try:
        async for item in simple_agent.execute(context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)
            elif isinstance(item, TaskUpdate):
                subtask_info = ""
                if item.subtask and isinstance(item.subtask, SubTask):
                    subtask_info = f" for subtask (first 30 chars): '{item.subtask.content[:30]}...'"
                print(f"\n[Task Update: Event='{item.event.value}'{subtask_info}]")
            elif isinstance(item, ToolCall):
                print(f"\n[Tool Call: {item.name} with args {item.args}]")
            # else:
            #     print(f"\n[Unknown item type: {type(item)} - {item}]")

    except Exception as e:
        print(f"\n--- Error during execution: {e} ---")
        import traceback

        traceback.print_exc()

    print("\n--- Execution Finished ---")

    results = simple_agent.get_results()
    print(f"\nFinal Results:\n{results}")
    print(f"\nWorkspace Directory: {context.workspace_dir}")

    if results:
        print("SimpleAgent smoke test: Agent produced a result.")
    else:
        print(
            "SimpleAgent smoke test: Agent did not produce a result or the result was None."
        )


if __name__ == "__main__":
    # Example usage with OpenAI
    # Ensure your .env file has OPENAI_API_KEY set for this to work
    print("Starting SimpleAgent smoke test...")
    asyncio.run(
        run_simple_agent_test(
            provider=get_provider(Provider.OpenAI),
            model="gpt-4o-mini",
            # provider=get_provider(Provider.Ollama), model="mistral"
            # provider=get_provider(Provider.Groq), model="mixtral-8x7b-32768"
        )
    )
    print("SimpleAgent smoke test finished.")
