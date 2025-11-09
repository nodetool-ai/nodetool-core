import asyncio
from nodetool.agents.simple_agent import SimpleAgent
from nodetool.providers import get_provider
from nodetool.agents.tools import GoogleSearchTool, BrowserTool
from nodetool.providers.base import BaseProvider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk
from nodetool.runtime.resources import ResourceScope
import dotenv

dotenv.load_dotenv()


async def run_simple_agent_test(provider: BaseProvider, model: str):
    """
    Tests the SimpleAgent with a basic objective.
    """
    context = ProcessingContext()
    objective = (
        "What is the weather in London according to https://www.bbc.com/weather/2643743"
    )

    print(f"Objective: {objective}")

    simple_agent = SimpleAgent(
        name="Weather Reporter Agent",
        objective=objective,
        provider=provider,
        model=model,
        tools=[
            BrowserTool(),
        ],
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
        max_iterations=10,
    )

    try:
        async for item in simple_agent.execute(context):
            if isinstance(item, Chunk):
                print(item.content, end="", flush=True)

    except Exception as e:
        print(f"\nError during execution: {e}")

    results = simple_agent.get_results()
    print(f"\nFinal Results:\n{results}")


async def main():
    async with ResourceScope():
        await run_simple_agent_test(
            provider=await get_provider(Provider.OpenAI),
            model="gpt-4o-mini",
            # provider=await get_provider(Provider.Ollama), model="mistral"
            # provider=await get_provider(Provider.Groq), model="mixtral-8x7b-32768"
        )


if __name__ == "__main__":
    asyncio.run(main())
