import asyncio
from nodetool.agents.graph_planner import GraphPlanner
from nodetool.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.runtime.resources import ResourceScope

async def test_safeguard():
    print("Testing GraphPlanner safeguard...")
    # This should fail with a helpful error message
    provider_coro = get_provider(Provider.HuggingFaceCerebras)
    try:
        GraphPlanner(
            provider=provider_coro,
            model="test",
            objective="test"
        )
        print("FAILED: Safeguard did not catch coroutine")
    except ValueError as e:
        print(f"SUCCESS: Caught expected error: {e}")
    except Exception as e:
        print(f"FAILED: Caught unexpected error: {type(e).__name__}: {e}")

async def test_correct_init():
    print("\nTesting correct initialization...")
    async with ResourceScope():
        provider = await get_provider(Provider.HuggingFaceCerebras)
        try:
            planner = GraphPlanner(
                provider=provider,
                model="test",
                objective="test"
            )
            print(f"SUCCESS: Initialized with provider type: {type(planner.provider).__name__}")
        except Exception as e:
            print(f"FAILED: Error during correct initialization: {e}")

if __name__ == "__main__":
    asyncio.run(test_safeguard())
    asyncio.run(test_correct_init())
