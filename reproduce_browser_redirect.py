
import asyncio
import aiohttp
from nodetool.runtime.resources import ResourceScope

async def main():
    async with ResourceScope():
        redirect_url = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF9eo17FVM7EuEdBB_LLJQw7jHctE4m7s24jZDf9OoM9OSuC5iU8_RWkqPLBynMwj5ihATEMYM0JPdA4iJOgYzFtrVNNCAHnXGeOmzmv6o5VEi9IIcrXh4-RUB3XcDjZLCMwph4lYYsfO64lJXDFCea3VFDk1Nycec7Rg=="
        
        print(f"Attempting to resolve redirect: {redirect_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(redirect_url, allow_redirects=True) as resp:
                    print(f"Final URL: {resp.url}")
                    print(f"Status: {resp.status}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
