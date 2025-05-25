import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

from nodetool.agents.agent import Agent
from nodetool.agents.tools import BrowserTool, GoogleSearchTool
from nodetool.chat.providers import get_provider
from nodetool.metadata.types import Provider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import Chunk

app = FastAPI()

class AgentRequest(BaseModel):
    objective: str

@app.post("/run")
async def run_agent(req: AgentRequest):
    """Run the research agent with the given objective."""
    context = ProcessingContext()
    agent = Agent(
        name="Research Agent",
        objective=req.objective,
        provider=get_provider(Provider.OpenAI),
        model="gpt-4o-mini",
        tools=[GoogleSearchTool(), BrowserTool()],
        enable_analysis_phase=False,
    )

    chunks: list[str] = []
    async for item in agent.execute(context):
        if isinstance(item, Chunk):
            chunks.append(item.content)

    return {"result": "".join(chunks), "workspace": context.workspace_dir}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
