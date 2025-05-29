# Docker Agent Endpoint

This guide shows how to package a simple NodeTool agent as a small API service.
The service exposes one HTTP endpoint that runs the agent and returns the result.

## Example Server

The example server is located in `examples/docker_agent_endpoint/main.py`:

```python
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
```

## Dockerfile

A minimal Dockerfile to run the server:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY main.py /app/main.py
RUN pip install --no-cache-dir nodetool-core fastapi uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Building and Running

```bash
docker build -t agent-endpoint examples/docker_agent_endpoint

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=YOUR_KEY \
  agent-endpoint
```

The service exposes a `/run` endpoint. Send a POST request with a JSON body:

```bash
curl -X POST http://localhost:8000/run \
     -H "Content-Type: application/json" \
     -d '{"objective": "Explain Docker"}'
```

The response contains the agent output and the workspace directory used during execution.
