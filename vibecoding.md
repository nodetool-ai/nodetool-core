# VibeCoding Agent Backend Implementation

## Prompt for Coding Agent

---

### Objective

Implement the backend infrastructure for a VibeCoding agent in Nodetool. This agent generates self-contained HTML/CSS/JS applications as custom frontends for workflows. Users interact with the agent through chat to describe their desired UI, and the agent produces complete HTML that can execute the workflow.

---

### Context & Rationale

**Why this feature exists:**
Nodetool workflows currently use a generic `MiniAppPage` UI that auto-generates forms from `input_schema` and displays results from `output_schema`. While functional, this UI is generic and doesn't allow users to create beautiful, branded, or workflow-specific interfaces. The VibeCoding agent solves this by letting users describe their ideal UI in natural language and generating production-ready HTML apps.

**How it fits into the architecture:**
- The `html_app` field already exists on the `Workflow` model (added in migration `20260127_000000`)
- The endpoint `GET /api/workflows/{id}/app` already serves stored HTML as `HTMLResponse`
- The existing chat infrastructure (`GlobalChatStore`, `GlobalWebSocketManager`) handles streaming LLM responses
- We need a specialized agent that understands workflow schemas and generates valid, executable HTML

**Key design decisions:**
1. **Reuse existing chat infrastructure** - Don't create a separate websocket; use the same `thread_id`-based routing
2. **Self-contained HTML** - Generated apps must work standalone with only a CDN dependency on msgpack-lite
3. **Workflow context injection** - The agent needs the workflow's input/output schemas to generate matching forms
4. **Streaming response** - HTML is returned in a code block within the chat stream, parsed on the frontend

---

### Existing Code Reference

**Workflow Model** (`nodetool-core/src/nodetool/models/workflow.py`):
```python
class Workflow(DBModel):
    # ... other fields ...
    html_app: str | None = DBField(default=None)  # HTML content for the workflow app
```

**Workflow Types** (`nodetool-core/src/nodetool/types/workflow.py`):
```python
class Workflow(BaseModel):
    # ... other fields ...
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    html_app: str | None = None

class WorkflowRequest(BaseModel):
    # ... other fields ...
    html_app: str | None = None
```

**Existing App Endpoint** (`nodetool-core/src/nodetool/api/workflow.py`):
```python
@router.get("/{id}/app", response_class=HTMLResponse)
async def get_workflow_app(id: str, user: str = Depends(current_user)) -> HTMLResponse:
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if not workflow.html_app:
        raise HTTPException(status_code=404, detail="No HTML app configured for this workflow")
    return HTMLResponse(content=workflow.html_app, status_code=200)
```

**Chat Message Types** (`nodetool-core/src/nodetool/types/chat.py` - reference existing patterns):
- Messages have `role` (user/assistant/system), `content` (list of content blocks)
- Content blocks can be `text`, `image`, `tool_call`, `tool_result`, etc.
- Streaming uses websocket with `thread_id` routing

**Existing Agent Pattern** - Look at how other agents are defined in the codebase for the registration and system prompt patterns.

---

### Implementation Tasks

#### Task 1: Create VibeCoding Agent Definition

**File:** `src/nodetool/agents/vibecoding.py`

Create an agent class that:
1. Accepts workflow context (id, name, description, input_schema, output_schema)
2. Has a specialized system prompt for HTML generation
3. Formats the workflow context into the system message
4. Returns streaming text responses containing HTML in code blocks

**System Prompt Template:**
```
You are VibeCoder, an expert frontend developer that creates beautiful, self-contained HTML applications for Nodetool workflows.

## Your Task
Generate a complete, production-ready HTML file that serves as a custom UI for the workflow described below. The HTML must be fully self-contained with embedded CSS and JavaScript.

## Workflow Details
- **Name:** {workflow_name}
- **Description:** {workflow_description}
- **Workflow ID:** {workflow_id}

## Input Schema
The workflow accepts these inputs (generate form fields for each):
```json
{input_schema_json}
```

## Output Schema  
The workflow produces these outputs (display them appropriately):
```json
{output_schema_json}
```

## Technical Requirements

### 1. HTML Structure
- Single HTML file with <!DOCTYPE html>
- Embedded <style> in <head>
- Embedded <script> before </body>
- Responsive design (mobile-friendly)
- Modern, clean aesthetic

### 2. Required JavaScript
Include this workflow runner (paste exactly):
```javascript
class NodeToolRunner {
  constructor(options) {
    this.apiUrl = options.apiUrl || window.NODETOOL_API_URL || 'http://localhost:7777/api';
    this.wsUrl = options.wsUrl || window.NODETOOL_WS_URL || 'ws://localhost:7777/ws';
    this.workflowId = options.workflowId || window.NODETOOL_WORKFLOW_ID;
    this.onProgress = options.onProgress || (() => {});
    this.onOutput = options.onOutput || (() => {});
    this.onError = options.onError || (() => {});
    this.onComplete = options.onComplete || (() => {});
    this.onStatusChange = options.onStatusChange || (() => {});
    this.socket = null;
    this.jobId = null;
  }

  async run(params = {}) {
    return new Promise((resolve, reject) => {
      this.socket = new WebSocket(this.wsUrl);
      this.socket.binaryType = 'arraybuffer';
      
      this.socket.onopen = () => {
        this.onStatusChange('running');
        const request = {
          type: 'run_job_request',
          api_url: this.apiUrl,
          workflow_id: this.workflowId,
          job_type: 'workflow',
          auth_token: 'local_token',
          params: params
        };
        this.socket.send(msgpack.encode({ command: 'run_job', data: request }));
      };
      
      this.socket.onmessage = (event) => {
        const data = msgpack.decode(new Uint8Array(event.data));
        if (data.job_id) this.jobId = data.job_id;
        
        switch (data.type) {
          case 'job_update':
            if (data.status === 'completed') {
              this.onStatusChange('completed');
              this.onComplete(data.result);
              resolve(data.result);
              this.socket.close();
            } else if (data.status === 'failed') {
              this.onStatusChange('error');
              this.onError(data.error || 'Job failed');
              reject(new Error(data.error || 'Job failed'));
              this.socket.close();
            }
            break;
          case 'node_progress':
            this.onProgress((data.progress / data.total) * 100);
            break;
          case 'output_update':
            this.onOutput({ name: data.output_name, value: data.value, type: data.output_type });
            break;
          case 'error':
            this.onStatusChange('error');
            this.onError(data.error);
            reject(new Error(data.error));
            this.socket.close();
            break;
        }
      };
      
      this.socket.onerror = (error) => {
        this.onStatusChange('error');
        this.onError('Connection failed');
        reject(error);
      };
    });
  }

  cancel() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN && this.jobId) {
      this.socket.send(msgpack.encode({ command: 'cancel_job', data: { job_id: this.jobId } }));
    }
  }
}
```

### 3. Required CDN
Add this in <head>:
```html
<script src="https://unpkg.com/msgpack-lite@0.1.26/dist/msgpack.min.js"></script>
```

### 4. Form Generation Rules
For each input in the schema:
- `string` → <input type="text"> or <textarea> for long text
- `integer` → <input type="number" step="1">
- `number`/`float` → <input type="number" step="any">
- `boolean` → <input type="checkbox"> or toggle switch
- `enum` → <select> with options
- Respect `minimum`, `maximum` constraints with min/max attributes
- Use `title` as label, `description` as help text
- Use `default` as initial value

### 5. Output Display Rules
For each output in the schema:
- `string` → <pre> or formatted text block
- `image` → <img> tag (value will be base64 or URL)
- `audio` → <audio> tag with controls
- `video` → <video> tag with controls
- `object`/`array` → formatted JSON display

### 6. UI States
Include visual states for:
- **Idle:** Form ready, submit button enabled
- **Running:** Show progress bar, disable form, show cancel button
- **Completed:** Show results, enable "Run Again" 
- **Error:** Show error message with retry option

### 7. Styling Guidelines
- Use CSS variables for theming
- Include dark mode support via `prefers-color-scheme`
- Smooth transitions and subtle animations
- Clear visual hierarchy
- Accessible (proper labels, focus states, contrast)

## Output Format
Return ONLY the complete HTML file wrapped in a code block:
```html
<!DOCTYPE html>
<html lang="en">
...
</html>
```

Do not include any explanation before or after the code block. The HTML must be immediately usable.
```

**Agent Class Structure:**
```python
from nodetool.types.chat import Message, MessageContent, TextContent
from nodetool.types.workflow import Workflow
from typing import AsyncIterator
import json

class VibeCodingAgent:
    """Agent that generates self-contained HTML apps for workflows."""
    
    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Construct system prompt with workflow context."""
        # Use the template above, substituting:
        # - {workflow_name} = self.workflow.name
        # - {workflow_description} = self.workflow.description or "No description"
        # - {workflow_id} = self.workflow.id
        # - {input_schema_json} = json.dumps(self.workflow.input_schema, indent=2)
        # - {output_schema_json} = json.dumps(self.workflow.output_schema, indent=2)
        pass
    
    async def generate(self, user_prompt: str, thread_id: str) -> AsyncIterator[str]:
        """
        Generate HTML based on user prompt.
        Yields streaming text chunks.
        """
        # Use the configured LLM (Claude, GPT-4, etc.)
        # Stream the response
        # The frontend will extract HTML from ```html ... ``` blocks
        pass
```

#### Task 2: Create VibeCoding API Router

**File:** `src/nodetool/api/vibecoding.py`

Create API endpoints:

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from nodetool.api.auth import current_user
from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.agents.vibecoding import VibeCodingAgent
from nodetool.api.workflow import from_model
import json

router = APIRouter(prefix="/api/vibecoding", tags=["vibecoding"])

class GenerateRequest(BaseModel):
    workflow_id: str
    prompt: str
    thread_id: str | None = None  # Optional, will create new if not provided

class GenerateResponse(BaseModel):
    thread_id: str
    # Response streams via websocket, this just confirms initiation

@router.post("/generate")
async def generate_html(
    request: GenerateRequest,
    user: str = Depends(current_user)
) -> StreamingResponse:
    """
    Generate HTML app for a workflow based on user prompt.
    
    Returns streaming response with the generated HTML.
    The HTML will be wrapped in ```html ... ``` code blocks.
    """
    # 1. Load workflow and verify access
    workflow = await WorkflowModel.get(request.workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # 2. Convert to API type with schemas
    workflow_data = await from_model(workflow)
    
    # 3. Create agent and generate
    agent = VibeCodingAgent(workflow_data)
    
    async def stream_response():
        async for chunk in agent.generate(request.prompt, request.thread_id or ""):
            yield chunk
    
    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )

@router.get("/templates")
async def get_templates() -> list[dict]:
    """
    Return starter templates for common workflow patterns.
    """
    return [
        {
            "id": "minimal",
            "name": "Minimal",
            "description": "Clean, minimal interface with basic styling",
            "prompt": "Create a minimal, clean interface with a white background and simple form styling."
        },
        {
            "id": "dark",
            "name": "Dark Mode",
            "description": "Dark themed interface",
            "prompt": "Create a dark-themed interface with a dark background, light text, and subtle accent colors."
        },
        {
            "id": "gradient",
            "name": "Gradient",
            "description": "Modern gradient backgrounds",
            "prompt": "Create a modern interface with subtle gradient backgrounds and rounded corners."
        },
        {
            "id": "professional",
            "name": "Professional",
            "description": "Business/enterprise styling",
            "prompt": "Create a professional, enterprise-style interface suitable for business applications."
        }
    ]
```

#### Task 3: Register Router in Main App

**File:** `src/nodetool/api/__init__.py` or `src/nodetool/api/app.py`

Add the vibecoding router to the FastAPI app:

```python
from nodetool.api.vibecoding import router as vibecoding_router

# In the app setup:
app.include_router(vibecoding_router)
```

#### Task 4: Enhance Workflow App Endpoint

**File:** `src/nodetool/api/workflow.py`

Update the existing `/app` endpoint to inject runtime configuration:

```python
from nodetool.config import get_api_url, get_ws_url  # You may need to create these

@router.get("/{id}/app", response_class=HTMLResponse)
async def get_workflow_app(id: str, user: str = Depends(current_user)) -> HTMLResponse:
    """
    Serve the HTML app for a workflow.
    
    Injects runtime configuration (API URL, WS URL, workflow ID) so the
    app works in any environment.
    """
    workflow = await WorkflowModel.get(id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if workflow.access != "public" and workflow.user_id != user:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if not workflow.html_app:
        raise HTTPException(status_code=404, detail="No HTML app configured for this workflow")
    
    # Inject runtime configuration
    # These should come from environment/config
    api_url = get_api_url()  # e.g., "http://localhost:7777/api" or production URL
    ws_url = get_ws_url()    # e.g., "ws://localhost:7777/ws" or production URL
    
    config_script = f"""
    <script>
      window.NODETOOL_API_URL = "{api_url}";
      window.NODETOOL_WS_URL = "{ws_url}";
      window.NODETOOL_WORKFLOW_ID = "{id}";
    </script>
    """
    
    # Inject before </head>
    html = workflow.html_app
    if '</head>' in html:
        html = html.replace('</head>', f'{config_script}</head>')
    else:
        # Fallback: inject at start of body
        html = html.replace('<body>', f'<body>{config_script}')
    
    return HTMLResponse(content=html, status_code=200)
```

#### Task 5: Add Configuration Helpers

**File:** `src/nodetool/config.py` (or wherever config is managed)

Add helpers to get runtime URLs:

```python
import os
from nodetool.common.environment import Environment

def get_api_url() -> str:
    """Get the API URL for client-side use."""
    if Environment.is_production():
        return os.getenv("NODETOOL_API_URL", "https://api.nodetool.ai/api")
    return os.getenv("NODETOOL_API_URL", "http://localhost:7777/api")

def get_ws_url() -> str:
    """Get the WebSocket URL for client-side use."""
    if Environment.is_production():
        return os.getenv("NODETOOL_WS_URL", "wss://api.nodetool.ai/ws")
    return os.getenv("NODETOOL_WS_URL", "ws://localhost:7777/ws")
```

---

### Testing Requirements

Create tests in `tests/api/test_vibecoding.py`:

```python
import pytest
from fastapi.testclient import TestClient
from nodetool.models.workflow import Workflow

@pytest.mark.asyncio
async def test_generate_requires_valid_workflow(client: TestClient, headers: dict):
    """Test that generate fails for non-existent workflow."""
    response = client.post(
        "/api/vibecoding/generate",
        json={"workflow_id": "nonexistent", "prompt": "Create a simple form"},
        headers=headers
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_generate_requires_access(client: TestClient, headers: dict):
    """Test that users can only generate for workflows they have access to."""
    # Create workflow owned by different user
    # Attempt to generate - should fail with 403
    pass

@pytest.mark.asyncio
async def test_generate_returns_streaming_response(client: TestClient, headers: dict, workflow: Workflow):
    """Test that generate returns a streaming response."""
    response = client.post(
        "/api/vibecoding/generate",
        json={"workflow_id": workflow.id, "prompt": "Create a minimal form"},
        headers=headers
    )
    assert response.status_code == 200
    assert response.headers.get("content-type") == "text/plain; charset=utf-8"

@pytest.mark.asyncio
async def test_templates_endpoint(client: TestClient, headers: dict):
    """Test that templates endpoint returns valid templates."""
    response = client.get("/api/vibecoding/templates", headers=headers)
    assert response.status_code == 200
    templates = response.json()
    assert isinstance(templates, list)
    assert len(templates) > 0
    assert all("id" in t and "name" in t and "prompt" in t for t in templates)

@pytest.mark.asyncio
async def test_workflow_app_injects_config(client: TestClient, headers: dict, workflow: Workflow):
    """Test that the /app endpoint injects runtime configuration."""
    # Set html_app on workflow
    workflow.html_app = "<!DOCTYPE html><html><head></head><body>Test</body></html>"
    await workflow.save()
    
    response = client.get(f"/api/workflows/{workflow.id}/app", headers=headers)
    assert response.status_code == 200
    html = response.text
    
    assert "window.NODETOOL_API_URL" in html
    assert "window.NODETOOL_WS_URL" in html
    assert f'window.NODETOOL_WORKFLOW_ID = "{workflow.id}"' in html
```

---

### Acceptance Criteria

1. **`POST /api/vibecoding/generate`** accepts `workflow_id` and `prompt`, returns streaming text containing HTML in code blocks
2. **VibeCodingAgent** correctly formats workflow context (name, description, input_schema, output_schema) into system prompt
3. **Generated HTML** includes the NodeToolRunner class and msgpack CDN
4. **`GET /api/workflows/{id}/app`** injects `NODETOOL_API_URL`, `NODETOOL_WS_URL`, `NODETOOL_WORKFLOW_ID` into served HTML
5. **All tests pass** including access control and streaming response verification
6. **Agent is registered** and available for use through the chat infrastructure

---

### Notes for Implementation

1. **LLM Integration**: Use whatever LLM client pattern exists in the codebase. Look for existing agent implementations to follow the same pattern for model selection, streaming, etc.

2. **Error Handling**: If the workflow has no `input_schema` or `output_schema`, generate sensible defaults or inform the user to run the workflow once to generate schemas.

3. **Schema Extraction**: If schemas are missing, you might need to extract them from the workflow graph by looking at input/output nodes. Check existing code that builds `input_schema` from workflow nodes.

4. **Streaming**: The response should stream character-by-character or chunk-by-chunk so the frontend can show typing animation. Use `AsyncIterator[str]` or similar.

5. **Thread Management**: For v1, don't worry about persistent thread history for VibeCoding. Each generate request is standalone. Thread support can be added later for iterative refinement.
