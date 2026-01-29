"""
VibeCoding Agent for generating self-contained HTML apps for workflows.

This agent generates beautiful, self-contained HTML/CSS/JS applications
as custom frontends for Nodetool workflows. Users interact through chat
to describe their desired UI, and the agent produces complete HTML.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message, Provider
from nodetool.providers import BaseProvider, get_provider
from nodetool.types.workflow import Workflow
from nodetool.workflows.types import Chunk

log = get_logger(__name__)

# System prompt template for HTML generation
SYSTEM_PROMPT_TEMPLATE = """You are VibeCoder, an expert frontend developer that creates beautiful, self-contained HTML applications for Nodetool workflows.

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
class NodeToolRunner {{
  constructor(options) {{
    this.apiUrl = options.apiUrl || window.NODETOOL_API_URL || 'http://localhost:7777/api';
    this.wsUrl = options.wsUrl || window.NODETOOL_WS_URL || 'ws://localhost:7777/ws';
    this.workflowId = options.workflowId || window.NODETOOL_WORKFLOW_ID;
    this.onProgress = options.onProgress || (() => {{}});
    this.onOutput = options.onOutput || (() => {{}});
    this.onError = options.onError || (() => {{}});
    this.onComplete = options.onComplete || (() => {{}});
    this.onStatusChange = options.onStatusChange || (() => {{}});
    this.socket = null;
    this.jobId = null;
  }}

  async run(params = {{}}) {{
    return new Promise((resolve, reject) => {{
      this.socket = new WebSocket(this.wsUrl);

      this.socket.onopen = () => {{
        this.onStatusChange('running');
        const request = {{
          workflow_id: this.workflowId,
          params: params
        }};
        this.socket.send(JSON.stringify({{ command: 'run_job', data: request }}));
      }};

      this.socket.onmessage = (event) => {{
        const data = JSON.parse(event.data);
        if (data.job_id) this.jobId = data.job_id;

        switch (data.type) {{
          case 'job_update':
            if (data.status === 'completed') {{
              this.onStatusChange('completed');
              this.onComplete(data.result);
              resolve(data.result);
              this.socket.close();
            }} else if (data.status === 'failed') {{
              this.onStatusChange('error');
              this.onError(data.error || 'Job failed');
              reject(new Error(data.error || 'Job failed'));
              this.socket.close();
            }}
            break;
          case 'node_progress':
            this.onProgress((data.progress / data.total) * 100);
            break;
          case 'output_update':
            this.onOutput({{ name: data.output_name, value: data.value, type: data.output_type }});
            break;
          case 'error':
            this.onStatusChange('error');
            this.onError(data.error);
            reject(new Error(data.error));
            this.socket.close();
            break;
        }}
      }};

      this.socket.onerror = (error) => {{
        this.onStatusChange('error');
        this.onError('Connection failed');
        reject(error);
      }};
    }});
  }}

  cancel() {{
    if (this.socket && this.socket.readyState === WebSocket.OPEN && this.jobId) {{
      this.socket.send(JSON.stringify({{ command: 'cancel_job', data: {{ job_id: this.jobId }} }}));
    }}
  }}
}}
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
"""


# Default max tokens for HTML generation (large to accommodate full HTML apps)
DEFAULT_MAX_TOKENS = 16384


class VibeCodingAgent:
    """Agent that generates self-contained HTML apps for workflows."""

    def __init__(
        self,
        workflow: Workflow,
        provider: BaseProvider | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the VibeCodingAgent.

        Args:
            workflow: The workflow to generate an HTML app for.
            provider: Optional provider instance. If not provided, will be created.
            model: The model to use for generation.
            max_tokens: Maximum tokens for the response. Defaults to DEFAULT_MAX_TOKENS.
        """
        self.workflow = workflow
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Construct system prompt with workflow context."""
        input_schema = self.workflow.input_schema or {}
        output_schema = self.workflow.output_schema or {}

        return SYSTEM_PROMPT_TEMPLATE.format(
            workflow_name=self.workflow.name or "Untitled Workflow",
            workflow_description=self.workflow.description or "No description provided",
            workflow_id=self.workflow.id,
            input_schema_json=json.dumps(input_schema, indent=2),
            output_schema_json=json.dumps(output_schema, indent=2),
        )

    async def generate(
        self,
        user_prompt: str,
        user_id: str = "1",
    ) -> AsyncIterator[str]:
        """
        Generate HTML based on user prompt.

        Args:
            user_prompt: The user's description of the desired UI.
            user_id: The user ID for provider initialization.

        Yields:
            Streaming text chunks containing the generated HTML.
        """
        # Get provider if not already set
        if self.provider is None:
            self.provider = await get_provider(Provider.Anthropic, user_id=user_id)

        # Create messages for the conversation
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_prompt),
        ]

        # Stream the response
        async for chunk in self.provider.generate_messages(
            messages=messages,
            model=self.model,
            max_tokens=self.max_tokens,
        ):
            if isinstance(chunk, Chunk) and chunk.content:
                yield chunk.content


def extract_html_from_response(response: str) -> str | None:
    """
    Extract HTML content from a response that may contain markdown code blocks.

    Args:
        response: The full response text.

    Returns:
        The extracted HTML content, or None if no HTML block found.
    """
    import re

    # Look for ```html ... ``` blocks
    pattern = r"```html\s*([\s\S]*?)\s*```"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()

    # If no code block, check if response starts with <!DOCTYPE
    if response.strip().startswith("<!DOCTYPE"):
        return response.strip()

    return None
