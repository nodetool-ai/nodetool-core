# Deploying a Workflow to AWS Lambda

This guide shows how to package a simple NodeTool workflow and run it as an AWS Lambda function.

## Lambda Handler Example

Create a handler that executes your workflow. Save the following as `examples/aws_lambda_handler.py`:

```python
#!/usr/bin/env python3
import asyncio
import json
from typing import Any, Dict

from nodetool.dsl.graph import graph, run_graph
from nodetool.dsl.providers.openai import ChatCompletion
from nodetool.metadata.types import OpenAIModel

async def run_workflow(prompt: str) -> str:
    workflow = ChatCompletion(
        model=OpenAIModel(model="gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
    )
    return await run_graph(graph(workflow))


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    prompt = event.get("prompt", "Hello from NodeTool!")
    result = asyncio.run(run_workflow(prompt))
    return {"statusCode": 200, "body": json.dumps({"result": result})}
```

## Packaging and Deployment

1. Install NodeTool Core and its dependencies into a local directory:

```bash
pip install nodetool-core -t ./package
```

2. Create the deployment package:

```bash
cd package
zip -r9 ../function.zip .
cd ..
zip -g function.zip examples/aws_lambda_handler.py
```

3. Deploy using the AWS CLI:

```bash
aws lambda create-function \
    --function-name nodetool-example \
    --runtime python3.11 \
    --handler aws_lambda_handler.lambda_handler \
    --zip-file fileb://function.zip \
    --role arn:aws:iam::123456789012:role/lambda-execution-role
```

Replace the role ARN with your own IAM role that has permission to run Lambda functions.

Once deployed, invoke the function by sending a JSON payload with a `prompt` field.
