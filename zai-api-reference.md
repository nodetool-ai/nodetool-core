# Z.ai API Reference

Complete API documentation for Z.ai (Zhipu AI) Open Platform.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Models](#models)
5. [Chat Completions API](#chat-completions-api)
6. [Special Features](#special-features)
7. [Error Handling](#error-handling)
8. [SDKs](#sdks)
9. [Rate Limits](#rate-limits)

---

## Getting Started

### API Endpoint

Z.ai Platform's general API endpoint:

```
https://api.z.ai/api/paas/v4
```

For GLM Coding Plan subscribers, use the dedicated coding endpoint:

```
https://api.z.ai/api/coding/paas/v4
```

### Quick Start

1. **Get API Key**: Visit [Z.AI Open Platform](https://open.bigmodel.cn/) to register and create an API key
2. **Choose Model**: Select from available models based on your needs
3. **Make API Call**: Use HTTP API or SDK to make requests

---

## Authentication

Z.ai API uses HTTP Bearer Token authentication.

### API Key Authentication

```http
Authorization: Bearer YOUR_API_KEY
```

### JWT Token Authentication (Optional)

For higher security scenarios, generate JWT tokens:

```python
import jwt
import time

def generate_token(apikey: str, exp_seconds: int):
    id, secret = apikey.split(".")
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"}
    )
```

### Required Headers

```http
Content-Type: application/json
Accept-Language: en-US,en
Authorization: Bearer YOUR_API_KEY
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/completions` | POST | Generate chat completions |
| `/embeddings` | POST | Generate text embeddings |
| `/images/generations` | POST | Generate images |
| `/videos/generations` | POST | Generate videos |
| `/files` | POST | Upload files |
| `/files/{file_id}` | GET | Retrieve file |
| `/files/{file_id}` | DELETE | Delete file |
| `/audio/transcriptions` | POST | Audio-to-text transcription |
| `/moderations` | POST | Content moderation |

---

## Models

### Language Models (LLM)

| Model | Context Length | Description |
|-------|----------------|-------------|
| `glm-4.7` | 128K | Latest flagship model with enhanced programming and multi-step reasoning |
| `glm-4.6` | 200K | Superior coding, long-context, and reasoning capabilities |
| `glm-4.5` | 128K | Balanced performance across domains |
| `glm-4-32b-0414-128k` | 128K | Large parameter model with extended context |

### Vision Language Models (VLM)

| Model | Context Length | Description |
|-------|----------------|-------------|
| `glm-4.6v` | 128K | Flagship multimodal model with native tool use |
| `glm-4.6v-flashx` | 128K | Lightweight, high-speed vision processing |
| `glm-4.6v-flash` | 128K | Lightweight, free basic vision tasks |
| `glm-ocr` | - | Specialized OCR for text extraction |

### Image Generation Models

| Model | Description |
|-------|-------------|
| `glm-image` | General image generation |
| `cogview-4` | High-quality visual generation |

### Video Generation Models

| Model | Description |
|-------|-------------|
| `cogvideox-3` | Latest video generation |
| `vidu-q1` | Fast, efficient video creation |
| `vidu-2` | Enhanced quality and features |

### Audio Models

| Model | Description |
|-------|-------------|
| `glm-asr-2512` | Speech recognition / Audio-to-text |

---

## Chat Completions API

### Endpoint

```
POST /chat/completions
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model ID to use (e.g., `glm-4.7`) |
| `messages` | array | Yes | Array of message objects |
| `temperature` | number | No | Sampling temperature (0.0-2.0), default: 1.0 |
| `max_tokens` | integer | No | Maximum tokens to generate (1-32768) |
| `top_p` | number | No | Nucleus sampling threshold (0.0-1.0) |
| `stream` | boolean | No | Enable streaming responses, default: false |
| `thinking` | object | No | Configure thinking mode |
| `tools` | array | No | Array of tool definitions |
| `tool_choice` | string | No | Tool use strategy: `auto`, `none`, `required` |
| `response_format` | object | No | Structured output format |
| `stop` | string/array | No | Stop sequences |
| `presence_penalty` | number | No | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | number | No | Frequency penalty (-2.0 to 2.0) |

### Message Object

```json
{
  "role": "system|user|assistant",
  "content": "string or array"
}
```

For vision models, content can be an array:

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "Describe this image"
    },
    {
      "type": "image_url",
      "image_url": {
        "url": "https://example.com/image.jpg"
      }
    }
  ]
}
```

### Tool Definition

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "Function description",
    "parameters": {
      "type": "object",
      "properties": {
        "param_name": {
          "type": "string",
          "description": "Parameter description"
        }
      },
      "required": ["param_name"]
    }
  }
}
```

### Thinking Configuration

```json
{
  "thinking": {
    "type": "enabled"
  }
}
```

### Response Schema (Non-Streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1629900000,
  "model": "glm-4.7",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 100
  }
}
```

### Response Schema (Streaming)

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion.chunk",
  "created": 1629900000,
  "model": "glm-4.7",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "Partial response"
      },
      "finish_reason": null
    }
  ]
}
```

### Finish Reasons

| Value | Description |
|-------|-------------|
| `stop` | Model completed normally |
| `length` | Max tokens reached |
| `tool_calls` | Model requested tool call |
| `content_filter` | Content was filtered |

---

## Special Features

### Deep Thinking

Enable the AI's reasoning process to be shown:

```json
{
  "model": "glm-4.7",
  "messages": [...],
  "thinking": {
    "type": "enabled"
  }
}
```

Response includes `reasoning_content` field:

```json
{
  "choices": [{
    "message": {
      "content": "Final answer",
      "reasoning_content": "Step-by-step thinking process"
    }
  }]
}
```

### Function Calling / Tools

Define tools for the model to call:

```json
{
  "model": "glm-4.7",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Streaming with Tool Calls

```json
{
  "model": "glm-4.7",
  "messages": [...],
  "tools": [...],
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

### Structured Output

Request JSON-formatted responses:

```json
{
  "model": "glm-4.7",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "analysis",
      "schema": {
        "type": "object",
        "properties": {
          "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
          },
          "score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
          }
        },
        "required": ["sentiment", "score"],
        "additionalProperties": false
      }
    }
  }
}
```

### Context Caching

Reduce token usage for repeated content:

```json
{
  "model": "glm-4.7",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant.",
      "cache_control": {"type": "enabled"}
    },
    ...
  ]
}
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Invalid authentication",
    "type": "invalid_request_error",
    "param": null,
    "code": "invalid_api_key"
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Types

| Type | Description |
|------|-------------|
| `invalid_request_error` | Invalid request parameters |
| `invalid_api_key` | Invalid or expired API key |
| `rate_limit_error` | Rate limit exceeded |
| `insufficient_quota` | Insufficient account quota |
| `content_filter` | Content was filtered |

---

## SDKs

### Python SDK (Official)

```python
from zai import ZaiClient

client = ZaiClient(api_key="your-api-key")

# Basic chat
response = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="glm-4.7",
    messages=[...],
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

### Python (OpenAI Compatible)

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-zai-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

completion = client.chat.completions.create(
    model="glm-4.7",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Java SDK

```java
import ai.z.openapi.ZaiClient;
import ai.z.openapi.service.model.*;

ZaiClient client = ZaiClient.builder()
    .ofZAI()
    .apiKey("your-api-key")
    .build();

ChatCompletionCreateParams request = ChatCompletionCreateParams.builder()
    .model("glm-4.7")
    .messages(Arrays.asList(
        ChatMessage.builder()
            .role(ChatMessageRole.USER.value())
            .content("Hello!")
            .build()
    ))
    .build();

ChatCompletionResponse response = client.chat().createChatCompletion(request);
```

### Node.js (OpenAI Compatible)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: 'your-zai-api-key',
  baseURL: 'https://api.z.ai/api/paas/v4/'
});

const completion = await client.chat.completions.create({
  model: 'glm-4.7',
  messages: [{ role: 'user', content: 'Hello!' }]
});
```

### cURL Examples

```bash
# Basic request
curl -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# Streaming
curl -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7",
    "messages": [...],
    "stream": true
  }'

# With vision
curl -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6v",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

---

## Rate Limits

Rate limits are based on your account tier and subscription.

| Plan | Requests per Minute | Concurrent Requests |
|------|---------------------|---------------------|
| Free | 60 | 3 |
| GLM Coding | Higher limits | Higher limits |
| Enterprise | Custom | Custom |

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1629900000
```

### Handling Rate Limits

Implement exponential backoff for 429 errors:

```python
import time

def make_request_with_retry(client, request, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**request)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                raise
```

---

## Best Practices

### Token Management

1. Monitor token usage with the `usage` field in responses
2. Use `max_tokens` to control response length
3. Implement context caching for repeated content
4. Choose appropriate model for your use case

### Error Handling

1. Implement retry logic for 429 errors with exponential backoff
2. Validate API keys before making requests
3. Handle streaming interruptions gracefully
4. Log errors for debugging

### Performance Optimization

1. Use streaming for real-time applications
2. Use `temperature=0` for deterministic responses
3. Implement request batching where possible
4. Use appropriate model for the task (don't over-provision)

### Security

1. Store API keys securely (environment variables)
2. Use HTTPS for all API calls
3. Implement proper access controls
4. Rotate API keys regularly
5. Never commit API keys to version control

---

## Resources

- **Official Documentation**: https://docs.z.ai/
- **Developer Platform**: https://open.bigmodel.cn/
- **GitHub**: https://github.com/zai-org
- **Python SDK**: https://github.com/zai-org/z-ai-sdk-python
- **Java SDK**: https://github.com/zai-org/z-ai-sdk-java

---

## Changelog

### 2026
- **GLM-4.7-Flash** released - Open source, free 30B parameter model
- **GLM-4.7** released - Latest flagship with enhanced coding

### 2025
- **GLM-4.6** released - 200K context, agent-oriented
- **GLM-4.6V** released - Flagship multimodal
- **CogVideoX-3** released - Latest video generation
- **GLM Coding Plan** launched - Specialized coding endpoint

---

*Document generated based on Z.ai API documentation as of February 2026.*
