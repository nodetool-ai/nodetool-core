# MiniMax API Complete Reference

> Official API documentation for MiniMax AI platform - a multi-modal AI service provider offering text, voice, video, image, and music generation capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Base URLs & Endpoints](#base-urls--endpoints)
- [Text Generation API](#text-generation-api)
- [Speech/TTS API](#speechtts-api)
- [Video Generation API](#video-generation-api)
- [Image Generation API](#image-generation-api)
- [Music Generation API](#music-generation-api)
- [File Management API](#file-management-api)
- [SDK Integration](#sdk-integration)
- [Rate Limits & Pricing](#rate-limits--pricing)

---

## Overview

**MiniMax** is a Chinese AI company providing state-of-the-art generative AI models through their open platform. Their services include:

| Capability | Description |
|------------|-------------|
| **Text Generation** | LLMs optimized for coding and agent workflows |
| **Speech Synthesis** | 300+ system voices, voice cloning, async TTS (up to 1M chars) |
| **Video Generation** | T2V and I2V with Hailuo models |
| **Image Generation** | Text-to-image and image-to-image |
| **Music Generation** | AI music with vocals from lyrics |

**Official Resources:**
- Website: https://www.minimax.io/
- Platform: https://platform.minimaxi.com/
- API Docs: https://platform.minimaxi.com/docs/api-reference/api-overview
- Global API: https://minimax-ai.chat/docs/api/

---

## Authentication

### API Key

All API requests require authentication via Bearer token.

```
Authorization: Bearer <YOUR_API_KEY>
```

### Getting an API Key

1. Go to [MiniMax Open Platform](https://platform.minimaxi.com/)
2. Navigate to **API Keys** (接口密钥)
3. Choose your plan:
   - **Pay-as-you-go** (按量付费): Create standard API Key
   - **Coding Plan**: Create Coding Plan Key

### Key Types

| Key Type | Usage |
|----------|-------|
| General API Key | Standard API access |
| Coding Plan Key | Optimized for coding workflows |
| Regional Keys | Separate keys for China vs Overseas regions |

### Required Headers

| Header | Value |
|--------|-------|
| `Authorization` | `Bearer <YOUR_API_KEY>` |
| `Content-Type` | `application/json` (JSON requests) |
| `Content-Type` | `multipart/form-data` (file uploads) |

---

## Base URLs & Endpoints

### Primary Base URL

```
https://api.minimaxi.com/v1
```

### OpenAI-Compatible URL

```
https://api.minimax.io/v1
```

### Common Endpoints

| API | Endpoint | Method |
|-----|----------|--------|
| Text Chat Completion | `/text/chatcompletion_v2` | POST |
| Sync TTS | `/tts` | POST |
| Async TTS | `/text_to_speech/async` | POST |
| Video Generation | `/video_generation` | POST |
| Image Generation | `/image_generation` | POST |
| Music Generation | `/music_generation` | POST |
| File Upload | `/files/upload` | POST |
| File Download | `/files/{file_id}` | GET |
| File List | `/files` | GET |
| File Delete | `/files/{file_id}` | DELETE |

---

## Text Generation API

### Supported Models

| Model | Context | Description | Speed |
|-------|---------|-------------|-------|
| `MiniMax-M2.1` | 204,800 tokens | Strong multilingual coding, upgraded programming experience | ~60 tps |
| `MiniMax-M2.1-lightning` | 204,800 tokens | M2.1 lightning: same quality, faster, more agile | ~100 tps |
| `MiniMax-M2` | 204,800 tokens | Optimized for efficient coding and Agent workflows | ~60 tps |
| `MiniMax-Text-01` | ~200K tokens | 456B total params, 45.9B activated per token | Standard |

### Request Format

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/text/chatcompletion_v2 \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "MiniMax-M2.1",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": false
  }'
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model name to use |
| `messages` | array | Yes | Array of message objects |
| `temperature` | float | No | Sampling temperature (0-1) |
| `max_tokens` | int | No | Maximum tokens to generate |
| `stream` | boolean | No | Enable streaming response |
| `top_p` | float | No | Nucleus sampling parameter |
| `tools` | array | No | Function calling definitions |

### Message Object

```json
{
  "role": "user|assistant|system",
  "content": "string",
  "name": "string (optional)"
}
```

### Response Format

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "MiniMax-M2.1",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Response content here"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Function Calling

MiniMax M2 supports function calling for tool use:

```json
{
  "model": "MiniMax-M2",
  "messages": [
    {"role": "user", "content": "What's the weather in Beijing?"}
  ],
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
              "description": "City name"
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
  ]
}
```

### Python Example (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_MINIMAX_API_KEY",
    base_url="https://api.minimax.io/v1"
)

response = client.chat.completions.create(
    model="MiniMax-M2.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### JavaScript Example

```javascript
const response = await fetch('https://api.minimaxi.com/v1/text/chatcompletion_v2', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'MiniMax-M2.1',
    messages: [
      { role: 'user', content: 'Hello!' }
    ],
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

---

## Speech/TTS API

### Supported Models

| Model | Description |
|-------|-------------|
| `speech-2.8-hd` | Latest HD model, accurate tone detail reproduction |
| `speech-2.8-turbo` | Latest Turbo model, faster response |
| `speech-2.6-hd` | HD model, excellent prosody, natural generation |
| `speech-2.6-turbo` | Ultra-low latency (~250ms), real-time optimized |
| `speech-02-hd` | Outstanding prosody, stability, cloning similarity |
| `speech-02-turbo` | Enhanced minor language support |

### Features

- **300+ system voices** available
- **Voice cloning** capabilities
- **40 languages** supported (see list below)
- **Streaming output** supported
- **Async TTS** for up to 1M characters

### Supported Languages

Chinese, Cantonese, English, Spanish, French, Russian, German, Portuguese, Arabic, Italian, Japanese, Korean, Indonesian, Vietnamese, Turkish, Dutch, Ukrainian, Thai, Polish, Romanian, Greek, Czech, Finnish, Hindi, Bulgarian, Danish, Hebrew, Malay, Persian, Slovak, Swedish, Croatian, Filipino, Hungarian, Norwegian, Slovenian, Catalan, Nynorsk, Tamil, Afrikaans

### Synchronous TTS Request

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/tts \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "speech-2.6-turbo",
    "text": "Hello, this is a test of MiniMax speech synthesis.",
    "voice": "female-gentle",
    "speed": 1.0,
    "volume": 1.0,
    "pitch": 1.0,
    "audio_format": "mp3"
  }'
```

### Synchronous TTS Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Speech model name |
| `text` | string | Yes | Text to synthesize (max 10,000 chars) |
| `voice` | string | Yes | Voice ID or name |
| `speed` | float | No | Speech speed (0.5-2.0, default 1.0) |
| `volume` | float | No | Audio volume (0.1-10.0, default 1.0) |
| `pitch` | float | No | Pitch adjustment (0.5-2.0, default 1.0) |
| `audio_format` | string | No | Output format: mp3, pcm, flac, wav |
| `sample_rate` | int | No | Sample rate (e.g., 24000, 48000) |
| `stream` | boolean | No | Enable streaming output |

### Asynchronous TTS Request

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/text_to_speech/async \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "speech-2.6-hd",
    "text": "Long text content...",
    "voice": "male-calm",
    "audio_format": "mp3"
  }'
```

### Async TTS Response

```json
{
  "task_id": "task_xxx",
  "status": "Processing",
  "file_id": null
}
```

### Query Async Task Status

```bash
curl --request GET \
  --url https://api.minimaxi.com/v1/text_to_speech/async/{task_id} \
  --header 'Authorization: Bearer <YOUR_API_KEY>'
```

### Voice Cloning

Upload reference audio and clone the voice:

```bash
# Step 1: Upload reference audio
curl --request POST \
  --url https://api.minimaxi.com/v1/voice_clone/upload \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: multipart/form-data' \
  --form 'file=@reference.mp3'

# Step 2: Clone voice
curl --request POST \
  --url https://api.minimaxi.com/v1/voice_clone/create \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "voice_id": "my_custom_voice",
    "file_id": "file_xxx"
  }'
```

**Important Notes:**
- Cloned voices are temporary (7 days expiration)
- First use triggers billing, not the cloning operation
- Use cloned voice within 168 hours to make it permanent

### Voice Design

Generate custom voices from text descriptions:

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/voice_design/create \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "speech-02-hd",
    "prompt": "A young female voice with gentle tone, slow speech pattern",
    "voice_id": "my_designed_voice"
  }'
```

### WebSocket Streaming

For real-time applications:

```javascript
const ws = new WebSocket('wss://api.minimaxi.com/v1/tts/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    model: 'speech-2.6-turbo',
    text: 'Streaming speech example',
    voice: 'female-gentle',
    audio_format: 'mp3'
  }));
};

ws.onmessage = (event) => {
  const audioData = event.data;
  // Process audio chunks
};
```

---

## Video Generation API

### Supported Models

| Model | Description |
|-------|-------------|
| `MiniMax-Hailuo-2.3` | Latest video model, upgraded body movement, physics, instruction following |
| `MiniMax-Hailuo-2.3-Fast` | Image-to-video, faster generation, cost-effective |
| `MiniMax-Hailuo-02` | 1080P resolution, 10s duration, strong instruction following |

### Text-to-Video (T2V)

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/video_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "MiniMax-Hailuo-2.3",
    "prompt": "A cat wearing sunglasses, walking down a street at sunset",
    "video_length": 6
  }'
```

### Image-to-Video (I2V)

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/video_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "MiniMax-Hailuo-2.3-Fast",
    "prompt": "The person starts walking forward",
    "image_url": "https://example.com/reference.jpg",
    "video_length": 5
  }'
```

### Video Generation Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Video model name |
| `prompt` | string | Yes | Video description |
| `image_url` | string | No | Reference image URL (I2V) |
| `first_frame_image_id` | string | No | First frame file_id |
| `last_frame_image_id` | string | No | Last frame file_id |
| `subject_image_id` | string | No | Subject reference |
| `video_length` | int | No | Duration in seconds |
| `video_ratio` | string | No | Aspect ratio (e.g., "16:9") |

### Video Generation Response

```json
{
  "task_id": "task_xxx",
  "status": "Processing",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Query Video Task Status

```bash
curl --request GET \
  --url https://api.minimaxi.com/v1/video_generation/{task_id} \
  --header 'Authorization: Bearer <YOUR_API_KEY>'
```

### Completed Task Response

```json
{
  "task_id": "task_xxx",
  "status": "Success",
  "file_id": "file_xxx",
  "video_url": "https://cdn.minimax.chat/...",
  "duration": 6,
  "width": 1080,
  "height": 1920
}
```

### Video Agent Templates

MiniMax provides pre-built video generation templates:

| Template ID | Name | Description |
|-------------|------|-------------|
| 392753057216684038 | Diving | Generate diving action from photo |
| 393881433990066176 | Rings | Pet completing gymnastic rings |
| 393769180141805569 | Survival | Pet wilderness survival video |
| 394246956137422856 | Labubu Swap | Labubu face swap video |
| 393857704283172864 | Love Letter Photo | Winter snow portrait |

### Create Video Agent Task

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/video_template_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "template_id": "392753057216684038",
    "media_inputs": ["file_xxx"],
    "text_inputs": {}
  }'
```

---

## Image Generation API

### Supported Models

| Model | Description |
|-------|-------------|
| `image-01` | Text-to-image, image-to-image with person reference |
| `image-01-live` | Extended with style options |

### Text-to-Image

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/image_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "image-01",
    "prompt": "A serene mountain landscape at sunset, digital art style",
    "image_size": "1024x1024"
  }'
```

### Image-to-Image

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/image_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "image-01",
    "prompt": "Transform into anime style",
    "reference_image_id": "file_xxx",
    "image_size": "1024x1024"
  }'
```

### Image Generation Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Image model name |
| `prompt` | string | Yes | Image description |
| `reference_image_id` | string | No | Reference image file_id |
| `image_size` | string | No | Dimensions (e.g., "1024x1024") |
| `image_ratio` | string | No | Aspect ratio |
| `style` | string | No | Art style (for image-01-live) |

---

## Music Generation API

### Supported Models

| Model | Description |
|-------|-------------|
| `music-2.0` | Generate AI music with vocals from prompt and lyrics |

### Create Music

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/music_generation \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "music-2.0",
    "prompt": "An upbeat pop song with catchy chorus",
    "lyrics": "Verse 1: ...\n\nChorus: ...",
    "duration": 120
  }'
```

### Music Generation Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Music model name |
| `prompt` | string | Yes | Music style/description |
| `lyrics` | string | No | Song lyrics |
| `duration` | int | No | Duration in seconds |

---

## File Management API

### Supported File Formats

| Type | Formats |
|------|---------|
| Documents | pdf, docx, txt, jsonl |
| Audio | mp3, m4a, wav |

### Limits

| Limit | Value |
|-------|-------|
| Total capacity | 100GB |
| Single document | 512MB |

### Upload File

```bash
curl --request POST \
  --url https://api.minimaxi.com/v1/files/upload \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --header 'Content-Type: multipart/form-data' \
  --form 'file=@/path/to/file.pdf' \
  --form 'purpose="text-to-speech"'
```

### Upload Response

```json
{
  "file_id": "file_xxx",
  "filename": "document.pdf",
  "bytes": 12345,
  "created_at": 1234567890,
  "status": "processed"
}
```

### List Files

```bash
curl --request GET \
  --url https://api.minimaxi.com/v1/files?limit=100 \
  --header 'Authorization: Bearer <YOUR_API_KEY>'
```

### Retrieve File Info

```bash
curl --request GET \
  --url https://api.minimaxi.com/v1/files/{file_id} \
  --header 'Authorization: Bearer <YOUR_API_KEY>'
```

### Download File

```bash
curl --request GET \
  --url https://api.minimaxi.com/v1/files/{file_id}/content \
  --header 'Authorization: Bearer <YOUR_API_KEY>' \
  --output downloaded_file.mp3
```

### Delete File

```bash
curl --request DELETE \
  --url https://api.minimaxi.com/v1/files/{file_id} \
  --header 'Authorization: Bearer <YOUR_API_KEY>'
```

---

## SDK Integration

### Python (LangChain)

```python
from langchain_community.embeddings import MiniMaxEmbeddings
import os

embeddings = MiniMaxEmbeddings(
    minimax_api_key=os.getenv("MINIMAX_API_KEY"),
    minimax_group_id=os.getenv("MINIMAX_GROUP_ID")
)

result = embeddings.embed_query("Hello, world!")
print(result)
```

### Python (Spring AI)

```python
from spring_ai.minimax import MiniMaxChatClient

client = MiniMaxChatClient(
    api_key="YOUR_API_KEY",
    base_url="https://api.minimax.io/v1"
)

response = client.chat(
    model="MiniMax-M2.1",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### JavaScript (LangChain.js)

```javascript
import { MiniMaxEmbeddings } from '@langchain/community/embeddings/minimax';

const embeddings = new MiniMaxEmbeddings({
  apiKey: process.env.MINIMAX_API_KEY,
  groupId: process.env.MINIMAX_GROUP_ID
});

const result = await embeddings.embedQuery("Hello world");
```

### Node.js (MCP Server)

MiniMax provides official MCP (Model Context Protocol) servers:

**Python Version:**
```bash
# Install via uv
pip install minimax-mcp

# or clone repo
git clone https://github.com/MiniMax-AI/MiniMax-MCP.git
```

**JavaScript Version:**
```bash
npm install @minimax-ai/mcp-js
```

### Cursor IDE Setup

1. Open Cursor Settings
2. Go to AI Providers
3. Select MiniMax
4. Configure:
   - Base URL: `https://api.minimaxi.com/v1`
   - API Key: Your MiniMax API key
   - Model: `MiniMax-M2.1` or `MiniMax-M2`

---

## Rate Limits & Pricing

### Rate Limits

| Plan | Requests/Minute | Concurrent |
|------|-----------------|------------|
| Free | Varies | Limited |
| Pay-as-you-go | Higher limits | Increased |
| Enterprise | Custom | Custom |

### File URL Expiration

- **Generated files**: URLs expire after **9 hours** (32,400 seconds)
- **Cloned voices**: Temporary voices expire after **168 hours** (7 days)

### Billing Notes

- **Voice Cloning**: Charged on first use, not during cloning
- **Voice Design**: Charged on first TTS generation
- Make voices permanent by using them within 7 days

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Invalid endpoint or resource |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Support & Resources

### Official Support
- Email: [email protected]
- GitHub Issues: https://github.com/MiniMax-AI
- Documentation: https://platform.minimaxi.com/docs/

### Community
- CSDN Tutorials (Chinese)
- GitHub Community Projects
- Official MCP Guide: https://platform.minimax.io/docs/guides/mcp-guide

### Integration Platforms
- **DeepInfra**: MiniMax model hosting
- **Azure AI**: Managed MiniMax endpoints
- **Google Vertex AI**: MaaS integration
- **fal.ai**: Video generation API
- **Segmind**: Serverless access

---

*Last Updated: 2025*

*Sources:*
- [MiniMax API Overview](https://platform.minimaxi.com/docs/api-reference/api-overview)
- [MiniMax AI API](https://minimax-ai.chat/docs/api/)
- [MiniMax Text Generation](https://platform.minimaxi.com/docs/guides/text-generation)
- [MiniMax Async Speech](https://platform.minimaxi.com/docs/guides/speech-t2a-async)
- [MiniMax Official Website](https://www.minimax.io/)
