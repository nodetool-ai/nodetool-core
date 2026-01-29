# xllamacpp Provider Integration

## Overview

This document describes the integration of xllamacpp as a high-performance local LLM inference provider for NodeTool Core. xllamacpp provides a Cython-based Python wrapper around llama.cpp with built-in optimizations and Pythonic management.

## Why xllamacpp?

xllamacpp offers several advantages over the previous external llama-server approach:

1. **Pythonic Management**: No external process management required - models are loaded directly in Python
2. **Memory Estimation**: Automatic GPU layer offloading using built-in memory estimation
3. **Platform Optimization**: Pre-built wheels for CPU, CUDA (NVIDIA), Vulkan (AMD/Intel), and Metal (Apple Silicon)
4. **Thread Safety**: Built-in continuous batching with thread-safe server
5. **Performance**: High-performance Cython bindings with optimizations enabled by default
6. **Simplicity**: Single pip install - no need to manage external binaries

## Implementation Details

### Provider Class: `XLlamaCppProvider`

Location: `src/nodetool/providers/xllamacpp_provider.py`

The provider implements:
- OpenAI-compatible API via xllamacpp's built-in Server class
- Automatic model server caching (one server instance per model path)
- Message normalization for llama.cpp chat template constraints
- Tool calling via text-based emulation (similar to Ollama fallback)
- Token counting using tiktoken
- Automatic GPU layer estimation when GPUs are available

### Key Features

1. **Model Loading**
   ```python
   # Models can be specified as:
   # - Path to .gguf file: "/path/to/model.gguf"
   # - HuggingFace repo: "ggml-org/Llama-3.2-1B-Instruct-Q8_0-GGUF"
   ```

2. **Automatic GPU Offloading**
   ```python
   # xllamacpp automatically estimates optimal GPU layers
   devices = xlc.get_device_info()
   estimate = xlc.estimate_gpu_layers(
       devices, model, [],
       context_length=8192,
       batch_size=512,
       num_parallel=1
   )
   # estimate.layers = optimal number of layers for GPU
   ```

3. **Server Management**
   - One xllamacpp Server instance is created per unique model
   - Servers are cached and reused across requests
   - Each server exposes an OpenAI-compatible HTTP endpoint
   - No TTL or cleanup - servers run for the lifetime of the provider

## Configuration

### Environment Variables

| Variable                   | Description                                | Default | Required |
|---------------------------|--------------------------------------------|---------|----------|
| `XLLAMACPP_N_CTX`         | Context window size in tokens             | 8192    | No       |
| `XLLAMACPP_N_GPU_LAYERS`  | GPU layers to offload (empty = auto)      | auto    | No       |
| `XLLAMACPP_N_THREADS`     | CPU threads (empty = auto)                | auto    | No       |
| `XLLAMACPP_PARALLEL`      | Parallel sequences                        | 1       | No       |
| `XLLAMACPP_CACHE_RAM_MIB` | RAM cache size in MiB                     | 2048    | No       |

### Installation

Platform-specific installation via pip:

```bash
# CPU / Mac (Metal) - Default PyPI
pip install xllamacpp

# NVIDIA GPU - CUDA 12.8
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128

# NVIDIA GPU - CUDA 12.4
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124

# AMD/Intel GPU - Vulkan
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan
```

### Optional Dependency

xllamacpp is configured as an optional dependency in `pyproject.toml`:

```toml
[project.optional-dependencies]
xllamacpp = [
    "xllamacpp>=0.1.0",
]
```

## Usage

### Basic Usage

```python
from nodetool.providers import get_provider
from nodetool.metadata.types import Provider, Message

# Get provider instance
provider = await get_provider(Provider.XLlamaCpp, user_id="1")

# Generate a response
messages = [Message(role="user", content="What is 2+2?")]
response = await provider.generate_message(
    messages=messages,
    model="ggml-org/Llama-3.2-1B-Instruct-Q8_0-GGUF",
    max_tokens=100
)

print(response.content)
```

### With Tool Calling

```python
from nodetool.agents.tools.math_tools import CalculatorTool

tools = [CalculatorTool()]

response = await provider.generate_message(
    messages=messages,
    model="model.gguf",
    tools=tools,
    max_tokens=512
)

# Tool calls are emulated via text parsing
if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.args}")
```

### Streaming

```python
async for chunk in provider.generate_messages(
    messages=messages,
    model="model.gguf",
    max_tokens=512
):
    if hasattr(chunk, 'content'):
        print(chunk.content, end='', flush=True)
```

## Testing

Tests are located in `tests/chat/providers/test_xllamacpp_provider.py`.

To run tests:
```bash
# Skip if xllamacpp not installed
pytest tests/chat/providers/test_xllamacpp_provider.py -v

# Install xllamacpp first
pip install xllamacpp
pytest tests/chat/providers/test_xllamacpp_provider.py -v
```

Tests cover:
- Server initialization and lifecycle
- Message normalization for chat templates
- Tool calling emulation
- Streaming responses
- GPU layer estimation
- Model caching
- Error handling

## Comparison with llama-server

| Feature                    | xllamacpp Provider          | llama-server (LlamaProvider)    |
|---------------------------|-----------------------------|----------------------------------|
| **Process Management**    | In-process (Python)         | External binary                  |
| **Installation**          | pip install                 | Separate binary installation     |
| **Platform Builds**       | Pre-built wheels            | Manual compilation               |
| **Memory Estimation**     | Built-in                    | Manual configuration             |
| **GPU Offloading**        | Automatic detection         | Manual layer count               |
| **Server Startup**        | Instant (lazy)              | Process spawn overhead           |
| **Thread Safety**         | Yes                         | Depends on configuration         |
| **Model Caching**         | Per-model in-memory         | External process per model       |
| **API Compatibility**     | OpenAI-compatible           | OpenAI-compatible                |

## Migration Guide

### From llama-server

1. **Install xllamacpp:**
   ```bash
   pip install xllamacpp
   # or with GPU support
   pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/cu128
   ```

2. **Update provider type:**
   ```python
   # Old
   provider = await get_provider(Provider.LlamaCpp, user_id)

   # New
   provider = await get_provider(Provider.XLlamaCpp, user_id)
   ```

3. **Update environment variables:**
   ```bash
   # Old
   LLAMA_CPP_URL=http://localhost:8080
   LLAMA_CPP_CONTEXT_LENGTH=8192

   # New
   XLLAMACPP_N_CTX=8192
   XLLAMACPP_N_GPU_LAYERS=  # leave empty for auto-detection
   ```

4. **Remove external process management:**
   - No need to start/manage llama-server binary
   - No need for LLAMA_SERVER_BINARY environment variable
   - No need for health checks or readiness probes

### Benefits of Migration

1. **Simpler Deployment**: Single pip install, no binary management
2. **Better Resource Management**: Automatic GPU layer estimation
3. **Improved Performance**: Optimized Cython bindings
4. **Cross-Platform**: Pre-built wheels for all major platforms
5. **Easier Development**: No external dependencies to manage

## Known Limitations

1. **Tool Calling**: Uses text-based emulation (no native tool support yet)
2. **Model Discovery**: Models loaded on demand, not pre-listed
3. **Memory Usage**: Server instances persist for provider lifetime
4. **Platform Support**: Pre-built wheels require compatible glibc (Linux)

## Future Enhancements

1. Model pre-warming and health checks
2. Server instance TTL and cleanup
3. Multi-model batching optimization
4. Native tool calling support (when available upstream)
5. Model metadata caching
6. Better error messages and diagnostics

## References

- xllamacpp GitHub: https://github.com/xorbitsai/xllamacpp
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Provider README: `src/nodetool/providers/README.md`
- Settings Documentation: `src/nodetool/config/settings.py`
