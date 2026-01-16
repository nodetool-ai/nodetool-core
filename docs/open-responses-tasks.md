# OpenResponses API Implementation Tasks

This document provides a detailed breakdown of tasks required to implement the OpenResponses API specification across all compatible providers in NodeTool.

## Milestone 1: Foundation (Core Types and Infrastructure)

**Goal**: Establish the foundational types, mixin class, and basic infrastructure.

### Task 1.1: Create Responses API Type Definitions

**File**: `src/nodetool/providers/responses_types.py`

**Description**: Create comprehensive Pydantic models for all OpenResponses API types.

**Subtasks**:
- [ ] Define input content types (`InputTextContent`, `InputImageContent`, `InputFileContent`, `InputVideoContent`)
- [ ] Define output content types (`OutputTextContent`, `RefusalContent`, `ReasoningTextContent`, `SummaryTextContent`)
- [ ] Define message item types (`ResponseMessage`, `FunctionCallItem`, `FunctionCallOutputItem`, `ReasoningItem`)
- [ ] Define tool types (`FunctionTool`, `ToolChoice`, `SpecificFunctionChoice`, `AllowedToolsChoice`)
- [ ] Define configuration types (`ReasoningConfig`, `TextConfig`, `ResponseUsage`)
- [ ] Define request/response types (`CreateResponseRequest`, `Response`)
- [ ] Define streaming event types (all `response.*` events)
- [ ] Add comprehensive docstrings and examples

**Acceptance Criteria**:
- All types match the OpenResponses OpenAPI specification
- Types are serializable/deserializable with Pydantic
- Type validation tests pass
- Types are properly exported in `__init__.py`

**Estimated Effort**: 4 hours

---

### Task 1.2: Create OpenResponses Mixin Class

**File**: `src/nodetool/providers/open_responses_mixin.py`

**Description**: Create the base mixin class that provides Responses API interface.

**Subtasks**:
- [ ] Define abstract `create_response()` method
- [ ] Define abstract `create_response_stream()` method
- [ ] Implement `convert_messages_to_items()` utility
- [ ] Implement `convert_tools_to_function_tools()` utility
- [ ] Implement `convert_response_to_message()` utility for backward compatibility
- [ ] Add helper methods for common operations
- [ ] Add comprehensive docstrings

**Acceptance Criteria**:
- Mixin follows Python best practices
- Abstract methods are properly defined
- Conversion utilities handle all message types
- Unit tests cover all utility methods

**Estimated Effort**: 3 hours

---

### Task 1.3: Update Base Provider

**File**: `src/nodetool/providers/base.py`

**Description**: Add Responses API capability detection to BaseProvider.

**Subtasks**:
- [ ] Add `RESPONSES_API` to `ProviderCapability` enum
- [ ] Add `supports_responses_api()` method
- [ ] Update capability detection logic
- [ ] Add documentation for new capability

**Acceptance Criteria**:
- Capability detection works correctly
- No breaking changes to existing code
- Tests verify capability detection

**Estimated Effort**: 1 hour

---

### Task 1.4: Create Unit Tests for Foundation

**Files**: 
- `tests/providers/test_responses_types.py`
- `tests/providers/test_open_responses_mixin.py`

**Description**: Create comprehensive unit tests for new types and mixin.

**Subtasks**:
- [ ] Test all type serialization/deserialization
- [ ] Test type validation (required fields, constraints)
- [ ] Test message-to-item conversion
- [ ] Test tool conversion
- [ ] Test response-to-message conversion
- [ ] Test edge cases (empty content, None values)

**Acceptance Criteria**:
- 90%+ code coverage for new modules
- All tests pass
- Edge cases are covered

**Estimated Effort**: 3 hours

---

## Milestone 2: OpenAI Provider Implementation

**Goal**: Implement full Responses API support for OpenAI provider.

### Task 2.1: Implement OpenAI Responses API

**File**: `src/nodetool/providers/openai_provider.py`

**Description**: Add OpenResponses mixin and implement Responses API methods.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Implement `create_response()` method
- [ ] Implement `create_response_stream()` method
- [ ] Implement `_convert_input_items()` helper
- [ ] Implement `_convert_tool_choice()` helper
- [ ] Implement `_parse_stream_event()` helper
- [ ] Add proper error handling
- [ ] Add logging for debugging
- [ ] Handle OpenAI-specific features (reasoning, citations)

**Acceptance Criteria**:
- Non-streaming responses work correctly
- Streaming responses emit all event types
- Tool calling works with function outputs
- Reasoning models are supported
- Error handling is comprehensive
- Backward compatibility maintained

**Estimated Effort**: 6 hours

---

### Task 2.2: Create OpenAI Integration Tests

**File**: `tests/providers/test_openai_responses.py`

**Description**: Create integration tests for OpenAI Responses API.

**Subtasks**:
- [ ] Test basic text completion
- [ ] Test multi-turn conversation
- [ ] Test function calling
- [ ] Test streaming responses
- [ ] Test reasoning model support
- [ ] Test multi-modal inputs (text + image)
- [ ] Test error handling
- [ ] Test response conversion utilities

**Acceptance Criteria**:
- Tests can run with mock API (for CI)
- Tests cover all major use cases
- Tests are skipped gracefully without API key

**Estimated Effort**: 4 hours

---

## Milestone 3: OpenRouter Provider Implementation

**Goal**: Implement Responses API support for OpenRouter.

### Task 3.1: Implement OpenRouter Responses API

**File**: `src/nodetool/providers/openrouter_provider.py`

**Description**: Extend OpenAI implementation for OpenRouter.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Override `create_response()` if needed for OpenRouter specifics
- [ ] Override `create_response_stream()` if needed
- [ ] Handle OpenRouter-specific headers
- [ ] Handle cost tracking from usage data
- [ ] Test with various OpenRouter models

**Acceptance Criteria**:
- OpenRouter-specific headers are included
- Cost tracking works correctly
- All OpenAI features work via OpenRouter
- Model-specific limitations handled gracefully

**Estimated Effort**: 3 hours

---

### Task 3.2: Create OpenRouter Integration Tests

**File**: `tests/providers/test_openrouter_responses.py`

**Description**: Create integration tests for OpenRouter Responses API.

**Subtasks**:
- [ ] Test basic responses
- [ ] Test streaming
- [ ] Test cost tracking
- [ ] Test model selection
- [ ] Test error handling

**Acceptance Criteria**:
- Tests validate OpenRouter-specific behavior
- Cost tracking is verified
- Tests handle API unavailability gracefully

**Estimated Effort**: 2 hours

---

## Milestone 4: HuggingFace Provider Implementation

**Goal**: Implement Responses API support for HuggingFace Inference Providers.

### Task 4.1: Implement HuggingFace Responses API

**File**: `src/nodetool/providers/huggingface_provider.py`

**Description**: Add Responses API support using AsyncInferenceClient.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Implement `create_response()` using chat_completion
- [ ] Implement `create_response_stream()` with streaming
- [ ] Map HuggingFace message format to Responses API items
- [ ] Handle multi-modal inputs (images)
- [ ] Handle tool calling where supported
- [ ] Add provider-specific error handling

**Acceptance Criteria**:
- Works with all HuggingFace inference providers
- Gracefully handles unsupported features
- Multi-modal inputs work correctly
- Streaming works with proper events

**Estimated Effort**: 5 hours

---

### Task 4.2: Create HuggingFace Integration Tests

**File**: `tests/providers/test_huggingface_responses.py`

**Description**: Create integration tests for HuggingFace Responses API.

**Subtasks**:
- [ ] Test with different inference providers
- [ ] Test text completion
- [ ] Test streaming
- [ ] Test multi-modal inputs
- [ ] Test tool calling (where supported)

**Acceptance Criteria**:
- Tests cover multiple inference providers
- Feature detection works correctly
- Unsupported features fail gracefully

**Estimated Effort**: 3 hours

---

## Milestone 5: Local Provider Implementations

**Goal**: Implement Responses API for Ollama, LMStudio, and vLLM.

### Task 5.1: Implement Ollama Responses API

**File**: `src/nodetool/providers/ollama_provider.py`

**Description**: Add Responses API support for Ollama.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Implement `create_response()` using Ollama client
- [ ] Implement `create_response_stream()` for streaming
- [ ] Handle tool calling (native and emulated)
- [ ] Map Ollama response format to Responses API
- [ ] Handle model-specific capabilities

**Acceptance Criteria**:
- Works with Ollama's chat API
- Tool calling works (native or emulated)
- Streaming works correctly
- Multi-modal inputs work (for vision models)

**Estimated Effort**: 4 hours

---

### Task 5.2: Implement LMStudio Responses API

**File**: `src/nodetool/providers/lmstudio_provider.py`

**Description**: Add Responses API support for LMStudio.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Implement `create_response()` using OpenAI-compat endpoint
- [ ] Implement `create_response_stream()` for streaming
- [ ] Handle LMStudio-specific limitations
- [ ] Test with various local models

**Acceptance Criteria**:
- Works with LMStudio's OpenAI-compatible API
- Handles unsupported features gracefully
- Streaming works correctly

**Estimated Effort**: 3 hours

---

### Task 5.3: Implement vLLM Responses API

**File**: `src/nodetool/providers/vllm_provider.py`

**Description**: Add Responses API support for vLLM.

**Subtasks**:
- [ ] Add `OpenResponsesMixin` to class inheritance
- [ ] Implement `create_response()` using vLLM OpenAI-compat API
- [ ] Implement `create_response_stream()` for streaming
- [ ] Handle vLLM-specific parameters
- [ ] Optimize for vLLM performance features

**Acceptance Criteria**:
- Works with vLLM's OpenAI-compatible API
- Performance optimizations utilized
- Streaming works correctly

**Estimated Effort**: 3 hours

---

### Task 5.4: Create Local Provider Integration Tests

**Files**:
- `tests/providers/test_ollama_responses.py`
- `tests/providers/test_lmstudio_responses.py`
- `tests/providers/test_vllm_responses.py`

**Description**: Create integration tests for local providers.

**Subtasks**:
- [ ] Test basic responses for each provider
- [ ] Test streaming for each provider
- [ ] Test tool calling where supported
- [ ] Test multi-modal where supported
- [ ] Handle provider unavailability gracefully

**Acceptance Criteria**:
- Tests skip gracefully when provider unavailable
- Tests cover core functionality
- Tests verify streaming behavior

**Estimated Effort**: 4 hours

---

## Milestone 6: Multi-Modal Support

**Goal**: Ensure comprehensive multi-modal input/output support.

### Task 6.1: Implement Multi-Modal Input Handling

**Files**: 
- `src/nodetool/providers/responses_types.py`
- `src/nodetool/providers/open_responses_mixin.py`

**Description**: Ensure all input modalities are properly handled.

**Subtasks**:
- [ ] Implement image input conversion (URL and base64)
- [ ] Implement file input conversion
- [ ] Implement video input conversion
- [ ] Add input validation
- [ ] Add size limits and format validation
- [ ] Handle mixed content arrays

**Acceptance Criteria**:
- All input types serialize correctly
- Images can be sent as URL or base64
- Files can be sent as URL or base64
- Input validation prevents malformed requests

**Estimated Effort**: 3 hours

---

### Task 6.2: Create Multi-Modal Tests

**File**: `tests/providers/test_responses_multimodal.py`

**Description**: Comprehensive tests for multi-modal inputs/outputs.

**Subtasks**:
- [ ] Test text-only inputs
- [ ] Test image inputs (URL)
- [ ] Test image inputs (base64)
- [ ] Test file inputs
- [ ] Test video inputs
- [ ] Test mixed content arrays
- [ ] Test provider-specific limitations

**Acceptance Criteria**:
- All input modalities tested
- Provider-specific limitations documented
- Edge cases covered (empty, oversized, invalid)

**Estimated Effort**: 4 hours

---

## Milestone 7: Integration and Documentation

**Goal**: Integrate with existing systems and document the implementation.

### Task 7.1: Update Agent System Integration

**Files**: 
- `src/nodetool/agents/` (various files)

**Description**: Optionally enable Responses API in agent workflows.

**Subtasks**:
- [ ] Add configuration option for Responses API usage
- [ ] Update agent provider selection logic
- [ ] Add conversion layer for backward compatibility
- [ ] Test agent workflows with Responses API

**Acceptance Criteria**:
- Agents can optionally use Responses API
- Backward compatibility maintained
- No breaking changes to existing workflows

**Estimated Effort**: 4 hours

---

### Task 7.2: Update Provider Exports

**File**: `src/nodetool/providers/__init__.py`

**Description**: Export all new types and utilities.

**Subtasks**:
- [ ] Export Responses API types
- [ ] Export mixin class
- [ ] Export utility functions
- [ ] Update docstrings

**Acceptance Criteria**:
- All public APIs are exported
- Import paths are clean and intuitive

**Estimated Effort**: 1 hour

---

### Task 7.3: Create API Documentation

**File**: `docs/providers/responses-api.md`

**Description**: Document the Responses API implementation.

**Subtasks**:
- [ ] Document available types
- [ ] Document mixin usage
- [ ] Document provider-specific features
- [ ] Add code examples
- [ ] Document migration path

**Acceptance Criteria**:
- Documentation is comprehensive
- Examples are working code
- Provider differences are explained

**Estimated Effort**: 3 hours

---

## Milestone 8: Stretch Goals

**Goal**: Implement advanced features beyond basic feature parity.

### Task 8.1: Response Retrieval and Management

**Files**: Provider files

**Description**: Implement response storage and retrieval.

**Subtasks**:
- [ ] Add `get_response(response_id)` method
- [ ] Add `list_responses()` method
- [ ] Add `delete_response(response_id)` method
- [ ] Add `cancel_response(response_id)` method
- [ ] Handle response storage configuration

**Acceptance Criteria**:
- Response retrieval works for stored responses
- Response listing with pagination
- Proper cleanup on delete

**Estimated Effort**: 4 hours

---

### Task 8.2: Background Response Generation

**Files**: Provider files

**Description**: Support for background/async response generation.

**Subtasks**:
- [ ] Implement background request submission
- [ ] Implement polling for completion
- [ ] Handle background response cancellation
- [ ] Add timeout handling

**Acceptance Criteria**:
- Background responses work for long operations
- Polling is efficient
- Cancellation works correctly

**Estimated Effort**: 4 hours

---

### Task 8.3: Response Caching

**Files**: Provider files

**Description**: Implement prompt caching with cache keys.

**Subtasks**:
- [ ] Support `prompt_cache_key` parameter
- [ ] Track cache hits in usage
- [ ] Handle cache invalidation

**Acceptance Criteria**:
- Caching reduces latency for repeated prompts
- Cache usage is tracked
- Cache key collisions handled

**Estimated Effort**: 3 hours

---

### Task 8.4: Extended Output Types

**Files**: Provider files + types

**Description**: Support for multi-modal outputs (images, audio, video).

**Subtasks**:
- [ ] Add output image content type
- [ ] Add output audio content type
- [ ] Add output video content type
- [ ] Integrate with existing generation capabilities

**Acceptance Criteria**:
- Output images work with DALL-E models
- Output audio works with TTS models
- Proper type handling in responses

**Estimated Effort**: 5 hours

---

## Test Plan Summary

### Unit Tests

| Test File | Coverage Target | Description |
|-----------|----------------|-------------|
| `test_responses_types.py` | 95% | Type serialization, validation |
| `test_open_responses_mixin.py` | 90% | Mixin methods, conversions |

### Integration Tests

| Test File | Coverage Target | Description |
|-----------|----------------|-------------|
| `test_openai_responses.py` | Core APIs | OpenAI Responses API |
| `test_openrouter_responses.py` | Core APIs | OpenRouter Responses API |
| `test_huggingface_responses.py` | Core APIs | HuggingFace Responses API |
| `test_ollama_responses.py` | Core APIs | Ollama Responses API |
| `test_lmstudio_responses.py` | Core APIs | LMStudio Responses API |
| `test_vllm_responses.py` | Core APIs | vLLM Responses API |

### Multi-Modal Tests

| Test File | Coverage Target | Description |
|-----------|----------------|-------------|
| `test_responses_multimodal.py` | All modalities | Input/output type handling |

### End-to-End Tests

| Test | Description |
|------|-------------|
| Conversation flow | Multi-turn conversation with context |
| Tool calling round-trip | Function call → execution → result |
| Streaming validation | All event types received correctly |
| Error recovery | Graceful handling of failures |

---

## Estimated Timeline

| Milestone | Estimated Duration | Dependencies |
|-----------|-------------------|--------------|
| M1: Foundation | 2 days | None |
| M2: OpenAI | 2 days | M1 |
| M3: OpenRouter | 1 day | M2 |
| M4: HuggingFace | 2 days | M1 |
| M5: Local Providers | 2 days | M1 |
| M6: Multi-Modal | 2 days | M2, M4, M5 |
| M7: Integration | 2 days | M2-M6 |
| M8: Stretch Goals | 3 days | M7 |

**Total Estimated Duration**: 16 days (main goals) + 3 days (stretch goals)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Provider API differences | Medium | Implement feature detection, graceful degradation |
| Streaming event compatibility | Medium | Test with real APIs, mock complex scenarios |
| Performance regression | Low | Benchmark against existing implementation |
| Breaking changes | High | Maintain backward compatibility layer |
| API specification changes | Low | Monitor OpenResponses spec updates |

---

## Success Criteria

1. **Feature Parity**: All current chat completion features work with Responses API
2. **Provider Coverage**: All 6 target providers implemented and tested
3. **Multi-Modal**: Text, image, and file inputs work correctly
4. **Streaming**: All streaming event types are properly handled
5. **Backward Compatible**: Existing code continues to work unchanged
6. **Well Tested**: 90%+ code coverage on new code
7. **Documented**: Comprehensive documentation and examples

---

## Notes

- Priority should be given to OpenAI and OpenRouter as they have native/near-native support
- Local providers (Ollama, LMStudio, vLLM) may require more emulation
- HuggingFace implementation depends on inference provider capabilities
- Multi-modal support varies significantly by provider and model
- Consider feature detection mechanisms for graceful degradation
