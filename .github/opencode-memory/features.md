# Feature Inventory

- **Workflow Engine**: Executes async workflows in Python
- **Node System**: Typed nodes with media refs and processing context
- **Agent Framework**: LLM-backed planning and tool execution
- **API Server**: FastAPI HTTP/WebSocket services
- **Storage Layer**: Pluggable persistence backends
- **Async Utilities**: `AsyncSemaphore` and `gather_with_concurrency` for async concurrency control
- **Async Iterators**: `AsyncByteStream` for async byte sequence iteration in chunks
- **Retry Utilities**: `retry_with_exponential_backoff` and `RetryPolicy` for handling transient failures
- **Timeout Utilities**: `timeout`, `with_timeout`, `TimeoutPolicy`, and `TimeoutContext` for async timeout control
- **Circuit Breaker**: `CircuitBreaker`, `CircuitBreakerError`, and `CircuitState` for protecting against cascading failures
