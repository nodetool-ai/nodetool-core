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
- **Batching Utilities**: `batched_async_iterable` and `process_in_batches` for async batch processing
- **Rate Limiting**: `AsyncTokenBucket` and `AsyncRateLimiter` for controlling operation rates (token bucket algorithm)
- **Circuit Breaker**: `CircuitBreaker`, `MultiCircuitBreaker`, and `CircuitBreakerConfig` for preventing cascading failures in distributed systems
- **Async Lock**: `AsyncLock` for exclusive resource access with timeout support, complementing `AsyncSemaphore`
- **Task Group Management**: `AsyncTaskGroup` for managing groups of related async tasks with spawn, run, cancel, and result collection
- **Async Event**: `AsyncEvent` for inter-task signaling with one-shot and auto-reset modes, supports value passing and predicate-based waiting
