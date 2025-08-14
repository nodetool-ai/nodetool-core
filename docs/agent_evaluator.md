## Agent Evaluator

A reusable, provider-agnostic evaluation framework for running agents across multiple models and problems, aggregating token usage and runtime, and validating correctness via a pluggable checker.

### Key Features

- Parallel, process-isolated execution to avoid stdout/stderr bleed and shared provider state
- Pluggable agent factory (`build_agent_fn`) and result checker (`result_checker`)
- Provider-agnostic with a default provider factory (`openai`, `gemini`, `anthropic`, `huggingface:<provider>`)
- Simple aggregated stats per model and per-run logs

### Quick Start

```python
from nodetool.agents.agent_evaluator import AgentEvaluator
from nodetool.agents.simple_agent import SimpleAgent
from nodetool.agents.tools.node_tool import NodeTool

def build_agent(provider, model, tools, problem):
    return SimpleAgent(
        name="My Agent",
        objective=str(problem),
        provider=provider,
        model=model,
        tools=list(tools),
        output_schema={"value": "string"},
    )

def checker(result, expected):
    # Example checker for equality on the "value" field
    value = result.get("value") if isinstance(result, dict) else result
    return value == expected

models = [("openai", "gpt-5-mini"), ("anthropic", "claude-3-5-haiku-20241022")]
problems = [("echo hello", "hello"), ("echo world", "world")]
tools = [NodeTool(...)]  # optional

evaluator = AgentEvaluator(
    models=models,
    problems=problems,
    build_agent_fn=build_agent,
    result_checker=checker,
    tools=tools,
    concurrency=4,
)

result = await evaluator.evaluate()
print(result.stats)
print(result.logs[0])
```

### API

```python
class AgentEvaluator:
    def __init__(
        self,
        models: Sequence[tuple[str, str]],
        problems: Iterable[Any],
        build_agent_fn: Callable[[ChatProvider, str, Sequence[Any], Any], Any],
        result_checker: Callable[[Any, Any], bool],
        tools: Optional[Sequence[Any]] = None,
        concurrency: int = 8,
        provider_factory: Callable[[str], ChatProvider] = default_provider_factory,
    ) -> None: ...

    async def evaluate(self) -> EvaluationResult: ...

@dataclass
class EvaluationResult:
    stats: dict[str, ModelStats]
    logs: list[LogEntry]

@dataclass
class ModelStats:
    finished: int
    correct: int
    input_tokens: int
    output_tokens: int
    total_runtime_seconds: float

@dataclass
class LogEntry:
    provider_key: str
    model: str
    problem: str
    result: Any | None
    correct: bool | None
    runtime_seconds: float
```

### Design Notes

- Each (model, problem) runs in a separate process via `ProcessPoolExecutor` for isolation.
- Token usage is read from `agent.subtask_context` if available and aggregated per model.
- Correctness is entirely determined by your `result_checker` function.
- Problems can be items or `(item, expected)` pairs; if no expected value is provided, `expected` is `None` and your checker can choose its own semantics.

### Extensibility

- Swap in a custom `provider_factory` for alternate provider bootstrapping.
- Extend `ModelStats` or `LogEntry` as needed and compose your own wrapper if additional metrics are required.
