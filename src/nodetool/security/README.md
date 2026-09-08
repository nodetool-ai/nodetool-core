# NodeTool Security

Secret resolution for the Python worker.

## Components

- `secret_helper.py` — `get_secret()`, `get_secret_required()`: Secret resolution (env var → default)

## Usage

```python
from nodetool.security.secret_helper import get_secret

api_key = await get_secret("OPENAI_API_KEY", user_id)
```

## Secret Resolution Order

For the lean Python worker, secrets come from environment variables. The TS
server handles database-stored secrets, including encryption at rest, and
passes the resolved values to the worker via env.

1. Environment variable (`os.environ`)
2. Not found (returns the provided default / `None`, or raises for `get_secret_required`)
