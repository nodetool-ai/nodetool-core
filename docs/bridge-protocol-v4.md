# Bridge protocol v4 — run identity and the run boundary

Worker-side reference for bridge protocol v4. The wire contract itself lives in
`docs/python-bridge-protocol.md` in the `nodetool` repo; this document covers
what the Python worker does with it and the decisions that are not visible from
the wire.

`BRIDGE_PROTOCOL_VERSION` is `4` (`src/nodetool/worker/__init__.py`), in
lockstep with `packages/protocol/src/bridge-protocol.ts` on the JS side. The
floor does not move: `MIN_BRIDGE_PROTOCOL_VERSION` stays at 1 and
`MIN_NODETOOL_CORE_VERSION` at 0.7.0. Nothing here is a wire break.

## The problem v4 solves

Before v4 the worker saw an unlabeled stream of single-node executions.
`execute` was `{node_type, fields, secrets, blobs}` and nothing more, with three
consequences:

- `_prepare_node()` called `node_class()` with no id, so `node._id` was `""` for
  every execution. A node calling `set_model(self._id, …)` registered under the
  empty string, `ModelManager._models_by_node` was effectively one bucket, and
  `release_nodes()` had nothing meaningful to release.
- `WorkerContext` never received a workflow or a user.
- Nothing ever fired at the point where "these nodes are retired, their models
  are eligible" is true, so the model cache grew across runs and was only
  trimmed reactively, under memory pressure.

## Run identity on `execute` / `execute.stream`

Five new keys on `data`, parsed by `executor.read_run_identity()`:

| key | effect |
|---|---|
| `node_id` | passed to `node_class(id=…)`, so `node._id` is the real graph id |
| `job_id` | attributes the execution to a run via `JobRegistry` |
| `workflow_id` | populates `WorkerContext` |
| `user_id` | populates `WorkerContext` |
| `requires_vram_gb` | sizes the pre-execution reclaim pass |

**Every key is optional and absent keys reproduce the pre-v4 behaviour
exactly.** The JS side omits a field it cannot name rather than sending `null`,
so `data.get("node_id") is None` is the normal old-client path, not an error.
Wrong-typed values degrade to "no identity" for the same reason — the identity
is an optimization, and a malformed one must not fail an otherwise valid
execution.

These keys arrive on *every* execute, including from a v4 JS runtime talking to
a pre-v4 worker: they are extra dict entries, not a new message, and a worker
that only reads the four keys it knows ignores them cleanly.

`node_id` is the load-bearing one. It is what turns `_models_by_node` from a
single bucket into a real map, which is what makes `release_nodes()` mean
anything.

## `job.start` / `job.end`

Handled by `worker/job_handler.py` over `worker/job_registry.py`. Both answer
with a normal `result` frame — the JS side waits on `result`/`error` and would
hang otherwise.

- **`job.start`** opens a run and does the one reclaim pass per run, instead of
  one per node. The worker does not need it to attribute an execution (every
  `execute` carries its own `job_id`); a job first seen on an `execute` is
  created implicitly.
- **`job.end`** retires the job's nodes and releases their models. This is the
  caller `release_nodes()` never had.

Three properties the implementation guarantees:

- **Idempotent, and tolerant of an unknown job.** A `job.end` for a job the
  worker never saw — a run whose nodes all failed before reaching the worker, a
  duplicate, a client that reconnected — is a no-op that reports
  `{"released_nodes": [], "known": false}`, not an error. A second `job.end` for
  a job already closed behaves the same way.
- **`reason` does not branch the release.** An abnormal end retires exactly the
  same nodes as a clean one. The label is carried into logs and the result
  frame only; branching on it would move the leak to the failure path.
- **A `job.end` that never arrives is backstopped.** See below.

The JS side logs and swallows `job.*` failures on purpose — a `job.end` that
fails against a worker already tearing down must not turn a finished run into a
failed one. That means a broken handler here is **silent** from the JS side, so
the tests assert on the worker's own state (`ModelManager._models`,
`_models_by_node`, `JobRegistry`), never on the fact that a frame came back.

### The backstop decision

The JS side sends `job.end` from the same `finally` that closes the bridge, so
cancel, timeout and failure all reach it. A hard-killed client sends nothing.
Three backstops were available; the worker uses the **TTL sweep**.

- **TTL sweep over jobs with no activity — chosen.** Transport-agnostic, needs
  no background timer (it runs opportunistically off `start` /
  `note_execution` / `end`, all already on the message path), and it cannot
  retire a live job because any activity refreshes the job's timestamp. A job
  idle longer than `NODETOOL_WORKER_JOB_TTL_SECONDS` (default 1800; `0`
  disables) is released as if `job.end` with `reason="abandoned"` had arrived.
  The TTL is deliberately long — it is a leak backstop for a dead client, not a
  scheduler — and models a currently-executing node holds are protected
  regardless by `ModelManager`'s execution scopes.
- **Release on transport disconnect — rejected.** The WebSocket worker shares
  one `WorkerProtocolServer` across connections, so one client dropping would
  retire another client's live jobs. Correct only for the strictly 1:1 stdio
  transport, and a backstop that is right for one transport and wrong for the
  other is worse than none.
- **Rely on the existing reactive reclaim — rejected as the only backstop.**
  That is exactly the behaviour v4 exists to stop relying on: it trims under
  memory pressure, after the fact.

## `models.evict`

`data` (all optional): `node_ids`, `job_id`, `target_vram_gb`. Response:
`{evicted: [...], freed_vram_gb?: number}`. Backed by
`ModelManager.evict_models()`.

- `node_ids` and `job_id` narrow *which* models are candidates (a `job_id`
  resolves to the nodes that job executed); `target_vram_gb` bounds *how much*
  gets dropped, stopping once the target is reclaimed rather than dropping every
  loaded weight. Eviction is coldest-first, so a partial pass keeps the hot
  models resident.
- With no scope at all, everything eligible is evicted.
- A scope that resolves to nothing (an unknown `job_id`) evicts **nothing**: a
  scope was asked for, so an empty scope must not silently widen into
  "everything".
- Keys another node outside the scope still references are kept, and models an
  in-flight execution pins are never dropped — that would be a correctness bug,
  not a memory win.

This is the path for what only the JS side knows: the user switched workflows,
another process wants the GPU, the worker is idle.

## `requires_vram_gb`

Nodes declare their approximate VRAM need via
`BaseNode.get_required_vram_gb()`, which returns `None` by default. `discover`
carries `requires_vram_gb` per node **only when a node declares one** — omitted,
not null, where the worker genuinely does not know, because the JS side treats
absent as "no hint" and an invented number is worse than none. A broken override
degrades to no hint rather than dropping the node from `discover`.

The JS side echoes the hint back on `execute`, where it becomes
`ModelManager.free_vram_if_needed(required_free_gb=…)`. With it the
pre-execution reclaim pass targets the amount the node is about to load and
reclaims once; without it it falls back to the percentage threshold, which can
only trickle because it has no idea what is coming.

## `BaseNode.initialize()` removed

`initialize(context, skip_cache=False)` had no caller anywhere in this repo. An
enumeration of every published node package — `nodetool-huggingface`,
`nodetool-mlx`, `nodetool-comfy`, `nodetool-lib-audio`, `nodetool-lib-ml`,
`nodetool-apple`, `nodetool-whispercpp`, `nodetool-chroma`, `nodetool-sdk` —
found **zero** overrides. The only two overrides anywhere are in the archived
`nodetool-base`, whose nodes moved to the TypeScript server, and one of those
two does not even match the documented signature. The `skip_cache` parameter
belonged to a workflow runner that no longer exists in Python.

The executor's lifecycle is `pre_process → preload_model → move_to_device →
process`, with `finalize` afterwards on every path. `initialize` did not slot
into it, so it was deleted rather than left as a third documented-but-dead hook —
the same shape of confusion that produced the `clear_unused` contract bug. Node
authors who want the old behaviour want `preload_model`.

## Verification

- `tests/worker/test_bridge_protocol_v4.py` covers run identity, the job
  boundary, the TTL backstop, `models.evict`, the VRAM hint, and a leak
  reproduction that runs five workflows and shows the cache at
  `[3, 6, 9, 12, 15]` models without `job.end` and `[0, 0, 0, 0, 0]` with it.
- `tests/worker/test_bridge_frame_contract.py` validates the frames the worker
  emits against `tests/worker/fixtures/bridge-frames.schema.json`, a verbatim
  copy of the `nodetool` repo's generated `dist/bridge-frames.schema.json`, so
  the two sides of the bridge cannot silently drift. See that fixture
  directory's README for how to refresh it.
