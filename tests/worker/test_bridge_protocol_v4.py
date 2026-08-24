"""Bridge protocol v4: run identity, the job.* boundary, models.evict.

The JS side deliberately logs and swallows `job.*` failures — a `job.end` that
fails against a worker already tearing down must not turn a finished run into a
failed one — so a broken handler here is *silent* from the JS side. Everything
below therefore asserts on the worker's own state (`ModelManager._models`,
`ModelManager._models_by_node`, `JobRegistry`), never on the fact that a frame
came back.
"""

import asyncio
from typing import Any

import pytest
from pydantic import Field

from nodetool.ml.core.model_manager import ModelManager
from nodetool.worker.executor import execute_node, read_run_identity
from nodetool.worker.job_registry import JobRegistry
from nodetool.worker.protocol import WorkerProtocolServer
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.processing_context import ProcessingContext


class FakeModel:
    """Stand-in for a loaded weight set.

    Exposes ``device``/``numel``/``element_size`` so ``ModelManager``'s size
    detection records a real byte count for it — that is what lets the
    ``target_vram_gb`` accounting be exercised without a GPU.
    """

    def __init__(self, size_bytes: int) -> None:
        self.device = "cpu"
        self._size = size_bytes
        self.moved_to_cpu = False

    def numel(self) -> int:
        return self._size

    def element_size(self) -> int:
        return 1

    def to(self, device: str) -> "FakeModel":
        if device == "cpu":
            self.moved_to_cpu = True
        return self


GB = 1024**3


class ModelLoadingNode(BaseNode):
    """Registers a model under its own node id, the way a real ML node does."""

    model_name: str = Field(default="weights")

    @classmethod
    def get_node_type(cls) -> str:
        return "test.ModelLoadingNode"

    async def process(self, context: ProcessingContext) -> str:
        ModelManager.set_model(self._id, self.model_name, FakeModel(GB))
        return self._id


class VramHintNode(BaseNode):
    """Declares a VRAM hint, so discover has something to report."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.VramHintNode"

    @classmethod
    def get_required_vram_gb(cls) -> float | None:
        return 6.5

    async def process(self, context: ProcessingContext) -> str:
        return "ok"


class QuietNode(BaseNode):
    """No VRAM hint — the honest default for a node that does not know."""

    @classmethod
    def get_node_type(cls) -> str:
        return "test.QuietNode"

    async def process(self, context: ProcessingContext) -> str:
        return "ok"


class RecordingTransport:
    """Minimal WorkerTransport that keeps every frame it was handed."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.frames.append(msg)


@pytest.fixture(autouse=True)
def clean_state():
    ModelManager.clear()
    JobRegistry.clear()
    yield
    ModelManager.clear()
    JobRegistry.clear()


# ---------------------------------------------------------------------------
# §2 — run identity on execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_models_by_node_keys_on_the_real_node_id():
    """The specific bug: every node used to register under "".

    A test that only asserted "release_nodes ran" would have passed before this
    change too — so assert on the *key*.
    """
    await execute_node(
        node_type="test.ModelLoadingNode",
        fields={"model_name": "weights-a"},
        secrets={},
        input_blobs={},
        node_id="graph-node-1",
    )
    await execute_node(
        node_type="test.ModelLoadingNode",
        fields={"model_name": "weights-b"},
        secrets={},
        input_blobs={},
        node_id="graph-node-2",
    )

    assert set(ModelManager._models_by_node) == {"graph-node-1", "graph-node-2"}
    assert "" not in ModelManager._models_by_node
    assert ModelManager._models_by_node["graph-node-1"] == {"weights-a"}
    assert ModelManager._models_by_node["graph-node-2"] == {"weights-b"}


@pytest.mark.asyncio
async def test_execute_without_identity_reproduces_pre_v4_behaviour():
    """No identity keys at all: exactly what the worker did before v4."""
    result = await execute_node(
        node_type="test.ModelLoadingNode",
        fields={"model_name": "weights"},
        secrets={},
        input_blobs={},
    )

    # The node still ran, and its id is still the empty string.
    assert result["outputs"] == {"output": ""}
    # The single "" bucket — the old behaviour, preserved byte for byte.
    assert set(ModelManager._models_by_node) == {""}
    # Nothing was attributed to a run.
    assert JobRegistry.open_job_ids() == []


@pytest.mark.asyncio
async def test_identity_populates_worker_context(monkeypatch):
    seen: dict[str, Any] = {}

    class ContextProbeNode(BaseNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "test.ContextProbeNode"

        async def process(self, context: ProcessingContext) -> str:
            seen["workflow_id"] = context.workflow_id
            seen["user_id"] = context.user_id
            seen["job_id"] = context.job_id
            seen["node_id"] = self._id
            return "ok"

    await execute_node(
        node_type="test.ContextProbeNode",
        fields={},
        secrets={},
        input_blobs={},
        node_id="n-7",
        job_id="job-7",
        workflow_id="wf-7",
        user_id="user-7",
    )

    assert seen == {
        "workflow_id": "wf-7",
        "user_id": "user-7",
        "job_id": "job-7",
        "node_id": "n-7",
    }


def test_read_run_identity_absent_keys_are_none():
    """An old-client payload: four known keys, nothing else."""
    identity = read_run_identity(
        {"node_type": "x", "fields": {}, "secrets": {}, "blobs": {}}
    )
    assert identity == {
        "node_id": None,
        "job_id": None,
        "workflow_id": None,
        "user_id": None,
        "requires_vram_gb": None,
    }


def test_read_run_identity_reads_all_five():
    identity = read_run_identity(
        {
            "node_type": "x",
            "node_id": "n1",
            "job_id": "j1",
            "workflow_id": "wf1",
            "user_id": "u1",
            "requires_vram_gb": 8,
        }
    )
    assert identity["node_id"] == "n1"
    assert identity["job_id"] == "j1"
    assert identity["workflow_id"] == "wf1"
    assert identity["user_id"] == "u1"
    assert identity["requires_vram_gb"] == 8.0


@pytest.mark.parametrize(
    "payload",
    [
        {"node_id": ""},          # empty string is not an id
        {"node_id": 42},          # wrong type
        {"requires_vram_gb": True},  # bool is an int subclass; must not be 1.0
        {"requires_vram_gb": -3},
        {"requires_vram_gb": "8"},
    ],
)
def test_read_run_identity_degrades_instead_of_failing(payload):
    """A malformed value degrades to "no identity", never to an error."""
    identity = read_run_identity({"node_type": "x", **payload})
    assert all(value is None for value in identity.values())


@pytest.mark.asyncio
async def test_pre_v4_handler_ignores_the_new_keys_cleanly():
    """A worker that predates v4 must ignore the extra dict entries.

    The JS side sends the identity keys unconditionally, including to workers
    that cannot read them, so this is a live path — not a hypothetical.
    """

    async def v3_style_handler(data, cancel_event, emit_progress, emit_chunk, emit_update):
        # Verbatim shape of the pre-v4 handler: four keys, nothing else.
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
            emit_progress=emit_progress,
            emit_chunk=emit_chunk,
            emit_update=emit_update,
        )

    server = WorkerProtocolServer(transport_name="test")
    server.set_execute_handler(v3_style_handler)
    transport = RecordingTransport()

    await server.dispatch(
        {
            "type": "execute",
            "request_id": "e1",
            "data": {
                "node_type": "test.QuietNode",
                "fields": {},
                "secrets": {},
                "blobs": {},
                "node_id": "n1",
                "job_id": "j1",
                "workflow_id": "wf1",
                "user_id": "u1",
                "requires_vram_gb": 4.0,
            },
        },
        transport,
    )

    assert [f["type"] for f in transport.frames] == ["result"]
    assert transport.frames[0]["data"]["outputs"] == {"output": "ok"}


# ---------------------------------------------------------------------------
# §3 — job.start / job.end
# ---------------------------------------------------------------------------


async def _dispatch(msg: dict[str, Any]) -> list[dict[str, Any]]:
    server = WorkerProtocolServer(transport_name="test")

    async def handler(data, cancel_event, emit_progress, emit_chunk, emit_update):
        from nodetool.worker.executor import read_run_identity as _identity

        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
            emit_progress=emit_progress,
            emit_chunk=emit_chunk,
            emit_update=emit_update,
            **_identity(data),
        )

    server.set_execute_handler(handler)
    transport = RecordingTransport()
    await server.dispatch(msg, transport)
    return transport.frames


@pytest.mark.asyncio
async def test_job_start_and_end_answer_with_a_result_frame():
    """The JS side waits on result/error and hangs otherwise."""
    frames = await _dispatch(
        {"type": "job.start", "request_id": "j1", "data": {"job_id": "job-a"}}
    )
    assert [f["type"] for f in frames] == ["result"]
    assert frames[0]["request_id"] == "j1"

    frames = await _dispatch(
        {
            "type": "job.end",
            "request_id": "j2",
            "data": {"job_id": "job-a", "reason": "completed"},
        }
    )
    assert [f["type"] for f in frames] == ["result"]


@pytest.mark.asyncio
async def test_job_end_releases_the_jobs_models():
    await _dispatch({"type": "job.start", "request_id": "s", "data": {"job_id": "job-b"}})
    await _dispatch(
        {
            "type": "execute",
            "request_id": "e",
            "data": {
                "node_type": "test.ModelLoadingNode",
                "fields": {"model_name": "big-weights"},
                "node_id": "node-b",
                "job_id": "job-b",
            },
        }
    )
    assert "big-weights" in ModelManager._models
    assert ModelManager._models_by_node["node-b"] == {"big-weights"}

    await _dispatch(
        {
            "type": "job.end",
            "request_id": "x",
            "data": {"job_id": "job-b", "reason": "completed"},
        }
    )

    assert ModelManager._models == {}
    assert ModelManager._models_by_node == {}
    assert JobRegistry.open_job_ids() == []


@pytest.mark.asyncio
async def test_job_end_for_unknown_job_is_a_no_op():
    ModelManager.set_model("someone-elses-node", "untouched", FakeModel(GB))

    frames = await _dispatch(
        {
            "type": "job.end",
            "request_id": "u1",
            "data": {"job_id": "never-seen", "reason": "failed"},
        }
    )

    assert frames[0]["type"] == "result"
    assert frames[0]["data"]["released_nodes"] == []
    assert frames[0]["data"]["known"] is False
    # Nothing else was collateral damage.
    assert "untouched" in ModelManager._models


@pytest.mark.asyncio
async def test_job_end_is_idempotent():
    await _dispatch({"type": "job.start", "request_id": "s", "data": {"job_id": "job-c"}})
    await _dispatch(
        {
            "type": "execute",
            "request_id": "e",
            "data": {
                "node_type": "test.ModelLoadingNode",
                "fields": {"model_name": "w-c"},
                "node_id": "node-c",
                "job_id": "job-c",
            },
        }
    )

    first = await _dispatch(
        {"type": "job.end", "request_id": "x1", "data": {"job_id": "job-c"}}
    )
    second = await _dispatch(
        {"type": "job.end", "request_id": "x2", "data": {"job_id": "job-c"}}
    )

    assert first[0]["data"]["released_nodes"] == ["node-c"]
    assert first[0]["data"]["known"] is True
    assert second[0]["type"] == "result"
    assert second[0]["data"]["released_nodes"] == []
    assert second[0]["data"]["known"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["completed", "failed", "cancelled", "abandoned"])
async def test_reason_does_not_branch_the_release(reason):
    """An abnormal end retires the same nodes as a clean one."""
    await _dispatch(
        {"type": "job.start", "request_id": "s", "data": {"job_id": f"job-{reason}"}}
    )
    await _dispatch(
        {
            "type": "execute",
            "request_id": "e",
            "data": {
                "node_type": "test.ModelLoadingNode",
                "fields": {"model_name": f"w-{reason}"},
                "node_id": f"node-{reason}",
                "job_id": f"job-{reason}",
            },
        }
    )

    frames = await _dispatch(
        {
            "type": "job.end",
            "request_id": "x",
            "data": {"job_id": f"job-{reason}", "reason": reason},
        }
    )

    assert frames[0]["data"]["released_nodes"] == [f"node-{reason}"]
    assert ModelManager._models == {}


@pytest.mark.asyncio
async def test_job_message_without_job_id_errors():
    frames = await _dispatch({"type": "job.end", "request_id": "bad", "data": {}})
    assert frames[0]["type"] == "error"
    assert "job_id" in frames[0]["data"]["error"]


@pytest.mark.asyncio
async def test_execute_without_job_start_still_attributes_to_the_job():
    """job.start is optional: every execute carries its own job_id."""
    await _dispatch(
        {
            "type": "execute",
            "request_id": "e",
            "data": {
                "node_type": "test.ModelLoadingNode",
                "fields": {"model_name": "w-d"},
                "node_id": "node-d",
                "job_id": "job-d",
            },
        }
    )
    assert JobRegistry.node_ids_for("job-d") == ["node-d"]

    await _dispatch({"type": "job.end", "request_id": "x", "data": {"job_id": "job-d"}})
    assert ModelManager._models == {}


# ---------------------------------------------------------------------------
# The leak itself: measured, not asserted about
# ---------------------------------------------------------------------------


async def _run_workflow(job_id: str, *, close: bool) -> None:
    """One 'workflow': three nodes, each loading its own model."""
    for index in range(3):
        await _dispatch(
            {
                "type": "execute",
                "request_id": f"{job_id}-{index}",
                "data": {
                    "node_type": "test.ModelLoadingNode",
                    "fields": {"model_name": f"{job_id}-model-{index}"},
                    "node_id": f"{job_id}-node-{index}",
                    "job_id": job_id,
                },
            }
        )
    if close:
        await _dispatch(
            {
                "type": "job.end",
                "request_id": f"{job_id}-end",
                "data": {"job_id": job_id, "reason": "completed"},
            }
        )


@pytest.mark.asyncio
async def test_cache_grows_without_job_end_and_stays_flat_with_it():
    """Reproduce the leak, then show job.end closes it.

    Without a run boundary the worker has no caller for `release_nodes()`, so
    every model any run ever loaded stays cached until memory pressure trims it
    reactively. The measurement is the deliverable here — the numbers below are
    the leak.
    """
    leaking: list[int] = []
    for run in range(5):
        await _run_workflow(f"leak-job-{run}", close=False)
        leaking.append(len(ModelManager._models))

    # Strictly monotonic: 3 models per run, never released.
    assert leaking == [3, 6, 9, 12, 15]

    ModelManager.clear()
    JobRegistry.clear()

    bounded: list[int] = []
    for run in range(5):
        await _run_workflow(f"bounded-job-{run}", close=True)
        bounded.append(len(ModelManager._models))

    # Flat at zero: each run's models are released at its boundary.
    assert bounded == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------------------
# §3 — the TTL backstop for a job.end that never arrives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ttl_sweep_releases_an_abandoned_job(monkeypatch):
    monkeypatch.setenv("NODETOOL_WORKER_JOB_TTL_SECONDS", "60")

    await execute_node(
        node_type="test.ModelLoadingNode",
        fields={"model_name": "orphan"},
        secrets={},
        input_blobs={},
        node_id="orphan-node",
        job_id="orphan-job",
    )
    assert "orphan" in ModelManager._models

    # Age the job past the TTL: the hard-killed client that never sent job.end.
    JobRegistry._jobs["orphan-job"].last_activity -= 120

    swept = JobRegistry.sweep()

    assert swept == ["orphan-job"]
    assert ModelManager._models == {}


@pytest.mark.asyncio
async def test_ttl_sweep_leaves_a_live_job_alone(monkeypatch):
    monkeypatch.setenv("NODETOOL_WORKER_JOB_TTL_SECONDS", "60")

    await execute_node(
        node_type="test.ModelLoadingNode",
        fields={"model_name": "live"},
        secrets={},
        input_blobs={},
        node_id="live-node",
        job_id="live-job",
    )

    assert JobRegistry.sweep() == []
    assert "live" in ModelManager._models


def test_ttl_of_zero_disables_the_sweep(monkeypatch):
    monkeypatch.setenv("NODETOOL_WORKER_JOB_TTL_SECONDS", "0")
    JobRegistry.start("j")
    JobRegistry._jobs["j"].last_activity -= 10**6
    assert JobRegistry.sweep() == []
    assert JobRegistry.has("j")


# ---------------------------------------------------------------------------
# §4 — models.evict
# ---------------------------------------------------------------------------


async def _evict(data: dict[str, Any]) -> dict[str, Any]:
    from nodetool.worker.model_handler import handle_models_message

    transport = RecordingTransport()
    await handle_models_message(
        msg_type="models.evict",
        request_id="ev",
        data=data,
        transport=transport,
        cancel_flags={},
    )
    assert transport.frames[0]["type"] == "result", transport.frames[0]
    return transport.frames[0]["data"]


@pytest.mark.asyncio
async def test_evict_with_no_scope_drops_everything():
    ModelManager.set_model("n1", "m1", FakeModel(GB))
    ModelManager.set_model("n2", "m2", FakeModel(GB))

    result = await _evict({})

    assert sorted(result["evicted"]) == ["m1", "m2"]
    assert ModelManager._models == {}


@pytest.mark.asyncio
async def test_evict_stops_at_target_vram_instead_of_clearing_everything():
    # Four 2 GiB models, distinct last-used times so "coldest first" is defined.
    for index in range(4):
        ModelManager.set_model(f"n{index}", f"m{index}", FakeModel(2 * GB))
        ModelManager._model_last_used[f"m{index}"] = float(index)

    result = await _evict({"target_vram_gb": 3.0})

    # 2 GiB is short of the target, 4 GiB clears it — so exactly two go.
    assert len(result["evicted"]) == 2
    assert result["freed_vram_gb"] == pytest.approx(4.0)
    # And the rest are still resident. This is the assertion that fails if the
    # target is ignored and everything is dropped.
    assert len(ModelManager._models) == 2
    # Coldest first: m0/m1 went, the two most recently used stayed.
    assert sorted(ModelManager._models) == ["m2", "m3"]


@pytest.mark.asyncio
async def test_evict_scoped_to_node_ids():
    ModelManager.set_model("keep-node", "keep", FakeModel(GB))
    ModelManager.set_model("drop-node", "drop", FakeModel(GB))

    result = await _evict({"node_ids": ["drop-node"]})

    assert result["evicted"] == ["drop"]
    assert sorted(ModelManager._models) == ["keep"]


@pytest.mark.asyncio
async def test_evict_keeps_a_key_another_node_still_references():
    ModelManager.set_model("node-a", "shared", FakeModel(GB))
    ModelManager.set_model("node-b", "shared", FakeModel(GB))

    result = await _evict({"node_ids": ["node-a"]})

    assert result["evicted"] == []
    assert "shared" in ModelManager._models


@pytest.mark.asyncio
async def test_evict_scoped_to_a_job():
    JobRegistry.start("job-e")
    JobRegistry.note_execution("job-e", "job-node")
    ModelManager.set_model("job-node", "job-model", FakeModel(GB))
    ModelManager.set_model("other-node", "other-model", FakeModel(GB))

    result = await _evict({"job_id": "job-e"})

    assert result["evicted"] == ["job-model"]
    assert sorted(ModelManager._models) == ["other-model"]


@pytest.mark.asyncio
async def test_evict_scoped_to_an_unknown_job_evicts_nothing():
    """An empty scope is still a scope — it must not widen to "everything"."""
    ModelManager.set_model("n", "m", FakeModel(GB))

    result = await _evict({"job_id": "never-seen"})

    assert result["evicted"] == []
    assert "m" in ModelManager._models


# ---------------------------------------------------------------------------
# §5/§6 — the requires_vram_gb hint, reported and then used
# ---------------------------------------------------------------------------


def test_discover_reports_the_vram_hint_when_a_node_declares_one():
    from nodetool.worker.node_loader import node_to_metadata

    assert node_to_metadata(VramHintNode)["requires_vram_gb"] == 6.5


def test_discover_omits_the_vram_hint_when_the_node_does_not_know():
    """Absent, not null: an invented number is worse than no hint."""
    from nodetool.worker.node_loader import node_to_metadata

    assert "requires_vram_gb" not in node_to_metadata(QuietNode)


def test_discover_survives_a_broken_vram_override():
    from nodetool.worker.node_loader import node_to_metadata

    class BrokenHintNode(QuietNode):
        @classmethod
        def get_node_type(cls) -> str:
            return "test.BrokenHintNode"

        @classmethod
        def get_required_vram_gb(cls) -> float | None:
            raise RuntimeError("package bug")

    metadata = node_to_metadata(BrokenHintNode)
    assert metadata["node_type"] == "test.BrokenHintNode"
    assert "requires_vram_gb" not in metadata


@pytest.mark.asyncio
async def test_reclaim_pass_targets_the_hinted_amount(monkeypatch):
    """With a hint the pass targets a real number, not just a threshold."""
    calls: list[dict[str, Any]] = []

    def record(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(ModelManager, "free_vram_if_needed", record)

    await execute_node(
        node_type="test.QuietNode",
        fields={},
        secrets={},
        input_blobs={},
        node_id="n",
        requires_vram_gb=12.0,
    )

    assert calls[0]["required_free_gb"] == 12.0


@pytest.mark.asyncio
async def test_reclaim_pass_without_a_hint_keeps_the_threshold_behaviour(monkeypatch):
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(ModelManager, "free_vram_if_needed", lambda **kw: calls.append(kw))

    await execute_node(
        node_type="test.QuietNode", fields={}, secrets={}, input_blobs={}
    )

    assert calls[0]["required_free_gb"] is None


@pytest.mark.asyncio
async def test_job_start_runs_one_reclaim_pass(monkeypatch):
    """The reason job.start exists: one pass per run, not one per node."""
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(ModelManager, "free_vram_if_needed", lambda **kw: calls.append(kw))

    await _dispatch({"type": "job.start", "request_id": "s", "data": {"job_id": "job-r"}})

    assert len(calls) == 1
    assert "job-r" in calls[0]["reason"]


@pytest.mark.asyncio
async def test_evict_skips_models_a_concurrent_execution_is_using():
    ModelManager.set_model("busy-node", "busy", FakeModel(GB))

    async def hold_the_model(started: asyncio.Event, release: asyncio.Event) -> None:
        with ModelManager.execution_scope():
            ModelManager.get_model("busy")
            started.set()
            await release.wait()

    started, release = asyncio.Event(), asyncio.Event()
    holder = asyncio.create_task(hold_the_model(started, release))
    await started.wait()
    try:
        result = await _evict({})
    finally:
        release.set()
        await holder

    assert result["evicted"] == []
    assert "busy" in ModelManager._models
