"""The stub must register a blob for every asset it produces.

This is one defect class, not four bugs. ``WorkerContext`` covers a ref type
per *entry point*, so any factory it does not override falls through to
``ProcessingContext``, which returns a ref carrying inline ``data`` and no
uri. The executor extracts blobs only from a ``blob://`` uri and the
serializer strips raw ``data`` at any depth, so those bytes are dropped while
the node reports success.

Two checks, because neither is enough alone:

* The behavioral half drives every ``*_from_io`` funnel — the method each
  type's other factories delegate to — with a real buffer, and follows the
  bytes through ``executor._extract_outputs``.
* The structural half walks the source: every producer on
  ``ProcessingContext`` must be overridden by the stub or delegate to
  something that is. It catches a factory added tomorrow. It cannot see
  *branches*, which is why the behavioral half exists — ``audio_from_segment``
  delegates to ``audio_from_io`` on one branch only, and this check would have
  called it covered while its other branch dropped the audio.
"""

import inspect
import re
from io import BytesIO
from typing import Any, Callable

import pytest

from nodetool.metadata.types import AssetRef
from nodetool.worker.context_stub import WorkerContext
from nodetool.worker.executor import ASSET_REF_TYPES, _extract_outputs
from nodetool.workflows.processing_context import ProcessingContext

PAYLOAD = b"\x00\x01payload bytes\x02\x03"
_REF_NAMES = {cls.__name__ for cls in ASSET_REF_TYPES}


def _return_type_name(method: Callable[..., Any]) -> str | None:
    annotation = inspect.signature(method).return_annotation
    return annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)


def _returns_asset_ref(method: Callable[..., Any]) -> bool:
    return _return_type_name(method) in _REF_NAMES


def _takes_asset_ref(method: Callable[..., Any]) -> bool:
    """A method fed an existing ref converts one; it does not produce output."""
    for param in inspect.signature(method).parameters.values():
        annotation = param.annotation
        name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", None)
        if name in _REF_NAMES:
            return True
    return False


def _producers() -> dict[str, Callable[..., Any]]:
    return {
        name: method
        for name, method in inspect.getmembers(ProcessingContext, inspect.isfunction)
        if _returns_asset_ref(method) and not _takes_asset_ref(method)
    }


def _io_funnels() -> dict[str, Callable[..., Any]]:
    return {name: method for name, method in _producers().items() if name.endswith("_from_io")}


def test_every_asset_ref_type_has_an_io_funnel():
    """Guards the behavioral check below against silently covering nothing."""
    funnels = _io_funnels()
    assert funnels, "no *_from_io producers found — the enumeration is broken"
    covered = {_return_type_name(m) for m in funnels.values()}
    missing = [cls.__name__ for cls in ASSET_REF_TYPES if cls is not AssetRef and cls.__name__ not in covered]
    assert not missing, f"no _from_io funnel enumerated for {missing}"


@pytest.mark.asyncio
@pytest.mark.parametrize("funnel", sorted(_io_funnels()))
async def test_io_funnel_registers_a_blob_that_reaches_the_executor(funnel: str):
    ctx = WorkerContext()
    ref = await getattr(ctx, funnel)(BytesIO(PAYLOAD))

    assert ref.uri.startswith("blob://"), f"{funnel} returned uri {ref.uri!r}, so its bytes are dropped"
    assert ctx.get_output_blobs()[ref.uri[len("blob://") :]] == PAYLOAD

    outputs, blobs = _extract_outputs(ref, ctx)
    assert blobs.get("output") == PAYLOAD, f"{funnel} bytes never reached the response blobs map"
    assert not outputs["output"].get("data")


def _covered(name: str, sources: dict[str, str], overridden: set[str], seen: tuple[str, ...] = ()) -> bool:
    if name in overridden:
        return True
    if name in seen or name not in sources:
        return False
    calls = set(re.findall(r"self\.([a-z_0-9]+)\(", sources[name]))
    return any(_covered(call, sources, overridden, seen + (name,)) for call in calls if call in sources)


def test_every_producer_is_overridden_or_delegates_to_one():
    producers = _producers()
    sources = {name: inspect.getsource(method) for name, method in producers.items()}
    overridden = set(WorkerContext.__dict__)

    gaps = sorted(name for name in producers if not _covered(name, sources, overridden))
    assert not gaps, (
        "these produce an asset the worker cannot deliver — override them on WorkerContext "
        f"so the bytes are registered as a blob: {gaps}"
    )
