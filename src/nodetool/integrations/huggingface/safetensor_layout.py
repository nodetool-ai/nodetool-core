"""Safetensors layout inspection helpers.

These helpers inspect only safetensors headers and a handful of tensor shapes.
They never load full weight payloads. The goal is to classify whether multiple
files represent shards of the same model or independent variants.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Iterable, Sequence

from safetensors import safe_open


class SafetensorLayoutHint(str, Enum):
    """Classification for a set of safetensors files."""

    EMPTY = "empty"
    SINGLE = "single"
    SHARDED_BUNDLE = "sharded_bundle"
    DISJOINT = "disjoint"
    MIXED = "mixed"


@dataclass(frozen=True)
class SafetensorSummary:
    """Lightweight snapshot of a safetensors file."""

    path: str
    key_count: int
    sampled_shapes: dict[str, tuple[int, ...]]


def summarize_safetensor(
    path: str | Path,
    sample_limit: int = 32,
) -> SafetensorSummary:
    """Read only the safetensors header and a few shapes for fingerprinting.

    Args:
        path: Path to a `.safetensors` file.
        sample_limit: Maximum number of keys to sample for shapes.

    Returns:
        SafetensorSummary containing key count and sampled shapes.
    """
    safe_path = str(path)
    with safe_open(safe_path, framework="numpy") as handle:
        keys = list(handle.keys())
        sampled = {}
        for key in keys[:sample_limit]:
            # This uses safetensors' mmap reader; only header/shape bytes are read.
            sampled[key] = tuple(handle.get_tensor(key).shape)
        return SafetensorSummary(
            path=safe_path,
            key_count=len(keys),
            sampled_shapes=sampled,
        )


def classify_safetensor_set(
    paths: Sequence[str | Path],
    sample_limit: int = 32,
) -> SafetensorLayoutHint:
    """Classify a set of safetensors files as shards or disjoint variants.

    Strategy (header-only):
    - If there are no files, return EMPTY.
    - If there is one file, return SINGLE.
    - If sampled key sets intersect and shapes match for intersections, treat as SHARDED_BUNDLE.
    - If there is no intersection of sampled keys, treat as DISJOINT.
    - Otherwise return MIXED.

    Args:
        paths: safetensors file paths to inspect.
        sample_limit: Number of keys to sample per file for shape comparison.

    Returns:
        SafetensorLayoutHint describing the relationship.
    """
    if not paths:
        return SafetensorLayoutHint.EMPTY

    if len(paths) == 1:
        return SafetensorLayoutHint.SINGLE

    summaries = [summarize_safetensor(path, sample_limit=sample_limit) for path in paths]

    sampled_key_sets = [set(summary.sampled_shapes.keys()) for summary in summaries]
    intersection = _intersect(sampled_key_sets)

    if intersection and _shapes_align(summaries, intersection):
        return SafetensorLayoutHint.SHARDED_BUNDLE

    if not intersection:
        return SafetensorLayoutHint.DISJOINT

    return SafetensorLayoutHint.MIXED


def _intersect(sets: Iterable[set[str]]) -> set[str]:
    sets = list(sets)
    if not sets:
        return set()
    result = sets[0].copy()
    for candidate in sets[1:]:
        result.intersection_update(candidate)
    return result


def _shapes_align(
    summaries: Sequence[SafetensorSummary],
    keys: Iterable[str],
) -> bool:
    keys = list(keys)
    for key in keys:
        shapes = [summary.sampled_shapes.get(key) for summary in summaries]
        if any(shape is None for shape in shapes):
            return False
        first_shape = shapes[0]
        if not all(shape == first_shape for shape in shapes[1:]):
            return False
    return True
