"""Tests for safetensors layout detection."""

from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors.numpy import save_file

from nodetool.integrations.huggingface.safetensor_layout import (
    SafetensorLayoutHint,
    classify_safetensor_set,
    summarize_safetensor,
)


def _write_safetensor(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a safetensors file using numpy backend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def _arrays(shape: Iterable[int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_sharded_bundle_detected(tmp_path: Path) -> None:
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    tensors = {
        "unet.conv.weight": _arrays((4, 4, 3, 3)),
        "unet.conv.bias": _arrays((4,)),
    }
    _write_safetensor(shard1, tensors)
    _write_safetensor(shard2, tensors)

    hint = classify_safetensor_set([shard1, shard2])

    assert hint == SafetensorLayoutHint.SHARDED_BUNDLE


def test_disjoint_variants_detected(tmp_path: Path) -> None:
    lora = tmp_path / "style-lora.safetensors"
    base = tmp_path / "base-model.safetensors"

    _write_safetensor(lora, {"lora.down": _arrays((1, 4))})
    _write_safetensor(base, {"base.weight": _arrays((4, 4, 3, 3))})

    hint = classify_safetensor_set([lora, base])

    assert hint == SafetensorLayoutHint.DISJOINT


def test_single_file_classification(tmp_path: Path) -> None:
    single = tmp_path / "single.safetensors"
    _write_safetensor(single, {"linear.weight": _arrays((2, 2))})

    hint = classify_safetensor_set([single])

    assert hint == SafetensorLayoutHint.SINGLE


def test_summary_reads_header_only(tmp_path: Path) -> None:
    """Ensure summarization returns key and shape info."""
    file_path = tmp_path / "model.safetensors"
    tensors = {"encoder.weight": _arrays((8, 8)), "encoder.bias": _arrays((8,))}
    _write_safetensor(file_path, tensors)

    summary = summarize_safetensor(file_path, sample_limit=1)

    assert summary.key_count == 2
    assert len(summary.sampled_shapes) == 1
