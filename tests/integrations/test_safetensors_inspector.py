"""Tests for safetensors_inspector detection heuristics."""

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from nodetool.integrations.huggingface.safetensors_inspector import (
    detect_model,
)


def _write(path: Path, tensors: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def test_detect_lora_adapter(tmp_path: Path) -> None:
    path = tmp_path / "lora.safetensors"
    _write(
        path,
        {
            "lora_down.weight": np.zeros((4, 2), dtype=np.float32),
            "lora_up.weight": np.zeros((2, 4), dtype=np.float32),
        },
    )

    result = detect_model(path, framework="np")

    assert result.component == "lora_adapter"
    assert result.family == "lora-adapter"
    assert result.confidence > 0.9


def test_detect_flux_transformer(tmp_path: Path) -> None:
    path = tmp_path / "dit.safetensors"
    _write(
        path,
        {
            "transformer_blocks.0.attn.to_q.weight": np.zeros((4, 4), dtype=np.float32),
            "x_embedder.proj.weight": np.zeros((4, 4), dtype=np.float32),
            "t_embedder.mlp.0.weight": np.zeros((4, 4), dtype=np.float32),
        },
    )

    result = detect_model(path, framework="np")

    assert result.component == "transformer_denoiser"
    assert result.family.startswith("flux")


def test_detect_llama_family(tmp_path: Path) -> None:
    path = tmp_path / "llama.safetensors"
    _write(
        path,
        {
            "model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
        },
    )

    result = detect_model(path, framework="np")

    assert result.component == "llm"
    assert result.family == "llama-family"


def test_detect_sdxl_base_unet(tmp_path: Path) -> None:
    path = tmp_path / "unet.safetensors"
    _write(
        path,
        {
            "down_blocks.0.resnets.0.conv1.weight": np.zeros(
                (4, 1280, 3, 3), dtype=np.float32
            ),
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": np.zeros(
                (4, 4), dtype=np.float32
            ),
            "text_model.encoder.layers.0.self_attn.q_proj.weight": np.zeros(
                (4, 4), dtype=np.float32
            ),
            "up_blocks.0.resnets.0.conv1.weight": np.zeros((4, 4, 3, 3), dtype=np.float32),
        },
    )

    result = detect_model(path, framework="np")

    assert result.component == "unet"
    assert result.family in {"sdxl-base", "sdxl-refiner"}
