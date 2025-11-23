"""Utilities for optional torch-dependent workflow features.

This module provides a thin abstraction that hides torch/comfy specific logic
behind helper classes so callers can operate without conditional imports.
"""

from __future__ import annotations

import asyncio
import gc
import random
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager
from nodetool.workflows.types import NodeProgress

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from PIL import Image

    from nodetool.metadata.types import TorchTensor
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.processing_context import ProcessingContext
    from nodetool.workflows.workflow_runner import WorkflowRunner

log = get_logger(__name__)

TORCH_AVAILABLE = False

torch: Any

try:  # pragma: no cover - optional dependency
    import torch as _torch  # type: ignore

    torch = _torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch not installed
    torch = None  # type: ignore


def is_cuda_available() -> bool:
    """Safely check if CUDA is available, handling cases where PyTorch is not compiled with CUDA support."""
    if not TORCH_AVAILABLE or torch is None:
        return False
    try:
        # Check if cuda module exists
        if not hasattr(torch, "cuda"):
            return False
        # Try to check availability - this can raise RuntimeError if CUDA is not compiled
        return torch.cuda.is_available()
    except (RuntimeError, AttributeError):
        # PyTorch not compiled with CUDA support or other CUDA-related error
        return False


class BaseTorchSupport:
    """Interface describing torch specific hooks used by ``WorkflowRunner``."""

    def __init__(self, *, base_delay: int, max_delay: int, max_retries: int) -> None:
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries

    def get_available_vram(self) -> int:
        return 0

    def log_vram_usage(self, runner: WorkflowRunner, message: str = "") -> None:
        return None

    @contextmanager
    def torch_context(
        self, runner: WorkflowRunner, context: ProcessingContext
    ) -> Generator[None, None, None]:
        yield

    async def process_with_gpu(
        self,
        runner: WorkflowRunner,
        context: ProcessingContext,
        node: BaseNode,
        retries: int = 0,
    ) -> Any:
        return await node.process(context)

    def is_cuda_oom_exception(self, exc: Exception) -> bool:
        return False

    def empty_cuda_cache(self) -> None:
        return None


class TorchWorkflowSupport(BaseTorchSupport):
    """Concrete torch-enabled implementation."""

    def get_available_vram(self) -> int:
        if not is_cuda_available():
            return 0
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory - torch.cuda.memory_allocated(0)
        except (RuntimeError, AttributeError):
            return 0

    def log_vram_usage(self, runner: WorkflowRunner, message: str = "") -> None:
        if not is_cuda_available():
            return
        try:
            torch.cuda.synchronize()
            vram = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
            log.info(f"{message} VRAM: {vram:.2f} GB")
        except (RuntimeError, AttributeError):
            # CUDA not available or not compiled, skip logging
            pass

    @contextmanager
    def torch_context(
        self, runner: WorkflowRunner, context: ProcessingContext
    ) -> Generator[None, None, None]:
        self.log_vram_usage(runner, "Before workflow")

        try:
            yield
        finally:
            self.log_vram_usage(runner, "After workflow")

        log.info("Exiting torch context")

    async def process_with_gpu(
        self,
        runner: WorkflowRunner,
        context: ProcessingContext,
        node: BaseNode,
        retries: int = 0,
    ) -> Any:
        try:
            if node._requires_grad:
                return await node.process(context)
            with torch.no_grad():
                return await node.process(context)
        except Exception as exc:
            if not self.is_cuda_oom_exception(exc):
                log.debug(
                    "Non-OOM error in process_with_gpu for node %s: %s",
                    node.get_title(),
                    exc,
                    exc_info=True,
                )
                raise

            log.error(
                "VRAM OOM error for node %s (%s): %s",
                node.get_title(),
                node._id,
                exc,
            )
            retries += 1

            if is_cuda_available():
                try:
                    torch.cuda.synchronize()
                    vram_before_cleanup = self.get_available_vram()
                    log.error(
                        "VRAM before cleanup: %.2f GB",
                        vram_before_cleanup / (1024**3),
                    )

                    snapshot = ModelManager.get_vram_snapshot()
                    target_free = None
                    if snapshot is not None:
                        target_free = max(4.0, snapshot.total_gb * 0.3)

                    ModelManager.free_vram_if_needed(
                        reason=(
                            f"CUDA OOM for node {node.get_title()} ({node._id})"
                        ),
                        required_free_gb=target_free,
                        aggressive=retries >= self.max_retries,
                    )
                    gc.collect()

                    self.empty_cuda_cache()
                    with suppress(RuntimeError, AttributeError):
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                    vram_after_cleanup = self.get_available_vram()
                    log.error(
                        "VRAM after cleanup: %.2f GB",
                        vram_after_cleanup / (1024**3),
                    )
                except (RuntimeError, AttributeError):
                    # CUDA not available or not compiled, skip cleanup
                    pass

            if retries >= self.max_retries:
                log.error(
                    "Max retries (%d) reached for OOM error on node %s. Raising error.",
                    self.max_retries,
                    node.get_title(),
                )
                raise

            delay = min(
                self.base_delay * (2 ** (retries - 1)) + random.uniform(0, 1),
                self.max_delay,
            )
            log.warning(
                "VRAM OOM encountered for node %s. Retrying in %.2f seconds. (Attempt %d/%d)",
                node._id,
                delay,
                retries,
                self.max_retries,
            )
            await asyncio.sleep(delay)
            return await self.process_with_gpu(runner, context, node, retries + 1)

    def is_cuda_oom_exception(self, exc: Exception) -> bool:
        if not TORCH_AVAILABLE or torch is None:
            return False
        try:
            if not is_cuda_available():
                return False
            return isinstance(exc, torch.cuda.OutOfMemoryError)
        except (RuntimeError, AttributeError):
            return False

    def empty_cuda_cache(self) -> None:
        if is_cuda_available():
            with suppress(RuntimeError, AttributeError):
                torch.cuda.empty_cache()


class NoopTorchSupport(BaseTorchSupport):
    """Stub used when torch is unavailable."""

    # Inherits no-op behaviour from BaseTorchSupport.


def build_torch_support(
    *, base_delay: int, max_delay: int, max_retries: int
) -> BaseTorchSupport:
    if TORCH_AVAILABLE:
        return TorchWorkflowSupport(
            base_delay=base_delay, max_delay=max_delay, max_retries=max_retries
        )
    return NoopTorchSupport(
        base_delay=base_delay, max_delay=max_delay, max_retries=max_retries
    )


def is_torch_tensor(value: Any) -> bool:
    """Return True when value is a torch tensor and torch is installed."""
    return bool(
        TORCH_AVAILABLE and torch is not None and isinstance(value, torch.Tensor)
    )


def detach_tensor(value: Any) -> Any:
    """Detach tensor from graph and move to CPU when possible."""
    if is_torch_tensor(value):
        return value.detach().cpu()
    return value


def detach_tensors_recursively(value: Any) -> Any:
    """Traverse common containers and detach any torch tensors within."""
    if is_torch_tensor(value):
        return detach_tensor(value)
    if isinstance(value, dict):
        return {k: detach_tensors_recursively(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detach_tensors_recursively(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detach_tensors_recursively(v) for v in value)
    return value


def tensor_from_array(array: np.ndarray) -> Any:
    """Create a float tensor in range [0,1] from a numpy array."""
    if not TORCH_AVAILABLE or torch is None:
        raise ImportError("torch is required for tensor conversion")
    return torch.tensor(array).float() / 255.0


def tensor_from_pil(image: Image.Image) -> Any:
    """Create a tensor from a PIL image."""
    return tensor_from_array(np.array(image))


def tensor_to_image_array(tensor: Any) -> np.ndarray:
    """Convert a torch tensor into a uint8 numpy image array."""
    if not is_torch_tensor(tensor):
        raise ImportError("torch is required for tensor conversion")
    data = tensor.detach().cpu().numpy()
    return np.clip(255.0 * data, 0, 255).astype(np.uint8)


def torch_tensor_to_metadata(tensor: Any) -> TorchTensor | Any:
    """Wrap a torch tensor into metadata representation when available."""
    if not is_torch_tensor(tensor):
        return tensor
    from nodetool.metadata.types import TorchTensor as TorchTensorModel

    return TorchTensorModel.from_tensor(tensor)
