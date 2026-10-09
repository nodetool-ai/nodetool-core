"""Utilities for optional torch-dependent workflow features.

This module provides a thin abstraction that hides torch/comfy specific logic
behind helper classes so callers can operate without conditional imports.
"""

from __future__ import annotations

import asyncio
import gc
import os
import random
import re
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from nodetool.config.logging_config import get_logger
from nodetool.ml.core.model_manager import ModelManager

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from PIL import Image

    from nodetool.metadata.types import TorchTensor
    from nodetool.workflows.base_node import BaseNode
    from nodetool.workflows.processing_context import ProcessingContext

    WorkflowRunner = Any

log = get_logger(__name__)

TORCH_AVAILABLE = False

torch: Any

try:  # pragma: no cover - optional dependency
    import torch as _torch  # type: ignore

    torch = _torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch not installed
    torch = None  # type: ignore


def _release_exception(exc: BaseException) -> None:
    """Drop an exception's traceback and chained-exception references.

    A live ``__traceback__`` pins every frame of the failed call, and with them
    the locals of the forward pass — including the activations that caused a
    CUDA OOM. Reclaiming VRAM while those frames are still reachable frees
    little or nothing, so the retry runs against the same occupied memory.
    Clearing the chain first lets the collector actually release them.

    The exception object itself stays usable (and re-raisable); only the frames
    it was holding go away.
    """
    with suppress(Exception):
        exc.__traceback__ = None
        exc.__cause__ = None
        exc.__context__ = None


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


TORCH_DEVICE_ENV = "NODETOOL_TORCH_DEVICE"
_DEVICE_OVERRIDE_PATTERN = re.compile(r"^(cpu|mps|cuda)(?::(\d+))?$")
_warned_device_overrides: set[str] = set()


def _warn_device_override_once(value: str, reason: str) -> None:
    if value in _warned_device_overrides:
        return
    _warned_device_overrides.add(value)
    log.warning(
        "Ignoring %s=%r: %s. Selecting the device automatically.",
        TORCH_DEVICE_ENV,
        value,
        reason,
    )


def _is_mps_available() -> bool:
    if not TORCH_AVAILABLE or torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except (RuntimeError, AttributeError):
        return False


def _override_problem(kind: str, index: str | None) -> str | None:
    """Explain why a parsed override cannot be used here, or None if it can."""
    if kind == "cpu":
        return "cpu takes no index" if index is not None else None
    if kind == "mps":
        if index is not None:
            return "mps takes no index"
        return None if _is_mps_available() else "MPS is not available"
    if not is_cuda_available():
        return "CUDA is not available"
    if index is not None:
        try:
            count = int(torch.cuda.device_count())
        except (RuntimeError, AttributeError):
            count = 0
        if int(index) >= count:
            return f"only {count} CUDA device(s) are visible"
    return None


def _device_from_override(value: str) -> str | None:
    """Validate a ``NODETOOL_TORCH_DEVICE`` value against the installed torch.

    Returns the device string, or None (after a one-time warning) when the value
    is malformed or names a device this machine does not have.
    """
    match = _DEVICE_OVERRIDE_PATTERN.match(value)
    if match is None:
        _warn_device_override_once(value, "expected cpu, mps, cuda or cuda:<index>")
        return None
    problem = _override_problem(match.group(1), match.group(2))
    if problem is not None:
        _warn_device_override_once(value, problem)
        return None
    return value


def resolve_torch_device(explicit_device: str | None = None) -> str:
    """Pick the torch device for node execution.

    Order: an explicit device from the caller, then ``NODETOOL_TORCH_DEVICE``
    (``cpu``, ``mps``, ``cuda`` or ``cuda:<index>``), then automatic selection:
    MPS, then CUDA, then CPU. An override naming an unavailable device logs a
    warning once and falls back to automatic selection. Always returns a
    concrete device name, so ``BaseNode.move_to_device`` never sees ``None``.
    """
    if explicit_device:
        return explicit_device

    override = os.environ.get(TORCH_DEVICE_ENV, "").strip().lower()
    if override:
        device = _device_from_override(override)
        if device is not None:
            return device

    if _is_mps_available():
        return "mps"
    if is_cuda_available():
        return "cuda"
    return "cpu"


def is_gpu_oom_exception(exc: BaseException) -> bool:
    """True for a CUDA or MPS out-of-memory error."""
    if not TORCH_AVAILABLE or torch is None:
        return False
    oom_types = tuple(
        t
        for t in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
        )
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return isinstance(exc, RuntimeError) and "MPS backend out of memory" in str(exc)


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
    def torch_context(self, runner: WorkflowRunner, context: ProcessingContext) -> Generator[None, None, None]:
        yield

    async def process_with_gpu(
        self,
        runner: WorkflowRunner,
        context: ProcessingContext,
        node: BaseNode,
        retries: int = 0,
        *,
        disable_grad: bool = True,
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
    def torch_context(self, runner: WorkflowRunner, context: ProcessingContext) -> Generator[None, None, None]:
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
        *,
        disable_grad: bool = True,
    ) -> Any:
        """Run ``node.process`` and retry after reclaiming memory on a GPU OOM.

        ``disable_grad=False`` leaves torch's grad mode alone. The worker passes
        it because grad mode is thread-local: two nodes awaiting inside
        ``torch.no_grad()`` on one event loop restore each other's saved state
        out of order and can leave grad disabled for the whole thread.
        """
        try:
            if node._requires_grad or not disable_grad:
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

            # Release the frames of the failed forward pass *before* reclaiming.
            # The message is already logged above; everything below only needs
            # the exception object itself, which stays re-raisable.
            _release_exception(exc)

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
                        reason=(f"CUDA OOM for node {node.get_title()} ({node._id})"),
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
            elif _is_mps_available():
                gc.collect()
                with suppress(RuntimeError, AttributeError):
                    torch.mps.empty_cache()

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
            return await self.process_with_gpu(runner, context, node, retries, disable_grad=disable_grad)

    def is_cuda_oom_exception(self, exc: Exception) -> bool:
        """True for a CUDA or MPS out-of-memory error (name kept for callers)."""
        return is_gpu_oom_exception(exc)

    def empty_cuda_cache(self) -> None:
        if is_cuda_available():
            with suppress(RuntimeError, AttributeError):
                torch.cuda.empty_cache()


class NoopTorchSupport(BaseTorchSupport):
    """Stub used when torch is unavailable."""

    # Inherits no-op behaviour from BaseTorchSupport.


def build_torch_support(*, base_delay: int, max_delay: int, max_retries: int) -> BaseTorchSupport:
    if TORCH_AVAILABLE:
        return TorchWorkflowSupport(base_delay=base_delay, max_delay=max_delay, max_retries=max_retries)
    return NoopTorchSupport(base_delay=base_delay, max_delay=max_delay, max_retries=max_retries)


def is_torch_tensor(value: Any) -> bool:
    """Return True when value is a torch tensor and torch is installed."""
    return bool(TORCH_AVAILABLE and torch is not None and isinstance(value, torch.Tensor))


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

    # ⚡ Bolt Optimization: Use torch.from_numpy() instead of torch.tensor() to avoid unnecessary byte-copying.
    # We must ensure the array is contiguous and writable because torch.from_numpy requires it.
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.ascontiguousarray(array) if not array.flags.c_contiguous else array.copy()

    return torch.from_numpy(array).float() / 255.0


def tensor_from_pil(image: Image.Image) -> Any:
    """Create a tensor from a PIL image."""
    return tensor_from_array(np.array(image))


def tensor_to_image_array(tensor: Any) -> np.ndarray:
    """Convert a torch tensor into a uint8 numpy image array."""
    if not is_torch_tensor(tensor):
        raise ImportError("torch is required for tensor conversion")

    # ⚡ Bolt Optimization: Use PyTorch's optimized C++ backend to scale and clip tensors
    # before converting to NumPy to avoid memory-intensive intermediate allocations.
    if tensor.is_floating_point():
        return tensor.detach().cpu().mul(255.0).clamp_(0, 255).byte().numpy()

    # Non-floating point tensors were historically multiplied by 255.0 and clipped.
    # To preserve correctness while avoiding OOM, we convert to float first, scale, clip, and byte.
    return tensor.detach().cpu().float().mul(255.0).clamp_(0, 255).byte().numpy()


def torch_tensor_to_metadata(tensor: Any) -> TorchTensor | Any:
    """Wrap a torch tensor into metadata representation when available."""
    if not is_torch_tensor(tensor):
        return tensor
    from nodetool.metadata.types import TorchTensor as TorchTensorModel

    return TorchTensorModel.from_tensor(tensor)
