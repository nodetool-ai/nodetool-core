"""Tests for workflows/torch_support.py - Torch utilities and device management."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nodetool.workflows.torch_support import (
    TORCH_AVAILABLE,
    BaseTorchSupport,
    NoopTorchSupport,
    TorchWorkflowSupport,
    build_torch_support,
    detach_tensor,
    detach_tensors_recursively,
    is_cuda_available,
    is_torch_tensor,
    tensor_from_array,
    tensor_to_image_array,
    torch_tensor_to_metadata,
)


class TestTorchAvailability:
    """Tests for torch availability detection."""

    def test_torch_available_flag(self):
        """Test TORCH_AVAILABLE reflects actual torch installation."""
        # Just check the flag is a boolean
        assert isinstance(TORCH_AVAILABLE, bool)

    def test_is_cuda_available_when_torch_not_available(self):
        """Test is_cuda_available returns False when torch is not available."""
        if not TORCH_AVAILABLE:
            assert is_cuda_available() is False


class TestTensorDetection:
    """Tests for tensor detection utilities."""

    def test_is_torch_tensor_with_none(self):
        """Test is_torch_tensor with None returns False."""
        assert is_torch_tensor(None) is False

    def test_is_torch_tensor_with_int(self):
        """Test is_torch_tensor with int returns False."""
        assert is_torch_tensor(42) is False

    def test_is_torch_tensor_with_list(self):
        """Test is_torch_tensor with list returns False."""
        assert is_torch_tensor([1, 2, 3]) is False

    def test_is_torch_tensor_with_numpy_array(self):
        """Test is_torch_tensor with numpy array returns False."""
        array = np.array([1, 2, 3])
        assert is_torch_tensor(array) is False


class TestTensorDetachment:
    """Tests for tensor detachment utilities."""

    def test_detach_tensor_with_none(self):
        """Test detach_tensor with None returns None."""
        assert detach_tensor(None) is None

    def test_detach_tensor_with_int(self):
        """Test detach_tensor with int returns the int."""
        assert detach_tensor(42) == 42

    def test_detach_tensor_with_list(self):
        """Test detach_tensor with list returns the list."""
        lst = [1, 2, 3]
        assert detach_tensor(lst) is lst

    def test_detach_tensors_recursively_with_dict(self):
        """Test detach_tensors_recursively with dict."""
        input_dict = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        result = detach_tensors_recursively(input_dict)
        assert result == input_dict

    def test_detach_tensors_recursively_with_list(self):
        """Test detach_tensors_recursively with list."""
        input_list = [1, 2, [3, 4]]
        result = detach_tensors_recursively(input_list)
        assert result == input_list

    def test_detach_tensors_recursively_with_tuple(self):
        """Test detach_tensors_recursively with tuple."""
        input_tuple = (1, 2, (3, 4))
        result = detach_tensors_recursively(input_tuple)
        assert result == input_tuple
        assert isinstance(result, tuple)

    def test_detach_tensors_recursively_with_nested_structures(self):
        """Test detach_tensors_recursively with nested structures."""
        input_data = {
            "list": [1, 2, 3],
            "dict": {"nested": [4, 5]},
            "tuple": (1, 2),
        }
        result = detach_tensors_recursively(input_data)
        assert result["list"] == [1, 2, 3]
        assert result["dict"] == {"nested": [4, 5]}
        assert result["tuple"] == (1, 2)


class TestTensorConversion:
    """Tests for tensor conversion utilities."""

    def test_tensor_from_array_raises_when_torch_not_available(self):
        """Test tensor_from_array raises ImportError when torch is not available."""
        if not TORCH_AVAILABLE:
            array = np.array([1, 2, 3])
            with pytest.raises(ImportError, match="torch is required"):
                tensor_from_array(array)

    def test_tensor_to_image_array_with_non_tensor(self):
        """Test tensor_to_image_array raises when input is not a tensor."""
        if not TORCH_AVAILABLE:
            array = np.array([1, 2, 3])
            with pytest.raises(ImportError, match="torch is required"):
                tensor_to_image_array(array)


class TestTorchMetadata:
    """Tests for torch metadata conversion."""

    def test_torch_tensor_to_metadata_with_non_tensor(self):
        """Test torch_tensor_to_metadata returns input when not a tensor."""
        value = 42
        result = torch_tensor_to_metadata(value)
        assert result == value

    def test_torch_tensor_to_metadata_with_none(self):
        """Test torch_tensor_to_metadata returns None."""
        result = torch_tensor_to_metadata(None)
        assert result is None

    def test_torch_tensor_to_metadata_with_list(self):
        """Test torch_tensor_to_metadata returns list unchanged."""
        value = [1, 2, 3]
        result = torch_tensor_to_metadata(value)
        assert result == value


class TestTorchSupportClasses:
    """Tests for torch support classes."""

    def test_base_torch_support_init(self):
        """Test BaseTorchSupport initialization."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.base_delay == 1
        assert support.max_delay == 10
        assert support.max_retries == 3

    def test_base_torch_support_get_available_vram(self):
        """Test BaseTorchSupport.get_available_vram returns 0."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.get_available_vram() == 0

    def test_base_torch_support_log_vram_usage(self):
        """Test BaseTorchSupport.log_vram_usage is a no-op."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        mock_runner = MagicMock()
        support.log_vram_usage(mock_runner, "Test message")  # Should not raise

    def test_base_torch_support_torch_context(self):
        """Test BaseTorchSupport.torch_context is a no-op context manager."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        mock_runner = MagicMock()
        mock_context = MagicMock()
        with support.torch_context(mock_runner, mock_context):
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_base_torch_support_process_with_gpu(self):
        """Test BaseTorchSupport.process_with_gpu calls node.process."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        mock_runner = MagicMock()
        mock_context = MagicMock()
        mock_node = MagicMock()
        mock_node.process = AsyncMock(return_value="result")

        result = await support.process_with_gpu(mock_runner, mock_context, mock_node)
        assert result == "result"
        mock_node.process.assert_called_once_with(mock_context)

    def test_base_torch_support_is_cuda_oom_exception(self):
        """Test BaseTorchSupport.is_cuda_oom_exception returns False."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.is_cuda_oom_exception(Exception("test")) is False

    def test_base_torch_support_empty_cuda_cache(self):
        """Test BaseTorchSupport.empty_cuda_cache is a no-op."""
        support = BaseTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        support.empty_cuda_cache()  # Should not raise

    def test_noop_torch_support_init(self):
        """Test NoopTorchSupport initialization."""
        support = NoopTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.base_delay == 1
        assert support.max_delay == 10
        assert support.max_retries == 3

    def test_noop_torch_support_inherits_base_behavior(self):
        """Test NoopTorchSupport inherits no-op behavior."""
        support = NoopTorchSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.get_available_vram() == 0
        assert support.is_cuda_oom_exception(Exception("test")) is False

    def test_torch_workflow_support_init(self):
        """Test TorchWorkflowSupport initialization."""
        support = TorchWorkflowSupport(base_delay=1, max_delay=10, max_retries=3)
        assert support.base_delay == 1
        assert support.max_delay == 10
        assert support.max_retries == 3

    def test_torch_workflow_support_get_available_vram_when_no_cuda(self):
        """Test TorchWorkflowSupport.get_available_vram returns 0 when CUDA unavailable."""
        if not is_cuda_available():
            support = TorchWorkflowSupport(base_delay=1, max_delay=10, max_retries=3)
            assert support.get_available_vram() == 0

    def test_torch_workflow_support_log_vram_usage_when_no_cuda(self):
        """Test TorchWorkflowSupport.log_vram_usage is no-op when CUDA unavailable."""
        if not is_cuda_available():
            support = TorchWorkflowSupport(base_delay=1, max_delay=10, max_retries=3)
            mock_runner = MagicMock()
            support.log_vram_usage(mock_runner, "Test message")  # Should not raise

    def test_torch_workflow_support_is_cuda_oom_exception_when_no_torch(self):
        """Test TorchWorkflowSupport.is_cuda_oom_exception returns False when no torch."""
        if not TORCH_AVAILABLE:
            support = TorchWorkflowSupport(base_delay=1, max_delay=10, max_retries=3)
            assert support.is_cuda_oom_exception(Exception("test")) is False

    def test_torch_workflow_support_empty_cuda_cache_when_no_cuda(self):
        """Test TorchWorkflowSupport.empty_cuda_cache is no-op when CUDA unavailable."""
        if not is_cuda_available():
            support = TorchWorkflowSupport(base_delay=1, max_delay=10, max_retries=3)
            support.empty_cuda_cache()  # Should not raise


class TestBuildTorchSupport:
    """Tests for build_torch_support factory function."""

    def test_build_torch_support_returns_noop_when_torch_unavailable(self):
        """Test build_torch_support returns NoopTorchSupport when torch unavailable."""
        if not TORCH_AVAILABLE:
            support = build_torch_support(base_delay=1, max_delay=10, max_retries=3)
            assert isinstance(support, NoopTorchSupport)

    def test_build_torch_support_returns_torch_when_available(self):
        """Test build_torch_support returns TorchWorkflowSupport when torch available."""
        support = build_torch_support(base_delay=1, max_delay=10, max_retries=3)
        if TORCH_AVAILABLE:
            assert isinstance(support, TorchWorkflowSupport)
        else:
            assert isinstance(support, NoopTorchSupport)
