import io
import os

import joblib
import numpy as np
import pytest

from nodetool.workflows.processing_offload import _joblib_load_from_io


def test_exploit_fails():
    """Test that loading an exploit payload raises a ValueError."""

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo 'VULNERABILITY EXPLOITED' > exploit_result_test.txt",))

    buffer = io.BytesIO()
    joblib.dump(Exploit(), buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="Unsafe or invalid model data"):
        _joblib_load_from_io(buffer)

    assert not os.path.exists("exploit_result_test.txt")


def test_exploit_eval_fails():
    """Test that loading an exploit payload using builtins.eval raises a ValueError."""

    class Exploit:
        def __reduce__(self):
            return (eval, ("print('VULNERABILITY EXPLOITED')",))

    buffer = io.BytesIO()
    joblib.dump(Exploit(), buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="Unsafe or invalid model data"):
        _joblib_load_from_io(buffer)


def test_valid_load_numpy():
    """Test that loading a valid numpy array works."""
    data = np.array([[1, 2], [3, 4]])
    buffer = io.BytesIO()
    joblib.dump(data, buffer)
    buffer.seek(0)

    loaded = _joblib_load_from_io(buffer)
    assert np.array_equal(data, loaded)


def test_valid_load_basic():
    """Test that loading basic python types works."""
    data = {"a": 1, "b": [2, 3], "c": "test"}
    buffer = io.BytesIO()
    joblib.dump(data, buffer)
    buffer.seek(0)

    loaded = _joblib_load_from_io(buffer)
    assert loaded == data


def test_valid_load_compressed():
    """Test that loading a compressed joblib file works."""
    data = np.array([[1, 2], [3, 4]])
    buffer = io.BytesIO()
    joblib.dump(data, buffer, compress=3)
    buffer.seek(0)

    loaded = _joblib_load_from_io(buffer)
    assert np.array_equal(data, loaded)
