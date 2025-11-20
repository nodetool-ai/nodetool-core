from nodetool.workflows.actor import _device_move_plan


def test_device_plan_cuda():
    assert _device_move_plan("cuda") == ("cuda", "cpu")


def test_device_plan_cuda_index():
    assert _device_move_plan("cuda:0") == ("cuda:0", "cpu")


def test_device_plan_mps():
    assert _device_move_plan("mps") == (None, None)


def test_device_plan_cpu():
    assert _device_move_plan("cpu") == (None, None)


def test_device_plan_none():
    assert _device_move_plan(None) == (None, None)
