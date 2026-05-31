import sys

from nodetool.worker.stdio_stdout_guard import (
    get_protocol_stdout_buffer,
    install_stdio_stdout_guard,
)


def test_install_stdio_stdout_guard_redirects_text_to_stderr(capfd):
    import os
    import nodetool.worker.stdio_stdout_guard as guard_mod

    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__
    assert real_stdout is not None
    assert real_stderr is not None

    original_fd1 = os.dup(1)

    install_stdio_stdout_guard()

    marker = "stdout-guard-test-marker"
    print(marker)
    sys.stdout.flush()
    get_protocol_stdout_buffer().write(b"binary-protocol-bytes")

    captured = capfd.readouterr()

    # Restore OS fd 1 and sys.stdout for other tests.
    os.dup2(original_fd1, 1)
    os.close(original_fd1)
    sys.stdout = real_stdout
    sys.stderr = real_stderr
    guard_mod._protocol_stdout_fd = None

    assert marker in captured.err
    assert marker not in captured.out
