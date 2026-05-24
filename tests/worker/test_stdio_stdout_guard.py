import io
import sys

from nodetool.worker.stdio_stdout_guard import install_stdio_stdout_guard


def test_install_stdio_stdout_guard_redirects_text_to_stderr(capsys):
    real_stdout = sys.__stdout__
    real_stderr = sys.__stderr__
    assert real_stdout is not None
    assert real_stderr is not None

    install_stdio_stdout_guard()

    marker = "stdout-guard-test-marker"
    print(marker)
    real_stdout.buffer.write(b"binary-protocol-bytes")

    captured = capsys.readouterr()
    assert marker in captured.err
    assert marker not in captured.out

    # Restore for other tests.
    sys.stdout = real_stdout
    sys.stderr = real_stderr
