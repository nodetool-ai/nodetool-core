import os
import tempfile
from pathlib import Path

from nodetool.io.get_files import get_content, get_files


def test_get_files_and_get_content():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a directory structure
        base_dir = Path(tmpdir)

        # Files that should match
        file1 = base_dir / "test1.py"
        file1.write_text("print('test1')")

        subdir = base_dir / "subdir"
        subdir.mkdir()

        file2 = subdir / "test2.js"
        file2.write_text("console.log('test2')")

        # Files that should be ignored
        file3 = base_dir / "ignore.txt"
        file3.write_text("ignore me")

        # Test get_files
        files = get_files(str(base_dir), [".py", ".js"])
        assert len(files) == 2
        assert str(file1) in files
        assert str(file2) in files
        assert str(file3) not in files

        # Test get_content
        content = get_content([str(base_dir)], [".py", ".js"])

        # Check that both files' content and headers are in the output
        assert "## " + str(file1) in content
        assert "print('test1')" in content
        assert "## " + str(file2) in content
        assert "console.log('test2')" in content
        assert "ignore me" not in content
