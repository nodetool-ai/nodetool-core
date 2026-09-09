import os

import pytest

from nodetool.io.get_files import get_content, get_files


def test_get_files(tmp_path):
    # Setup temporary directory structure with dummy files
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    subdir_path = dir_path / "subdir"
    subdir_path.mkdir()

    file1 = dir_path / "a.py"
    file1.write_text("print('a')")

    file2 = dir_path / "b.js"
    file2.write_text("console.log('b')")

    file3 = subdir_path / "c.py"
    file3.write_text("print('c')")

    file4 = subdir_path / "d.txt"
    file4.write_text("d")

    # Test getting files from directory
    files = get_files(str(dir_path), [".py", ".js"])

    # Sort for deterministic comparison
    files.sort()

    assert len(files) == 3
    assert str(file1) in files
    assert str(file2) in files
    assert str(file3) in files
    assert str(file4) not in files

    # Test getting a specific file
    single_file = get_files(str(file1), [".py"])
    assert len(single_file) == 1
    assert single_file[0] == str(file1)

    # Test getting a specific file with wrong extension
    wrong_ext_file = get_files(str(file4), [".py"])
    assert len(wrong_ext_file) == 0


def test_get_content(tmp_path):
    # Setup temporary directory structure with dummy files
    dir_path = tmp_path / "test_dir"
    dir_path.mkdir()

    file1 = dir_path / "a.py"
    file1.write_text("print('a')")

    file2 = dir_path / "b.js"
    file2.write_text("console.log('b')")

    content = get_content([str(dir_path)], [".py"])

    assert "##" in content
    assert str(file1) in content
    assert "print('a')" in content
    assert str(file2) not in content
