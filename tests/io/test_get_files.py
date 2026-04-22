import os
import pytest
from nodetool.io.get_files import get_files, get_content

def test_get_files(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p1 = d / "file1.py"
    p1.write_text("print('test')")
    p2 = d / "file2.js"
    p2.write_text("console.log('test')")
    p3 = d / "file3.txt"
    p3.write_text("test")

    files = get_files(str(d))
    assert len(files) == 2
    assert str(p1) in files
    assert str(p2) in files

def test_get_content(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p1 = d / "file1.py"
    p1.write_text("print('test1')")
    p2 = d / "file2.js"
    p2.write_text("console.log('test2')")

    content = get_content([str(d)])
    assert "print('test1')" in content
    assert "console.log('test2')" in content
    assert f"## {str(p1)}" in content
    assert f"## {str(p2)}" in content
