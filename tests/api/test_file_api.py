import os
import tempfile
from fastapi.testclient import TestClient


def test_list_files_excludes_hidden(tmp_path, client: TestClient):
    # Set the base directory for file API to the test temp path
    os.environ["FILE_API_BASE_DIR"] = str(tmp_path)
    directory = tmp_path / "files"
    directory.mkdir()
    (directory / "visible.txt").write_text("data")
    (directory / ".hidden.txt").write_text("hidden")

    response = client.get("/api/files/list", params={"path": str(directory)})
    assert response.status_code == 200
    names = [f["name"] for f in response.json()]
    assert "visible.txt" in names
    assert ".hidden.txt" not in names


def test_get_file_info(tmp_path, client: TestClient):
    # Set the base directory for file API to the test temp path
    os.environ["FILE_API_BASE_DIR"] = str(tmp_path)
    file_path = tmp_path / "info.txt"
    file_path.write_text("hello")
    response = client.get("/api/files/info", params={"path": str(file_path)})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "info.txt"
    assert data["is_dir"] is False


def test_upload_and_download_file(tmp_path, client: TestClient):
    # Set the base directory for file API to the test temp path
    os.environ["FILE_API_BASE_DIR"] = str(tmp_path)
    target = tmp_path / "upload.txt"
    content = b"hello world"
    response = client.post(
        f"/api/files/upload/{target}",
        files={"file": ("upload.txt", content, "text/plain")},
    )
    assert response.status_code == 200
    assert os.path.exists(target)

    download = client.get(f"/api/files/download/{target}")
    assert download.status_code == 200
    assert download.content == content
