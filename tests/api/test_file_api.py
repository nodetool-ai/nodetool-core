from unittest.mock import patch
import os

from fastapi.testclient import TestClient


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_list_files_excludes_hidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    directory = tmp_path / "files"
    directory.mkdir()
    (directory / "visible.txt").write_text("data")
    (directory / ".hidden.txt").write_text("hidden")

    response = client.get("/api/files/list", params={"path": str(directory)}, headers=headers)
    assert response.status_code == 200
    names = [f["name"] for f in response.json()]
    assert "visible.txt" in names
    assert ".hidden.txt" not in names


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_get_file_info(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    file_path = tmp_path / "info.txt"
    file_path.write_text("hello")
    response = client.get("/api/files/info", params={"path": str(file_path)}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "info.txt"
    assert data["is_dir"] is False


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_upload_and_download_file(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    target = tmp_path / "upload.txt"
    content = b"hello world"
    response = client.post(
        f"/api/files/upload/{target}",
        files={"file": ("upload.txt", content, "text/plain")},
        headers=headers,
    )
    assert response.status_code == 200
    assert os.path.exists(target)

    download = client.get(f"/api/files/download/{target}", headers=headers)
    assert download.status_code == 200
    assert download.content == content


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_hidden_file_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    # Create a hidden file
    hidden_file = tmp_path / ".secret"
    hidden_file.write_text("secret_data")

    # Try to download it
    response = client.get(f"/api/files/download/{hidden_file}", headers=headers)

    # This should fail with 403 Forbidden
    assert response.status_code == 403
    assert response.json()["detail"] == "Access to this path is forbidden"


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_env_file_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    # Create a .env file
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_KEY=12345")

    # Try to download it
    response = client.get(f"/api/files/download/{env_file}", headers=headers)

    # This should fail with 403 Forbidden
    assert response.status_code == 403
    assert response.json()["detail"] == "Access to this path is forbidden"


@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_file_in_hidden_dir_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))
    # Create a hidden directory and a file inside
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    secret_file = hidden_dir / "secret.txt"
    secret_file.write_text("secret_data")

    # Try to download it
    response = client.get(f"/api/files/download/{secret_file}", headers=headers)

    # This should fail with 403 Forbidden because a path component starts with .
    assert response.status_code == 403
    assert response.json()["detail"] == "Access to this path is forbidden"


def test_upload_to_system_dir_forbidden(client: TestClient, headers: dict[str, str]):
    """Ensure uploading to sensitive system directories is forbidden."""
    # We use a mocked path or a real system path that should be blocked
    # /usr/bin/pwned.txt
    target_file = "/usr/bin/pwned.txt"
    content = b"hacked"

    response = client.post(
        f"/api/files/upload/{target_file}",
        files={"file": ("pwned.txt", content, "text/plain")},
        headers=headers,
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Access to this path is forbidden"


def test_download_from_system_dir_forbidden(client: TestClient, headers: dict[str, str]):
    """Ensure downloading from sensitive system directories is forbidden."""
    # We request a file that likely doesn't exist but is in a blocked path
    # The check happens before existence check
    target_file = "/usr/bin/secret"

    response = client.get(f"/api/files/download/{target_file}", headers=headers)

    assert response.status_code == 403
    assert response.json()["detail"] == "Access to this path is forbidden"
