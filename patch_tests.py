import re

with open("tests/api/test_file_api.py") as f:
    content = f.read()

# Add mock import if not present
if "from unittest.mock import patch" not in content:
    content = "from unittest.mock import patch\n" + content

# Patch test_list_files_excludes_hidden
content = content.replace("def test_list_files_excludes_hidden(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_list_files_excludes_hidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

# Patch test_get_file_info
content = content.replace("def test_get_file_info(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_get_file_info(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

# Patch test_upload_and_download_file
content = content.replace("def test_upload_and_download_file(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_upload_and_download_file(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

# Patch test_download_hidden_file_forbidden
content = content.replace("def test_download_hidden_file_forbidden(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_hidden_file_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

# Patch test_download_env_file_forbidden
content = content.replace("def test_download_env_file_forbidden(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_env_file_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

# Patch test_download_file_in_hidden_dir_forbidden
content = content.replace("def test_download_file_in_hidden_dir_forbidden(tmp_path, client: TestClient, headers: dict[str, str]):",
"""@patch("nodetool.api.file.SAFE_ROOTS", new_callable=list)
def test_download_file_in_hidden_dir_forbidden(mock_safe_roots, tmp_path, client: TestClient, headers: dict[str, str]):
    mock_safe_roots.append(str(tmp_path))""")

with open("tests/api/test_file_api.py", "w") as f:
    f.write(content)
