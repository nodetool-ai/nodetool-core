from fastapi.testclient import TestClient
from nodetool.api.app import app

client = TestClient(app)
headers = {"Authorization": "Bearer admin"} # Or however auth is mocked

from nodetool.api.file import _is_safe_path
print("is safe /tmp/hack?", _is_safe_path("/tmp/hack"))
print("is safe /etc/passwd?", _is_safe_path("/etc/passwd"))
print("is safe ~/.ssh/id_rsa?", _is_safe_path("~/.ssh/id_rsa"))
