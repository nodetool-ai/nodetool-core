from fastapi.testclient import TestClient
from nodetool.api.app import app

client = TestClient(app)
headers = {"Authorization": "Bearer admin"} # Or however auth is mocked

response = client.get("/api/files/list", params={"path": "/etc"}, headers=headers)
print("status:", response.status_code)
if response.status_code == 200:
    names = [f["name"] for f in response.json()]
    print("files in /etc:", names[:5])
