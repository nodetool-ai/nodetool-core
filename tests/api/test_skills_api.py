from fastapi.testclient import TestClient


def test_list_skills(tmp_path, client: TestClient, headers: dict[str, str], monkeypatch):
    skill_root = tmp_path / "skills"
    valid_skill = skill_root / "data-review"
    valid_skill.mkdir(parents=True)
    (valid_skill / "SKILL.md").write_text(
        """---
name: data-review
description: Analyze datasets and summarize anomalies
---
# Data Review

Always compute aggregate statistics before final conclusions.
""",
        encoding="utf-8",
    )

    invalid_skill = skill_root / "bad-skill"
    invalid_skill.mkdir(parents=True)
    (invalid_skill / "SKILL.md").write_text(
        """---
name: bad_skill
description: Invalid because underscore in name
---
# Bad
Do not load.
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("NODETOOL_AGENT_SKILL_DIRS", str(skill_root))

    response = client.get("/api/skills", params={"skill_dir": str(skill_root)}, headers=headers)
    assert response.status_code == 200
    payload = response.json()
    names = [skill["name"] for skill in payload["skills"]]
    assert "data-review" in names
    data_review = next(skill for skill in payload["skills"] if skill["name"] == "data-review")
    assert data_review["instructions"] is None


def test_get_skill(tmp_path, client: TestClient, headers: dict[str, str], monkeypatch):
    skill_root = tmp_path / "skills"
    valid_skill = skill_root / "workflow-audit"
    valid_skill.mkdir(parents=True)
    (valid_skill / "SKILL.md").write_text(
        """---
name: workflow-audit
description: Review workflow graphs for structure issues
---
# Workflow Audit

Check for cycles, missing outputs, and invalid node types.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("NODETOOL_AGENT_SKILL_DIRS", str(skill_root))

    response = client.get(
        "/api/skills/workflow-audit",
        params={"skill_dir": str(skill_root)},
        headers=headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "workflow-audit"
    assert "Check for cycles" in payload["instructions"]


def test_get_skill_not_found(client: TestClient, headers: dict[str, str]):
    response = client.get("/api/skills/nonexistent-skill", headers=headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
