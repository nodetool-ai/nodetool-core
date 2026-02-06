from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from .utils import current_user

_INVALID_SKILL_NAME_RE = re.compile(r"[^a-z0-9-]")
_XML_TAG_RE = re.compile(r"<[^>]+>")
_SKILL_RESERVED_TERMS = ("anthropic", "claude")


class SkillResponse(BaseModel):
    name: str
    description: str
    path: str
    instructions: Optional[str] = None


class SkillListResponse(BaseModel):
    count: int
    skills: list[SkillResponse]


router = APIRouter(prefix="/api/skills", tags=["skills"])


def _parse_frontmatter(frontmatter: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in frontmatter.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip().strip('"').strip("'")
    return parsed


def _is_valid_skill_name(name: str) -> bool:
    if not name or len(name) > 64:
        return False
    if _INVALID_SKILL_NAME_RE.search(name):
        return False
    lowered = name.lower()
    if any(term in lowered for term in _SKILL_RESERVED_TERMS):
        return False
    return True


def _is_valid_skill_description(description: str) -> bool:
    if not description or len(description) > 1024:
        return False
    if _XML_TAG_RE.search(description):
        return False
    return True


def _resolve_skill_dirs(skill_dirs: list[str] | None = None) -> list[Path]:
    resolved: list[Path] = []
    if skill_dirs:
        resolved.extend(Path(path).expanduser() for path in skill_dirs)

    env_dirs = os.environ.get("NODETOOL_AGENT_SKILL_DIRS")
    if env_dirs:
        resolved.extend(Path(path).expanduser() for path in env_dirs.split(os.pathsep) if path.strip())

    resolved.extend(
        [
            Path.cwd() / ".claude" / "skills",
            Path.home() / ".claude" / "skills",
            Path.home() / ".codex" / "skills",
        ]
    )
    ordered_paths = [Path(path) for path in dict.fromkeys(str(path) for path in resolved)]
    return [path for path in ordered_paths if path.exists()]


def _load_skill(skill_file: Path, *, include_instructions: bool) -> SkillResponse | None:
    try:
        content = skill_file.read_text(encoding="utf-8")
    except Exception:
        return None

    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    metadata = _parse_frontmatter(parts[1])
    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    instructions = parts[2].strip()

    if not _is_valid_skill_name(name):
        return None
    if not _is_valid_skill_description(description):
        return None
    if not instructions:
        return None

    return SkillResponse(
        name=name,
        description=description,
        path=str(skill_file),
        instructions=instructions if include_instructions else None,
    )


def _discover_skills(skill_dirs: list[Path], *, include_instructions: bool) -> dict[str, SkillResponse]:
    discovered: dict[str, SkillResponse] = {}
    for skill_dir in skill_dirs:
        for skill_file in skill_dir.rglob("SKILL.md"):
            skill = _load_skill(skill_file, include_instructions=include_instructions)
            if not skill:
                continue
            if skill.name in discovered:
                continue
            discovered[skill.name] = skill
    return discovered


@router.get("", response_model=SkillListResponse)
async def list_skills(
    include_instructions: bool = Query(default=False),
    skill_dir: list[str] | None = Query(default=None),
    user: str = Depends(current_user),
) -> SkillListResponse:
    del user
    discovered = _discover_skills(_resolve_skill_dirs(skill_dir), include_instructions=include_instructions)
    skills = sorted(discovered.values(), key=lambda s: s.name)
    return SkillListResponse(count=len(skills), skills=skills)


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(
    skill_name: str,
    skill_dir: list[str] | None = Query(default=None),
    user: str = Depends(current_user),
) -> SkillResponse:
    del user
    discovered = _discover_skills(_resolve_skill_dirs(skill_dir), include_instructions=True)
    skill = discovered.get(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
    return skill
