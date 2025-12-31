"""
Lua Runner (Docker-backed with local subprocess fallback)
========================================================

Executes user-supplied Lua code and streams raw stdout/stderr lines. A
lightweight sandbox is applied by constructing a restricted environment that
excludes I/O and OS access by default. The user code is executed via `load`
with a provided environment table when available (Lua 5.2+), or using
`loadstring` + `setfenv` as a compatibility fallback (Lua 5.1).

Behavior matches other runners: attempts to run inside Docker (default image
"lua:5.4"), and when Docker is unavailable, falls back to a local
subprocess using the configured executable ("lua" or "luajit").
"""

from __future__ import annotations

from typing import Any

from nodetool.code_runners.runtime_base import StreamRunnerBase


def _lua_escape_string(value: str) -> str:
    """Return a Lua string literal representing value.

    Uses long-bracket quoting with a single equals level to minimize escaping.
    If the input happens to contain the closing `]===]` sequence, falls back to
    standard quoted string with escaped characters.
    """
    # Prefer long-bracket quoting to preserve newlines and avoid heavy escaping
    if "]===]" not in value:
        return "[===[" + value + "]===]"
    # Fallback: escape backslashes and quotes for standard quoting
    escaped = (
        value.replace("\\", r"\\\\")
        .replace("\n", r"\n")
        .replace("\t", r"\t")
        .replace("\r", r"\r")
        .replace('"', r"\\\"")
    )
    return '"' + escaped + '"'


def _lua_literal(value: Any, depth: int = 0) -> str:
    """Best-effort conversion of a Python value to a Lua literal.

    Supports None, bool, int, float, str, list/tuple, and dict with string keys.
    Falls back to string() via tostring semantics for unknown types.
    Depth is limited to avoid excessively deep recursion.
    """
    if depth > 10:
        return "nil"
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return _lua_escape_string(value)
    if isinstance(value, list | tuple):
        items = ", ".join(_lua_literal(v, depth + 1) for v in value)
        return "{" + items + "}"
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            if isinstance(k, str) and k.isidentifier():
                parts.append(f"{k} = {_lua_literal(v, depth + 1)}")
            else:
                parts.append(f"[{_lua_literal(k, depth + 1)}] = {_lua_literal(v, depth + 1)}")
        return "{" + ", ".join(parts) + "}"
    # Fallback: coerce to string
    return _lua_escape_string(str(value))


class LuaRunner(StreamRunnerBase):
    """Execute Lua code with Docker or local subprocess, streaming stdout/stderr."""

    def __init__(
        self,
        image: str = "nickblah/lua:5.2.4-luarocks-ubuntu",
        executable: str = "lua",
        **kwargs,
    ) -> None:
        super().__init__(image=image, **kwargs)
        self.executable = executable

    def build_container_command(self, user_code: str, env_locals: dict[str, Any]) -> list[str]:
        """Return a `lua -e` command that evaluates sandboxed user code.

        The generated snippet:
        - builds a restricted environment `_ENV` with safe standard libs only
        - injects provided locals into `_ENV`
        - loads and runs the user code within that environment
        """
        injected_locals_lines: list[str] = []
        for key, val in (env_locals or {}).items():
            if not isinstance(key, str) or not key.isidentifier():
                # Skip invalid identifiers
                continue
            injected_locals_lines.append(f"_ENV[{_lua_literal(key)}] = {_lua_literal(val)}")

        # Compose the sandbox prelude and user code executor
        lua_snippet = (
            "do "
            "local _G = _G; "
            "local base = {assert=assert, error=error, ipairs=ipairs, next=next, pairs=pairs, pcall=pcall, select=select, tonumber=tonumber, tostring=tostring, type=type, utf8=utf8, math=math, string=string, table=table, print=print}; "
            "local _ENV = setmetatable({}, { __index = base }); "
            + ("; ".join(injected_locals_lines) + "; " if injected_locals_lines else "")
            + "local src = "
            + _lua_escape_string(user_code)
            + "; "
            + (
                # Prefer Lua 5.1-compatible loadstring first; fall back to 5.2+ load with env
                "local fn, err = nil, nil; "
                "if _G.loadstring then fn, err = _G.loadstring(src); if fn and _G.setfenv then _G.setfenv(fn, _ENV) end end; "
                "if not fn and _G.load then fn, err = _G.load(src, 'user', 't', _ENV) end; "
                "if not fn then error(err or 'failed to load code') end; "
                "fn(); "
            )
            + "end"
        )

        return [self.executable, "-e", lua_snippet]


class LuaSubprocessRunner(LuaRunner):
    """Convenience runner that always uses local subprocess mode."""

    def __init__(self, executable: str = "lua", **kwargs) -> None:
        super().__init__(executable=executable, mode="subprocess", **kwargs)
