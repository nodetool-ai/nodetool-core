# Security Audit: Build Script False Positives

**Insight**: Ruff's security rules (S603, S310) flag `build.py` but these are false positives in context.

**Rationale**:
- **S603 (subprocess call)**: The `_run()` function uses `subprocess.run()` with a list of strings, not shell expansion. This is the safe pattern. Commands are hardcoded internal build commands (e.g., `["uv", "build"]`), not user input.

- **S310 (URL open)**: The `_req.Request()` and `urlopen()` calls are for GitHub API HTTPS requests with Bearer token authentication. The URL scheme is always `https://`, not `file:` or arbitrary schemes.

**Why It Matters**: 
- Suppressing these warnings in `build.py` would hide real security issues in other files
- The build script is a trusted internal tool, not a user-facing service
- The warnings appear in CI logs but don't indicate actual vulnerabilities

**Recommendation**: 
- Document these as expected warnings in build.py comments
- Do NOT add noqa comments that would suppress real issues elsewhere
- Continue using ruff's security rules for the main codebase

**Files**: `build.py`

**Date**: 2026-01-19
