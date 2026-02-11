# Shell Escaping Best Practices for Security

**Insight**: When constructing shell commands programmatically, always use `shlex.quote()` to escape user-supplied arguments. This prevents shell metacharacters from being interpreted.

**Rationale**: Direct string interpolation into shell commands is a critical security vulnerability. Shell metacharacters like `;`, `&`, `|`, `$`, `()` can be used to execute arbitrary commands. Using proper quoting ensures user input is treated as a single literal argument.

**Example**:
\`\`\`python
# VULNERABLE:
ssh.execute(f"cat {user_path}")  # user_path = "../../etc/passwd; rm -rf /"

# SECURE:
safe_path = shlex.quote(user_path)
ssh.execute(f"cat {safe_path}")  # Treated as literal argument
\`\`\`

**Impact**: Prevents command injection vulnerabilities in any code that executes shell commands with user-controlled input.

**Files**:
- `src/nodetool/deploy/remote_users.py`
- `src/nodetool/deploy/self_hosted.py`

**Date**: 2026-02-11
