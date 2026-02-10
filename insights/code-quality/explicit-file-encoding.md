# Explicit File Encoding Best Practice

**Insight**: Always specify explicit encoding (typically `encoding="utf-8"`) when opening files in text mode using `open()` or `Path.open()`.

**Rationale**: Python's default text encoding varies by platform:
- Modern Linux: UTF-8 (via `utf-8-mode`)
- Windows: Can be `cp1252`, `gbk`, or other legacy encodings depending on locale
- Older systems: May use system locale encoding

Relying on implicit defaults leads to:
1. Cross-platform bugs where code works on Linux but fails on Windows
2. Silent data corruption when non-ASCII characters are involved
3. Unpredictable behavior in different deployment environments

**Example**:
```python
# Bad - relies on system default encoding
with open("config.json") as f:
    data = json.load(f)

# Good - explicit UTF-8 encoding
with open("config.json", encoding="utf-8") as f:
    data = json.load(f)

# Same for Path.open()
with Path("config.json").open("r", encoding="utf-8") as f:
    data = json.load(f)
```

**Detection**: Use ruff with PLW1514 rule to detect missing encoding:
```bash
ruff check --preview --select=PLW1514
```

**Fix**: Use ruff's auto-fix with unsafe fixes:
```bash
ruff check --preview --select=PLW1514 --fix --unsafe-fixes
```

**Reference**: PEP 597 - https://peps.python.org/pep-0597/

**Date**: 2026-02-10
