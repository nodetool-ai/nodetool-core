# OpenCode Long-Term Memory

This directory contains persistent knowledge for OpenCode workflows in **nodetool-core**. Content is organized into folders by topic, with individual files per issue/insight to keep context focused and reduce merge conflicts.

## Directory Structure

```
opencode-memory/
├── README.md              # This file - overview and usage
├── features.md            # List of user-facing features
├── build-test-lint.md     # Quality requirements and commands
├── tech-stack.md          # Key technologies and versions
├── repository-context.md  # Project structure and conventions
│
├── issues/                # Known issues organized by topic
│   ├── README.md          # How to add new issues
│   ├── testing/
│   ├── linting/
│   ├── typing/
│   ├── workflows/
│   ├── api/
│   ├── storage/
│   ├── dependencies/
│   ├── config/
│   └── ci/
│
└── insights/              # Best practices organized by topic
    ├── README.md          # How to add new insights
    ├── architecture/
    ├── performance/
    ├── testing/
    ├── code-quality/
    ├── async/
    ├── workflows/
    ├── api/
    ├── storage/
    └── deployment/
```

## How to Use

### Before Making Changes

1. Check `features.md` to avoid duplicate feature work
2. List `issues/` to discover relevant topics, then read descriptive files
3. List `insights/` to discover best practices, then read descriptive files
4. Read `repository-context.md`

### After Completing Work

1. New feature? → Add ONE line to `features.md`
2. Solved a tricky issue? → Create a file in `issues/<topic>/`
3. Discovered a best practice? → Create a file in `insights/<topic>/`

## File Naming Convention

Use descriptive, kebab-case names that make the topic obvious at a glance:
- `issues/testing/pytest-async-timeouts.md`
- `issues/workflows/workflow-runner-state-leaks.md`
- `insights/async/non-blocking-db-queries.md`

## File Format

### Issue Files (`issues/<topic>/*.md`)

```markdown
# Issue Title

**Problem**: One sentence describing the issue

**Solution**: One sentence or brief code snippet

**Why**: Brief explanation (optional)

**Files**: Related files (optional)

**Date**: YYYY-MM-DD
```

### Insight Files (`insights/<topic>/*.md`)

```markdown
# Insight Title

**Insight**: What was learned or discovered

**Rationale**: Why it matters

**Example**: Code example (if applicable)

**Impact**: Measurable benefit (if known)

**Files**: Related files (optional)

**Date**: YYYY-MM-DD
```

## Benefits

✅ Fewer merge conflicts with file-based entries
✅ Focused context per issue or insight
✅ Easier discovery by listing topic folders
✅ Cleaner diffs for updates and removals
