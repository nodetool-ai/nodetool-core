# PR Merge Rate Analysis: Claude-Created PRs

**Date:** 2026-02-24  
**Analyzed:** 94 total Claude PRs (37 merged, 57 open/unmerged)  
**Merge rate:** ~39%

## What Gets Merged ✅

1. **Direct QA fixes**: typecheck failures, ruff lint errors, import ordering
2. **Targeted bug fixes**: async I/O blocking, exception handling, type annotations
3. **Minimal scope**: changes that touch only what's needed
4. **Clear validation**: PR explicitly shows `make typecheck`, `make lint`, `make test` pass

## What Doesn't Get Merged ❌

1. **Speculative new features**: e.g., `AsyncCountDownLatch` (#669), `AsyncGeneratorOrchestrator` (#516)
   - These were invented without an explicit issue requesting them
   - They add API surface that nobody needed
2. **Duplicate work**: e.g., #641 "fix: resolve ruff linting issues" — same fixes as #672 which was merged
   - The agent checked branches but not existing open PRs
3. **PR backlog overflow**: 57 open PRs is too many; reviewers can't keep up

## Root Causes

- `assistant-features.yaml` prompt was too open-ended ("choose a small high-impact feature")
- All scheduled workflows checked branches but NOT existing open PRs for duplicates
- No backlog limit to stop PR creation when queue is already large

## Fixes Applied (2026-02-24)

1. **Features workflow**: Now issue-driven only — must find a GitHub issue or TODO/FIXME before implementing
2. **All scheduled workflows**: Added PR backlog check (`gh pr list --state open --author "claude[bot]"`) — stops if >= 5 open PRs
3. **Improve/Test workflows**: Added PR title scan to detect duplicate open PRs before creating a new one

## Key Lessons for Future Sessions

- **Never add new utilities/abstractions unless explicitly requested** — they won't be merged
- **Always run `gh pr list --state open` before creating a new PR** — check if your fix is already covered
- **If there are 5+ open Claude PRs, stop** — the backlog needs to be cleared first
- **QA fixes (lint, typecheck) get merged quickly** when they're genuine fixes, not duplicates
