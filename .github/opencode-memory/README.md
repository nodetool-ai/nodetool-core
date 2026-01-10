# OpenCode Memory System

This directory contains long-term memory for OpenCode AI agents working on the nodetool-core repository.

## Purpose

The memory system helps AI agents:
1. **Avoid redundant work** - Check if a problem has been solved before
2. **Maintain consistency** - Follow established patterns and conventions
3. **Learn from history** - Build on past insights and solutions
4. **Work more efficiently** - Access repository context immediately

## Memory Files

### `repository-context.md`
Essential information about the repository structure, conventions, and development practices. Read this FIRST before starting any work.

**When to use:**
- Starting a new task
- Unfamiliar with the codebase
- Need to verify conventions or commands

### `common-issues.md`
Documented issues and their solutions. Saves time by providing proven fixes.

**When to use:**
- Before debugging a problem (check if it's already solved)
- After solving a non-trivial issue (document it for others)
- When encountering test/build failures

### `insights.md`
Important architectural decisions, patterns, and learnings about the codebase.

**When to use:**
- Before making architectural changes
- When designing new features
- To understand why things are structured a certain way

## Usage Guidelines for AI Agents

### Before Starting Work
1. **ALWAYS read `repository-context.md`** for project setup and conventions
2. Check `common-issues.md` for known problems related to your task
3. Review `insights.md` for architectural guidance

### During Work
- Refer back to memory files when making decisions
- Note any patterns or issues you discover
- Validate your changes follow documented conventions

### After Completing Work
1. **Document new issues** in `common-issues.md` if you solved something non-trivial
2. **Record insights** in `insights.md` if you discovered important patterns
3. **Update repository-context.md** if you added new major conventions

## What to Document

### DO Document
✅ Non-obvious solutions to common problems
✅ Architectural decisions and their rationale
✅ Patterns that should be followed consistently
✅ Environment-specific quirks or configurations
✅ Commands that must be run for validation
✅ Security considerations and pitfalls

### DON'T Document
❌ Temporary workarounds that should be fixed properly
❌ Personal preferences without justification
❌ Information that's already in official documentation
❌ Overly specific details that change frequently
❌ Secrets or sensitive information

## Maintenance

- Review memory files monthly to keep them current
- Archive outdated information
- Consolidate duplicate or conflicting information
- Keep entries concise and actionable

## Memory File Format

Each entry should be:
- **Dated**: Include when the information was added
- **Categorized**: Use appropriate sections
- **Actionable**: Provide clear guidance
- **Referenced**: Include file paths or examples where relevant
- **Concise**: Focus on key information

## Benefits

By maintaining this memory system, we:
- Reduce duplicate work across workflow runs
- Maintain consistency in code quality
- Build institutional knowledge
- Speed up onboarding for new tasks
- Improve overall code quality over time
