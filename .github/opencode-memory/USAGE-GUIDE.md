# Memory System Usage Guide for AI Agents

This guide explains how AI agents working on nodetool-core should use the OpenCode memory system.

## Quick Start

### Before Every Task

1. **Read repository-context.md** (Required)
   - Contains essential project information
   - Explains structure, conventions, and commands
   - Provides environment setup details

2. **Check issues/ folder** (If encountering issues)
    - See if your problem has been solved before
    - Avoid redundant debugging
    - Learn from past solutions

3. **Review insights.md** (For architectural work)
   - Understand key design patterns
   - Learn important architectural decisions
   - Follow established conventions

## When to Update Memory Files

### Update repository-context.md When:
- Adding new major project conventions
- Changing build/test/lint commands
- Updating development environment setup
- Adding new core dependencies or system requirements
- Documenting new architectural components

### Add files to issues/ When:
- You solve a non-trivial problem that took >10 minutes
- You encounter a confusing error with a clear solution
- You fix an issue that's likely to recur
- You discover an environment-specific quirk
- You find a workaround for a known limitation

### Update insights.md When:
- You discover an important architectural pattern
- You learn why something is designed a certain way
- You identify a performance consideration
- You find a security best practice specific to this codebase
- You document a design decision that should be followed

## What NOT to Document

Avoid documenting:
- Temporary workarounds that should be fixed properly
- Personal preferences without justification
- Information already in official docs (README, CONTRIBUTING)
- Overly specific details that change frequently
- One-off issues unlikely to recur
- Secrets or sensitive information

## Examples of Good Memory Entries

### Good Common Issue Entry
```markdown
### Import Errors After Adding Dependencies
**Date Discovered**: 2024-01-10
**Context**: New dependencies not found after adding to pyproject.toml
**Solution**: Run `uv sync --all-extras --dev` or `pip install -e .`
**Related Files**: `pyproject.toml`
**Prevention**: Document dependency installation in PR description
```

### Good Insight Entry
```markdown
### Async-First Design
**Date**: 2024-01-10
**Category**: Architecture
**Insight**: The codebase is heavily async-oriented. Most subsystems use asyncio.
**Impact**: New code should prefer async/await patterns. Avoid blocking calls.
**Examples**: `src/nodetool/workflows/`, `src/nodetool/agents/`
```

### Bad Entry (Too Specific)
```markdown
### Fixed typo on line 42 of utils.py
**Date**: 2024-01-10
**Solution**: Changed "teh" to "the"
```
This is too specific and not worth documenting.

### Bad Entry (Temporary Workaround)
```markdown
### Temporarily disable failing test
**Solution**: Comment out test_complicated_function() until we fix it
```
This is a workaround, not a proper solution.

## Memory File Maintenance

### Monthly Review (Suggested)
- Remove outdated information
- Consolidate duplicate entries
- Update dates and examples
- Archive old entries that are no longer relevant
- Verify all file paths and examples still exist

### Keep It Lean
- Each file should be readable in 2-3 minutes
- Focus on actionable information
- Use clear, concise language
- Link to detailed docs when appropriate

## Integration with Workflows

### opencode-hourly-test.yaml
- Reads memory before running tests
- Adds files to issues/ if fixing recurring problems
- Records test-specific patterns in insights/

### opencode-hourly-improve.yaml
- Consults memory before scanning for issues
- Documents discovered patterns in insights/
- Adds files to issues/

### opencode.yml (Comment-triggered)
- System prompt directs agent to memory files
- Agent should read relevant memory before starting
- Updates memory after completing significant work

## Benefits of Proper Memory Usage

✅ **Faster problem solving** - Check if issue is already solved
✅ **Consistent code quality** - Follow documented patterns
✅ **Institutional knowledge** - Build understanding over time
✅ **Reduced redundancy** - Don't solve the same problem twice
✅ **Better onboarding** - New agents learn from past work
✅ **Pattern recognition** - Identify recurring issues

## Measuring Success

Good memory usage leads to:
- Fewer repeated issues in PRs
- More consistent code patterns
- Faster resolution times
- Better PR descriptions (referencing past solutions)
- Proactive prevention of known issues

## Questions?

If the memory system isn't working well:
1. Are entries too vague or too specific?
2. Is information duplicated across files?
3. Are entries kept up-to-date?
4. Are agents actually reading the files?
5. Is the format easy to scan quickly?

Adjust the system based on what actually helps with coding work.
