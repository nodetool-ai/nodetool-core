# ChatCLI Missing Type Attributes

**Problem**: The `ChatCLI` class was missing several attributes (`debug_mode`, `all_tools`, `enabled_tools`, `refresh_tools`) that command classes expected to access. This caused 36 type checking diagnostics to be reported.

**Files Affected**:
- `src/nodetool/chat/chat_cli.py` - Missing attributes and refresh_tools method
- `src/nodetool/chat/commands/debug.py` - Using cli.console with Confirm.ask() type mismatch
- `src/nodetool/chat/commands/tools.py` - Using cli.all_tools, cli.enabled_tools, cli.refresh_tools()

**Solution**:
1. Added `debug_mode: bool = False` attribute to `ChatCLI.__init__()`
2. Added tool management attributes:
   - `all_tools: list[Tool] = []`
   - `enabled_tools: dict[str, bool] = {}`
3. Added `refresh_tools()` method to populate the tool list from `create_tools()`
4. Updated `save_settings()` to persist `debug_mode` and `enabled_tools`
5. Updated `load_settings()` to restore `debug_mode` and `enabled_tools`
6. Moved `Tool` import into `TYPE_CHECKING` block to avoid TC001 lint error
7. Fixed type issue in `debug.py` by using `cli.console._base` for `Confirm.ask()`

**Why**: The tool management commands were defined but the corresponding infrastructure on `ChatCLI` was incomplete, leading to type errors and potential runtime failures.

**Date**: 2026-02-07
