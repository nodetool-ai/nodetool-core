# MiniMax Provider Typecheck Warnings

**Problem**: Two typecheck warnings in `src/nodetool/providers/minimax_provider.py`:
1. Line 638: Cannot subscript None (file_info could be None)
2. Line 814: Invalid method override (missing **kwargs parameter)

**Solution**:
1. Add `if file_info and` check before accessing `file_info.get("download_url")`
2. Add `**kwargs: Any` parameter to `image_to_video()` method signature

**Why**: 
1. The `file_info` variable could be an empty dict `{}`, which when used in a boolean context is falsy, but the type checker couldn't guarantee safe subscript access
2. The base class `BaseProvider.image_to_video()` has `**kwargs: Any` but the MiniMax override didn't include it, violating Liskov Substitution Principle

**Files**:
- `src/nodetool/providers/minimax_provider.py:638` (file_info subscript)
- `src/nodetool/providers/minimax_provider.py:814-821` (image_to_video signature)

**Date**: 2026-02-08
