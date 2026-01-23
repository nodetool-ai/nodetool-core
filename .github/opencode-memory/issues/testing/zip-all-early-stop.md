# Zip All Early Stop Bug

**Problem**: Commit `cf0ae160` ("workflow logging") introduced `any_closed_and_empty()` check that breaks `test_zip_all_pairs_items_across_two_handles_in_order`. The check prematurely stops the zip_all loop when handles are marked as done, even though items exist in the inbox.

**Solution**: Removed the `any_closed_and_empty()` function and its call site from `src/nodetool/workflows/actor.py`. The original behavior allowed the iterator to naturally drain all items before exiting.

**Files**: `src/nodetool/workflows/actor.py`, `tests/workflows/test_actor.py`

**Date**: 2026-01-22
