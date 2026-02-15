# MiniMax Test Mock Missing params Argument

**Problem**: MiniMax provider video generation tests failed because mock `MockSession.get()` didn't accept the `params` keyword argument.

**Solution**: Update both `MockSession.get()` method signatures in tests to accept `**kwargs` including `params`.

**Why**: The real `MiniMaxProvider._resolve_video_download_url()` code calls `session.get()` with `params={"file_id": file_id}` to query parameters, but the test mocks only accepted `url` and `headers`.

**Files**:
- `tests/chat/providers/test_minimax_provider.py:795-801` (test_text_to_video_success mock)
- `tests/chat/providers/test_minimax_provider.py:928-932` (test_image_to_video_success mock)

**Related Tests**:
- `tests/chat/providers/test_minimax_provider.py::TestMiniMaxVideoGeneration::test_text_to_video_success`
- `tests/chat/providers/test_minimax_provider.py::TestMiniMaxVideoGeneration::test_image_to_video_success`

**Date**: 2026-02-08
