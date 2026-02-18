# PyMuPDF Document Resource Leak

**Problem**: PyMuPDF document objects were opened without proper cleanup in example code, potentially causing resource leaks when processing multiple PDFs.

**Solution**: Added `try/finally` block to ensure `doc.close()` is called after processing each PDF.

**Why**: PyMuPDF documents hold file resources and memory that need to be explicitly released. Without proper cleanup, this can lead to memory leaks and file handle leaks when processing multiple PDFs in a loop.

**Files**: `examples/chromadb_research_agent.py`

**Impact**:
- Ensures PyMuPDF document resources are properly released
- Prevents potential memory leaks when processing multiple PDFs
- Example code now demonstrates proper resource management

**Date**: 2026-02-18
