# PyMuPDF Document Resource Leaks

**Problem**: PyMuPDF document objects were not being closed after use, causing resource leaks.

**Solution**: Added try/finally blocks to ensure `doc.close()` is called in all PDF processing functions.

**Why**: PyMuPDF document objects hold file handles and memory. Without proper cleanup, repeated PDF processing can lead to file descriptor exhaustion and memory leaks.

**Files**:
- `src/nodetool/agents/tools/pdf_tools.py` - Fixed 3 functions:
  - `ExtractPDFTextTool.process()` (line 44)
  - `ExtractPDFTablesTool.process()` (line 100)
  - `ConvertPDFToMarkdownTool.process()` (line 185)
- `src/nodetool/indexing/ingestion.py` - Fixed sync version:
  - `default_ingestion_workflow()` (line 120)

**Date**: 2026-02-09
