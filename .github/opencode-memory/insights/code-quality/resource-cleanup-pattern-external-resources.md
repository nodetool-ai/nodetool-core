# Resource Cleanup Pattern for External Resources

**Insight**: Always use try/finally blocks to ensure cleanup of external resources (file handles, network connections, document objects) even when exceptions occur.

**Rationale**: External resources like PyMuPDF documents, file handles, and database connections hold system resources that must be released. Without proper cleanup, long-running processes can exhaust file descriptors or memory.

**Example**:
```python
# Before - resource leak on exception
doc = pymupdf.open(path)
text = doc.get_text()
return text  # If exception occurs above, doc never closes

# After - proper cleanup
doc = pymupdf.open(path)
try:
    text = doc.get_text()
    return text
finally:
    doc.close()  # Always executed, even on exception
```

**Impact**: Fixed 4 resource leaks in PDF processing code that could cause file descriptor exhaustion in production workloads.

**Files**:
- `src/nodetool/agents/tools/pdf_tools.py`
- `src/nodetool/indexing/ingestion.py`

**Date**: 2026-02-09
