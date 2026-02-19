# Resource Leak Prevention in Python

**Insight**: Always use context managers or explicit cleanup for resources that hold file handles, network connections, or other system resources.

**Rationale**: Resource leaks can accumulate over time, especially in long-running processes or when processing many files. Even if a function appears to work correctly, unhandled exceptions can bypass cleanup code, leaving resources open.

**Best Practices**:
1. **Context Managers**: Use `with` statements for PIL Images, file handles, and database connections
2. **Try/Finally**: When context managers aren't available, use `try/finally` to ensure cleanup
3. **Async Resources**: Ensure async resources are properly closed even when exceptions occur

**Example - PIL Image**:
```python
# BAD - Resource leak if exception occurs
image = PIL_Image.open(buffer)
image.load()
return image.convert("RGB")

# GOOD - Context manager ensures cleanup
with PIL_Image.open(buffer) as image:
    image.load()
    return image.convert("RGB")
```

**Example - PyMuPDF**:
```python
# BAD - Resource leak if exception occurs
doc = pymupdf.open(path)
md_text = pymupdf4llm.to_markdown(doc)
input_files.append(md_text)

# GOOD - Try/finally ensures cleanup
doc = pymupdf.open(path)
try:
    md_text = pymupdf4llm.to_markdown(doc)
    input_files.append(md_text)
finally:
    doc.close()
```

**Impact**: Prevents file descriptor leaks, memory leaks, and "too many open files" errors in production.

**Files**: See related issues in `issues/code-quality/pil-image-resource-leak.md` and `issues/code-quality/pymupdf-resource-leak.md`

**Date**: 2026-02-18
